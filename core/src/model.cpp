#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include "model.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

// Fenwick (binary indexed) tree for O(log N) weighted sampling without
// replacement.  Supports: build from weights, sample one index from the
// cumulative distribution, zero the chosen weight — all in O(log N).
struct FenwickSampler {
    std::vector<double> tree; // 1-indexed prefix sums
    int n;

    void build(const double* w, int len) {
        n = len;
        tree.assign(n + 1, 0.0);
        for (int i = 0; i < n; i++) tree[i + 1] = w[i];
        for (int i = 1; i <= n; i++) {
            int j = i + (i & -i);
            if (j <= n) tree[j] += tree[i];
        }
    }

    double total() const {
        double s = 0;
        for (int i = n; i > 0; i -= i & -i) s += tree[i];
        return s;
    }

    // Find smallest index whose prefix sum >= target. O(log N).
    int sample(double target) const {
        int pos = 0;
        for (int bit = 1 << (31 - __builtin_clz(n)); bit; bit >>= 1) {
            int next = pos + bit;
            if (next <= n && tree[next] < target) {
                target -= tree[next];
                pos = next;
            }
        }
        return std::min(pos, n - 1); // 0-indexed result
    }

    // Subtract delta from element at 0-indexed position idx.
    void update(int idx, double delta) {
        for (int i = idx + 1; i <= n; i += i & -i) tree[i] -= delta;
    }

    // Sample and remove: returns 0-indexed donor, zeroes its weight.
    int draw(double u) {
        int idx = sample(u);
        // Read the element weight = prefix(idx+1) - prefix(idx)
        double w = weight_at(idx);
        update(idx, w);
        return idx;
    }

    double weight_at(int idx) const {
        double s = tree[idx + 1];
        int z = idx + 1 - ((idx + 1) & -(idx + 1));
        for (int i = idx; i > z; i -= i & -i) s -= tree[i];
        return s;
    }
};

namespace {

constexpr float EDGE_SCORE_WEIGHT = 0.25f;
constexpr int EDGE_MIN_VIEW_SAMPLES = 10;
constexpr int IGS_ERROR_CANDIDATE_FACTOR = 4;

bool strategyUsesHybridRefine(Strategy strategy) {
    return strategy == Strategy::Hybrid || strategy == Strategy::MRNF;
}

inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float logitf(float p) {
    p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
    return std::log(p / (1.0f - p));
}

void expandVisCountsToActive(Model &model) {
    if (!model.visCounts.defined()) return;
    if (model.visCounts.size(0) >= model.num_active) return;
    int old_n = (int)model.visCounts.size(0);
    MTensor expanded = gpu_zeros({model.num_active}, DType::Float32);
    memcpy(expanded.data_ptr(), model.visCounts.data_ptr(), old_n * sizeof(float));
    model.visCounts = std::move(expanded);
}

float positiveMedian(const std::vector<float> &values) {
    std::vector<float> positives;
    positives.reserve(values.size());
    for (float v : values) {
        if (v > 0.0f) positives.push_back(v);
    }
    if (positives.empty()) return 1.0f;
    size_t mid = positives.size() / 2;
    std::nth_element(positives.begin(), positives.begin() + mid, positives.end());
    return std::max(positives[mid], 1e-6f);
}

std::vector<float> normalizeByPositiveMedian(const std::vector<float> &values) {
    std::vector<float> normalized(values);
    const float median = positiveMedian(values);
    const float inv_median = 1.0f / median;
    for (float &v : normalized) {
        if (v > 0.0f) v *= inv_median;
    }
    return normalized;
}

std::vector<int> sampleCameraIndices(int numCameras, int step, int minSamples = EDGE_MIN_VIEW_SAMPLES) {
    if (numCameras <= 0) return {};
    int numSamples = 0;
    if (numCameras < minSamples) {
        numSamples = numCameras;
    } else {
        const int minFromDataset = static_cast<int>(0.08f * static_cast<float>(numCameras));
        numSamples = std::max(minSamples, minFromDataset);
    }
    std::vector<int> indices(numCameras);
    std::iota(indices.begin(), indices.end(), 0);
    std::default_random_engine rng(static_cast<unsigned>(step));
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(numSamples);
    return indices;
}

std::vector<float> computePerSplatEdgeScores(Model &model, std::vector<Camera> &cameras, int step) {
    std::vector<float> scores(model.num_active, 0.0f);
    if (model.num_active <= 0 || cameras.empty()) return scores;

    const std::vector<int> sampleIndices = sampleCameraIndices(static_cast<int>(cameras.size()), step);
    if (sampleIndices.empty()) return scores;

    const int ds = model.getDownscaleFactor(step);
    std::vector<MTensor> perViewScores;
    perViewScores.reserve(sampleIndices.size());
    for (int cameraIndex : sampleIndices) {
        Camera &camera = cameras[cameraIndex];
        auto setup = model.prepareCam(camera, step);
        MTensor &edgeMap = camera.getGPUEdgeMap(ds);
        if (!edgeMap.defined() || edgeMap.size(0) != setup.height || edgeMap.size(1) != setup.width)
            continue;
        perViewScores.push_back(msplat_render_edge_scores(
            model.num_active, model.means, model.scales, 1.0f,
            model.quats, camera.cachedViewMat, camera.cachedProjViewMat,
            setup.fx, setup.fy, setup.cx, setup.cy,
            setup.height, setup.width, setup.tileBounds, 0.01f,
            setup.degree, setup.degreesToUse, setup.cam_pos,
            model.featuresDc, model.featuresRest,
            model.opacities, edgeMap));
    }
    if (perViewScores.empty()) return scores;

    msplat_gpu_sync();
    std::vector<float> viewScores(model.num_active);
    for (MTensor &viewTensor : perViewScores) {
        const float *viewData = viewTensor.data<float>();
        std::copy(viewData, viewData + model.num_active, viewScores.begin());
        std::vector<float> normalized = normalizeByPositiveMedian(viewScores);
        for (int i = 0; i < model.num_active; ++i) scores[i] += normalized[i];
    }

    const float invSamples = 1.0f / static_cast<float>(perViewScores.size());
    for (float &score : scores) score *= invSamples;
    return scores;
}

void schedulersStepClassicOrHybrid(Model &model, int step) {
    float t = std::clamp((float)step / (float)model.maxSteps, 0.f, 1.f);
    model.adam_lr[0] = std::exp(std::log(model.means_lr_init) * (1.f - t)
                              + std::log(model.means_lr_final) * t);
}

void afterTrainClassicOrHybrid(Model &model, int step) {
    if (!model.radii.defined()) return;
    const bool refineStep = step % model.refineEvery == 0 && step > model.warmupLength;
    if (model.strategy == Strategy::Hybrid && model.xysGradNorm.defined() && model.refineWeightMax.defined()) {
        msplat_accumulate_refine_max(model.num_active, model.xysGradNorm, model.refineWeightMax);
    }
    if (refineStep) {
        int resetInterval = model.resetAlphaEvery * model.refineEvery;
        bool doDensification = step < model.stopSplitAt
                            && step % resetInterval > model.numCameras + model.refineEvery;

        if (!model.hybridRefine && doDensification){
            int numPointsBefore = model.num_active;
            model.ensureCapacity(3 * model.num_active);  // worst case: every gaussian splits

            // Fill random samples for splits (CPU randn, shared memory)
            {
                std::mt19937 rng(step);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                float *p = model.densify_random_samples.data<float>();
                for (int64_t i = 0; i < 2 * model.num_active * 3; i++) p[i] = dist(rng);
            }

            float half_max_dim = 0.5f * static_cast<float>((std::max)(model.lastWidth, model.lastHeight));
            int check_screen = (step < model.stopScreenSizeAt) ? 1 : 0;
            bool checkHuge = step > model.refineEvery * model.resetAlphaEvery;
            int fr_stride = (int)model.featuresRest_buf.stride0();

            int new_count = msplat_densify(
                model.num_active, model.buf_capacity,
                model.densifyGradThresh, model.densifySizeThresh, model.splitScreenSize, check_screen,
                0.1f, 0.5f, 0.15f, checkHuge ? 1 : 0,
                model.xysGradNorm, model.visCounts, model.max2DSize, half_max_dim,
                model.means_buf, model.scales_buf, model.quats_buf,
                model.featuresDc_buf, model.featuresRest_buf, model.opacities_buf, fr_stride,
                model.adam_exp_avg_buf, model.adam_exp_avg_sq_buf,
                model.densify_split_flag, model.densify_dup_flag,
                model.densify_split_prefix, model.densify_dup_prefix,
                model.densify_keep_flag, model.densify_keep_prefix,
                model.densify_block_totals, model.densify_compact_scratch,
                model.densify_random_samples,
                0
            );

            model.num_active = new_count;
            model.refreshViews();
            std::cout << "Densified: " << numPointsBefore << " -> " << model.num_active << " gaussians" << std::endl;
        }

        if (!model.hybridRefine && !model.maskAwareData
            && step < model.stopSplitAt && step % resetInterval == model.refineEvery){
            msplat_gpu_sync();
            constexpr float resetLogit = -1.3862943611198906f;
            float *op = model.opacities.data<float>();
            for (int64_t i = 0; i < model.opacities.numel(); i++)
                if (op[i] > resetLogit) op[i] = resetLogit;

            model.adam_exp_avg[5].zero();
            model.adam_exp_avg_sq[5].zero();
            fprintf(stderr, "Opacity reset at step %d\n", step);
        }
    }

    if (model.hybridRefine && refineStep) {
        msplat_gpu_sync();
        static constexpr float MIN_OPACITY = 1.0f / 255.0f;
        float train_t = std::clamp((float)step / (float)model.maxSteps, 0.0f, 1.0f);
        float shrink_strength = 1.0f - train_t;
        float minus_opac = model.opacityDecay * shrink_strength;
        float scale_factor = 1.0f - model.scaleDecay * shrink_strength;

        float *op = model.opacities.data<float>();
        float *sc = model.scales.data<float>();
        float *mn = model.means.data<float>();
        int n = model.num_active;
        std::vector<float> savedVisCounts(n, 0.0f);
        std::vector<float> savedGradWeights(n, 0.0f);
        if (model.refineWeightMax.defined() && model.visCounts.defined()) {
            const float *gn = model.refineWeightMax.data<float>();
            const float *vc = model.visCounts.data<float>();
            for (int i = 0; i < n; ++i) {
                savedVisCounts[i] = vc[i];
                savedGradWeights[i] = (vc[i] > 0.0f) ? gn[i] : 0.0f;
            }
        }

        if (!model.boundsValid) {
            model.computeBounds();
            mn = model.means.data<float>();
            sc = model.scales.data<float>();
            op = model.opacities.data<float>();
        }

        if (minus_opac > 0 || scale_factor < 1.0f) {
            float log_sf = std::log(scale_factor);
            for (int i = 0; i < n; i++) {
                float sig = 1.0f / (1.0f + std::exp(-op[i]));
                float new_sig = std::clamp(sig - minus_opac, 1e-6f, 1.0f - 1e-6f);
                op[i] = std::log(new_sig / (1.0f - new_sig));
                sc[i*3] += log_sf; sc[i*3+1] += log_sf; sc[i*3+2] += log_sf;
            }
        }

        int pre_prune = model.num_active;
        int dst = 0;
        std::vector<float> compactVis(pre_prune, 0.0f);
        std::vector<float> compactGrad(pre_prune, 0.0f);
        float *qt = model.quats.data<float>();
        float *dc = model.featuresDc.data<float>();
        float *fr = model.featuresRest.data<float>();
        int fr_stride = (int)model.featuresRest_buf.stride0();
        int strides[6] = {3, 3, 4, 3, fr_stride, 1};
        float *param_ptrs[6] = {mn, sc, qt, dc, fr, op};
        float *adam_ea_ptrs[6], *adam_es_ptrs[6];
        for (int g = 0; g < Model::N_ADAM_GROUPS; g++) {
            adam_ea_ptrs[g] = model.adam_exp_avg[g].data<float>();
            adam_es_ptrs[g] = model.adam_exp_avg_sq[g].data<float>();
        }

        for (int i = 0; i < n; i++) {
            float sig = 1.0f / (1.0f + std::exp(-op[i]));
            float s0 = std::exp(sc[i*3]), s1 = std::exp(sc[i*3+1]), s2 = std::exp(sc[i*3+2]);
            float max_s = std::max({s0, s1, s2});
            float min_s = std::min({s0, s1, s2});
            float dx = mn[i*3] - model.bounds.center[0];
            float dy = mn[i*3+1] - model.bounds.center[1];
            float dz = mn[i*3+2] - model.bounds.center[2];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            float scale_limit = model.boundsValid ? model.bounds.maxExtent * 100.0f : model.scale * 100.0f;
            float dist_limit = model.boundsValid ? model.bounds.maxExtent * 100.0f : std::numeric_limits<float>::infinity();

            bool dead = sig < MIN_OPACITY || min_s < 1e-10f || max_s > scale_limit || dist > dist_limit;
            if (dead) continue;

            compactVis[dst] = savedVisCounts[i];
            compactGrad[dst] = savedGradWeights[i];

            if (dst != i) {
                for (int g = 0; g < Model::N_ADAM_GROUPS; g++) {
                    memmove(&param_ptrs[g][dst * strides[g]], &param_ptrs[g][i * strides[g]], strides[g] * sizeof(float));
                    memmove(&adam_ea_ptrs[g][dst * strides[g]], &adam_ea_ptrs[g][i * strides[g]], strides[g] * sizeof(float));
                    memmove(&adam_es_ptrs[g][dst * strides[g]], &adam_es_ptrs[g][i * strides[g]], strides[g] * sizeof(float));
                }
            }
            dst++;
        }

        if (dst < model.num_active) {
            model.num_active = dst;
            model.refreshViews();
        }
        compactVis.resize(model.num_active);
        compactGrad.resize(model.num_active);

        int pruned = pre_prune - model.num_active;
        int above_threshold_count = 0;
        int replacement = 0;
        int growth = 0;
        int selected_total = 0;
        if (model.num_active > 0) {
            replacement = std::min(pruned, model.num_active);
            if (model.maxSplats > 0)
                replacement = std::min(replacement, std::max(0, model.maxSplats - model.num_active));

            for (int i = 0; i < model.num_active; ++i) {
                if (compactVis[i] > 0.0f && compactGrad[i] > model.growthGradThreshold)
                    above_threshold_count++;
            }

            int total_growth = 0;
            if (step < model.growUntilIter) {
                total_growth = (int)std::lround((double)above_threshold_count * (double)model.growFraction);
                if (model.hybridGrowthFloorDivisor > 0.0f)
                    total_growth = std::max(total_growth, (int)(model.num_active / model.hybridGrowthFloorDivisor));
            }
            growth = std::max(0, total_growth - replacement);
            if (model.maxSplats > 0)
                growth = std::min(growth, std::max(0, model.maxSplats - (model.num_active + replacement)));
            growth = std::min(growth, std::max(0, model.num_active - replacement));

            int budget = replacement + growth;
            if (budget > 0) {
                model.ensureCapacity(model.num_active + budget);
                fr_stride = (int)model.featuresRest_buf.stride0();
                op = model.opacities_buf.data<float>();

                std::vector<int32_t> donorIdx(budget);
                std::vector<float> rndBuf(budget * 3);
                std::vector<uint8_t> chosen(model.num_active, 0);
                int selected = 0;

                auto sample_donors = [&](const std::vector<double> &weights, uint32_t seed, int count) {
                    if (count <= 0) return;
                    FenwickSampler sampler;
                    sampler.build(weights.data(), model.num_active);
                    double running = sampler.total();
                    if (running <= 0.0) return;

                    std::mt19937 rng(seed);
                    std::uniform_real_distribution<double> uni(0.0, 1.0);
                    std::normal_distribution<float> randn(0.0f, 1.0f);

                    for (int d = 0; d < count; ++d) {
                        running = sampler.total();
                        if (running <= 0.0) break;
                        int c = sampler.draw(uni(rng) * running);
                        donorIdx[selected] = c;
                        rndBuf[selected*3]   = randn(rng);
                        rndBuf[selected*3+1] = randn(rng);
                        rndBuf[selected*3+2] = randn(rng);
                        chosen[c] = 1;
                        selected++;
                    }
                };

                if (replacement > 0) {
                    std::vector<double> replaceWeights(model.num_active, 0.0);
                    for (int i = 0; i < model.num_active; ++i) {
                        if (compactVis[i] <= 0.0f) continue;
                        replaceWeights[i] = sigmoidf(op[i]);
                    }
                    sample_donors(replaceWeights, step ^ 0xABCD1234u, replacement);
                }

                if (growth > 0) {
                    std::vector<double> growthWeights(model.num_active, 0.0);
                    for (int i = 0; i < model.num_active; ++i) {
                        if (chosen[i] || compactVis[i] <= 0.0f || compactGrad[i] <= model.growthGradThreshold)
                            continue;
                        growthWeights[i] = compactGrad[i];
                    }
                    sample_donors(growthWeights, step ^ 0xDEADBEEFu, growth);
                }

                if (selected > 0) {
                    MTensor donorBuf = gpu_empty({(int64_t)selected}, DType::Int32);
                    memcpy(donorBuf.data_ptr(), donorIdx.data(), selected * sizeof(int32_t));
                    memcpy(model.densify_random_samples.data_ptr(), rndBuf.data(), selected * 3 * sizeof(float));
                    msplat_hybrid_refine(
                        model.num_active, selected, fr_stride,
                        donorBuf, model.densify_random_samples,
                        model.means_buf, model.scales_buf, model.quats_buf,
                        model.featuresDc_buf, model.featuresRest_buf, model.opacities_buf,
                        model.adam_exp_avg_buf, model.adam_exp_avg_sq_buf
                    );
                    model.num_active += selected;
                    model.refreshViews();
                }
                selected_total = selected;
            }
        }

        model.computeBounds();

        std::cout << "Hybrid refine: " << pre_prune << " -> " << model.num_active << " gaussians"
                  << " (pruned=" << pruned
                  << ", above=" << above_threshold_count
                  << ", replace=" << replacement
                  << ", growth=" << growth
                  << ", selected=" << selected_total
                  << ")" << std::endl;
    }

    if (refineStep) {
        model.xysGradNorm.reset();
        model.visCounts.reset();
        model.max2DSize.reset();
        if (model.refineWeightMax.defined()) model.refineWeightMax.zero();
    }
}

int compactActiveMRNF(Model &model) {
    int pre_prune = model.num_active;
    if (pre_prune <= 0) return 0;

    float min_opacity = 1.0f / 255.0f;
    float scale_limit = model.boundsValid ? model.bounds.maxExtent * 100.0f : model.scale * 100.0f;
    float dist_limit = model.boundsValid ? model.bounds.maxExtent * 100.0f : std::numeric_limits<float>::infinity();

    float *mn = model.means.data<float>();
    float *sc = model.scales.data<float>();
    float *qt = model.quats.data<float>();
    float *dc = model.featuresDc.data<float>();
    float *fr = model.featuresRest.data<float>();
    float *op = model.opacities.data<float>();
    float *vw = model.visCounts.defined() ? model.visCounts.data<float>() : nullptr;
    float *rw = model.refineWeightMax.defined() ? model.refineWeightMax.data<float>() : nullptr;
    int fr_stride = (int)model.featuresRest_buf.stride0();

    int param_strides[6] = {3, 3, 4, 3, fr_stride, 1};
    float *param_ptrs[6] = {mn, sc, qt, dc, fr, op};
    float *adam_ea_ptrs[Model::N_ADAM_GROUPS], *adam_es_ptrs[Model::N_ADAM_GROUPS];
    for (int g = 0; g < Model::N_ADAM_GROUPS; ++g) {
        adam_ea_ptrs[g] = model.adam_exp_avg[g].data<float>();
        adam_es_ptrs[g] = model.adam_exp_avg_sq[g].data<float>();
    }

    int dst = 0;
    for (int i = 0; i < pre_prune; ++i) {
        float sig = sigmoidf(op[i]);
        float sx = std::exp(sc[i*3]);
        float sy = std::exp(sc[i*3+1]);
        float sz = std::exp(sc[i*3+2]);
        float min_s = std::min({sx, sy, sz});
        float max_s = std::max({sx, sy, sz});
        float dx = mn[i*3] - model.bounds.center[0];
        float dy = mn[i*3+1] - model.bounds.center[1];
        float dz = mn[i*3+2] - model.bounds.center[2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

        bool dead = sig < min_opacity
                 || min_s < 1e-10f
                 || max_s > scale_limit
                 || dist > dist_limit;
        if (dead) continue;

        if (dst != i) {
            for (int g = 0; g < Model::N_ADAM_GROUPS; ++g) {
                memmove(&param_ptrs[g][dst * param_strides[g]], &param_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
                memmove(&adam_ea_ptrs[g][dst * param_strides[g]], &adam_ea_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
                memmove(&adam_es_ptrs[g][dst * param_strides[g]], &adam_es_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
            }
            if (vw) vw[dst] = vw[i];
            if (rw) rw[dst] = rw[i];
        }
        dst++;
    }

    if (dst < model.num_active) {
        model.num_active = dst;
        model.refreshViews();
    }
    return pre_prune - model.num_active;
}

void afterTrainMRNF(Model &model, std::vector<Camera> &cameras, int step) {
    if (!model.radii.defined()) return;
    if (step % model.refineEvery != 0 || step < model.warmupLength) return;
    if (!model.xysGradNorm.defined() || !model.visCounts.defined()) return;

    msplat_gpu_sync();

    float *refine = model.refineWeightMax.defined() ? model.refineWeightMax.data<float>() : nullptr;
    const float *gn = model.xysGradNorm.data<float>();
    const float *vc = model.visCounts.data<float>();
    if (!refine) return;

    for (int i = 0; i < model.num_active; ++i) {
        float score = (vc[i] > 0.0f) ? gn[i] : 0.0f;
        refine[i] = std::max(refine[i], score);
    }

    if (model.boundsValid && model.meanNoiseWeight > 0.0f
        && (model.noiseStopAt <= 0 || step < model.noiseStopAt)) {
        msplat_apply_mean_noise(
            model.num_active,
            model.meanNoiseWeight,
            model.adam_lr[0],
            model.means,
            model.scales,
            model.quats,
            model.opacities,
            model.radii,
            model.visCounts.defined() ? &model.visCounts : nullptr,
            model.bounds.medianSize,
            static_cast<uint32_t>(step ^ 0x9E3779B9u)
        );
    }

    if (!model.boundsValid || model.refineWindowsSinceBounds >= 5) {
        model.computeBounds();
        model.refineWindowsSinceBounds = 0;
    } else {
        model.refineWindowsSinceBounds++;
    }

    int pruned_count = compactActiveMRNF(model);
    std::vector<float> edge_scores = computePerSplatEdgeScores(model, cameras, step);
    std::vector<float> normalized_edge = normalizeByPositiveMedian(edge_scores);
    std::vector<double> edge_guidance(model.num_active, 1.0);
    for (int i = 0; i < model.num_active; ++i)
        edge_guidance[i] = 1.0 + static_cast<double>(normalized_edge[i]) * EDGE_SCORE_WEIGHT;

    float *op = model.opacities.data<float>();
    float *vis = model.visCounts.defined() ? model.visCounts.data<float>() : nullptr;
    float *weights_max = model.refineWeightMax.defined() ? model.refineWeightMax.data<float>() : nullptr;

    int above_threshold_count = 0;
    int replacement = 0;
    int growth = 0;
    int selected_splits = 0;

    if (step < model.growUntilIter && model.num_active > 0 && vis && weights_max) {
        int capacity_budget = (model.maxSplats > 0) ? std::max(0, model.maxSplats - model.num_active) : model.num_active;
        replacement = std::min({pruned_count, capacity_budget, model.num_active});

        for (int i = 0; i < model.num_active; ++i) {
            if (vis[i] > 0.0f && weights_max[i] > model.growthGradThreshold)
                above_threshold_count++;
        }

        int total_growth = (int)std::lround((double)above_threshold_count * (double)model.growFraction);
        growth = std::max(0, total_growth - replacement);
        growth = std::min(growth, std::max(0, capacity_budget - replacement));
        growth = std::min(growth, std::max(0, model.num_active - replacement));

        int total_splits = replacement + growth;
        if (total_splits > 0) {
            std::vector<int32_t> donors(total_splits);
            std::vector<uint8_t> chosen(model.num_active, 0);
            int selected = 0;

            if (replacement > 0) {
                std::vector<double> repl_weights(model.num_active, 0.0);
                for (int i = 0; i < model.num_active; ++i)
                    repl_weights[i] = (vis[i] > 0.0f) ? sigmoidf(op[i]) * edge_guidance[i] : 0.0f;
                FenwickSampler sampler;
                sampler.build(repl_weights.data(), model.num_active);
                std::mt19937 rng(step ^ 0x13579BDFu);
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                for (int i = 0; i < replacement; ++i) {
                    double total = sampler.total();
                    if (total <= 0.0) break;
                    int idx = sampler.draw(uni(rng) * total);
                    donors[selected++] = idx;
                    chosen[idx] = 1;
                }
            }

            if (growth > 0) {
                std::vector<double> grow_weights(model.num_active, 0.0);
                for (int i = 0; i < model.num_active; ++i) {
                    if (!chosen[i] && vis[i] > 0.0f && weights_max[i] > model.growthGradThreshold)
                        grow_weights[i] = static_cast<double>(weights_max[i]) * edge_guidance[i];
                }
                FenwickSampler sampler;
                sampler.build(grow_weights.data(), model.num_active);
                std::mt19937 rng(step ^ 0x2468ACE0u);
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                for (int i = 0; i < growth; ++i) {
                    double total = sampler.total();
                    if (total <= 0.0) break;
                    int idx = sampler.draw(uni(rng) * total);
                    donors[selected++] = idx;
                    chosen[idx] = 1;
                }
            }

            if (selected > 0) {
                model.ensureCapacity(model.num_active + selected);
                MTensor donor_buf = gpu_empty({(int64_t)selected}, DType::Int32);
                memcpy(donor_buf.data_ptr(), donors.data(), selected * sizeof(int32_t));
                int fr_stride = (int)model.featuresRest_buf.stride0();
                msplat_long_axis_split(
                    selected, model.num_active, fr_stride,
                    donor_buf,
                    model.means_buf, model.scales_buf, model.quats_buf,
                    model.featuresDc_buf, model.featuresRest_buf, model.opacities_buf,
                    model.adam_exp_avg_buf, model.adam_exp_avg_sq_buf
                );
                model.num_active += selected;
                model.refreshViews();
                expandVisCountsToActive(model);
            }
            selected_splits = selected;
        }
    }

    msplat_gpu_sync();
    static constexpr float MIN_OPACITY = 1.0f / 255.0f;
    float train_t = std::clamp((float)step / (float)model.maxSteps, 0.0f, 1.0f);
    float shrink_strength = 1.0f - train_t;
    float minus_opac = model.opacityDecay * shrink_strength;
    float scale_factor = 1.0f - model.scaleDecay * shrink_strength;
    op = model.opacities.data<float>();
    float *sc = model.scales.data<float>();

    if (minus_opac > 0.0f || scale_factor < 1.0f) {
        float log_sf = std::log(std::max(scale_factor, 1e-6f));
        for (int i = 0; i < model.num_active; ++i) {
            float sig = sigmoidf(op[i]);
            float new_sig = std::clamp(sig - minus_opac, 1e-6f, 1.0f - 1e-6f);
            op[i] = logitf(new_sig);
            sc[i*3] += log_sf;
            sc[i*3+1] += log_sf;
            sc[i*3+2] += log_sf;
        }
    }
    const int pruned_after_decay = 0;

    static int mrnf_debug_enabled = -1;
    if (mrnf_debug_enabled < 0)
        mrnf_debug_enabled = (std::getenv("MSPLAT_DEBUG_MRNF") != nullptr) ? 1 : 0;
    if (mrnf_debug_enabled) {
        float max_score = 0.0f;
        for (int i = 0; i < model.num_active; ++i)
            max_score = std::max(max_score, refine[i]);
        std::fprintf(stderr,
                     "[MRNF] step=%d active=%d pruned=%d above=%d replace=%d growth=%d selected=%d max_score=%.6f\n",
                     step, model.num_active, pruned_count, above_threshold_count,
                     replacement, growth, selected_splits, max_score);
    }

    if (model.refineWeightMax.defined()) model.refineWeightMax.zero();
    model.xysGradNorm.reset();
    model.visCounts.reset();
    model.max2DSize.reset();
}

int compactActiveIGSPlus(Model &model) {
    int pre_prune = model.num_active;
    if (pre_prune <= 0) return 0;

    float *mn = model.means.data<float>();
    float *sc = model.scales.data<float>();
    float *qt = model.quats.data<float>();
    float *dc = model.featuresDc.data<float>();
    float *fr = model.featuresRest.data<float>();
    float *op = model.opacities.data<float>();
    float *err = model.errorScoreMax.defined() ? model.errorScoreMax.data<float>() : nullptr;
    int fr_stride = (int)model.featuresRest_buf.stride0();

    int param_strides[6] = {3, 3, 4, 3, fr_stride, 1};
    float *param_ptrs[6] = {mn, sc, qt, dc, fr, op};
    float *adam_ea_ptrs[Model::N_ADAM_GROUPS], *adam_es_ptrs[Model::N_ADAM_GROUPS];
    for (int g = 0; g < Model::N_ADAM_GROUPS; ++g) {
        adam_ea_ptrs[g] = model.adam_exp_avg[g].data<float>();
        adam_es_ptrs[g] = model.adam_exp_avg_sq[g].data<float>();
    }

    int dst = 0;
    for (int i = 0; i < pre_prune; ++i) {
        float sig = sigmoidf(op[i]);
        float sx = std::exp(sc[i*3]);
        float sy = std::exp(sc[i*3+1]);
        float sz = std::exp(sc[i*3+2]);
        float max_s = std::max({sx, sy, sz});
        bool dead = sig < 0.005f || max_s > 0.1f;
        if (dead) continue;

        if (dst != i) {
            for (int g = 0; g < Model::N_ADAM_GROUPS; ++g) {
                memmove(&param_ptrs[g][dst * param_strides[g]], &param_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
                memmove(&adam_ea_ptrs[g][dst * param_strides[g]], &adam_ea_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
                memmove(&adam_es_ptrs[g][dst * param_strides[g]], &adam_es_ptrs[g][i * param_strides[g]], param_strides[g] * sizeof(float));
            }
            if (err) err[dst] = err[i];
        }
        dst++;
    }

    if (dst < model.num_active) {
        model.num_active = dst;
        model.refreshViews();
    }
    return pre_prune - model.num_active;
}

void afterTrainIGSPlus(Model &model, std::vector<Camera> &cameras, int step) {
    if (!model.radii.defined()) return;
    if (step % model.refineEvery != 0 || step < model.warmupLength) return;

    if (model.errorScoreMax.defined() && model.xysGradNorm.defined() && model.visCounts.defined()) {
        msplat_gpu_sync();
        float half_max_dim = 0.5f * static_cast<float>((std::max)(model.lastWidth, model.lastHeight));
        const float *gn = model.xysGradNorm.data<float>();
        const float *vc = model.visCounts.data<float>();
        float *err = model.errorScoreMax.data<float>();
        for (int i = 0; i < model.num_active; ++i) {
            float avg_grad = (vc[i] > 0.0f) ? (gn[i] / vc[i]) * half_max_dim : 0.0f;
            err[i] = std::max(err[i], avg_grad);
        }
    }

    if (step <= model.stopSplitAt) {
        if (model.igsCurrentStep < model.igsTotalSteps && model.num_active > 0) {
            int64_t budget = model.budgetSchedule.empty()
                           ? model.num_active
                           : model.budgetSchedule[std::min((int)model.budgetSchedule.size() - 1, model.igsCurrentStep + 1)];
            int budget_for_alloc = (int)std::max<int64_t>(0, budget - model.num_active);
            if (budget_for_alloc > 0 && model.errorScoreMax.defined()) {
                float *err = model.errorScoreMax.data<float>();
                std::vector<float> error_scores(model.num_active, 0.0f);
                for (int i = 0; i < model.num_active; ++i)
                    error_scores[i] = err[i];
                std::vector<float> normalized_error = normalizeByPositiveMedian(error_scores);
                std::vector<float> edge_scores = computePerSplatEdgeScores(model, cameras, step);
                std::vector<float> normalized_edge = normalizeByPositiveMedian(edge_scores);

                const int candidate_budget = std::min(
                    model.num_active,
                    std::max(budget_for_alloc, budget_for_alloc * IGS_ERROR_CANDIDATE_FACTOR));
                std::vector<uint8_t> candidate_mask(model.num_active, 1);
                if (candidate_budget < model.num_active) {
                    std::vector<float> sorted_error = normalized_error;
                    const auto nth = sorted_error.begin() + (candidate_budget - 1);
                    std::nth_element(sorted_error.begin(), nth, sorted_error.end(), std::greater<float>());
                    const float threshold = *nth;
                    for (int i = 0; i < model.num_active; ++i)
                        candidate_mask[i] = normalized_error[i] >= threshold ? 1 : 0;
                }

                std::vector<double> weights(model.num_active, 0.0);
                int selectable = 0;
                for (int i = 0; i < model.num_active; ++i) {
                    if (!candidate_mask[i]) continue;
                    weights[i] = static_cast<double>(normalized_error[i]) *
                                 (1.0 + static_cast<double>(normalized_edge[i]) * EDGE_SCORE_WEIGHT);
                    if (weights[i] > 0.0) selectable++;
                }

                if (selectable < budget_for_alloc) {
                    selectable = 0;
                    for (int i = 0; i < model.num_active; ++i) {
                        weights[i] = normalized_edge[i];
                        if (weights[i] > 0.0) selectable++;
                    }
                    if (selectable == 0) {
                        std::fill(weights.begin(), weights.end(), 1.0);
                        selectable = model.num_active;
                    }
                }

                FenwickSampler sampler;
                sampler.build(weights.data(), model.num_active);
                std::vector<int32_t> donors(budget_for_alloc);
                std::mt19937 rng(step ^ 0xBADC0FFEu);
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                int selected = 0;
                for (int i = 0; i < budget_for_alloc; ++i) {
                    double total = sampler.total();
                    if (total <= 0.0) break;
                    donors[selected++] = sampler.draw(uni(rng) * total);
                }

                if (selected > 0) {
                    model.ensureCapacity(model.num_active + selected);
                    MTensor donor_buf = gpu_empty({(int64_t)selected}, DType::Int32);
                    memcpy(donor_buf.data_ptr(), donors.data(), selected * sizeof(int32_t));
                    int fr_stride = (int)model.featuresRest_buf.stride0();
                    msplat_long_axis_split(
                        selected, model.num_active, fr_stride,
                        donor_buf,
                        model.means_buf, model.scales_buf, model.quats_buf,
                        model.featuresDc_buf, model.featuresRest_buf, model.opacities_buf,
                        model.adam_exp_avg_buf, model.adam_exp_avg_sq_buf
                    );
                    model.num_active += selected;
                    model.refreshViews();
                }
            }
        }

        compactActiveIGSPlus(model);
        model.igsCurrentStep++;
    }

    int resetInterval = model.resetAlphaEvery * model.refineEvery;
    if (resetInterval > 0 && step % resetInterval == 0) {
        msplat_gpu_sync();
        float resetLogit = logitf(0.1f);
        float *op = model.opacities.data<float>();
        for (int i = 0; i < model.num_active; ++i)
            if (op[i] > resetLogit) op[i] = resetLogit;
        model.adam_exp_avg[5].zero();
        model.adam_exp_avg_sq[5].zero();
    }

    if (model.errorScoreMax.defined()) model.errorScoreMax.zero();
    model.xysGradNorm.reset();
    model.visCounts.reset();
    model.max2DSize.reset();
}

}  // namespace


static const double C0 = 0.28209479177387814;

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt, const MTensor* mask) {
    int64_t pixels = rendered.numel() / 3;
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    const float *m = mask ? mask->data<float>() : nullptr;
    double wse = 0, wsum = 0;
    for (int64_t p = 0; p < pixels; p++) {
        float w = m ? m[p] : 1.0f;
        if (w <= 0.0f) continue;
        for (int c = 0; c < 3; c++) {
            double d = r[p*3+c] - g[p*3+c];
            wse += w * d * d;
        }
        wsum += w * 3.0;
    }
    if (wsum <= 0.0) return 0.0f;
    return 10.0f * std::log10(1.0 / (wse / wsum));
}

float l1_loss(const MTensor& rendered, const MTensor& gt, const MTensor* mask) {
    int64_t pixels = rendered.numel() / 3;
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    const float *m = mask ? mask->data<float>() : nullptr;
    double wsum_err = 0, wsum = 0;
    for (int64_t p = 0; p < pixels; p++) {
        float w = m ? m[p] : 1.0f;
        if (w <= 0.0f) continue;
        for (int c = 0; c < 3; c++)
            wsum_err += w * std::abs(r[p*3+c] - g[p*3+c]);
        wsum += w * 3.0;
    }
    if (wsum <= 0.0) return 0.0f;
    return (float)(wsum_err / wsum);
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs, float meanNoiseWeight, int noiseStopAt,
    bool mipSplatting,
    Strategy strategy, int maxSplats, float hybridGrowthFloorDivisor,
    float growthGradThreshold, float growFraction, int growUntilIter,
    float opacityDecay, float scaleDecay, float boundsPercentile,
    float scalesLrInit, float scalesLrFinal,
    const float* bgColor)
    : numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), keepCrs(keepCrs), mipSplatting(mipSplatting), meanNoiseWeight(meanNoiseWeight), noiseStopAt(noiseStopAt),
      strategy(strategy), hybridRefine(strategyUsesHybridRefine(strategy)), maxSplats(maxSplats),
      hybridGrowthFloorDivisor(hybridGrowthFloorDivisor),
      growthGradThreshold(growthGradThreshold), growFraction(growFraction), growUntilIter(growUntilIter),
      opacityDecay(opacityDecay), scaleDecay(scaleDecay), boundsPercentile(boundsPercentile),
      scaleLrCurrent(scalesLrInit),
      scaleLrGamma((scalesLrInit > 0.0f && scalesLrFinal > 0.0f)
            ? std::pow((double)scalesLrFinal / (double)scalesLrInit, 1.0 / std::max(1, maxSteps))
            : 1.0) {
    if (strategy == Strategy::MRNF) {
        if (this->refineEvery == 100) this->refineEvery = 200;
        if (this->warmupLength == 500) this->warmupLength = 0;
        this->stopSplitAt = std::min(this->maxSteps, 28500);
        // LichtFeld MRNF preset parity: growFraction 0.07, maxSplats 5M.
        if (this->growFraction == 0.2f) this->growFraction = 0.07f;
        if (this->maxSplats == 10000000) this->maxSplats = 5000000;
        if (scalesLrInit == 0.005f && scalesLrFinal == 0.00005f) {
            this->scaleLrCurrent = 7e-3f;
            this->scaleLrGamma = std::pow(5e-3 / 7e-3, 1.0 / std::max(1, this->maxSteps));
        }
    } else if (strategy == Strategy::IGSPlus) {
        if (this->refineEvery == 100 || this->refineEvery == 200) this->refineEvery = 500;
        // LichtFeld IGS+ preset parity: maxSplats 4M (override global default too).
        if (this->maxSplats <= 0 || this->maxSplats == 10000000) this->maxSplats = 4000000;
        this->stopSplitAt = std::min(this->maxSteps, 15000);
    }

    // Set mip-splatting mode (affects pipeline specialization)
    msplat_set_mip_splatting(mipSplatting);

    int64_t numPoints = inputData.points.count;
    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));
    maskAwareData = std::any_of(inputData.cameras.begin(), inputData.cameras.end(),
                                [](const Camera& cam) { return cam.hasMask(); });

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), inputData.points.xyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KD-tree nearest neighbor distances, log'd, repeated 3x
    {
        PointsTensor pt(inputData.points.xyz.data(), numPoints);
        auto sc = pt.scales();  // vector<float> of length numPoints
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float v = std::log(sc[i]);
            sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
        }
    }

    // Brush uses identity rotations for point-cloud initialization.
    {
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            qp[i*4+0] = 1.0f;
            qp[i*4+1] = 0.0f;
            qp[i*4+2] = 0.0f;
            qp[i*4+3] = 0.0f;
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        const uint8_t *rgb = inputData.points.rgb.data();
        for (int64_t i = 0; i < numPoints; i++) {
            for (int c = 0; c < 3; c++)
                dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // Brush initializes point-cloud splats at opacity 0.5.
    {
        float logit05 = 0.0f;
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < numPoints; i++) op[i] = logit05;
    }

    // Background color defaults to black, matching Brush masked-training behavior.
    backgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.0f, 0.0f, 0.0f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();


    num_active = means.size(0);
    buf_capacity = num_active * 4;
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    float lr_init[N_ADAM_GROUPS] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f};
    switch (strategy) {
        case Strategy::Classic:
            break;
        case Strategy::Hybrid:
            lr_init[0] = 2e-5f;
            lr_init[1] = 7e-3f;
            lr_init[2] = 2e-3f;
            lr_init[3] = 2e-3f;
            lr_init[4] = 2e-3f / 20.0f;  // SH rest: DC / 20 (matches Brush)
            lr_init[5] = 0.012f;
            break;
        case Strategy::MRNF:
            lr_init[0] = 2e-5f;
            lr_init[1] = 7e-3f;
            lr_init[2] = 2e-3f;
            lr_init[3] = 2e-3f;
            lr_init[4] = 2e-3f / 20.0f;  // SH rest: DC / 20 (matches Brush)
            lr_init[5] = 0.012f;
            break;
        case Strategy::IGSPlus:
            lr_init[0] = 1.6e-5f;
            lr_init[1] = 2e-2f;
            lr_init[2] = 1.5e-3f;
            lr_init[3] = 5e-3f;
            lr_init[4] = 5e-3f / 20.0f;  // SH rest: DC / 20 (matches Brush)
            lr_init[5] = 0.025f;
            break;
    }
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    if (strategy == Strategy::IGSPlus) {
        scaleLrCurrent = lr_init[1];
        scaleLrGamma = std::pow(0.1, 1.0 / std::max(1, maxSteps));
    }
    adam_lr[1] = scaleLrCurrent;
    adam_step_count = 0;
    means_lr_init = lr_init[0];
    means_lr_final = lr_init[0] * 0.01f;
    if (strategy == Strategy::MRNF) {
        means_lr_final = 2e-7f;
    } else if (strategy == Strategy::IGSPlus) {
        means_lr_final = 1.6e-7f;
    }
    meansLrUnscaled = means_lr_init;
    meansLrGamma = std::pow((double)means_lr_final / (double)means_lr_init, 1.0 / std::max(1, maxSteps));
    adam_lr[0] = meansLrUnscaled;

    densify_split_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_empty_private({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_empty_private({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);
    refineWeightMax = gpu_zeros({buf_capacity}, DType::Float32);
    errorScoreMax = gpu_zeros({buf_capacity}, DType::Float32);
    igsInitialPoints = num_active;
    igsCurrentStep = 0;
    computeBudgetSchedule();

    refreshViews();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    densify_split_flag.reset(); densify_dup_flag.reset();
    densify_split_prefix.reset(); densify_dup_prefix.reset();
    densify_keep_flag.reset(); densify_keep_prefix.reset();
    densify_block_totals.reset(); densify_compact_scratch.reset(); densify_random_samples.reset();
    refineWeightMax.reset();
    errorScoreMax.reset();
}

void Model::schedulersStep(int step){
    switch (strategy) {
        case Strategy::Classic:
        case Strategy::Hybrid:
            schedulersStepClassicOrHybrid(*this, step);
            break;
        case Strategy::MRNF:
            meansLrUnscaled *= meansLrGamma;
            scaleLrCurrent = (float)(scaleLrCurrent * scaleLrGamma);
            adam_lr[1] = scaleLrCurrent;
            adam_lr[0] = meansLrUnscaled * (boundsValid ? bounds.medianSize : 1.0f);
            break;
        case Strategy::IGSPlus: {
            double gamma = std::pow(0.1, 1.0 / std::max(1, maxSteps));
            adam_lr[0] = (float)(adam_lr[0] * gamma);
            adam_lr[1] = (float)(adam_lr[1] * gamma);
            break;
        }
    }
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

void Model::ensureCapacity(int needed){
    if (needed <= buf_capacity) return;
    int new_cap = std::max(needed, buf_capacity * 2);

    auto grow = [&](MTensor &buf) {
        auto shape = buf.shape();
        shape[0] = new_cap;
        MTensor new_buf = gpu_zeros(shape, DType::Float32);
        size_t copy_bytes = num_active * buf.stride0() * sizeof(float);
        memcpy(new_buf.data_ptr(), buf.data_ptr(), copy_bytes);
        buf = std::move(new_buf);
    };
    grow(means_buf); grow(scales_buf); grow(quats_buf);
    grow(featuresDc_buf); grow(featuresRest_buf); grow(opacities_buf);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        grow(adam_exp_avg_buf[g]);
        grow(adam_exp_avg_sq_buf[g]);
    }
    densify_split_flag = gpu_empty_private({new_cap}, DType::Int32);
    densify_dup_flag = gpu_empty_private({new_cap}, DType::Int32);
    densify_split_prefix = gpu_empty_private({new_cap}, DType::Int32);
    densify_dup_prefix = gpu_empty_private({new_cap}, DType::Int32);
    densify_keep_flag = gpu_empty_private({new_cap}, DType::Int32);
    densify_keep_prefix = gpu_zeros({new_cap}, DType::Int32);
    int max_blocks = (new_cap + 1023) / 1024;
    densify_block_totals = gpu_empty_private({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest_buf.stride0();
    densify_compact_scratch = gpu_empty_private({(int64_t)new_cap * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({new_cap, 3}, DType::Float32);
    {
        MTensor new_refine = gpu_zeros({new_cap}, DType::Float32);
        if (refineWeightMax.defined())
            memcpy(new_refine.data_ptr(), refineWeightMax.data_ptr(), num_active * sizeof(float));
        refineWeightMax = std::move(new_refine);
    }
    {
        MTensor new_error = gpu_zeros({new_cap}, DType::Float32);
        if (errorScoreMax.defined())
            memcpy(new_error.data_ptr(), errorScoreMax.data_ptr(), num_active * sizeof(float));
        errorScoreMax = std::move(new_error);
    }


    buf_capacity = new_cap;
    refreshViews();
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

void Model::afterTrain(std::vector<Camera>& cameras, int step){
    switch (strategy) {
        case Strategy::Classic:
        case Strategy::Hybrid:
            afterTrainClassicOrHybrid(*this, step);
            break;
        case Strategy::MRNF:
            afterTrainMRNF(*this, cameras, step);
            break;
        case Strategy::IGSPlus:
            afterTrainIGSPlus(*this, cameras, step);
            break;
    }
}

void Model::applyMaskOpacityPenalty(std::vector<Camera>& cameras, int step) {
    if (!maskAwareData || num_active <= 0) return;
    // Only run every refineEvery steps during training
    if (step % refineEvery != 0 || step <= warmupLength) return;

    msplat_gpu_sync();

    // LichtFeld-style: penalize opacity for splats whose centers project
    // into background (masked) regions. Sample a subset of cameras.
    static constexpr float PENALTY_WEIGHT = 0.02f;  // opacity reduction per step
    static constexpr int MAX_SAMPLE_CAMS = 8;

    std::mt19937 rng(step ^ 0x55AA55AAu);
    int nCams = (int)cameras.size();
    int nSample = std::min(nCams, MAX_SAMPLE_CAMS);

    // Pick random camera indices
    std::vector<int> camIdx(nCams);
    std::iota(camIdx.begin(), camIdx.end(), 0);
    std::shuffle(camIdx.begin(), camIdx.end(), rng);
    camIdx.resize(nSample);

    float *op = opacities.data<float>();
    const float *mn = means.data<float>();

    for (int i = 0; i < num_active; i++) {
        float px = mn[i*3], py = mn[i*3+1], pz = mn[i*3+2];
        float bg_sum = 0;
        int valid = 0;

        for (int ci = 0; ci < nSample; ci++) {
            Camera &cam = cameras[camIdx[ci]];
            int dsf = getDownscaleFactor(step);
            if (!cam.hasMask()) continue;

            // World-to-camera: invert camToWorld
            const float *d = cam.camToWorld;
            float R[3][3], T[3];
            for (int r = 0; r < 3; r++) {
                R[r][0] = d[r*4+0]; R[r][1] = -d[r*4+1]; R[r][2] = -d[r*4+2];
                T[r] = d[r*4+3];
            }
            // cam_pos = R^T * (point - T)
            float dx = px - T[0], dy = py - T[1], dz = pz - T[2];
            float cx = R[0][0]*dx + R[1][0]*dy + R[2][0]*dz;
            float cy = R[0][1]*dx + R[1][1]*dy + R[2][1]*dz;
            float cz = R[0][2]*dx + R[1][2]*dy + R[2][2]*dz;

            if (cz <= 0.01f) continue;  // behind camera
            float fx_s = cam.fx / dsf, fy_s = cam.fy / dsf;
            float cx_s = cam.cx / dsf, cy_s = cam.cy / dsf;
            int w = (int)(cam.width / dsf), h = (int)(cam.height / dsf);

            float u = fx_s * (cx / cz) + cx_s;
            float v = fy_s * (cy / cz) + cy_s;
            int iu = (int)(u + 0.5f), iv = (int)(v + 0.5f);
            if (iu < 0 || iu >= w || iv < 0 || iv >= h) continue;

            // Look up mask
            MTensor &maskTensor = cam.getGPUMask(dsf);
            const float *msk = maskTensor.data<float>();
            float mask_val = msk[iv * w + iu];
            bg_sum += (1.0f - mask_val);  // background weight
            valid++;
        }

        if (valid > 0) {
            float bg_frac = bg_sum / valid;  // 0=on object, 1=in background
            if (bg_frac > 0.3f) {
                // Apply penalty: stronger for more background-covered splats
                float penalty = PENALTY_WEIGHT * bg_frac * bg_frac;
                float sig = 1.0f / (1.0f + std::exp(-op[i]));
                float new_sig = std::clamp(sig - penalty, 1e-6f, 1.0f - 1e-6f);
                op[i] = std::log(new_sig / (1.0f - new_sig));
            }
        }
    }
}

void Model::applyMeanNoise(int step) {
    if (strategy != Strategy::Hybrid) return;
    if (!radii.defined() || meanNoiseWeight <= 0.0f) return;
    if (noiseStopAt > 0 && step >= noiseStopAt) return;

    const int numPoints = means.size(0);
    if (numPoints <= 0) return;

    // Noise is generated on-GPU via PCG hash + Box-Muller — no shared buffer,
    // no CPU write, no sync needed. Just pass the step as seed.
    msplat_apply_mean_noise(
        numPoints,
        meanNoiseWeight,
        adam_lr[0],
        means,
        scales,
        quats,
        opacities,
        radii,
        nullptr,
        0.0f,
        (uint32_t)(step ^ 0x9E3779B9u)
    );
}

void Model::computeBounds() {
    if (num_active <= 0) {
        bounds = SceneBounds{};
        boundsValid = false;
        return;
    }

    msplat_gpu_sync();
    const float *mn = means.data<float>();
    float p = std::clamp(boundsPercentile, 0.5f, 0.999f);
    int low_idx = (int)std::floor((double)(num_active - 1) * (1.0 - p));
    int high_idx = (int)std::floor((double)(num_active - 1) * p);

    for (int axis = 0; axis < 3; ++axis) {
        std::vector<float> vals(num_active);
        for (int i = 0; i < num_active; ++i) vals[i] = mn[i*3 + axis];
        std::nth_element(vals.begin(), vals.begin() + low_idx, vals.end());
        float lo = vals[low_idx];
        std::nth_element(vals.begin(), vals.begin() + high_idx, vals.end());
        float hi = vals[high_idx];
        bounds.center[axis] = 0.5f * (lo + hi);
        bounds.extent[axis] = std::max(0.5f * (hi - lo), 1e-6f);
    }

    float extents[3] = {bounds.extent[0], bounds.extent[1], bounds.extent[2]};
    std::sort(extents, extents + 3);
    bounds.medianSize = extents[1];
    bounds.maxExtent = std::max({bounds.extent[0], bounds.extent[1], bounds.extent[2], 1e-6f});
    boundsValid = bounds.medianSize > 0.0f && bounds.maxExtent > 0.0f;
}

void Model::computeBudgetSchedule() {
    budgetSchedule.clear();
    igsTotalSteps = 0;
    int64_t scheduleInitial = (igsInitialPoints > 0) ? igsInitialPoints : num_active;
    igsInitialPoints = scheduleInitial;

    if (strategy != Strategy::IGSPlus || maxSplats <= 0 || scheduleInitial <= 0) {
        budgetSchedule.push_back(num_active);
        igsCurrentStep = 0;
        return;
    }

    igsTotalSteps = std::max(1, ((stopSplitAt - warmupLength) / std::max(refineEvery, 1)) + 2);

    budgetSchedule.resize(igsTotalSteps + 1);
    budgetSchedule[0] = scheduleInitial;
    int64_t growth_total = std::max<int64_t>(0, (int64_t)maxSplats - scheduleInitial);
    double slope_lower_bound = (double)growth_total / (double)igsTotalSteps;
    double k = 2.0 * slope_lower_bound;
    double a = ((double)growth_total - k * (double)igsTotalSteps) /
               ((double)igsTotalSteps * (double)igsTotalSteps);
    double b = k;
    double c = (double)scheduleInitial;
    for (int i = 1; i <= igsTotalSteps; ++i) {
        double budget_f = a * (double)i * (double)i + b * (double)i + c;
        int64_t budget = (int64_t)std::llround(budget_f);
        budget = std::max<int64_t>(budget, scheduleInitial);
        budgetSchedule[i] = std::min<int64_t>(budget, maxSplats);
    }
    budgetSchedule.back() = maxSplats;
    igsCurrentStep = std::clamp(igsCurrentStep, 0, igsTotalSteps);
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step);
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSplat(filename, p);
}

int Model::loadPly(const std::string &filename){
    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
static constexpr uint32_t CKPT_VERSION = 2;

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static MTensor readTensor(std::ifstream &f) {
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++)
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    uint64_t bytes;
    f.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    MTensor t = gpu_empty(shape, DType::Float32);
    f.read(reinterpret_cast<char*>(t.data_ptr()), bytes);
    return t;
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg_sq[g]);

    // Version 2 trailer: strategy + strategy-specific runtime state.
    uint32_t strategy_u = static_cast<uint32_t>(strategy);
    uint32_t bounds_valid_u = boundsValid ? 1u : 0u;
    uint32_t refine_windows_u = static_cast<uint32_t>(std::max(refineWindowsSinceBounds, 0));
    uint32_t igs_step_u = static_cast<uint32_t>(std::max(igsCurrentStep, 0));
    f.write(reinterpret_cast<const char*>(&strategy_u), sizeof(strategy_u));
    f.write(reinterpret_cast<const char*>(&bounds_valid_u), sizeof(bounds_valid_u));
    f.write(reinterpret_cast<const char*>(bounds.center), sizeof(bounds.center));
    f.write(reinterpret_cast<const char*>(bounds.extent), sizeof(bounds.extent));
    f.write(reinterpret_cast<const char*>(&bounds.medianSize), sizeof(bounds.medianSize));
    f.write(reinterpret_cast<const char*>(&bounds.maxExtent), sizeof(bounds.maxExtent));
    f.write(reinterpret_cast<const char*>(&refine_windows_u), sizeof(refine_windows_u));
    f.write(reinterpret_cast<const char*>(&meansLrUnscaled), sizeof(meansLrUnscaled));
    f.write(reinterpret_cast<const char*>(&scaleLrCurrent), sizeof(scaleLrCurrent));
    f.write(reinterpret_cast<const char*>(&meansLrGamma), sizeof(meansLrGamma));
    f.write(reinterpret_cast<const char*>(&scaleLrGamma), sizeof(scaleLrGamma));
    f.write(reinterpret_cast<const char*>(&igs_step_u), sizeof(igs_step_u));
    f.write(reinterpret_cast<const char*>(&igsInitialPoints), sizeof(igsInitialPoints));

    f.close();
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians, "
              << fs::file_size(filename) / (1024*1024) << " MB)" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file: " + filename);

    // Header
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != CKPT_MAGIC) throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version < 1 || version > CKPT_VERSION)
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));

    // Scalar state
    uint32_t step, numPts, shDeg, adamSteps;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&numPts), sizeof(numPts));
    f.read(reinterpret_cast<char*>(&shDeg), sizeof(shDeg));
    f.read(reinterpret_cast<char*>(&adamSteps), sizeof(adamSteps));

    f.read(reinterpret_cast<char*>(adam_lr), sizeof(adam_lr));
    f.read(reinterpret_cast<char*>(&means_lr_init), sizeof(means_lr_init));
    f.read(reinterpret_cast<char*>(&means_lr_final), sizeof(means_lr_final));
    adam_step_count = (int)adamSteps;

    // Gaussian parameters — read into fresh tensors
    means = readTensor(f);
    scales = readTensor(f);
    quats = readTensor(f);
    featuresDc = readTensor(f);
    featuresRest = readTensor(f);
    opacities = readTensor(f);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg[g] = readTensor(f);
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg_sq[g] = readTensor(f);

    if (version >= 2) {
        uint32_t strategy_u = 0, bounds_valid_u = 0, refine_windows_u = 0, igs_step_u = 0;
        f.read(reinterpret_cast<char*>(&strategy_u), sizeof(strategy_u));
        f.read(reinterpret_cast<char*>(&bounds_valid_u), sizeof(bounds_valid_u));
        f.read(reinterpret_cast<char*>(bounds.center), sizeof(bounds.center));
        f.read(reinterpret_cast<char*>(bounds.extent), sizeof(bounds.extent));
        f.read(reinterpret_cast<char*>(&bounds.medianSize), sizeof(bounds.medianSize));
        f.read(reinterpret_cast<char*>(&bounds.maxExtent), sizeof(bounds.maxExtent));
        f.read(reinterpret_cast<char*>(&refine_windows_u), sizeof(refine_windows_u));
        f.read(reinterpret_cast<char*>(&meansLrUnscaled), sizeof(meansLrUnscaled));
        f.read(reinterpret_cast<char*>(&scaleLrCurrent), sizeof(scaleLrCurrent));
        f.read(reinterpret_cast<char*>(&meansLrGamma), sizeof(meansLrGamma));
        f.read(reinterpret_cast<char*>(&scaleLrGamma), sizeof(scaleLrGamma));
        f.read(reinterpret_cast<char*>(&igs_step_u), sizeof(igs_step_u));
        f.read(reinterpret_cast<char*>(&igsInitialPoints), sizeof(igsInitialPoints));

        if (strategy_u > static_cast<uint32_t>(Strategy::IGSPlus))
            throw std::runtime_error("Unsupported checkpoint strategy: " + std::to_string(strategy_u));
        strategy = static_cast<Strategy>(strategy_u);
        boundsValid = bounds_valid_u != 0;
        refineWindowsSinceBounds = static_cast<int>(refine_windows_u);
        igsCurrentStep = static_cast<int>(igs_step_u);
    } else {
        strategy = hybridRefine ? Strategy::Hybrid : Strategy::Classic;
        bounds = SceneBounds{};
        boundsValid = false;
        refineWindowsSinceBounds = 0;
        meansLrUnscaled = adam_lr[0];
        scaleLrCurrent = adam_lr[1];
        meansLrGamma = std::pow((double)means_lr_final / (double)means_lr_init, 1.0 / std::max(1, maxSteps));
        igsCurrentStep = 0;
        igsInitialPoints = numPts;
    }
    hybridRefine = strategyUsesHybridRefine(strategy);
    if (strategy == Strategy::IGSPlus && maxSplats <= 0)
        throw std::runtime_error("IGSPlus checkpoint requires maxSplats > 0 in the current config");

    f.close();

    // Rebuild backing buffers with loaded data (don't call setupOptimizers —
    // it would zero the optimizer state we just loaded)
    num_active = (int)numPts;
    buf_capacity = num_active * 4;

    // Copy gaussian params into oversized backing buffers
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    // Copy optimizer state into oversized backing buffers
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = adam_exp_avg[g].shape();
        shape[0] = buf_capacity;
        MTensor avg_buf = gpu_zeros(shape, DType::Float32);
        MTensor sq_buf = gpu_zeros(shape, DType::Float32);
        memcpy(avg_buf.data_ptr(), adam_exp_avg[g].data_ptr(), adam_exp_avg[g].nbytes());
        memcpy(sq_buf.data_ptr(), adam_exp_avg_sq[g].data_ptr(), adam_exp_avg_sq[g].nbytes());
        adam_exp_avg_buf[g] = std::move(avg_buf);
        adam_exp_avg_sq_buf[g] = std::move(sq_buf);
    }

    // Allocate densification scratch buffers
    densify_split_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_empty_private({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_empty_private({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_empty_private({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);
    refineWeightMax = gpu_zeros({buf_capacity}, DType::Float32);
    errorScoreMax = gpu_zeros({buf_capacity}, DType::Float32);
    if (strategy != Strategy::IGSPlus || igsInitialPoints <= 0) igsInitialPoints = num_active;
    if (strategy != Strategy::IGSPlus) igsCurrentStep = 0;
    computeBudgetSchedule();

    refreshViews();

    std::cout << "Checkpoint loaded: " << filename << " (step " << step
              << ", " << num_active << " gaussians)" << std::endl;

    return (int)step;
}

Model::CamSetup Model::prepareCam(Camera& cam, int step) {
    const float sf = getDownscaleFactor(step);
    CamSetup s;
    s.fx = cam.fx / sf; s.fy = cam.fy / sf;
    s.cx = cam.cx / sf; s.cy = cam.cy / sf;
    s.height = static_cast<int>(cam.height / sf);
    s.width = static_cast<int>(cam.width / sf);

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    if (!cam.cachedViewMat.defined() || cam.cachedFovX != fovX || cam.cachedFovY != fovY) {
        const float *d = cam.camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] = d[i*4+0]; R[i][1] = -d[i*4+1]; R[i][2] = -d[i*4+2]; T[i] = d[i*4+3];
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++) Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);
        float vm[16] = { Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1 };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        cam.cachedViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0] = T[0]; cam.cachedCamPos[1] = T[1]; cam.cachedCamPos[2] = T[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
    }

    s.degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

MTensor Model::render(Camera& cam, int step){
    auto s = prepareCam(cam, step);
    return msplat_render(
        means.size(0), means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor);
}

void Model::fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight, MTensor *mask){
    auto s = prepareCam(cam, step);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    // Final 10% of training: switch to pure L1 for tighter per-pixel accuracy (matches Brush aux_loss_time=0.9)
    constexpr float auxLossTime = 0.9f;
    if (maxSteps > 0 && step > (int)(auxLossTime * maxSteps))
        ssimWeight = 0.0f;

    // Initialize SSIM window (once)
    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
    MTensor adam_p[N_ADAM_GROUPS];
    MTensor adam_ea[N_ADAM_GROUPS], adam_eas[N_ADAM_GROUPS];
    float adam_ss[N_ADAM_GROUPS], adam_bc2s[N_ADAM_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    if (!xysGradNorm.defined()) {
    
        xysGradNorm = gpu_zeros({numPoints}, DType::Float32);
        visCounts = gpu_zeros({numPoints}, DType::Float32);
        max2DSize = gpu_zeros({numPoints}, DType::Float32);
    }

    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    float lossInvN = 1.0f / (float)(s.height * s.width * 3);
    int densificationMode = strategy == Strategy::Hybrid ? 2 : (strategyUsesHybridRefine(strategy) ? 1 : 0);

    // Background noise: add per-step random perturbation to prevent splats from
    // relying on a fixed background color (matches Brush background_noise_strength=0.1).
    constexpr float bgNoiseStrength = 0.1f;
    float *bgPtr = backgroundColor.data<float>();
    float bgOrig[3] = {bgPtr[0], bgPtr[1], bgPtr[2]};
    {
        std::mt19937 rng(step ^ 0xB601u);
        std::uniform_real_distribution<float> dist(-bgNoiseStrength, bgNoiseStrength);
        for (int c = 0; c < 3; c++)
            bgPtr[c] = std::clamp(bgPtr[c] + dist(rng), 0.0f, 1.0f);
    }
    MTensor trainBackground = backgroundColor;

    MTensor r = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, trainBackground, gt, window2d, ssimWeight,
        lossInvN, (int)featuresRest.size(-2),
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        visCounts, xysGradNorm, max2DSize, invMaxDim, densificationMode, mask);

    // Restore original background color after noisy training step
    bgPtr[0] = bgOrig[0]; bgPtr[1] = bgOrig[1]; bgPtr[2] = bgOrig[2];

    radii = r;
    applyMeanNoise(step);
}
