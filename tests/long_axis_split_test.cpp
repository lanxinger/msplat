#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "bindings.h"

namespace {

constexpr float kTol = 1e-4f;

bool approx(float a, float b, float tol = kTol) {
    return std::fabs(a - b) <= tol;
}

float logit(float x) {
    return std::log(x / (1.0f - x));
}

void require(bool cond, const char *message) {
    if (!cond) {
        std::cerr << "long_axis_split_test: " << message << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main() {
    constexpr int capacity = 6;
    constexpr int num_splits = 2;
    constexpr int dst_offset = 4;
    constexpr int fr_bases = 2;
    constexpr int fr_stride = fr_bases * 3;

    MTensor donor_indices = gpu_empty({num_splits}, DType::Int32);
    auto *donors = donor_indices.data<int32_t>();
    donors[0] = 0;
    donors[1] = 1;

    MTensor means_buf = gpu_zeros({capacity, 3}, DType::Float32);
    MTensor scales_buf = gpu_zeros({capacity, 3}, DType::Float32);
    MTensor quats_buf = gpu_zeros({capacity, 4}, DType::Float32);
    MTensor features_dc_buf = gpu_zeros({capacity, 3}, DType::Float32);
    MTensor features_rest_buf = gpu_zeros({capacity, fr_bases, 3}, DType::Float32);
    MTensor opacities_buf = gpu_zeros({capacity, 1}, DType::Float32);

    MTensor adam_exp_avg[6];
    MTensor adam_exp_avg_sq[6];
    const int group_strides[6] = {3, 3, 4, 3, fr_stride, 1};
    for (int g = 0; g < 6; ++g) {
        adam_exp_avg[g] = gpu_zeros({capacity, group_strides[g]}, DType::Float32);
        adam_exp_avg_sq[g] = gpu_zeros({capacity, group_strides[g]}, DType::Float32);
        float *ea = adam_exp_avg[g].data<float>();
        float *es = adam_exp_avg_sq[g].data<float>();
        for (int i = 0; i < capacity * group_strides[g]; ++i) {
            ea[i] = 10.0f + (float)g;
            es[i] = 20.0f + (float)g;
        }
    }

    float *means = means_buf.data<float>();
    float *scales = scales_buf.data<float>();
    float *quats = quats_buf.data<float>();
    float *fdc = features_dc_buf.data<float>();
    float *frest = features_rest_buf.data<float>();
    float *opac = opacities_buf.data<float>();

    means[0] = 1.0f; means[1] = 2.0f; means[2] = 3.0f;
    scales[0] = std::log(2.0f); scales[1] = 0.0f; scales[2] = std::log(0.5f);
    quats[0] = 1.0f; quats[1] = 0.0f; quats[2] = 0.0f; quats[3] = 0.0f;
    fdc[0] = 10.0f; fdc[1] = 11.0f; fdc[2] = 12.0f;
    for (int i = 0; i < fr_stride; ++i) frest[i] = 100.0f + (float)i;
    opac[0] = logit(0.5f);

    means[3] = -1.0f; means[4] = 5.0f; means[5] = 0.0f;
    scales[3] = std::log(4.0f); scales[4] = 0.0f; scales[5] = 0.0f;
    float q = std::sqrt(0.5f);
    quats[4] = q; quats[5] = 0.0f; quats[6] = 0.0f; quats[7] = q;
    fdc[3] = 20.0f; fdc[4] = 21.0f; fdc[5] = 22.0f;
    for (int i = 0; i < fr_stride; ++i) frest[fr_stride + i] = 200.0f + (float)i;
    opac[1] = logit(0.8f);

    msplat_long_axis_split(
        num_splits, dst_offset, fr_stride,
        donor_indices,
        means_buf, scales_buf, quats_buf,
        features_dc_buf, features_rest_buf, opacities_buf,
        adam_exp_avg, adam_exp_avg_sq
    );
    msplat_gpu_sync();

    const float log_half = std::log(0.5f);
    const float log_minor = std::log(0.85f);

    require(approx(means[0], 2.0f) && approx(means[1], 2.0f) && approx(means[2], 3.0f),
            "donor 0 first child mean mismatch");
    require(approx(means[12], 0.0f) && approx(means[13], 2.0f) && approx(means[14], 3.0f),
            "donor 0 second child mean mismatch");
    require(approx(scales[0], std::log(2.0f) + log_half), "donor 0 major-axis scale mismatch");
    require(approx(scales[1], log_minor), "donor 0 minor-axis y scale mismatch");
    require(approx(scales[2], std::log(0.5f) + log_minor), "donor 0 minor-axis z scale mismatch");
    require(approx(scales[12], scales[0]) && approx(scales[13], scales[1]) && approx(scales[14], scales[2]),
            "donor 0 child scales mismatch");
    require(approx(opac[0], logit(0.3f)) && approx(opac[4], logit(0.3f)),
            "donor 0 opacity mismatch");

    require(approx(means[3], -1.0f) && approx(means[4], 7.0f) && approx(means[5], 0.0f),
            "donor 1 first child mean mismatch");
    require(approx(means[15], -1.0f) && approx(means[16], 3.0f) && approx(means[17], 0.0f),
            "donor 1 second child mean mismatch");
    require(approx(scales[3], std::log(4.0f) + log_half), "donor 1 major-axis scale mismatch");
    require(approx(scales[4], log_minor), "donor 1 minor-axis y scale mismatch");
    require(approx(scales[5], log_minor), "donor 1 minor-axis z scale mismatch");
    require(approx(scales[15], scales[3]) && approx(scales[16], scales[4]) && approx(scales[17], scales[5]),
            "donor 1 child scales mismatch");
    require(approx(opac[1], logit(0.48f)) && approx(opac[5], logit(0.48f)),
            "donor 1 opacity mismatch");

    for (int c = 0; c < 4; ++c) {
        require(approx(quats[dst_offset * 4 + c], quats[c]), "child 0 quat copy mismatch");
        require(approx(quats[(dst_offset + 1) * 4 + c], quats[4 + c]), "child 1 quat copy mismatch");
    }
    for (int c = 0; c < 3; ++c) {
        require(approx(fdc[dst_offset * 3 + c], fdc[c]), "child 0 feature dc copy mismatch");
        require(approx(fdc[(dst_offset + 1) * 3 + c], fdc[3 + c]), "child 1 feature dc copy mismatch");
    }
    for (int c = 0; c < fr_stride; ++c) {
        require(approx(frest[dst_offset * fr_stride + c], frest[c]), "child 0 feature rest copy mismatch");
        require(approx(frest[(dst_offset + 1) * fr_stride + c], frest[fr_stride + c]), "child 1 feature rest copy mismatch");
    }

    for (int g = 0; g < 6; ++g) {
        float *ea = adam_exp_avg[g].data<float>();
        float *es = adam_exp_avg_sq[g].data<float>();
        int stride = group_strides[g];
        for (int idx : {0, 1, dst_offset, dst_offset + 1}) {
            for (int c = 0; c < stride; ++c) {
                require(approx(ea[idx * stride + c], 0.0f), "adam exp avg not zeroed");
                require(approx(es[idx * stride + c], 0.0f), "adam exp avg sq not zeroed");
            }
        }
    }

    cleanup_msplat_metal();
    std::cout << "long_axis_split_test: ok" << std::endl;
    return 0;
}
