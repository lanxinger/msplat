#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

static fs::path resolveMasksDir(const std::string &imagePath, const std::string &maskDir) {
    if (!maskDir.empty()) return fs::path(maskDir);
    fs::path imgPath(imagePath);
    return imgPath.parent_path().parent_path() / "masks";
}

static std::shared_ptr<const std::unordered_map<std::string, std::string>>
getMaskIndex(const fs::path &masksDir) {
    static std::mutex cacheMutex;
    static std::unordered_map<std::string,
        std::shared_ptr<const std::unordered_map<std::string, std::string>>> cache;

    std::string key = masksDir.lexically_normal().string();
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }

    auto index = std::make_shared<std::unordered_map<std::string, std::string>>();
    if (fs::exists(masksDir) && fs::is_directory(masksDir)) {
        for (const auto &entry : fs::directory_iterator(masksDir)) {
            if (!entry.is_regular_file()) continue;
            index->try_emplace(entry.path().stem().string(), entry.path().string());
        }
    }

    std::lock_guard<std::mutex> lock(cacheMutex);
    auto [it, inserted] = cache.emplace(key, index);
    if (!inserted) return it->second;
    return index;
}

// Try to find a mask file matching the image filename stem.
// If maskDir is provided, search there; otherwise look for a sibling masks/ directory.
static std::string findMaskPath(const std::string &imagePath, const std::string &maskDir) {
    fs::path imgPath(imagePath);
    std::string stem = imgPath.stem().string();
    fs::path masksDir = resolveMasksDir(imagePath, maskDir);
    auto index = getMaskIndex(masksDir);
    auto it = index->find(stem);
    if (it != index->end()) return it->second;
    return "";
}

static float positiveMedian(std::vector<float> values) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](float v) { return !(v > 0.0f); }),
        values.end());
    if (values.empty()) return 1.0f;
    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return std::max(values[mid], 1e-6f);
}

static MTensor buildEdgeMapFromImage(const MTensor &image) {
    const int h = static_cast<int>(image.size(0));
    const int w = static_cast<int>(image.size(1));
    const float *rgb = image.data<float>();

    std::vector<float> gray(h * w, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int idx = (y * w + x) * 3;
            gray[y * w + x] = 0.299f * rgb[idx] + 0.587f * rgb[idx + 1] + 0.114f * rgb[idx + 2];
        }
    }

    // Match the LFS edge-guidance intent closely enough for scoring:
    // light Gaussian smoothing, Sobel gradients, then Canny-style NMS.
    static constexpr float kBlur[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
    auto clampX = [w](int x) { return std::clamp(x, 0, w - 1); };
    auto clampY = [h](int y) { return std::clamp(y, 0, h - 1); };

    std::vector<float> blurX(h * w, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -2; k <= 2; ++k)
                acc += gray[y * w + clampX(x + k)] * kBlur[k + 2];
            blurX[y * w + x] = acc;
        }
    }

    std::vector<float> blur(h * w, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -2; k <= 2; ++k)
                acc += blurX[clampY(y + k) * w + x] * kBlur[k + 2];
            blur[y * w + x] = acc;
        }
    }

    std::vector<float> magnitude(h * w, 0.0f);
    std::vector<uint8_t> direction(h * w, 0);
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float gx =
                -blur[(y - 1) * w + (x - 1)] + blur[(y - 1) * w + (x + 1)] +
                -2.0f * blur[y * w + (x - 1)] + 2.0f * blur[y * w + (x + 1)] +
                -blur[(y + 1) * w + (x - 1)] + blur[(y + 1) * w + (x + 1)];
            float gy =
                 blur[(y - 1) * w + (x - 1)] + 2.0f * blur[(y - 1) * w + x] + blur[(y - 1) * w + (x + 1)] -
                 blur[(y + 1) * w + (x - 1)] - 2.0f * blur[(y + 1) * w + x] - blur[(y + 1) * w + (x + 1)];
            float mag = std::sqrt(gx * gx + gy * gy);
            magnitude[y * w + x] = mag;

            float angle = std::atan2(gy, gx) * (180.0f / static_cast<float>(M_PI));
            if (angle < 0.0f) angle += 180.0f;
            if (angle < 22.5f || angle >= 157.5f) direction[y * w + x] = 0;
            else if (angle < 67.5f) direction[y * w + x] = 1;
            else if (angle < 112.5f) direction[y * w + x] = 2;
            else direction[y * w + x] = 3;
        }
    }

    MTensor edges = gpu_zeros({h, w}, DType::Float32);
    float *out = edges.data<float>();
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const int idx = y * w + x;
            const float mag = magnitude[idx];
            float n0 = 0.0f, n1 = 0.0f;
            switch (direction[idx]) {
                case 0:
                    n0 = magnitude[idx - 1];
                    n1 = magnitude[idx + 1];
                    break;
                case 1:
                    n0 = magnitude[(y - 1) * w + (x + 1)];
                    n1 = magnitude[(y + 1) * w + (x - 1)];
                    break;
                case 2:
                    n0 = magnitude[(y - 1) * w + x];
                    n1 = magnitude[(y + 1) * w + x];
                    break;
                default:
                    n0 = magnitude[(y - 1) * w + (x - 1)];
                    n1 = magnitude[(y + 1) * w + (x + 1)];
                    break;
            }
            out[idx] = (mag >= n0 && mag >= n1) ? mag : 0.0f;
        }
    }

    std::vector<float> positives(out, out + static_cast<size_t>(h) * static_cast<size_t>(w));
    const float median = positiveMedian(std::move(positives));
    const float invMedian = 1.0f / median;
    for (int i = 0; i < h * w; ++i)
        if (out[i] > 0.0f) out[i] *= invMedian;
    return edges;
}

void Camera::loadImage(float downscaleFactor, const std::string &maskDir) {
    // Save original metadata dimensions for computing final target
    int metaW = width, metaH = height;

    // Decode close to the final target when metadata dimensions are known.
    // Any residual 1px mismatch still goes through the regular resize path below.
    int thumbMaxDim = 0;
    if (downscaleFactor > 1.0f && metaW > 0 && metaH > 0) {
        int targetW = (int)(metaW / downscaleFactor);
        int targetH = (int)(metaH / downscaleFactor);
        thumbMaxDim = std::max(targetW, targetH);
    }

    Image raw = imreadRGB(filePath, thumbMaxDim);
    if (raw.empty()) return;

    // Try loading a matching mask (with same subsample hint). If no explicit
    // mask exists, fall back to the image alpha channel when present.
    std::string maskPath = findMaskPath(filePath, maskDir);
    Mask rawMask;
    if (!maskPath.empty()) {
        rawMask = imreadMask(maskPath, thumbMaxDim);
        if (!rawMask.empty()) maskPath_ = maskPath;
    } else {
        rawMask = imreadAlphaMask(filePath, thumbMaxDim);
        if (!rawMask.empty()) maskFromAlpha_ = true;
    }

    // Ensure mask matches image dimensions (resize if mismatched)
    if (!rawMask.empty() && (rawMask.width != raw.width || rawMask.height != raw.height))
        rawMask = resizeAreaMask(rawMask, raw.width, raw.height);

    // Rescale intrinsics from metadata to decoded dimensions
    if (metaW > 0 && metaH > 0 && (raw.width != metaW || raw.height != metaH)) {
        float sx = (float)raw.width / (float)metaW;
        float sy = (float)raw.height / (float)metaH;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = raw.width; height = raw.height;
    } else if (metaW == 0 || metaH == 0) {
        metaW = raw.width; metaH = raw.height;
        width = raw.width; height = raw.height;
    }

    // Downscale to final target (computed from original metadata, not intermediate)
    if (downscaleFactor > 1.0f) {
        int newW = (int)(metaW / downscaleFactor);
        int newH = (int)(metaH / downscaleFactor);
        if (newW != width || newH != height) {
            raw = resizeArea(raw, newW, newH);
            if (!rawMask.empty()) rawMask = resizeAreaMask(rawMask, newW, newH);
        }
        float sx = (float)newW / (float)width;
        float sy = (float)newH / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = newW; height = newH;
    }

    // Undistort if needed — save state for lazy reload
    if (hasDistortion()) {
        reload_.hadDistortion = true;
        reload_.k1 = k1; reload_.k2 = k2; reload_.k3 = k3;
        reload_.p1 = p1; reload_.p2 = p2;
        reload_.fx = fx; reload_.fy = fy;
        reload_.cx = cx; reload_.cy = cy;
        reload_.width = width; reload_.height = height;

        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        raw = std::move(result.image);

        reload_.roiX = result.roiX; reload_.roiY = result.roiY;
        reload_.roiW = result.width; reload_.roiH = result.height;

        if (!rawMask.empty())
            rawMask = undistortMask(rawMask, fx, fy, cx, cy, k1, k2, p1, p2, k3,
                                    result.roiX, result.roiY, result.width, result.height);
        fx = result.fx; fy = result.fy;
        cx = result.cx; cy = result.cy;
        width = result.width; height = result.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    image = std::move(raw);
    if (!rawMask.empty()) mask = std::move(rawMask);
}

void Camera::loadMetadataOnly(float downscaleFactor, const std::string &maskDir) {
    int metaW = width, metaH = height;

    // Read actual file dimensions to correct intrinsics when metadata
    // doesn't match (e.g. --colmap-image-path points to resized copies).
    int fileW = 0, fileH = 0;
    if (imageFileDimensions(filePath, fileW, fileH)) {
        if (metaW <= 0 || metaH <= 0) {
            metaW = fileW; metaH = fileH;
            width = fileW; height = fileH;
        } else if (fileW != metaW || fileH != metaH) {
            float sx = (float)fileW / (float)metaW;
            float sy = (float)fileH / (float)metaH;
            fx *= sx; fy *= sy; cx *= sx; cy *= sy;
            metaW = fileW; metaH = fileH;
            width = fileW; height = fileH;
        }
    } else if (metaW <= 0 || metaH <= 0) {
        // Can't determine dimensions at all — fall back to full decode.
        loadImage(downscaleFactor, maskDir);
        releaseCPUData();
        return;
    }

    // Detect mask file (filename matching only, no pixel decode)
    std::string maskPath = findMaskPath(filePath, maskDir);
    if (!maskPath.empty()) {
        maskPath_ = maskPath;
    } else if (imageHasAlphaChannel(filePath)) {
        maskFromAlpha_ = true;
    }

    // Apply downscale to intrinsics
    if (downscaleFactor > 1.0f) {
        int newW = (int)(metaW / downscaleFactor);
        int newH = (int)(metaH / downscaleFactor);
        float sx = (float)newW / (float)width;
        float sy = (float)newH / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = newW; height = newH;
    }

    // Compute undistortion geometry (no pixel data needed)
    if (hasDistortion()) {
        reload_.hadDistortion = true;
        reload_.k1 = k1; reload_.k2 = k2; reload_.k3 = k3;
        reload_.p1 = p1; reload_.p2 = p2;
        reload_.fx = fx; reload_.fy = fy;
        reload_.cx = cx; reload_.cy = cy;
        reload_.width = width; reload_.height = height;

        auto roi = computeUndistortROI(width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3);

        reload_.roiX = roi.roiX; reload_.roiY = roi.roiY;
        reload_.roiW = roi.width; reload_.roiH = roi.height;

        fx = roi.fx; fy = roi.fy;
        cx = roi.cx; cy = roi.cy;
        width = roi.width; height = roi.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }
}

void Camera::reloadImage() {
    if (filePath.empty() || !image.empty()) return;

    // Use maxDim=0 (full decode) to match the original loadImage path exactly.
    // The thumbnail decode path (maxDim>0) can produce subtly different pixels.
    Image raw = imreadRGB(filePath, 0);
    if (raw.empty()) return;

    if (reload_.hadDistortion) {
        if (raw.width != reload_.width || raw.height != reload_.height)
            raw = resizeArea(raw, reload_.width, reload_.height);
        auto result = undistortImage(raw,
            reload_.fx, reload_.fy, reload_.cx, reload_.cy,
            reload_.k1, reload_.k2, reload_.p1, reload_.p2, reload_.k3);
        raw = std::move(result.image);
        // Ensure exact match (ROI crop may differ by ±1px across runs)
        if (raw.width != width || raw.height != height)
            raw = resizeArea(raw, width, height);
    } else {
        if (raw.width != width || raw.height != height)
            raw = resizeArea(raw, width, height);
    }

    image = std::move(raw);
}

void Camera::reloadMask() {
    if (!mask.empty()) return;
    if (maskPath_.empty() && !maskFromAlpha_) return;

    // Use maxDim=0 (full decode) to match the original loadImage path.
    Mask rawMask;
    if (!maskPath_.empty())
        rawMask = imreadMask(maskPath_, 0);
    else
        rawMask = imreadAlphaMask(filePath, 0);
    if (rawMask.empty()) {
        // Alpha channel exists but is fully opaque — not a real mask.
        maskFromAlpha_ = false;
        return;
    }

    if (reload_.hadDistortion) {
        if (rawMask.width != reload_.width || rawMask.height != reload_.height)
            rawMask = resizeAreaMask(rawMask, reload_.width, reload_.height);
        rawMask = undistortMask(rawMask,
            reload_.fx, reload_.fy, reload_.cx, reload_.cy,
            reload_.k1, reload_.k2, reload_.p1, reload_.p2, reload_.k3,
            reload_.roiX, reload_.roiY, reload_.roiW, reload_.roiH);
        if (rawMask.width != width || rawMask.height != height)
            rawMask = resizeAreaMask(rawMask, width, height);
    } else {
        if (rawMask.width != width || rawMask.height != height)
            rawMask = resizeAreaMask(rawMask, width, height);
    }

    mask = std::move(rawMask);
}

Image Camera::getImage(int downscaleFactor) {
    // If base image was released and we need a downscaled version,
    // reload directly at target size without storing the full-res base.
    if (image.empty() && !filePath.empty() && downscaleFactor > 1) {
        auto it = imagePyramids.find(downscaleFactor);
        if (it != imagePyramids.end()) return it->second;
        reloadImage();
        int newW = image.width / downscaleFactor;
        int newH = image.height / downscaleFactor;
        Image scaled = resizeArea(image, newW, newH);
        image.data.clear(); image.data.shrink_to_fit();  // don't keep full-res
        imagePyramids[downscaleFactor] = std::move(scaled);
        return imagePyramids[downscaleFactor];
    }
    if (image.empty() && !filePath.empty()) reloadImage();
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = image.width / downscaleFactor;
    int newH = image.height / downscaleFactor;
    Image scaled = resizeArea(image, newW, newH);
    imagePyramids[downscaleFactor] = scaled;
    return scaled;
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    auto it = mtensorImageCache.find(downscaleFactor);
    if (it != mtensorImageCache.end()) return it->second;
    Image img = getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    memcpy(mt.data_ptr(), img.ptr(), img.width * img.height * 3 * sizeof(float));
    mtensorImageCache[downscaleFactor] = std::move(mt);
    return mtensorImageCache[downscaleFactor];
}

Mask Camera::getMask(int downscaleFactor) {
    if (mask.empty() && (maskPath_.size() || maskFromAlpha_) && downscaleFactor > 1) {
        auto it = maskPyramids.find(downscaleFactor);
        if (it != maskPyramids.end()) return it->second;
        reloadMask();
        if (mask.empty()) return {};
        int newW = mask.width / downscaleFactor;
        int newH = mask.height / downscaleFactor;
        Mask scaled = resizeAreaMask(mask, newW, newH);
        mask.data.clear(); mask.data.shrink_to_fit();
        maskPyramids[downscaleFactor] = std::move(scaled);
        return maskPyramids[downscaleFactor];
    }
    if (mask.empty()) reloadMask();
    if (mask.empty()) return {};
    if (downscaleFactor <= 1) return mask;

    auto it = maskPyramids.find(downscaleFactor);
    if (it != maskPyramids.end()) return it->second;

    int newW = mask.width / downscaleFactor;
    int newH = mask.height / downscaleFactor;
    Mask scaled = resizeAreaMask(mask, newW, newH);
    maskPyramids[downscaleFactor] = scaled;
    return scaled;
}

MTensor& Camera::getGPUMask(int downscaleFactor) {
    auto it = mtensorMaskCache.find(downscaleFactor);
    if (it != mtensorMaskCache.end()) return it->second;
    Mask m = getMask(downscaleFactor);
    MTensor mt = gpu_empty({m.height, m.width, 1}, DType::Float32);
    memcpy(mt.data_ptr(), m.ptr(), m.width * m.height * sizeof(float));
    mtensorMaskCache[downscaleFactor] = std::move(mt);
    return mtensorMaskCache[downscaleFactor];
}

MTensor& Camera::getGPUEdgeMap(int downscaleFactor) {
    auto it = mtensorEdgeCache.find(downscaleFactor);
    if (it != mtensorEdgeCache.end()) return it->second;
    MTensor &imageTensor = getGPUImage(downscaleFactor);
    mtensorEdgeCache[downscaleFactor] = buildEdgeMapFromImage(imageTensor);
    return mtensorEdgeCache[downscaleFactor];
}

void Camera::releaseCPUData() {
    image.data.clear(); image.data.shrink_to_fit();
    image.width = image.height = 0;
    mask.data.clear(); mask.data.shrink_to_fit();
    mask.width = mask.height = 0;
    imagePyramids.clear();
    maskPyramids.clear();
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    fs::path root(path);

    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json"))
        return loaders::loadNerfstudio(path);

    // COLMAP: cameras.bin (direct or in sparse/0/)
    if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin"))
        return loaders::loadColmap(path, colmapImagePath);

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json"))
        return loaders::loadPolycam(path);

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)");
}
