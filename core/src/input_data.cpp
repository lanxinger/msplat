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

    // Try loading a matching mask (with same subsample hint)
    std::string maskPath = findMaskPath(filePath, maskDir);
    Mask rawMask;
    if (!maskPath.empty()) rawMask = imreadMask(maskPath, thumbMaxDim);

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

    // Undistort if needed
    if (hasDistortion()) {
        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        raw = std::move(result.image);
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

Image Camera::getImage(int downscaleFactor) {
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
