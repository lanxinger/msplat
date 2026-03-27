#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --- macOS build (host) ---
echo "=== Building msplat (macOS) ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
cmake --build build -j

# --- iOS build (cross-compile for arm64 device) ---
echo "=== Building msplat (iOS arm64) ==="
IOS_SDK=$(xcrun --sdk iphoneos --show-sdk-path)
cmake -B build-ios \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$IOS_SDK" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0
cmake --build build-ios --target msplat_core -j

# Compile Metal shaders for iOS separately
echo "=== Compiling Metal shaders (iOS) ==="
xcrun -sdk iphoneos metal -O3 -c \
    core/metal/msplat_metal.metal \
    -o build-ios/msplat_metal.air
xcrun -sdk iphoneos metallib \
    build-ios/msplat_metal.air \
    -o build-ios/default.metallib
rm -f build-ios/msplat_metal.air

# --- XCFramework ---
echo "=== Creating XCFramework ==="

# Prepare headers with modulemap
rm -rf build/xcf-headers
mkdir -p build/xcf-headers
cp core/include/msplat_c_api.h build/xcf-headers/
cat > build/xcf-headers/module.modulemap <<'MAP'
module MsplatCore {
    header "msplat_c_api.h"
    export *
}
MAP

# Create XCFramework with both slices
rm -rf MsplatCore.xcframework
xcodebuild -create-xcframework \
    -library build/libmsplat_core.a \
    -headers build/xcf-headers \
    -library build-ios/libmsplat_core.a \
    -headers build/xcf-headers \
    -output MsplatCore.xcframework

# Copy both metallibs as Swift package resources (runtime platform selection)
mkdir -p swift/Sources/Msplat/Resources
cp build/default.metallib swift/Sources/Msplat/Resources/default-macos.metallib
cp build-ios/default.metallib swift/Sources/Msplat/Resources/default-ios.metallib

echo "=== Done ==="
echo "  MsplatCore.xcframework (macOS + iOS)"
echo "  swift/Sources/Msplat/Resources/default.metallib"
