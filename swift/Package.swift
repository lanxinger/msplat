// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Msplat",
    platforms: [.macOS("26.0"), .iOS(.v17)],
    products: [
        .library(name: "Msplat", targets: ["Msplat"]),
    ],
    targets: [
        // Pre-built core library (run scripts/build-xcframework.sh to generate)
        .binaryTarget(
            name: "MsplatCore",
            path: "../MsplatCore.xcframework"
        ),

        // Swift API
        .target(
            name: "Msplat",
            dependencies: ["MsplatCore"],
            path: "Sources/Msplat",
            resources: [
                .copy("Resources/default-macos.metallib"),
                .copy("Resources/default-ios.metallib"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Foundation"),
                .linkedFramework("ImageIO"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Accelerate"),
                .unsafeFlags(["-lc++"]),
            ]
        ),

        // Tests
        .testTarget(
            name: "MsplatTests",
            dependencies: ["Msplat"],
            path: "Tests"
        ),
    ]
)
