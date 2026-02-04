// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JarvisOverlay",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "JarvisOverlay", targets: ["JarvisOverlay"])
    ],
    targets: [
        .executableTarget(
            name: "JarvisOverlay",
            path: "Sources",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "JarvisOverlayTests",
            dependencies: ["JarvisOverlay"],
            path: "Tests"
        )
    ]
)
