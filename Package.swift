// swift-tools-version: 5.9
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PackageDescription

let package = Package(
  name: "LiteRTLM",
  platforms: [
    .iOS(.v15),
    .macOS(.v12),
  ],
  products: [
    .library(
      name: "LiteRTLM",
      targets: ["LiteRTLM"]
    ),
    .library(
      name: "LiteRTLMFoundationModels",
      targets: ["LiteRTLMFoundationModels"]
    ),
  ],
  targets: [
    // The Prebuilt Binary Target for iOS (Official Stable Release)
    // For nightly builds or pinning to specific nightly dates, switch to the 'nightly' branch or see:
    // https://github.com/google-ai-edge/LiteRT-LM/tree/main/samples/ios_and_mac#1-add-dependency
    .binaryTarget(
      name: "CLiteRTLM",
      url:
        "https://storage.googleapis.com/litert/litert_lm/swift/nightly/CLiteRTLM_0.19.0-20261019.xcframework.zip",
      checksum: "d68cd0b58ca7c5e39b95b59abcdb1e6c8f3ee42f00754229328e4bb8d561ebc4"
    ),
    // The Prebuilt Binary Target for Mac
    .binaryTarget(
      name: "CLiteRTLM_mac",
      url:
        "https://storage.googleapis.com/litert/litert_lm/swift/nightly/CLiteRTLM_mac_0.19.0-20261019.xcframework.zip",
      checksum: "1ffe15e38222b07fe73546aa67c2ce8d2e01c054658292c6d1a7073347d24f9a"
    ),
    // The Swift Wrapper Target
    .target(
      name: "LiteRTLM",
      dependencies: [
        .target(name: "CLiteRTLM", condition: .when(platforms: [.iOS])),
        .target(name: "CLiteRTLM_mac", condition: .when(platforms: [.macOS])),
      ],
      path: "swift",
      exclude: [
        "apple_fm",
        "device_tests",
        "ModelInfoTests.swift",
        "EngineTests.swift",
        "EmbeddingEngineTests.swift",
        "ConversationTests.swift",
        "ToolTests.swift",
        "MessageTests.swift",
        "BUILD",
        "Info.plist",
      ],
    ),
    // Apple Foundation Models Adapter
    .target(
      name: "LiteRTLMFoundationModels",
      dependencies: ["LiteRTLM"],
      path: "swift/apple_fm",
      exclude: [
        "BUILD",
        "main.swift",
        "AdapterTests.swift",
      ]
    ),
    // Separate test targets for each file to avoid naming conflicts:
    .testTarget(
      name: "ModelInfoTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ModelInfoTests.swift"]
    ),
    .testTarget(
      name: "ConversationTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ConversationTests.swift"]
    ),
    .testTarget(
      name: "ToolTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ToolTests.swift"]
    ),
    .testTarget(
      name: "EngineTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["EngineTests.swift"]
    ),
    .testTarget(
      name: "EmbeddingEngineTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["EmbeddingEngineTests.swift"]
    ),
    .testTarget(
      name: "MessageTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["MessageTests.swift"]
    ),
  ]
)
