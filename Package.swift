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
        "https://storage.googleapis.com/litert/litert_lm/swift/nightly/CLiteRTLM_0.19.0-20261029.xcframework.zip",
      checksum: "9696017f737d9a4a5f1faf54dcb595251f0fadaf8c375e933e4aa45cca262dee"
    ),
    // The Prebuilt Binary Target for Mac
    .binaryTarget(
      name: "CLiteRTLM_mac",
      url:
        "https://storage.googleapis.com/litert/litert_lm/swift/nightly/CLiteRTLM_mac_0.19.0-20261029.xcframework.zip",
      checksum: "08dfdd6095b6501da14fa4243c136dd7879c188be9a950c90ec2b686a967e5e1"
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
