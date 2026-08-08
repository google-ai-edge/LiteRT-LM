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
//
// Acknowledgements:
// This implementation was originally authored by @john-rocky and ported
// from the open-source repository: https://github.com/john-rocky/swift-litert-lm/tree/main
//
// Tests that need no `.litertlm` on disk.
//
// `LiteRTLMExecutor.init` only builds a `LazyEngine`; the weights are read on the
// first `respond`. So executor/engine accounting, configuration identity, and
// capability derivation are all observable against a model path that does not
// exist — which makes them safe to run in CI.

#if canImport(FoundationModels) && compiler(>=6.4)

  import FoundationModels
  import LiteRTLM
  import XCTest

  @testable import LiteRTLMFoundationModels

  @available(iOS 27.0, macOS 27.0, *)
  private struct StubTool: FoundationModels.Tool {
    let name = "get_temperature"
    let description = "Get the current temperature for a city."
    @Generable struct Arguments {
      @Guide(description: "The city name")
      var city: String
    }
    func call(arguments: Arguments) async throws -> String { "21°C" }
  }

  @available(iOS 27.0, macOS 27.0, *)
  final class AdapterTests: XCTestCase {
    private static let modelPath = "/tmp/litertlm-tests-nonexistent.litertlm"

    override func setUp() async throws {
      try await super.setUp()
      await LiteRTLanguageModel.releaseCachedEngines()
    }

    override func tearDown() async throws {
      await LiteRTLanguageModel.releaseCachedEngines()
      try await super.tearDown()
    }

    // MARK: - Engine sharing

    /// Foundation Models builds one executor per session — a plain session and a
    /// tool-enabled session over the same model yield two. They must resolve to a
    /// single engine, or the multi-GB weights load twice and the app OOMs.
    func testPlainAndToolSessionsShareOneEngine() throws {
      let model = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      XCTAssertEqual(EngineCache.shared.count, 0)

      let plain = LanguageModelSession(model: model)
      plain.prewarm()
      let tooled = LanguageModelSession(model: model, tools: [StubTool()])
      tooled.prewarm()

      withExtendedLifetime((plain, tooled)) {
        XCTAssertEqual(
          EngineCache.shared.count, 1,
          "two sessions over one model must share a single engine")
      }
    }

    /// Two models over the *same* file but different backends genuinely need
    /// different engines, and must not collide in the cache.
    func testDifferentBackendsDoNotShareAnEngine() throws {
      let cpu = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      let gpu = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))

      let a = LanguageModelSession(model: cpu)
      a.prewarm()
      let b = LanguageModelSession(model: gpu)
      b.prewarm()

      withExtendedLifetime((a, b)) {
        XCTAssertEqual(EngineCache.shared.count, 2)
      }
    }

    /// `visualTokenBudget` is a conversation-level setting, so two models over the
    /// same engine configuration with different budgets share one engine (the
    /// multi-GB weights load once).
    func testDifferentVisualTokenBudgetsShareOneEngine() throws {
      let config = try EngineConfig(modelPath: Self.modelPath, backend: .cpu())
      let small = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 70)
      let large = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 280)

      let a = LanguageModelSession(model: small)
      a.prewarm()
      let b = LanguageModelSession(model: large)
      b.prewarm()

      withExtendedLifetime((a, b)) {
        XCTAssertEqual(EngineCache.shared.count, 1)
      }
    }

    // MARK: - Configuration identity

    /// Regression: `Configuration` once hashed on `modelPath` alone, so a `.gpu`
    /// model silently received a `.cpu` engine.
    func testConfigurationDistinguishesBackend() throws {
      let cpu = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
      let gpu = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      XCTAssertNotEqual(cpu, gpu)
    }

    func testConfigurationDistinguishesMaxNumTokens() throws {
      let small = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 512))
      let large = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 4096))
      XCTAssertNotEqual(small, large)
    }

    func testIdenticalConfigurationsCompareEqual() throws {
      let a = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      let b = LiteRTLMExecutor.Configuration(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
      XCTAssertEqual(a, b)
      XCTAssertEqual(a.hashValue, b.hashValue)
    }

    /// Regression: the adapter used to mirror a subset of `EngineConfig`'s fields
    /// into its own `Configuration`, so `cacheDir` / `loraRank` / `audioLoraRank` /
    /// `maxNumImages` were dropped on the way to the engine.
    func testConfigurationCarriesEveryEngineConfigField() throws {
      let engineConfig = try EngineConfig(
        modelPath: Self.modelPath,
        backend: .gpu,
        visionBackend: .gpu,
        audioBackend: .cpu(threadCount: 2),
        maxNumTokens: 4096,
        cacheDir: "/tmp/litertlm-tests-cache",
        loraRank: 8,
        audioLoraRank: 4)

      let carried = LiteRTLanguageModel(engineConfig: engineConfig)
        .executorConfiguration.engineConfig

      XCTAssertEqual(carried, engineConfig)
      XCTAssertEqual(carried.cacheDir, "/tmp/litertlm-tests-cache")
      XCTAssertEqual(carried.loraRank, 8)
      XCTAssertEqual(carried.audioLoraRank, 4)
      XCTAssertEqual(carried.maxNumTokens, 4096)
      XCTAssertEqual(carried.audioBackend, .cpu(threadCount: 2))
    }

    /// `init(engineConfig:)` honours the caller's `cacheDir`; the convenience
    /// initializer supplies the app's Caches directory when none is given.
    func testCacheDirIsHonouredAndDefaulted() throws {
      let explicit = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, cacheDir: "/tmp/explicit"))
      XCTAssertEqual(explicit.executorConfiguration.engineConfig.cacheDir, "/tmp/explicit")

      let sugared = try LiteRTLanguageModel(modelPath: Self.modelPath)
      let defaulted = try XCTUnwrap(sugared.executorConfiguration.engineConfig.cacheDir)
      XCTAssertTrue(defaulted.contains("Caches"), "got \(defaulted)")
    }

    // MARK: - Capabilities

    func testVisionCapabilityTracksVisionBackend() throws {
      let textOnly = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath))
      XCTAssertFalse(textOnly.capabilities.contains(.vision))
      XCTAssertTrue(textOnly.capabilities.contains(.guidedGeneration))
      XCTAssertTrue(textOnly.capabilities.contains(.toolCalling))

      let vision = LiteRTLanguageModel(
        engineConfig: try EngineConfig(modelPath: Self.modelPath, visionBackend: .gpu))
      XCTAssertTrue(vision.capabilities.contains(.vision))
    }
  }

#endif
