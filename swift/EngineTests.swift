// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import LiteRTLM
import XCTest

/// Returns the full path to a test data resource.
func testDataPath(forResource resource: String) -> String {
  guard let testSrcdir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] else {
    fatalError("TEST_SRCDIR not set.")
  }
  return "\(testSrcdir)/\(resource)"
}

class EngineTests: XCTestCase {

  func testEngineConfig_IsCorrectlySet() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())

    let engine = Engine(engineConfig: engineConfig)

    let config = await engine.engineConfig

    XCTAssertEqual(config.modelPath, modelPath)
    XCTAssertEqual(config.maxNumTokens, 16)
    XCTAssertEqual(config.cacheDir, NSTemporaryDirectory())
  }

  func testEngineConfigThrowsErrorWithInvalidMaxNumTokens() throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    XCTAssertThrowsError(
      try EngineConfig(
        modelPath: modelPath, maxNumTokens: 0, cacheDir: NSTemporaryDirectory())
    ) { error in
      XCTAssertEqual(error as? LiteRTLMError, LiteRTLMError.config(.invalidMaxNumTokens))
    }
  }

  func testIsInitialized_IsFalseForNewEngine() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
    let engine = Engine(engineConfig: engineConfig)

    let isInitialized = await engine.isInitialized()
    XCTAssertFalse(isInitialized)
  }

  func testInitialize_SetsIsInitializedToTrue() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
    let engine = Engine(engineConfig: engineConfig)
    try await engine.initialize()
    let isInitialized = await engine.isInitialized()
    XCTAssertTrue(isInitialized)
  }

  func testInitialize_ThrowsIfCalledTwice() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
    let engine = Engine(engineConfig: engineConfig)

    // First initialization should succeed.
    try await engine.initialize()
    let isInitialized = await engine.isInitialized()
    XCTAssertTrue(isInitialized)

    // Second initialization should throw an error. XCTAssertThrowsError doesn't support async
    // functions, so we need to use try-catch here.
    do {
      try await engine.initialize()
      XCTFail("Second init should throw error.")
    } catch let error as LiteRTLMError {
      XCTAssertEqual(error, LiteRTLMError.engine(.alreadyInitialized))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testInitialize_ThrowsWithInvalidModelPath() async throws {
    let engineConfig = try EngineConfig(
      modelPath: "/non/existent/path", maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
    let engine = Engine(engineConfig: engineConfig)

    // Initialization with a non-existent model path should throw an error.
    // XCTAssertThrowsError doesn't support async functions, so we need to use try-catch here.
    do {
      try await engine.initialize()
      XCTFail("Initialization with a non-existent model path should throw an error.")
    } catch let error as LiteRTLMError {
      XCTAssertEqual(error, LiteRTLMError.engine(.failedToCreateEngine))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }

    let isInitialized = await engine.isInitialized()
    XCTAssertFalse(isInitialized)
  }

  func testBenchmark_returnsBenchmarkInfo() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    let info = try await benchmark(
      modelPath: modelPath,
      backend: .cpu(),
      prefillTokens: 16,
      decodeTokens: 16
    )

    XCTAssertGreaterThan(info.initTimeInSecond, 0)
    XCTAssertGreaterThan(info.timeToFirstTokenInSecond, 0)
    XCTAssertGreaterThan(info.lastPrefillTokenCount, 0)
    XCTAssertGreaterThan(info.lastDecodeTokenCount, 0)
  }

  /// Tests that deinit runs without crashing after initialization.
  func testDeinitDoesNotCrashAfterInitialize() async throws {
    // This function creates, initializes, and then implicitly deinits an engine
    // when it goes out of scope. If deinit (which calls close()) has an issue,
    // this test will crash or fail.
    func scopeToTriggerDeinit() async throws {
      // swift-format-ignore
      let modelResource =
        "runtime/testdata/test_lm_new_metadata.task"
      let modelPath = testDataPath(forResource: modelResource)
      let engineConfig = try EngineConfig(
        modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
      let engine = Engine(engineConfig: engineConfig)

      try await engine.initialize()
      let isInitialized = await engine.isInitialized()
      XCTAssertTrue(isInitialized)
      // 'engine' goes out of scope here, triggering deinit.
    }

    try await scopeToTriggerDeinit()
    // If we reached this point, deinit completed without crashing.
  }
}
