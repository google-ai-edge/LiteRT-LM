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

  override func setUp() {
    super.setUp()
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.gpuEnableMetalResidencySet = nil
  }

  override func tearDown() {
    ExperimentalFlags.gpuEnableMetalResidencySet = nil
    super.tearDown()
  }

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

  func testInitialize_WithVisualTokenBudget_WithoutVisionBackend_Succeeds() async throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath,
      maxNumTokens: 16,
      cacheDir: NSTemporaryDirectory()
    )
    let engine = Engine(engineConfig: engineConfig)

    ExperimentalFlags.optIntoExperimentalAPIs()
    let originalBudget = ExperimentalFlags.visualTokenBudget
    defer { ExperimentalFlags.visualTokenBudget = originalBudget }
    ExperimentalFlags.visualTokenBudget = 280

    try await engine.initialize()

    let isInitialized = await engine.isInitialized()
    XCTAssertTrue(isInitialized)
  }

  func testUpdateGPUEnableMetalResidencySetSucceeds() async throws {
    ExperimentalFlags.gpuEnableMetalResidencySet = true

    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm_new_metadata.task"
    let modelPath = testDataPath(forResource: modelResource)
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
    let engine = Engine(engineConfig: engineConfig)
    try await engine.initialize()

    try await engine.updateGPUEnableMetalResidencySet(false)
    try await engine.updateGPUEnableMetalResidencySet(true)
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
      guard case .engine(.failedToCreateEngine(let message)) = error else {
        XCTFail("Expected failedToCreateEngine error, got \(error)")
        return
      }
      XCTAssertFalse(message.isEmpty, "Expected non-empty error message from native layer")
      XCTAssertTrue(error.localizedDescription.contains(message))
      XCTAssertNil(LiteRTLMError.getLastErrorMessage())
      XCTAssertEqual(LiteRTLMError.getLastErrorCode(), 0)
    } catch {
      XCTFail("Unexpected error: \(error)")
    }

    let isInitialized = await engine.isInitialized()
    XCTAssertFalse(isInitialized)
  }

  func testNativeErrorReporting_ClearAndGetError() {
    LiteRTLMError.clearLastError()
    XCTAssertNil(LiteRTLMError.getLastErrorMessage())
    XCTAssertEqual(LiteRTLMError.getLastErrorCode(), 0)
  }

  func testLiteRTLMError_DescriptionFormatting() {
    let engineErrorWithoutDetails = LiteRTLMError.engine(.failedToCreateEngine(""))
    XCTAssertEqual(engineErrorWithoutDetails.localizedDescription, "Failed to create engine.")

    let engineErrorWithDetails = LiteRTLMError.engine(.failedToCreateEngine("file not found"))
    XCTAssertEqual(
      engineErrorWithDetails.localizedDescription, "Failed to create engine: file not found")

    let embeddingErrorWithoutDetails = LiteRTLMError.embeddingEngine(
      .failedToCreateEngine(""))
    XCTAssertEqual(
      embeddingErrorWithoutDetails.localizedDescription, "Failed to create embedding engine.")

    let embeddingErrorWithDetails = LiteRTLMError.embeddingEngine(
      .failedToCreateEngine("model corrupt"))
    XCTAssertEqual(
      embeddingErrorWithDetails.localizedDescription,
      "Failed to create embedding engine: model corrupt")

    let streamErrorWithoutDetails = LiteRTLMError.conversation(
      .failedToStartStream(status: 1))
    XCTAssertEqual(
      streamErrorWithoutDetails.localizedDescription, "Failed to start stream. Status: 1")

    let streamErrorWithDetails = LiteRTLMError.conversation(
      .failedToStartStream(status: 1, message: "backend error"))
    XCTAssertEqual(
      streamErrorWithDetails.localizedDescription,
      "Failed to start stream (status 1): backend error")
  }

  func testLiteRTLMError_EquatableBackwardsCompatibility() {
    // EngineError: legacy static property matches instance with error message
    XCTAssertEqual(
      LiteRTLMError.EngineError.failedToCreateEngine("native error"),
      .failedToCreateEngine
    )
    XCTAssertEqual(
      LiteRTLMError.engine(.failedToCreateEngine("native error")),
      LiteRTLMError.engine(.failedToCreateEngine)
    )
    // Same error messages are equal
    XCTAssertEqual(
      LiteRTLMError.EngineError.failedToCreateEngine("native error"),
      .failedToCreateEngine("native error")
    )
    // Different non-empty error messages are not equal
    XCTAssertNotEqual(
      LiteRTLMError.EngineError.failedToCreateEngine("msg1"),
      .failedToCreateEngine("msg2")
    )
    // Different cases are not equal
    XCTAssertNotEqual(
      LiteRTLMError.EngineError.failedToCreateEngine("native error"),
      .failedToCreateSettings
    )

    // EmbeddingEngineError backwards compatibility
    XCTAssertEqual(
      LiteRTLMError.EmbeddingEngineError.failedToCreateEngine("native error"),
      .failedToCreateEngine
    )

    // ConversationError backwards compatibility
    XCTAssertEqual(
      LiteRTLMError.ConversationError.failedToStartStream(status: 2, message: "native error"),
      .failedToStartStream(status: 2)
    )
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

  func testEngineTeardownAndHandleNilDoesNotCrash() async throws {
    func scopeToTriggerEngineTeardown() async throws {
      let modelResource =
        + "runtime/testdata/test_lm_new_metadata.task"
      let modelPath = testDataPath(forResource: modelResource)
      let engineConfig = try EngineConfig(
        modelPath: modelPath, maxNumTokens: 16, cacheDir: NSTemporaryDirectory())
      var engine: Engine? = Engine(engineConfig: engineConfig)

      try await engine?.initialize()
      let isInitialized = await engine?.isInitialized() == true
      XCTAssertTrue(isInitialized)

      // Releasing engine runs Engine.deinit where self.handle = nil right before calling litert_lm_engine_delete.
      engine = nil
    }

    try await scopeToTriggerEngineTeardown()
  }
}
