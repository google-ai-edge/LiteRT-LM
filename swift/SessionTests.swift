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

import XCTest
import CLiteRTLM

@testable import LiteRTLM

/// Returns the full path of the test model.
private func testModelPath() -> String {
  guard let testSrcdir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] else {
    fatalError("TEST_SRCDIR not set.")
  }
  // swift-format-ignore
  let modelResource =
    "runtime/testdata/test_lm.litertlm"
  return "\(testSrcdir)/\(modelResource)"
}

/// Asserts that `expression` throws an error, and calls `handler` with it.
private func assertThrowsAsync<T>(
  _ expression: @autoclosure () async throws -> T,
  _ message: String = "Expected an error.",
  file: StaticString = #filePath,
  line: UInt = #line,
  _ handler: (Error) -> Void = { _ in }
) async {
  do {
    _ = try await expression()
    XCTFail(message, file: file, line: line)
  } catch {
    handler(error)
  }
}

/// Whether `error` ends a native stream that reached the engine's `maxNumTokens`.
///
/// Without `maxOutputTokens`, the test model decodes until the context is full, and the native
/// stream then ends with this error instead of finishing normally.
private func isMaxNumTokensReached(_ error: Error) -> Bool {
  guard case .session(.streamFailed(let details)) = error as? LiteRTLMError else { return false }
  return details.contains("Max number of tokens reached")
}

/// Collects the chunks of `stream`, which may end because the context is full.
private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> [String] {
  var chunks: [String] = []
  do {
    for try await chunk in stream {
      chunks.append(chunk)
    }
  } catch where isMaxNumTokensReached(error) {
    // The stream finished.
  }
  return chunks
}

/// Returns the first chunk of the stream created by `makeStream`, then releases the stream before
/// it finishes, as a consumer that breaks out of its loop does.
private func firstChunk(
  of makeStream: () -> AsyncThrowingStream<String, Error>
) async throws -> String? {
  var iterator = makeStream().makeAsyncIterator()
  return try await iterator.next()
}

class SessionTests: XCTestCase {

  var engine: Engine!

  /// A greedy sampler, so that the outputs of the test model are deterministic.
  private let greedySampler = try! SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0, seed: 42)

  override func setUp() async throws {
    try await super.setUp()
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.enableBenchmark = true
    let engineConfig = try EngineConfig(
      modelPath: testModelPath(), maxNumTokens: 32, cacheDir: NSTemporaryDirectory())
    engine = Engine(engineConfig: engineConfig)
    try await engine.initialize()
  }

  /// Creates a session with the greedy sampler and the given output budget.
  private func makeSession(maxOutputTokens: Int? = nil) async throws -> Session {
    return try await engine.createSession(
      with: try SessionConfig(samplerConfig: greedySampler, maxOutputTokens: maxOutputTokens))
  }

  /// Asserts that `session` was cancelled by terminating one of its streams early, or, if the
  /// stream finished before it was terminated on a fast machine, that `session` is still usable.
  private func assertCancelledOrUsable(
    _ session: Session, file: StaticString = #filePath, line: UInt = #line
  ) {
    if session.isAlive {
      XCTAssertNoThrow(try session.clone(), file: file, line: line)
    } else {
      XCTAssertThrowsError(try session.clone(), file: file, line: line) { error in
        XCTAssertEqual(error as? LiteRTLMError, .session(.notAlive), file: file, line: line)
      }
    }
  }

  // MARK: - SessionConfig

  func testSessionConfigDefaultsAreUnset() throws {
    let config = try SessionConfig()
    XCTAssertNil(config.samplerConfig)
    XCTAssertNil(config.maxOutputTokens)
    XCTAssertNil(config.applyPromptTemplate)
    XCTAssertNil(config.loraPath)
    XCTAssertNil(config.audioLoraPath)
    XCTAssertNil(config.speculativeDecodingConfig)
  }

  func testSessionConfigStoresValues() throws {
    let config = try SessionConfig(
      samplerConfig: greedySampler,
      maxOutputTokens: 8,
      applyPromptTemplate: false,
      loraPath: "lora.tflite",
      audioLoraPath: "audio_lora.tflite",
      speculativeDecodingConfig: SpeculativeDecodingConfig(mode: .off, lenience: 0.5))
    XCTAssertEqual(config.samplerConfig?.topK, 1)
    XCTAssertEqual(config.maxOutputTokens, 8)
    XCTAssertEqual(config.applyPromptTemplate, false)
    XCTAssertEqual(config.loraPath, "lora.tflite")
    XCTAssertEqual(config.audioLoraPath, "audio_lora.tflite")
    XCTAssertEqual(
      config.speculativeDecodingConfig, SpeculativeDecodingConfig(mode: .off, lenience: 0.5))
  }

  func testSessionConfigRejectsOutOfRangeMaxOutputTokens() {
    for maxOutputTokens in [0, -1, Int(Int32.max) + 1, Int.max] {
      XCTAssertThrowsError(try SessionConfig(maxOutputTokens: maxOutputTokens)) { error in
        XCTAssertEqual(error as? LiteRTLMError, .config(.invalidMaxOutputTokens))
      }
    }
  }

  func testSessionConfigAcceptsMaxOutputTokensUpToInt32Max() throws {
    XCTAssertEqual(try SessionConfig(maxOutputTokens: 1).maxOutputTokens, 1)
    XCTAssertEqual(
      try SessionConfig(maxOutputTokens: Int(Int32.max)).maxOutputTokens, Int(Int32.max))
  }

  // MARK: - Engine.createSession

  func testCreateSessionBeforeInitializeThrows() async throws {
    let uninitializedEngine = Engine(
      engineConfig: try EngineConfig(modelPath: testModelPath(), maxNumTokens: 32))
    await assertThrowsAsync(try await uninitializedEngine.createSession()) { error in
      XCTAssertEqual(error as? LiteRTLMError, .engine(.notInitialized))
    }
  }

  func testCreateSessionWithDefaultConfig() async throws {
    let session = try await engine.createSession()
    XCTAssertTrue(session.isAlive)
    let response = try await session.generateContent([.text("Hello")])
    XCTAssertFalse(response.isEmpty)
  }

  func testCreateSessionWithAllConfigFields() async throws {
    let session = try await engine.createSession(
      with: try SessionConfig(
        samplerConfig: try SamplerConfig(topK: 10, topP: 0.5, temperature: 0.5, seed: 123),
        maxOutputTokens: 2,
        applyPromptTemplate: false,
        speculativeDecodingConfig: SpeculativeDecodingConfig(mode: .off)))
    XCTAssertTrue(session.isAlive)
    let response = try await session.generateContent([.text("Hello")])
    XCTAssertFalse(response.isEmpty)
  }

  func testMaxOutputTokensLimitsTheResponse() async throws {
    let shortSession = try await makeSession(maxOutputTokens: 1)
    let longSession = try await makeSession(maxOutputTokens: 4)
    let shortResponse = try await shortSession.generateContent([.text("Hello")])
    let longResponse = try await longSession.generateContent([.text("Hello")])
    XCTAssertFalse(shortResponse.isEmpty)
    XCTAssertTrue(
      longResponse.hasPrefix(shortResponse),
      "Expected '\(longResponse)' to start with '\(shortResponse)'")
    XCTAssertGreaterThan(longResponse.count, shortResponse.count)
  }

  func testMaxOutputTokensAllowsSeveralGenerations() async throws {
    let session = try await makeSession(maxOutputTokens: 4)
    let first = try await session.generateContent([.text("Hello")])
    let second = try await session.generateContent([.text(" world!")])
    XCTAssertFalse(first.isEmpty)
    XCTAssertFalse(second.isEmpty)
  }

  func testCreateSessionRejectsInvalidSpeculativeLenience() async throws {
    let config = try SessionConfig(
      speculativeDecodingConfig: SpeculativeDecodingConfig(lenience: 1.5))
    await assertThrowsAsync(try await engine.createSession(with: config)) { error in
      guard case .engine(.failedToCreateSessionConfig) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  func testCreateSessionWithMissingLoraFileThrows() async throws {
    let config = try SessionConfig(loraPath: "/nonexistent/lora.tflite")
    await assertThrowsAsync(try await engine.createSession(with: config)) { error in
      guard case .engine(.failedToSetLoraPath) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  func testCreateSessionWithMissingAudioLoraFileThrows() async throws {
    let config = try SessionConfig(audioLoraPath: "/nonexistent/audio_lora.tflite")
    await assertThrowsAsync(try await engine.createSession(with: config)) { error in
      guard case .engine(.failedToSetAudioLoraPath) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  func testMultipleSessionsAreIndependent() async throws {
    let session1 = try await makeSession()
    let session2 = try await makeSession()
    try await session1.runPrefill([.text("Hello")])
    try await session2.runPrefill([.text("Hello")])
    let response1 = try await session1.runDecode()
    let response2 = try await session2.runDecode()
    XCTAssertFalse(response1.isEmpty)
    XCTAssertEqual(response1, response2)
  }

  func testSessionKeepsEngineAlive() async throws {
    var localEngine: Engine? = Engine(
      engineConfig: try EngineConfig(
        modelPath: testModelPath(), maxNumTokens: 32, cacheDir: NSTemporaryDirectory()))
    try await localEngine?.initialize()
    let session = try await localEngine!.createSession()
    // The session holds a strong reference to the engine, which stays alive.
    localEngine = nil
    let response = try await session.generateContent([.text("Hello")])
    XCTAssertFalse(response.isEmpty)
  }

  // MARK: - Prefill and decode

  func testRunPrefillAndRunDecode() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello"), .text(" world!")])
    let response = try await session.runDecode()
    XCTAssertFalse(response.isEmpty)
  }

  func testRunPrefillSeveralTimes() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    try await session.runPrefill([.text(" world!")])
    let response = try await session.runDecode()
    XCTAssertFalse(response.isEmpty)
  }

  func testRunPrefillWithEmptyInputsThrows() async throws {
    let session = try await makeSession()
    await assertThrowsAsync(try await session.runPrefill([])) { error in
      guard case .session(.invalidInput) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  func testImageInputWithoutVisionBackendThrows() async throws {
    let session = try await makeSession()
    await assertThrowsAsync(try await session.generateContent([.image(Data([1, 2, 3]))])) {
      error in
      guard case .session(.operationFailed) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  // MARK: - generateContent

  func testGenerateContentMatchesPrefillAndDecode() async throws {
    let session1 = try await makeSession()
    let generated = try await session1.generateContent([.text("Hello")])
    XCTAssertFalse(generated.isEmpty)

    let session2 = try await makeSession()
    try await session2.runPrefill([.text("Hello")])
    let decoded = try await session2.runDecode()
    XCTAssertEqual(generated, decoded)
  }

  func testGenerateContentAfterPrefill() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let generated = try await session.generateContent([.text(" world!")])
    XCTAssertFalse(generated.isEmpty)
  }

  func testGenerateContentWithEmptyInputsThrows() async throws {
    let session = try await makeSession()
    await assertThrowsAsync(try await session.generateContent([])) { error in
      guard case .session(.invalidInput) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
    await assertThrowsAsync(try await collect(session.generateContentStream([]))) { error in
      guard case .session(.invalidInput) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
    // The session is still usable.
    let generated = try await session.generateContent([.text("Hello")])
    XCTAssertFalse(generated.isEmpty)
  }

  // MARK: - Streams

  func testRunDecodeStreamYieldsTheDecodedText() async throws {
    let session = try await makeSession(maxOutputTokens: 4)
    try await session.runPrefill([.text("Hello")])
    let clone = try session.clone()

    let chunks = try await collect(session.runDecodeStream())
    XCTAssertFalse(chunks.isEmpty)
    let decoded = try await clone.runDecode()
    XCTAssertEqual(chunks.joined(), decoded)
  }

  func testGenerateContentStreamYieldsTheGeneratedText() async throws {
    let session = try await makeSession()
    let chunks = try await collect(session.generateContentStream([.text("Hello")]))
    XCTAssertFalse(chunks.isEmpty)

    let otherSession = try await makeSession()
    let generated = try await otherSession.generateContent([.text("Hello")])
    XCTAssertEqual(chunks.joined(), generated)
  }

  func testSessionIsReusableAfterStream() async throws {
    let session = try await makeSession()
    _ = try await collect(session.generateContentStream([.text("Hello")]))
    // Other operations can run once the stream has finished.
    XCTAssertTrue(session.isAlive)
    XCTAssertNoThrow(try session.clone())
  }

  func testOperationDuringStreamThrowsStreamInProgressOrSucceedsAfterIt() async throws {
    let session = try await makeSession()
    let stream = session.generateContentStream([.text("Hello")])
    // The stream may already have ended on a fast machine, so only check the error if any.
    do {
      _ = try session.clone()
    } catch {
      XCTAssertEqual(error as? LiteRTLMError, .session(.streamInProgress))
    }
    _ = try await collect(stream)
  }

  func testStreamOnClosedSessionThrows() async throws {
    let session = try await makeSession()
    session.close()
    await assertThrowsAsync(try await collect(session.runDecodeStream())) { error in
      XCTAssertEqual(error as? LiteRTLMError, .session(.notAlive))
    }
    await assertThrowsAsync(try await collect(session.generateContentStream([.text("Hello")]))) {
      error in
      XCTAssertEqual(error as? LiteRTLMError, .session(.notAlive))
    }
  }

  func testReleasingStreamEarlyCancelsTheSession() async throws {
    let session = try await makeSession()
    let chunk = try await firstChunk { session.generateContentStream([.text("Hello")]) }
    XCTAssertNotNil(chunk)
    assertCancelledOrUsable(session)
  }

  func testReleasingRunDecodeStreamEarlyCancelsTheSession() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let chunk = try await firstChunk { session.runDecodeStream() }
    XCTAssertNotNil(chunk)
    assertCancelledOrUsable(session)
  }

  func testCancellingConsumingTaskCancelsTheSession() async throws {
    let session = try await makeSession()
    let task = Task {
      var chunkCount = 0
      do {
        for try await _ in session.generateContentStream([.text("Hello")]) {
          chunkCount += 1
          // Cancel the consuming task after the first chunk, which ends the loop.
          withUnsafeCurrentTask { $0?.cancel() }
        }
      } catch where isMaxNumTokensReached(error) {
        // The stream finished before the task was cancelled on a fast machine.
      }
      return chunkCount
    }
    let chunkCount = try await task.value
    XCTAssertGreaterThanOrEqual(chunkCount, 1)
    assertCancelledOrUsable(session)
  }

  // MARK: - cancel

  func testCancelDuringStream() async throws {
    let session = try await makeSession()
    var chunkCount = 0
    do {
      for try await _ in session.generateContentStream([.text("Hello")]) {
        chunkCount += 1
        if chunkCount == 1 {
          try session.cancel()
        }
      }
      // The stream may have finished before `cancel()` ran on a fast machine.
    } catch where isMaxNumTokensReached(error) {
      // The stream finished before `cancel()` ran on a fast machine.
    } catch {
      XCTAssertEqual(error as? LiteRTLMError, .session(.cancelled))
    }
    XCTAssertGreaterThanOrEqual(chunkCount, 1)
    XCTAssertFalse(session.isAlive)
  }

  func testCancelDuringRunDecodeStream() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    var chunkCount = 0
    do {
      for try await _ in session.runDecodeStream() {
        chunkCount += 1
        if chunkCount == 1 {
          try session.cancel()
        }
      }
      // The stream may have finished before `cancel()` ran on a fast machine.
    } catch where isMaxNumTokensReached(error) {
      // The stream finished before `cancel()` ran on a fast machine.
    } catch {
      XCTAssertEqual(error as? LiteRTLMError, .session(.cancelled))
    }
    XCTAssertGreaterThanOrEqual(chunkCount, 1)
    XCTAssertFalse(session.isAlive)
  }

  func testSessionIsNotAliveAfterCancel() async throws {
    let session = try await makeSession()
    try session.cancel()
    // Cancelling again has no effect.
    try session.cancel()
    XCTAssertFalse(session.isAlive)
    await assertThrowsAsync(try await session.generateContent([.text("Hello")])) { error in
      XCTAssertEqual(error as? LiteRTLMError, .session(.notAlive))
    }
    session.close()
  }

  // MARK: - clone

  func testCloneIsIndependent() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let clone = try session.clone()
    XCTAssertTrue(clone.isAlive)

    let response = try await session.runDecode()
    // The clone has the state from before the original decoded.
    let cloneResponse = try await clone.runDecode()
    XCTAssertEqual(response, cloneResponse)

    // The clone keeps working after the original is closed.
    session.close()
    XCTAssertTrue(clone.isAlive)
    XCTAssertNoThrow(try clone.clone())
  }

  func testCloneOfClosedSessionThrows() async throws {
    let session = try await makeSession()
    session.close()
    XCTAssertThrowsError(try session.clone()) { error in
      XCTAssertEqual(error as? LiteRTLMError, .session(.notAlive))
    }
  }

  // MARK: - Text scoring

  func testRunTextScoringWithTokenLengths() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let scores = try await session.runTextScoring([" world"], storeTokenLengths: true)
    XCTAssertEqual(scores.count, 1)
    XCTAssertEqual(scores[0].text, " world")
    XCTAssertLessThanOrEqual(scores[0].score, 0, "A log likelihood is at most 0.")
    let tokenLength = try XCTUnwrap(scores[0].tokenLength)
    XCTAssertGreaterThan(tokenLength, 0)
    XCTAssertEqual(scores[0].tokenScores.count, tokenLength)
  }

  func testRunTextScoringWithoutTokenLengths() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let scores = try await session.runTextScoring([" world"])
    XCTAssertEqual(scores.count, 1)
    XCTAssertEqual(scores[0].text, " world")
    XCTAssertNil(scores[0].tokenLength)
  }

  func testRunTextScoringIsDeterministic() async throws {
    let session1 = try await makeSession()
    try await session1.runPrefill([.text("Hello")])
    let session2 = try await makeSession()
    try await session2.runPrefill([.text("Hello")])
    let score1 = try await session1.runTextScoring([" world"])[0].score
    let score2 = try await session2.runTextScoring([" world"])[0].score
    XCTAssertEqual(score1, score2, accuracy: 1e-4)
  }

  func testRunTextScoringInCloneMatchesTheOriginal() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    let clone = try session.clone()
    let score = try await session.runTextScoring([" world"])[0].score
    let cloneScore = try await clone.runTextScoring([" world"])[0].score
    XCTAssertEqual(score, cloneScore, accuracy: 1e-4)
  }

  func testRunTextScoringWithEmptyTargetsThrows() async throws {
    let session = try await makeSession()
    try await session.runPrefill([.text("Hello")])
    await assertThrowsAsync(try await session.runTextScoring([])) { error in
      guard case .session(.invalidInput) = error as? LiteRTLMError else {
        return XCTFail("Unexpected error: \(error)")
      }
    }
  }

  // MARK: - Benchmark info

  func testGetBenchmarkInfo() async throws {
    let session = try await makeSession(maxOutputTokens: 4)
    let emptyInfo = try session.getBenchmarkInfo()
    XCTAssertEqual(emptyInfo.lastPrefillTokenCount, 0)
    XCTAssertEqual(emptyInfo.lastDecodeTokenCount, 0)

    _ = try await session.generateContent([.text("Hello")])
    let info = try session.getBenchmarkInfo()
    XCTAssertGreaterThan(info.lastPrefillTokenCount, 0)
    XCTAssertGreaterThan(info.lastDecodeTokenCount, 0)
    XCTAssertGreaterThan(info.lastDecodeTokensPerSecond, 0)
  }

  func testGetBenchmarkInfoWithoutBenchmarkFlagThrows() async throws {
    let session = try await makeSession()
    ExperimentalFlags.enableBenchmark = false
    defer { ExperimentalFlags.enableBenchmark = true }
    XCTAssertThrowsError(try session.getBenchmarkInfo()) { error in
      XCTAssertEqual(error as? LiteRTLMError, .session(.benchmarkNotEnabled))
    }
  }

  // MARK: - close

  func testCloseIsIdempotent() async throws {
    let session = try await makeSession()
    XCTAssertTrue(session.isAlive)
    session.close()
    XCTAssertFalse(session.isAlive)
    session.close()
    XCTAssertFalse(session.isAlive)
  }

  func testOperationsAfterCloseThrowNotAlive() async throws {
    let session = try await makeSession()
    session.close()
    let notAlive = LiteRTLMError.session(.notAlive)
    await assertThrowsAsync(try await session.runPrefill([.text("Hello")])) {
      XCTAssertEqual($0 as? LiteRTLMError, notAlive)
    }
    await assertThrowsAsync(try await session.runDecode()) {
      XCTAssertEqual($0 as? LiteRTLMError, notAlive)
    }
    await assertThrowsAsync(try await session.generateContent([.text("Hello")])) {
      XCTAssertEqual($0 as? LiteRTLMError, notAlive)
    }
    await assertThrowsAsync(try await session.runTextScoring(["Hello"])) {
      XCTAssertEqual($0 as? LiteRTLMError, notAlive)
    }
    XCTAssertThrowsError(try session.cancel()) { XCTAssertEqual($0 as? LiteRTLMError, notAlive) }
    XCTAssertThrowsError(try session.getBenchmarkInfo()) {
      XCTAssertEqual($0 as? LiteRTLMError, notAlive)
    }
  }

  func testCloseDuringStreamEndsTheStream() async throws {
    let session = try await makeSession()
    let stream = session.generateContentStream([.text("Hello")])
    session.close()
    XCTAssertFalse(session.isAlive)
    // The stream is cancelled, unless it already ended on a fast machine.
    do {
      _ = try await collect(stream)
    } catch {
      XCTAssertEqual(error as? LiteRTLMError, .session(.cancelled))
    }
  }

  // MARK: - InputData and errors

  func testInputDataCreatesNativeInputData() throws {
    let inputs: [InputData] = [.text("Hello"), .text(""), .image(Data([1, 2])), .audio(Data())]
    for input in inputs {
      let cInputData = try input.makeCInputData()
      litert_lm_input_data_delete(cInputData)
    }
  }

  func testInputDataEquality() {
    XCTAssertEqual(InputData.text("a"), .text("a"))
    XCTAssertNotEqual(InputData.text("a"), .text("b"))
    XCTAssertEqual(InputData.image(Data([1])), .image(Data([1])))
    XCTAssertNotEqual(InputData.image(Data([1])), .audio(Data([1])))
  }

  func testSessionErrorDescriptions() {
    let errors: [LiteRTLMError.SessionError] = [
      .notAlive, .streamInProgress, .invalidInput("x"), .operationFailed("x"),
      .failedToStartStream(status: 3, message: "x"), .streamFailed("x"), .cancelled,
      .benchmarkNotEnabled, .benchmarkInfoUnavailable,
    ]
    for error in errors {
      XCTAssertFalse(LiteRTLMError.session(error).errorDescription?.isEmpty ?? true)
    }
    XCTAssertNotEqual(LiteRTLMError.SessionError.cancelled, .streamFailed("CANCELLED"))
    XCTAssertEqual(
      LiteRTLMError.EngineError.failedToCreateSession("details"), .failedToCreateSession)
    XCTAssertFalse(
      LiteRTLMError.engine(.failedToCreateSession("x")).errorDescription?.isEmpty ?? true)
    XCTAssertFalse(LiteRTLMError.config(.invalidMaxOutputTokens).errorDescription?.isEmpty ?? true)
  }
}
