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

import Foundation
import CLiteRTLM

/// An input to a `Session`.
public enum InputData: Equatable, Hashable, Sendable {
  /// Text input.
  case text(String)
  /// Image input, as the bytes of an encoded image (e.g. PNG or JPEG).
  case image(Data)
  /// Audio input, as the bytes of an encoded audio file (e.g. WAV).
  case audio(Data)

  /// Creates the native input data of this input. The caller owns the returned handle and must
  /// release it with `litert_lm_input_data_delete`.
  func makeCInputData() throws -> OpaquePointer {
    let makeError: (String) -> LiteRTLMError = { .session(.invalidInput($0)) }
    switch self {
    case .text(let text):
      return try LiteRTLMError.create("litert_lm_input_data_create", makeError) { out in
        text.withCString { cString in
          litert_lm_input_data_create(kLiteRtLmInputDataTypeText, cString, text.utf8.count, out)
        }
      }
    case .image(let data):
      return try Self.makeCInputData(kLiteRtLmInputDataTypeImage, data, makeError)
    case .audio(let data):
      return try Self.makeCInputData(kLiteRtLmInputDataTypeAudio, data, makeError)
    }
  }

  private static func makeCInputData(
    _ type: LiteRtLmInputDataType, _ data: Data, _ makeError: (String) -> LiteRTLMError
  ) throws -> OpaquePointer {
    return try LiteRTLMError.create("litert_lm_input_data_create", makeError) { out in
      data.withUnsafeBytes { buffer in
        litert_lm_input_data_create(type, buffer.baseAddress, buffer.count, out)
      }
    }
  }
}

/// The score of one target text, returned by `Session.runTextScoring(_:storeTokenLengths:)`.
public struct TextScore: Equatable, Sendable {
  /// The scored target text.
  public let text: String

  /// The log likelihood of `text` given the prefilled context.
  public let score: Float

  /// The number of tokens of `text`, or `nil` unless `storeTokenLengths` was `true`.
  public let tokenLength: Int?

  /// The log likelihood of each token of `text`, or an empty array if the runtime didn't report
  /// them.
  public let tokenScores: [Float]
}

/// A session with a LiteRT-LM model, for raw text, image and audio inputs and raw text outputs.
///
/// A session is the lower-level alternative to a `Conversation`: it doesn't apply a chat
/// template to messages or run tools, but gives control over the prefill and decode steps.
/// Create one with `Engine.createSession(with:)`.
///
/// Example usage:
/// ```swift
/// let engine = Engine(engineConfig: try EngineConfig(modelPath: "/path/to/model.litertlm"))
/// try await engine.initialize()
/// let session = try await engine.createSession()
/// defer { session.close() }
///
/// // Prefill and decode in one call.
/// let answer = try await session.generateContent([.text("What is the capital of France?")])
///
/// // Or prefill and decode separately, e.g. to add input in several chunks.
/// try await session.runPrefill([.text("How about Japan?")])
/// let answer2 = try await session.runDecode()
///
/// // Or stream the response.
/// for try await chunk in session.generateContentStream([.text("And Italy?")]) {
///   print(chunk, terminator: "")
/// }
/// ```
///
/// A session keeps its engine alive until it is closed or deallocated. Its operations must not
/// overlap: while a stream is running, only `cancel()` and `close()` can be called, and other
/// operations throw `LiteRTLMError.session(.streamInProgress)`. The blocking operations
/// (`runPrefill(_:)`, `runDecode()`, `generateContent(_:)` and
/// `runTextScoring(_:storeTokenLengths:)`) are `async` so that they don't run on the caller's
/// actor, e.g. the main actor.
public final class Session: @unchecked Sendable {
  private let engine: Engine

  /// Guards `handle`, `isClosed`, `isCancelled`, `activeStreamCount` and
  /// `SessionStreamContext.isRunning`. It is only held for short periods, so that `cancel()` can
  /// run while a blocking operation holds `operationLock`.
  private let stateLock = NSLock()

  /// Serializes the native calls that use the session, and `close()` with them.
  private let operationLock = NSLock()

  /// The native session. `nil` once it is deleted.
  private var handle: OpaquePointer?
  private var isClosed = false
  private var isCancelled = false
  private var activeStreamCount = 0

  init(handle: OpaquePointer, engine: Engine) {
    self.handle = handle
    self.engine = engine
  }

  deinit {
    // Running streams retain the session, so none is running here.
    if let handle {
      litert_lm_session_delete(handle)
    }
  }

  /// Whether the session can be used, i.e. it was neither closed nor cancelled.
  public var isAlive: Bool {
    stateLock.lock()
    defer { stateLock.unlock() }
    return handle != nil && !isClosed && !isCancelled
  }

  /// Closes the session and releases its native resources. Calling it again has no effect.
  ///
  /// It waits for a running blocking operation to finish. If a stream is running, it cancels the
  /// stream, which throws `LiteRTLMError.session(.cancelled)`, and releases the native resources
  /// when the stream ends. After closing, the session's operations throw
  /// `LiteRTLMError.session(.notAlive)`.
  public func close() {
    operationLock.lock()
    defer { operationLock.unlock() }
    stateLock.lock()
    guard !isClosed else {
      stateLock.unlock()
      return
    }
    isClosed = true
    var handleToDelete: OpaquePointer? = nil
    if let handle {
      if activeStreamCount > 0 {
        // The last stream to end deletes the handle; see `streamDidEnd(_:)`.
        if !isCancelled {
          _ = litert_lm_session_cancel_process(handle)
          LiteRTLMError.clearLastError()
        }
      } else {
        handleToDelete = handle
        self.handle = nil
      }
    }
    stateLock.unlock()
    if let handleToDelete {
      litert_lm_session_delete(handleToDelete)
    }
  }

  /// Adds inputs to the session and runs the prefill step on them.
  ///
  /// It can be called several times to add the inputs in chunks before calling `runDecode()` or
  /// `runDecodeStream()`. It returns when the prefill step is done.
  ///
  /// - Parameter inputs: The inputs to prefill. Must not be empty: the native session rejects
  ///   empty inputs.
  /// - Throws: A `LiteRTLMError` if the session is not alive, `inputs` is empty or invalid, or the
  ///   prefill step fails.
  public func runPrefill(_ inputs: [InputData]) async throws {
    guard !inputs.isEmpty else {
      throw LiteRTLMError.session(.invalidInput("inputs must not be empty"))
    }
    try withOperation { handle in
      try withCInputData(inputs) { cInputs, count in
        try LiteRTLMError.check(
          litert_lm_session_run_prefill(handle, cInputs, count), "litert_lm_session_run_prefill",
          operationError)
      }
    }
  }

  /// Runs the decode step to generate a response to the inputs added by `runPrefill(_:)`.
  ///
  /// It returns when the decode step is done.
  ///
  /// - Returns: The generated text.
  /// - Throws: A `LiteRTLMError` if the session is not alive or the decode step fails.
  public func runDecode() async throws -> String {
    return try withOperation { handle in
      let responses = try LiteRTLMError.create("litert_lm_session_run_decode", operationError) {
        out in
        litert_lm_session_run_decode(handle, out)
      }
      defer { litert_lm_responses_delete(responses) }
      return try responseText(responses)
    }
  }

  /// Runs the decode step to generate a response to the inputs added by `runPrefill(_:)`, and
  /// streams the response.
  ///
  /// If the consuming task is cancelled, or the stream is released before it finishes, e.g.
  /// because the consumer breaks out of its `for try await` loop, the decoding is cancelled like
  /// with `cancel()`, so the session can't be used anymore.
  ///
  /// - Returns: A stream of the generated text chunks. It finishes after the last chunk, or throws
  ///   `LiteRTLMError.session(.cancelled)` if it is cancelled with `cancel()` or `close()`, or
  ///   another `LiteRTLMError` if the stream can't be started or fails.
  public func runDecodeStream() -> AsyncThrowingStream<String, Error> {
    return startStream("litert_lm_session_run_decode_async") { handle, contextPointer in
      litert_lm_session_run_decode_async(handle, sessionStreamCallback, contextPointer)
    }
  }

  /// Adds inputs to the session and generates a response, running both the prefill and the
  /// decode steps.
  ///
  /// It returns when the response is generated.
  ///
  /// - Parameter inputs: The inputs to prefill. They are added after the inputs added by
  ///   `runPrefill(_:)`, if any. Must not be empty: the native session rejects empty inputs. To
  ///   only run the decode step on the inputs added by `runPrefill(_:)`, call `runDecode()` or
  ///   `runDecodeStream()` instead.
  /// - Returns: The generated text.
  /// - Throws: A `LiteRTLMError` if the session is not alive, `inputs` is empty or invalid, or
  ///   generation fails.
  public func generateContent(_ inputs: [InputData]) async throws -> String {
    guard !inputs.isEmpty else {
      throw LiteRTLMError.session(.invalidInput("inputs must not be empty"))
    }
    return try withOperation { handle in
      try withCInputData(inputs) { cInputs, count in
        let responses = try LiteRTLMError.create(
          "litert_lm_session_generate_content", operationError
        ) { out in
          litert_lm_session_generate_content(handle, cInputs, count, out)
        }
        defer { litert_lm_responses_delete(responses) }
        return try responseText(responses)
      }
    }
  }

  /// Adds inputs to the session and generates a response, running both the prefill and the
  /// decode steps, and streams the response.
  ///
  /// If the consuming task is cancelled, or the stream is released before it finishes, e.g.
  /// because the consumer breaks out of its `for try await` loop, the generation is cancelled
  /// like with `cancel()`, so the session can't be used anymore.
  ///
  /// - Parameter inputs: The inputs to prefill. They are added after the inputs added by
  ///   `runPrefill(_:)`, if any. Must not be empty: the native session rejects empty inputs. To
  ///   only run the decode step on the inputs added by `runPrefill(_:)`, call `runDecode()` or
  ///   `runDecodeStream()` instead.
  /// - Returns: A stream of the generated text chunks. It finishes after the last chunk, or throws
  ///   `LiteRTLMError.session(.cancelled)` if it is cancelled with `cancel()` or `close()`, or
  ///   another `LiteRTLMError` if `inputs` is empty or the stream can't be started or fails.
  public func generateContentStream(_ inputs: [InputData]) -> AsyncThrowingStream<String, Error> {
    guard !inputs.isEmpty else {
      return AsyncThrowingStream {
        $0.finish(throwing: LiteRTLMError.session(.invalidInput("inputs must not be empty")))
      }
    }
    return startStream("litert_lm_session_generate_content_stream") { handle, contextPointer in
      // The native call copies the inputs, so they can be released when it returns.
      try withCInputData(inputs) { cInputs, count in
        litert_lm_session_generate_content_stream(
          handle, cInputs, count, sessionStreamCallback, contextPointer)
      }
    }
  }

  /// Scores target texts given the inputs added by `runPrefill(_:)`.
  ///
  /// Pass one target text, or several target texts that are each a single token: several targets
  /// are all scored from the logits of one decode step.
  ///
  /// Every call changes the session's state: to score targets one by one from the same prefilled
  /// inputs, score each of them in a clone of the session made with `clone()` before scoring.
  ///
  /// - Parameters:
  ///   - targets: The target texts to score. Must not be empty.
  ///   - storeTokenLengths: Whether to return the number of tokens of each target text in
  ///     `TextScore.tokenLength`.
  /// - Returns: The scores of the target texts, in the order of `targets`.
  /// - Throws: A `LiteRTLMError` if the session is not alive, `targets` is empty or invalid, or
  ///   scoring fails.
  public func runTextScoring(
    _ targets: [String], storeTokenLengths: Bool = false
  ) async throws -> [TextScore] {
    guard !targets.isEmpty else {
      throw LiteRTLMError.session(.invalidInput("targets must not be empty"))
    }
    return try withOperation { handle in
      let cTargets = targets.map { strdup($0) }
      defer { cTargets.forEach { free($0) } }
      var cTargetPointers = cTargets.map { $0.map { UnsafePointer($0) } }
      let responses = try LiteRTLMError.create(
        "litert_lm_session_run_text_scoring", operationError
      ) { out in
        cTargetPointers.withUnsafeMutableBufferPointer { buffer in
          litert_lm_session_run_text_scoring(
            handle, buffer.baseAddress, buffer.count, storeTokenLengths, out)
        }
      }
      defer { litert_lm_responses_delete(responses) }
      return try textScores(responses, targets: targets)
    }
  }

  /// Cancels the running operation, e.g. a stream, which then throws
  /// `LiteRTLMError.session(.cancelled)`.
  ///
  /// The session can't be used afterwards (`isAlive` is `false`), because the native session must
  /// not be reused after a cancellation: its operations throw `LiteRTLMError.session(.notAlive)`.
  /// Clone the session before running an operation that may be cancelled to keep its state.
  /// Calling it again has no effect.
  ///
  /// - Throws: `LiteRTLMError.session(.notAlive)` if the session is closed, or a `LiteRTLMError`
  ///   if the cancellation request fails.
  public func cancel() throws {
    stateLock.lock()
    defer { stateLock.unlock() }
    guard let handle, !isClosed else {
      throw LiteRTLMError.session(.notAlive)
    }
    guard !isCancelled else { return }
    try LiteRTLMError.check(
      litert_lm_session_cancel_process(handle), "litert_lm_session_cancel_process",
      operationError)
    isCancelled = true
  }

  /// Clones the session, copying its state so the new session can be used independently.
  ///
  /// - Returns: A new session with the same state and configuration, e.g. the same prefilled
  ///   inputs.
  /// - Throws: A `LiteRTLMError` if the session is not alive or cloning fails.
  public func clone() throws -> Session {
    return try withOperation { handle in
      let clonedHandle = try LiteRTLMError.create("litert_lm_session_clone", operationError) {
        out in
        litert_lm_session_clone(handle, out)
      }
      return Session(handle: clonedHandle, engine: engine)
    }
  }

  /// Retrieves the benchmark information of the session.
  ///
  /// Note: This is an experimental API. It requires setting `ExperimentalFlags.enableBenchmark`
  /// to `true` before initializing the engine.
  ///
  /// - Returns: The benchmark information.
  /// - Throws: `LiteRTLMError.session(.benchmarkNotEnabled)` if the benchmark flag is not enabled,
  ///   or a `LiteRTLMError` if the session is not alive or the info is unavailable.
  public func getBenchmarkInfo() throws -> BenchmarkInfo {
    if !ExperimentalFlags.enableBenchmark {
      throw LiteRTLMError.session(.benchmarkNotEnabled)
    }
    return try withOperation { handle in
      let makeError: (String) -> LiteRTLMError = { _ in .session(.benchmarkInfoUnavailable) }
      let benchmarkInfo = try LiteRTLMError.create(
        "litert_lm_session_get_benchmark_info", makeError
      ) { out in
        litert_lm_session_get_benchmark_info(handle, out)
      }
      defer { litert_lm_benchmark_info_delete(benchmarkInfo) }
      return try BenchmarkInfo(cBenchmarkInfo: benchmarkInfo, makeError: makeError)
    }
  }

  // MARK: - Private

  private let operationError: (String) -> LiteRTLMError = { .session(.operationFailed($0)) }

  /// Runs `body` with the native session while holding `operationLock`.
  ///
  /// - Throws: `LiteRTLMError.session(.notAlive)` if the session is closed or cancelled,
  ///   `LiteRTLMError.session(.streamInProgress)` if a stream is running, or the error thrown by
  ///   `body`.
  private func withOperation<T>(_ body: (OpaquePointer) throws -> T) throws -> T {
    operationLock.lock()
    defer { operationLock.unlock() }
    stateLock.lock()
    guard let handle, !isClosed, !isCancelled else {
      stateLock.unlock()
      throw LiteRTLMError.session(.notAlive)
    }
    guard activeStreamCount == 0 else {
      stateLock.unlock()
      throw LiteRTLMError.session(.streamInProgress)
    }
    stateLock.unlock()
    return try body(handle)
  }

  /// Starts a stream with `start`, which calls the native streaming function `functionName` with
  /// the native session and the stream context to pass to `sessionStreamCallback`.
  ///
  /// When the stream is terminated by its consumer, i.e. the consuming task is cancelled or the
  /// stream is released before it finishes, the native stream is cancelled if it is still
  /// running, and the session can't be used anymore.
  private func startStream(
    _ functionName: String,
    _ start: @escaping (OpaquePointer, UnsafeMutableRawPointer) throws -> LiteRtLmStatusCode
  ) -> AsyncThrowingStream<String, Error> {
    return AsyncThrowingStream { continuation in
      let context = SessionStreamContext(continuation: continuation, session: self)
      // The consumer can only terminate the stream after this closure returns, i.e. after the
      // native stream is started or has failed to start.
      continuation.onTermination = { [weak context] termination in
        guard case .cancelled = termination, let context else { return }
        context.session.cancelStream(context)
      }
      do {
        try self.withOperation { handle in
          let contextPointer = Unmanaged.passRetained(context).toOpaque()
          self.stateLock.lock()
          self.activeStreamCount += 1
          context.isRunning = true
          self.stateLock.unlock()

          let status: LiteRtLmStatusCode
          do {
            status = try start(handle, contextPointer)
          } catch {
            self.abortStream(contextPointer)
            throw error
          }
          guard status == kLiteRtLmStatusOk else {
            let message = LiteRTLMError.consumeLastError() ?? "\(functionName) failed"
            self.abortStream(contextPointer)
            throw LiteRTLMError.session(
              .failedToStartStream(status: Int(status.rawValue), message: message))
          }
        }
      } catch {
        continuation.finish(throwing: error)
      }
    }
  }

  /// Undoes the stream registration of `startStream` for a stream that didn't start.
  private func abortStream(_ contextPointer: UnsafeMutableRawPointer) {
    let context = Unmanaged<SessionStreamContext>.fromOpaque(contextPointer)
    stateLock.lock()
    activeStreamCount -= 1
    context.takeUnretainedValue().isRunning = false
    stateLock.unlock()
    context.release()
  }

  /// Cancels the native stream of `context`, whose consumer terminated it, if it is still running.
  /// The session can't be used afterwards, as after `cancel()`.
  fileprivate func cancelStream(_ context: SessionStreamContext) {
    stateLock.lock()
    defer { stateLock.unlock() }
    guard context.isRunning, let handle, !isClosed, !isCancelled else { return }
    _ = litert_lm_session_cancel_process(handle)
    LiteRTLMError.clearLastError()
    isCancelled = true
  }

  /// Unregisters the stream of `context`, which ended.
  ///
  /// - Returns: The native session to delete if the session was closed while the stream was
  ///   running and no other stream is running. The caller must delete it outside of the stream
  ///   callback.
  fileprivate func streamDidEnd(_ context: SessionStreamContext) -> OpaquePointer? {
    stateLock.lock()
    defer { stateLock.unlock() }
    activeStreamCount -= 1
    context.isRunning = false
    guard isClosed, activeStreamCount == 0, let handle else { return nil }
    self.handle = nil
    return handle
  }
}

/// Context passed to `sessionStreamCallback`, retained until the stream ends.
private final class SessionStreamContext: @unchecked Sendable {
  let continuation: AsyncThrowingStream<String, Error>.Continuation
  let session: Session

  /// Whether the native stream is running. Guarded by the session's state lock.
  var isRunning = false

  init(continuation: AsyncThrowingStream<String, Error>.Continuation, session: Session) {
    self.continuation = continuation
    self.session = session
  }

  /// Ends the stream and releases the context retained at `contextPointer`.
  func finish(throwing error: Error?, contextPointer: UnsafeMutableRawPointer) {
    let handleToDelete = session.streamDidEnd(self)
    if let error {
      continuation.finish(throwing: error)
    } else {
      continuation.finish()
    }
    // The stream callback must not call LiteRT-LM functions on the session, e.g. delete it, so
    // the last reference to the session, held by the context, is not released on its thread.
    DispatchQueue.global().async {
      if let handleToDelete {
        litert_lm_session_delete(handleToDelete)
      }
      Unmanaged<SessionStreamContext>.fromOpaque(contextPointer).release()
    }
  }
}

/// Bridges the native stream callback of a session to its `AsyncThrowingStream`.
private func sessionStreamCallback(userData: UnsafeMutableRawPointer?, chunk: OpaquePointer?) {
  guard let userData else { return }
  let context = Unmanaged<SessionStreamContext>.fromOpaque(userData).takeUnretainedValue()
  let makeError: (String) -> LiteRTLMError = { .session(.streamFailed($0)) }
  do {
    let errorMessage: UnsafePointer<CChar>? = try LiteRTLMError.get(
      nil, "litert_lm_stream_chunk_get_error", makeError
    ) { litert_lm_stream_chunk_get_error(chunk, $0) }
    if let errorMessage {
      throw streamError(String(cString: errorMessage))
    }
    let text: UnsafePointer<CChar>? = try LiteRTLMError.get(
      nil, "litert_lm_stream_chunk_get_text", makeError
    ) { litert_lm_stream_chunk_get_text(chunk, $0) }
    if let text {
      let string = String(cString: text)
      if !string.isEmpty {
        context.continuation.yield(string)
      }
    }
    let isFinal = try LiteRTLMError.get(false, "litert_lm_stream_chunk_is_final", makeError) {
      litert_lm_stream_chunk_is_final(chunk, $0)
    }
    if isFinal {
      context.finish(throwing: nil, contextPointer: userData)
    }
  } catch {
    context.finish(throwing: error, contextPointer: userData)
  }
}

/// Returns the error of a stream whose chunk carries the native error message `message`.
///
/// The message is the string form of the native status, e.g. "CANCELLED: ..." for a cancelled
/// stream, which maps to `LiteRTLMError.session(.cancelled)`.
private func streamError(_ message: String) -> LiteRTLMError {
  if message.hasPrefix("CANCELLED") {
    return .session(.cancelled)
  }
  return .session(.streamFailed(message))
}

/// Returns the text of the first candidate of the native responses `responses`, which are not
/// consumed, or an empty string if there is none. The Swift API doesn't configure several
/// candidates, so there is at most one.
private func responseText(_ responses: OpaquePointer) throws -> String {
  let makeError: (String) -> LiteRTLMError = { .session(.operationFailed($0)) }
  let numCandidates = try LiteRTLMError.get(
    0, "litert_lm_responses_get_num_candidates", makeError
  ) { litert_lm_responses_get_num_candidates(responses, $0) }
  guard numCandidates > 0 else { return "" }
  var text: UnsafePointer<CChar>? = nil
  let status = litert_lm_responses_get_response_text_at(responses, 0, &text)
  if status == kLiteRtLmStatusNotFound {
    LiteRTLMError.clearLastError()
    return ""
  }
  try LiteRTLMError.check(status, "litert_lm_responses_get_response_text_at", makeError)
  return text.map { String(cString: $0) } ?? ""
}

/// Reads the text scores of `targets` out of the native responses `responses`, which are not
/// consumed.
private func textScores(_ responses: OpaquePointer, targets: [String]) throws -> [TextScore] {
  let makeError: (String) -> LiteRTLMError = { .session(.operationFailed($0)) }
  let numCandidates = try LiteRTLMError.get(
    0, "litert_lm_responses_get_num_candidates", makeError
  ) { litert_lm_responses_get_num_candidates(responses, $0) }
  guard numCandidates == targets.count else {
    throw makeError("expected \(targets.count) scores but got \(numCandidates)")
  }
  return try targets.enumerated().map { index, target in
    let hasScore = try LiteRTLMError.get(false, "litert_lm_responses_has_score_at", makeError) {
      litert_lm_responses_has_score_at(responses, index, $0)
    }
    guard hasScore else {
      throw makeError("no score for target \(index)")
    }
    let score = try LiteRTLMError.get(Float(0), "litert_lm_responses_get_score_at", makeError) {
      litert_lm_responses_get_score_at(responses, index, $0)
    }

    var tokenLength: Int? = nil
    let hasTokenLength = try LiteRTLMError.get(
      false, "litert_lm_responses_has_token_length_at", makeError
    ) { litert_lm_responses_has_token_length_at(responses, index, $0) }
    if hasTokenLength {
      tokenLength = try LiteRTLMError.get(
        0, "litert_lm_responses_get_token_length_at", makeError
      ) { litert_lm_responses_get_token_length_at(responses, index, $0) }
    }

    var tokenScores: [Float] = []
    let hasTokenScores = try LiteRTLMError.get(
      false, "litert_lm_responses_has_token_scores_at", makeError
    ) { litert_lm_responses_has_token_scores_at(responses, index, $0) }
    if hasTokenScores {
      let numTokenScores = try LiteRTLMError.get(
        0, "litert_lm_responses_get_num_token_scores_at", makeError
      ) { litert_lm_responses_get_num_token_scores_at(responses, index, $0) }
      let cTokenScores: UnsafePointer<Float>? = try LiteRTLMError.get(
        nil, "litert_lm_responses_get_token_scores_at", makeError
      ) { litert_lm_responses_get_token_scores_at(responses, index, $0) }
      if let cTokenScores {
        tokenScores = Array(UnsafeBufferPointer(start: cTokenScores, count: numTokenScores))
      }
    }

    return TextScore(
      text: target, score: score, tokenLength: tokenLength, tokenScores: tokenScores)
  }
}

/// Creates the native input data of `inputs` and calls `body` with them, then releases them.
private func withCInputData<T>(
  _ inputs: [InputData], _ body: (UnsafePointer<OpaquePointer?>?, Int) throws -> T
) throws -> T {
  var cInputs: [OpaquePointer?] = []
  cInputs.reserveCapacity(inputs.count)
  defer { cInputs.forEach { litert_lm_input_data_delete($0) } }
  for input in inputs {
    cInputs.append(try input.makeCInputData())
  }
  return try cInputs.withUnsafeBufferPointer { buffer in
    try body(buffer.baseAddress, buffer.count)
  }
}
