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
import OSLog
import CLiteRTLM

/// Data struct to hold benchmark information. Note that this is an experimental API and may change
/// in the future.
public struct BenchmarkInfo {
  /// The time in seconds to initialize the engine and the conversation.
  public let initTimeInSecond: Double

  /// The time in seconds to the first token.
  public let timeToFirstTokenInSecond: Double

  /// The number of tokens in the last prefill turn.
  public let lastPrefillTokenCount: Int

  /// The number of tokens in the last decode turn.
  public let lastDecodeTokenCount: Int

  /// The number of tokens processed per second in the last prefill.
  public let lastPrefillTokensPerSecond: Double

  /// The number of tokens processed per second in the last decode.
  public let lastDecodeTokensPerSecond: Double
}

/// Runs a benchmark on the LiteRT-LM engine.
///
/// **Note:** This function can take a significant amount of time depending on the model size,
/// device hardware, and the number of prefill and decode tokens. It is strongly recommended to
/// call this method on a background thread to avoid blocking the main thread.
///
/// - Parameters:
///   - modelPath: The path to the model file.
///   - backend: The backend to use for the engine.
///   - prefillTokens: The number of tokens to prefill.
///   - decodeTokens: The number of tokens to decode.
///   - cacheDir: The directory for placing cache files. Set to ":nocache" to disable caching.
///   - prompt: The custom prompt string to tokenize and run. If the tokenized prompt is shorter
///       than `prefillTokens`, the remaining tokens are padded with zero. If it is longer, the
///       prompt is truncated to `prefillTokens`.
/// - Returns: The benchmark info.
/// - Throws: A `LiteRTLMError` if the engine fails to initialize or generate benchmark info.
public func benchmark(
  modelPath: String,
  backend: Backend,
  prefillTokens: Int = 256,
  decodeTokens: Int = 256,
  cacheDir: String? = nil,
  prompt: String = "How are you"
) async throws -> BenchmarkInfo {
  ExperimentalFlags.optIntoExperimentalAPIs()

  // temporarily enable benchmark flag so that Conversation API correctly passes checks later
  let previousBenchmarkFlag = ExperimentalFlags.enableBenchmark
  ExperimentalFlags.enableBenchmark = true
  defer {
    ExperimentalFlags.enableBenchmark = previousBenchmarkFlag
  }

  let engineConfig = try EngineConfig(
    modelPath: modelPath,
    backend: backend,
    maxNumTokens: max(prefillTokens, decodeTokens) + 32,
    cacheDir: cacheDir
  )
  let engine = Engine(engineConfig: engineConfig)
  try await engine.initializeForBenchmark(prefillTokens: prefillTokens, decodeTokens: decodeTokens)

  let conversation = try await engine.createConversation()
  _ = try await conversation.sendMessage(Message(prompt))
  return try conversation.getBenchmarkInfo()
}
