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
  /// The time in seconds to create the engine. Session and conversation creation are not included.
  /// 0 if the engine didn't report it.
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

  /// The max number of tokens, i.e. the KV cache size, that the engine used. It can differ from the
  /// requested one, e.g. for a model whose KV cache size is fixed when it is compiled. 0 if the
  /// engine didn't report it.
  public let maxNumTokens: Int
}

extension BenchmarkInfo {
  /// Reads the benchmark info out of the native benchmark info `cBenchmarkInfo`, which is not
  /// consumed.
  ///
  /// - Parameters:
  ///   - cBenchmarkInfo: The native benchmark info.
  ///   - makeError: Builds the error to throw from a description of a failed native call.
  /// - Throws: The error built by `makeError` if a native accessor fails.
  init(cBenchmarkInfo: OpaquePointer, makeError: (String) -> LiteRTLMError) throws {
    let numPrefillTurns = try LiteRTLMError.get(
      0, "litert_lm_benchmark_info_get_num_prefill_turns", makeError
    ) { litert_lm_benchmark_info_get_num_prefill_turns(cBenchmarkInfo, $0) }
    let numDecodeTurns = try LiteRTLMError.get(
      0, "litert_lm_benchmark_info_get_num_decode_turns", makeError
    ) { litert_lm_benchmark_info_get_num_decode_turns(cBenchmarkInfo, $0) }

    let initTimeInSecond = try LiteRTLMError.get(
      0.0, "litert_lm_benchmark_info_get_total_init_time_in_second", makeError
    ) { litert_lm_benchmark_info_get_total_init_time_in_second(cBenchmarkInfo, $0) }
    let timeToFirstTokenInSecond = try LiteRTLMError.get(
      0.0, "litert_lm_benchmark_info_get_time_to_first_token", makeError
    ) { litert_lm_benchmark_info_get_time_to_first_token(cBenchmarkInfo, $0) }

    var lastPrefillTokenCount = 0
    var lastPrefillTokensPerSec = 0.0
    if numPrefillTurns > 0 {
      lastPrefillTokenCount = try Int(
        LiteRTLMError.get(
          0, "litert_lm_benchmark_info_get_prefill_token_count_at", makeError
        ) {
          litert_lm_benchmark_info_get_prefill_token_count_at(
            cBenchmarkInfo, numPrefillTurns - 1, $0)
        })
      lastPrefillTokensPerSec = try LiteRTLMError.get(
        0.0, "litert_lm_benchmark_info_get_prefill_tokens_per_sec_at", makeError
      ) {
        litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
          cBenchmarkInfo, numPrefillTurns - 1, $0)
      }
    }

    var lastDecodeTokenCount = 0
    var lastDecodeTokensPerSec = 0.0
    if numDecodeTurns > 0 {
      lastDecodeTokenCount = try Int(
        LiteRTLMError.get(
          0, "litert_lm_benchmark_info_get_decode_token_count_at", makeError
        ) {
          litert_lm_benchmark_info_get_decode_token_count_at(
            cBenchmarkInfo, numDecodeTurns - 1, $0)
        })
      lastDecodeTokensPerSec = try LiteRTLMError.get(
        0.0, "litert_lm_benchmark_info_get_decode_tokens_per_sec_at", makeError
      ) {
        litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
          cBenchmarkInfo, numDecodeTurns - 1, $0)
      }
    }

    let maxNumTokens = try Int(
      LiteRTLMError.get(
        Int32(0), "litert_lm_benchmark_info_get_max_num_tokens", makeError
      ) { litert_lm_benchmark_info_get_max_num_tokens(cBenchmarkInfo, $0) })

    self.init(
      initTimeInSecond: initTimeInSecond,
      timeToFirstTokenInSecond: timeToFirstTokenInSecond,
      lastPrefillTokenCount: lastPrefillTokenCount,
      lastDecodeTokenCount: lastDecodeTokenCount,
      lastPrefillTokensPerSecond: lastPrefillTokensPerSec,
      lastDecodeTokensPerSecond: lastDecodeTokensPerSec,
      maxNumTokens: maxNumTokens
    )
  }
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
///   - visionBackend: The backend to use for the vision executor. If `nil`, the vision executor
///       is not initialized.
///   - audioBackend: The backend to use for the audio executor. If `nil`, the audio executor is
///       not initialized.
///   - prefillTokens: The number of tokens to prefill.
///   - decodeTokens: The number of tokens to decode.
///   - cacheDir: The directory for placing cache files. Set to ":nocache" to disable caching.
///   - prompt: The custom prompt string to tokenize and run. If the tokenized prompt is shorter
///       than `prefillTokens`, the remaining tokens are padded with zero. If it is longer, the
///       prompt is truncated to `prefillTokens`.
///   - maxNumTokens: The maximum number of the sum of input and output tokens, i.e. the size of the
///       KV cache, like `EngineConfig.maxNumTokens`. When `nil`, use the default value from the
///       model or the engine. Note that the KV cache size affects the prefill and decode speed.
///   - speculativeDecodingConfig: The speculative decoding configuration of the benchmarked engine
///       (see `EngineConfig.speculativeDecodingConfig`). `nil` keeps the runtime default (`.off`),
///       unless the deprecated `ExperimentalFlags.enableSpeculativeDecoding` is set.
///   - samplerConfig: The sampler configuration of the benchmarked conversation. `nil` uses the
///       model's default sampler, like `ConversationConfig.samplerConfig`.
/// - Returns: The benchmark info.
/// - Throws: A `LiteRTLMError` if the engine fails to initialize or generate benchmark info.
public func benchmark(
  modelPath: String,
  backend: Backend,
  visionBackend: Backend? = nil,
  audioBackend: Backend? = nil,
  prefillTokens: Int = 256,
  decodeTokens: Int = 256,
  cacheDir: String? = nil,
  prompt: String = "How are you",
  maxNumTokens: Int? = nil,
  speculativeDecodingConfig: SpeculativeDecodingConfig? = nil,
  samplerConfig: SamplerConfig? = nil
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
    visionBackend: visionBackend,
    audioBackend: audioBackend,
    maxNumTokens: maxNumTokens,
    cacheDir: cacheDir,
    speculativeDecodingConfig: speculativeDecodingConfig
  )
  let engine = Engine(engineConfig: engineConfig)
  try await engine.initializeForBenchmark(prefillTokens: prefillTokens, decodeTokens: decodeTokens)

  let conversation = try await engine.createConversation(
    with: ConversationConfig(samplerConfig: samplerConfig))
  _ = try await conversation.sendMessage(Message(prompt))
  return try conversation.getBenchmarkInfo()
}
