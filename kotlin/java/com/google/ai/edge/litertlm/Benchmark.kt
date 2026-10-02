/*
 * Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.litertlm

/**
 * Data class to hold benchmark information.
 *
 * @property initTimeInSecond The time in seconds to initialize the engine and the conversation.
 * @property timeToFirstTokenInSecond The time in seconds to the first token.
 * @property lastPrefillTokenCount The number of tokens in the last prefill. Returns 0 if there was
 *   no prefill.
 * @property lastDecodeTokenCount The number of tokens in the last decode. Returns 0 if there was no
 *   decode.
 * @property lastPrefillTokensPerSecond The number of tokens processed per second in the last
 *   prefill.
 * @property lastDecodeTokensPerSecond The number of tokens processed per second in the last decode.
 * @property markDurationsInSecond Map of runtime stage mark names (recorded via pairs of
 *   `BenchmarkInfo::TimeMarkDelta(mark_name)` calls in the C++ runtime) to the duration in seconds
 *   of the most recent measured interval for that stage. Common keys recorded during `benchmark()`
 *   execution include:
 *   - `"vision_executor"`: Time spent encoding an input image into vision embeddings via the vision
 *     executor during prefill (present when `visionBackend` is configured and an image input is
 *     provided).
 *   - `"audio_executor"`: Time spent encoding an input audio clip into audio embeddings via the
 *     audio executor during prefill (present when `audioBackend` is configured and an audio input
 *     is provided).
 *   - `"executor_decode"`: Time spent in the LLM executor `Decode` call for the last decoded token
 *     (when using an external sampler).
 *   - `"sampling"`: Time spent sampling the next token from logits for the last decoded token (when
 *     using an external sampler).
 *   - `"executor_decode_and_sample"`: Time spent in the combined `Decode` and sampling call for the
 *     last decoded token (when the executor performs sampling internally).
 */
data class BenchmarkInfo(
  val initTimeInSecond: Double,
  val timeToFirstTokenInSecond: Double,
  val lastPrefillTokenCount: Int,
  val lastDecodeTokenCount: Int,
  val lastPrefillTokensPerSecond: Double,
  val lastDecodeTokensPerSecond: Double,
  val markDurationsInSecond: Map<String, Double> = emptyMap(),
)

/**
 * Runs a benchmark on the LiteRT-LM engine.
 *
 * **Note:** This function can take a significant amount of time depending on the model size, device
 * hardware and the number of prefill and decode tokens. If applied, it is strongly recommended to
 * call this method on a background thread to avoid blocking the main thread.
 *
 * @param modelPath The path to the model file.
 * @param backend The backend to use for the engine.
 * @param visionBackend The backend to use for the vision executor. If null, the vision executor
 *   will not be initialized.
 * @param audioBackend The backend to use for the audio executor. If null, the audio executor will
 *   not be initialized.
 * @param prefillTokens The number of tokens to prefill.
 * @param decodeTokens The number of tokens to decode.
 * @param cacheDir The directory for placing cache files. It should be a directory with write
 *   access. If not set, it uses the directory of the [modelPath]. Set to ":nocache" to disable
 *   caching at all.
 * @param contents The contents to send to the conversation, defaulting to `Contents.of("How are
 *   you")`. May include multimodal inputs (such as images or audio alongside text) when
 *   benchmarking vision or audio encoders. For the last non-empty text chunk, if the tokenized text
 *   is shorter than [prefillTokens], the remaining tokens are padded with zero; if it is longer, it
 *   is truncated to [prefillTokens].
 * @param repetitionPenaltyConfig Applies repetition, presence and/or frequency penalties to every
 *   decode step. This installs a `RepetitionPenaltyConstraint`, whose sparse mask goes through the
 *   same logit-mask runner as grammar constraints, so it is a cheap way to make a benchmark
 *   exercise the masking path. `null` disables the penalties.
 * @return The benchmark info.
 */
@ExperimentalApi
fun benchmark(
  modelPath: String,
  backend: Backend,
  visionBackend: Backend? = null,
  audioBackend: Backend? = null,
  prefillTokens: Int = 256,
  decodeTokens: Int = 256,
  cacheDir: String? = null,
  contents: Contents = Contents.of("How are you"),
  repetitionPenaltyConfig: RepetitionPenaltyConfig? = null,
): BenchmarkInfo {
  val enginePointer =
    LiteRtLmJni.nativeCreateBenchmark(
      modelPath,
      backend.name,
      visionBackend?.name ?: "",
      audioBackend?.name ?: "",
      prefillTokens,
      decodeTokens,
      cacheDir ?: "",
      (backend as? Backend.NPU)?.nativeLibraryDir ?: "",
      (visionBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
      (audioBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
      ExperimentalFlags.enableSpeculativeDecoding,
      ExperimentalFlags.enableYnnpack,
    )

  try {
    // Keep backward compatibility with deprecated ExperimentalFlags.overwritePromptTemplate;
    // this fallback will be removed when the deprecated property is deleted.
    @Suppress("DEPRECATION")
    val conversationHandle =
      LiteRtLmJni.nativeCreateConversation(
        enginePointer,
        null, // SamplerConfig
        "[]", // messagesJsonString
        "[]", // toolsDescriptionJsonString
        null, // channelsJsonString
        "{}", // extraContextJsonString
        ExperimentalFlags.enableConversationConstrainedDecoding,
        ExperimentalFlags.filterChannelContentFromKvCache,
        ExperimentalFlags.overwritePromptTemplate,
        null, // loraPath
        null, // audioLoraPath
        false, // prefillPrefaceOnInit
        -1, // maxOutputToken
        null, // thinkingConfig
        false, // enableResponseFormat
        null, // enableSpeculativeDecoding
      )

    Conversation(conversationHandle).use { conversation ->
      val unused =
        conversation.sendMessage(contents, repetitionPenaltyConfig = repetitionPenaltyConfig)
      return conversation.getBenchmarkInfo()
    }
  } finally {
    LiteRtLmJni.nativeDeleteEngine(enginePointer)
  }
}
