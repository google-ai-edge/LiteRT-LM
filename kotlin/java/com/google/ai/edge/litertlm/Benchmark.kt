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
 */
data class BenchmarkInfo(
  val initTimeInSecond: Double,
  val timeToFirstTokenInSecond: Double,
  val lastPrefillTokenCount: Int,
  val lastDecodeTokenCount: Int,
  val lastPrefillTokensPerSecond: Double,
  val lastDecodeTokensPerSecond: Double,
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
 * @param prefillTokens The number of tokens to prefill.
 * @param decodeTokens The number of tokens to decode.
 * @param cacheDir The directory for placing cache files. It should be a directory with write
 *   access. If not set, it uses the directory of the [modelPath]. Set to ":nocache" to disable
 *   caching at all.
 * @return The benchmark info.
 */
@ExperimentalApi
fun benchmark(
  modelPath: String,
  backend: Backend,
  prefillTokens: Int = 256,
  decodeTokens: Int = 256,
  cacheDir: String? = null,
): BenchmarkInfo {
  val enginePointer =
    LiteRtLmJni.nativeCreateBenchmark(
      modelPath,
      backend.name,
      prefillTokens,
      decodeTokens,
      cacheDir ?: "",
      (backend as? Backend.NPU)?.nativeLibraryDir ?: "",
    )

  try {
    val conversationHandle =
      LiteRtLmJni.nativeCreateConversation(
        enginePointer,
        null, // SamplerConfig
        "[]", // messagesJsonString
        "[]", // toolsDescriptionJsonString
        null, // channelsJsonString
        "{}", // extraContextJsonString
        false, // enableConversationConstrainedDecoding
        ExperimentalFlags.filterChannelContentFromKvCache,
        ExperimentalFlags.overwritePromptTemplate,
      )

    Conversation(conversationHandle).use { conversation ->
      val unused = conversation.sendMessage("Engine ignore this message in this mode.")
      return conversation.getBenchmarkInfo()
    }
  } finally {
    LiteRtLmJni.nativeDeleteEngine(enginePointer)
  }
}
