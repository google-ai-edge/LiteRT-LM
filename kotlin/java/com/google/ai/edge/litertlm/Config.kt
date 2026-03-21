/*
 * Copyright 2025 Google LLC
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
 * Backend for the LiteRT-LM engine.
 *
 * This is the Kotlin version of the C++'s `litert::lm::Backend`.
 */
sealed class Backend(val name: String) {

  /**
   * @property numOfThreads The number of threads to use for CPU backend. When `null` or 0, use the
   *   default value from the native engine.
   */
  data class CPU(val numOfThreads: Int? = null) : Backend("CPU")

  class GPU : Backend("GPU")

  /**
   * @property nativeLibraryDir The directory contains the NPU libraries for [Backend.NPU]. On
   *   Android, for apps with built-in NPU libraries, including NPU libraries delivered as Google
   *   Play Feature modules, set it to [Context.applicationInfo.nativeLibraryDir]. If NPU libraries
   *   are not built-in (downloaded separately or on JVM Desktop), set this path to the directory
   *   containing the libraries.
   */
  data class NPU(val nativeLibraryDir: String = "") : Backend("NPU")
}

/**
 * Returns the directory containing the native libraries for the NPU backend. Returns an empty
 * string if not applicable.
 */
internal fun Backend.NPU.getNativeLibraryDir(): String {
  @OptIn(ExperimentalApi::class)
  // Uses ExperimentalFlags.npuLibrariesDir for backward compatibility.
  return this.nativeLibraryDir.ifEmpty { ExperimentalFlags.npuLibrariesDir }
}

/**
 * Configuration for the LiteRT-LM engine.
 *
 * @property modelPath The file path to the LiteRT-LM model.
 * @property backend The backend to use for the engine.
 * @property visionBackend The backend to use for the vision executor. If null, vision executor will
 *   not be initialized.
 * @property audioBackend The backend to use for the audio executor. If null, audio executor will
 *   not be initialized.
 * @property maxNumTokens The maximum number of the sum of input and output tokens. It is equivalent
 *   to the size of the kv-cache. When `null`, use the default value from the model or the engine.
 * @property cacheDir The directory for placing cache files. It should be a directory with write
 *   access. If not set, it uses the directory of the [modelPath]. Set to ":nocache" to disable
 *   caching at all.
 * @property llmMetaData Optional: A string in textproto format (LlmMetadata) to override the
 *   model's existing metadata. If not set, the model's default metadata is used.
 */
data class EngineConfig(
  val modelPath: String,
  val backend: Backend = Backend.CPU(),
  val visionBackend: Backend? = null,
  val audioBackend: Backend? = null,
  val maxNumTokens: Int? = null,
  val cacheDir: String? = null,
  val llmMetaData: String? = null,
) {
  init {
    require(maxNumTokens == null || maxNumTokens > 0) {
      "maxNumToken must be positive or null (use the default from model or engine)."
    }
  }
}

/**
 * Configuration for a LiteRT-LM [Conversation].
 *
 * @property systemInstruction The system instruction for the conversation.
 * @property initialMessages The initial messages for the conversation.
 * @property tools A list of tool objects to be used in the conversation.
 * @property samplerConfig Configuration for the sampling process. If `null`, then uses the engine's
 *   default values.
 * @property systemMessage The system message to be used in the conversation. If set, it will
 *   prepend to [initialMessages].
 */
data class ConversationConfig(
  val systemInstruction: Contents? = null,
  val initialMessages: List<Message> = listOf(),
  // TODO(b/476130607): Switch the type to List<ToolProvider>.
  val tools: List<Any> = listOf(),
  val samplerConfig: SamplerConfig? = null,
  @Deprecated("Use systemInstruction instead. e.g., systemInstrction = Contents.of(\"Be helpful\")")
  val systemMessage: Message? = null,
  val automaticToolCalling: Boolean = true,
)

/**
 * Configuration for the sampling process.
 *
 * @property topK The number of top logits used during sampling.
 * @property topP The cumulative probability threshold for nucleus sampling.
 * @property temperature The temperature to use for sampling.
 * @property seed The seed to use for randomization. Default to 0 (same default as engine code).
 */
data class SamplerConfig(
  val topK: Int,
  val topP: Double,
  val temperature: Double,
  val seed: Int = 0,
) {
  init {
    require(topK > 0) { "topK should be positive, but got $topK." }
    require(topP >= 0 && topP <= 1) { "topP should between 0 and 1 inclusively, but got $topP." }
    require(temperature >= 0) { "temperature should be non-negative, but got $temperature." }
  }
}

/**
 * Configuration for a LiteRT-LM [Session].
 *
 * @property samplerConfig Configuration for the sampling process. If `null`, then uses the engine's
 *   default values.
 */
data class SessionConfig(val samplerConfig: SamplerConfig? = null)
