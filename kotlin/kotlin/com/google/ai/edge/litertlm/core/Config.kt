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

import com.google.gson.JsonObject

/**
 * Definition of a channel for responses, e.g. thinking channel.
 *
 * @property channelName The channel name. Text from this channel will be written to
 *   [Message.channels] with the [channelName] as the key.
 * @property start A string that marks the start of the channel.
 * @property end A string that marks the end of the channel.
 */
data class Channel(val channelName: String, val start: String, val end: String) {
  internal fun toJson() =
    JsonObject().apply {
      addProperty("channel_name", channelName)
      addProperty("start", start)
      addProperty("end", end)
    }
}

/**
 * Configuration for repetition penalty.
 *
 * When multiple penalties are configured and active, the order of application to output logits
 * during decoding is:
 * 1. Multiplicative penalty ([repetitionPenalty])
 * 2. Subtractive penalties ([presencePenalty] and [frequencyPenalty])
 *
 * @property repetitionPenalty A multiplicative penalty applied to a token's logit if that token has
 *   appeared at least once inside the generated window history (e.g., 1.0 = no penalty, 1.2 =
 *   moderate penalty). Positive logits are divided by this parameter, and negative logits are
 *   multiplied (HuggingFace style). Must be >= 1.0f; values less than 1.0f are automatically
 *   clamped to 1.0f during execution. Defaults to 1.0f when `null`.
 * @property presencePenalty A scalar subtracted from a token's logit if that token has appeared at
 *   least once inside the generated window history. Positive values discourage repetition, while
 *   negative values reward repeating tokens (OpenAI style). Defaults to 0.0f when `null`.
 * @property frequencyPenalty A scalar subtracted from a token's logit, scaled linearly by the
 *   number of times that token has previously appeared inside the generated window history.
 *   Positive values discourage repetition, while negative values reward repeating tokens (OpenAI
 *   style). Defaults to 0.0f when `null`.
 * @property windowSize The maximum number of recent tokens in generation history to consider when
 *   computing penalization. Tokens generated prior to this window are forgotten. A value of 0 means
 *   tracking all infinite generation history. Must be >= 0; negative values are clamped to 0 during
 *   execution. Defaults to 0 when `null`.
 */
data class RepetitionPenaltyConfig
@JvmOverloads
constructor(
  val repetitionPenalty: Float? = null,
  val presencePenalty: Float? = null,
  val frequencyPenalty: Float? = null,
  val windowSize: Int? = null,
) {
  init {
    require(repetitionPenalty == null || repetitionPenalty >= 1.0f) {
      "repetitionPenalty should be >= 1.0, but got $repetitionPenalty."
    }
    require(windowSize == null || windowSize >= 0) {
      "windowSize should be >= 0, but got $windowSize."
    }
  }
}

/**
 * Configuration for no repeat ngram banning.
 *
 * When [noRepeatNgramSize] is set greater than 0, any sequence of tokens (an ngram of that exact
 * length) generated during decoding or present inside the window history can only occur at most
 * once. If generating a candidate token would complete a repeating ngram, that candidate token's
 * logit is set to -inf.
 *
 * @property noRepeatNgramSize The size of ngrams (consecutive token sequences) that are banned from
 *   repeating within the generation history window. If set > 0, when generating the next token
 *   would complete an already observed [noRepeatNgramSize] sequence, the logit of the candidate
 *   token is set to -inf. If set <= 0, no repeat ngram banning is disabled. Negative values are
 *   clamped to 0. Defaults to 0 when `null`.
 * @property windowSize The maximum number of recent tokens in generation history to consider when
 *   checking for repeating ngrams. Tokens generated prior to this window are forgotten. A value of
 *   0 means tracking all infinite generation history. Must be >= 0; negative values are clamped
 *   to 0. If [windowSize] is greater than 0 but less than [noRepeatNgramSize], it is automatically
 *   clamped to [noRepeatNgramSize] during execution so that the ngrams can fit and be tracked.
 *   Defaults to 0 when `null`.
 */
data class NoRepeatNgramConfig
@JvmOverloads
constructor(val noRepeatNgramSize: Int? = null, val windowSize: Int? = null) {
  init {
    require(noRepeatNgramSize == null || noRepeatNgramSize >= 0) {
      "noRepeatNgramSize should be >= 0, but got $noRepeatNgramSize."
    }
    require(windowSize == null || windowSize >= 0) {
      "windowSize should be >= 0, but got $windowSize."
    }
  }
}

/**
 * Configuration for suppressing specific tokens during decoding.
 *
 * @property suppressTokens A list of token IDs to suppress. Banned tokens will have their logit set
 *   to -inf.
 */
data class SuppressTokensConfig(val suppressTokens: Collection<Int>) {
  internal fun getSuppressTokensArray(): IntArray = suppressTokens.toIntArray()
}

/**
 * Configuration for thinking/reasoning generation.
 *
 * @property enableThinking Whether thinking/reasoning generation is enabled.
 * @property thinkingTokenBudget The token budget for thinking/reasoning generation. Defaults to -1
 *   (infinite budget).
 */
data class ThinkingConfig
@JvmOverloads
constructor(val enableThinking: Boolean = true, val thinkingTokenBudget: Int = -1)

/**
 * Backend for the LiteRT-LM engine.
 *
 * This is the Kotlin version of the C++'s `litert::lm::Backend`.
 */
sealed class Backend(val name: String) {

  /**
   * @property threadCount The number of threads to use for CPU backend. When `null` or 0, use the
   *   default value from the native engine.
   * @property numOfThreads Deprecated. Use [threadCount] instead.
   */
  data class CPU(
    val threadCount: Int? = null,
    @Deprecated("Use threadCount instead", ReplaceWith("threadCount")) val numOfThreads: Int? = null,
  ) : Backend("CPU")

  class GPU : Backend("GPU")

  /**
   * @property nativeLibraryDir The directory contains the NPU libraries for [Backend.NPU]. On
   *   Android, for apps with built-in NPU libraries, including NPU libraries delivered as Google
   *   Play Feature modules, set it to [Context.applicationInfo.nativeLibraryDir]. If NPU libraries
   *   are not built-in (downloaded separately or on JVM Desktop), set this path to the directory
   *   containing the libraries.
   */
  data class NPU(val nativeLibraryDir: String = "") : Backend("NPU")

  class GOOGLE_TENSOR : Backend("GOOGLE_TENSOR_ARTISAN")
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
 * @property maxNumImages The maximum number of images the model can handle. When `null`, use the
 *   default value from the model or the engine.
 * @property cacheDir The directory for placing cache files. It should be a directory with write
 *   access. If not set, it uses the directory of the [modelPath]. Set to ":nocache" to disable
 *   caching at all.
 */
data class EngineConfig(
  val modelPath: String,
  val backend: Backend = Backend.CPU(),
  val visionBackend: Backend? = null,
  val audioBackend: Backend? = null,
  val maxNumTokens: Int? = null,
  val maxNumImages: Int? = null,
  val cacheDir: String? = null,
) {
  init {
    require(maxNumTokens == null || maxNumTokens > 0) {
      "maxNumToken must be positive or null (use the default from model or engine)."
    }
    require(maxNumImages == null || maxNumImages > 0) {
      "maxNumImages must be positive or null (use the default from model or engine)."
    }
  }
}

/**
 * Configuration for a LiteRT-LM [Conversation].
 *
 * @property systemInstruction The system instruction for the conversation. If set, it will prepend
 *   to [initialMessages].
 * @property initialMessages The initial messages for the conversation.
 * @property tools A list of tool objects to be used in the conversation.
 * @property samplerConfig Configuration for the sampling process. If `null`, then uses the engine's
 *   default values.
 * @property automaticToolCalling If true, tools will be called automatically. If false, tool calls
 *   will be returned to the user to execute.
 * @property channels A list of channels for the conversation. Each [Channel] is a part of the
 *   model's output that is separate from the primary response, such as a 'thinking' channel.
 *   Channel content will be written to [Message.channels] with the [Channel.channelName] as the
 *   key. If `null`, uses the default channel configuration from the `LlmMetadata`. If empty,
 *   channels will be disabled.
 * @property extraContext Optional context passed to the prompt template rendering.
 * @property loraConfig Configuration for LoRA weights.
 * @property prefillPrefaceOnInit Whether to prefill the preface on initialization. Defaults to
 *   false. Note that this will make createConversation() take longer to finish, so you may want to
 *   call it in a background thread.
 * @property maxOutputToken The maximum number of output tokens per decode step. When `null`, use
 *   the default value from the model or the engine.
 * @property thinkingConfig Configuration for thinking/reasoning generation.
 * @property enableResponseFormat Whether to enable response format (constrained decoding). If true,
 *   initializes the constraint provider LLGuidance.
 */
data class ConversationConfig
@JvmOverloads
constructor(
  val systemInstruction: Contents? = null,
  val initialMessages: List<Message> = listOf(),
  val tools: List<ToolProvider> = listOf(),
  val samplerConfig: SamplerConfig? = null,
  val automaticToolCalling: Boolean = true,
  val channels: List<Channel>? = null,
  val extraContext: Map<String, Any> = emptyMap(),
  val loraConfig: LoraConfig? = null,
  val prefillPrefaceOnInit: Boolean = false,
  val maxOutputToken: Int? = null,
  val thinkingConfig: ThinkingConfig? = null,
  val enableResponseFormat: Boolean = false,
) {
  init {
    require(maxOutputToken == null || maxOutputToken > 0) {
      "maxOutputToken must be positive or null (use the default from model or engine)."
    }
  }
}

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
 * Configuration for LoRA (Low-Rank Adaptation) weights.
 *
 * @property loraPath Optional file path to the LoRA weights file.
 * @property audioLoraPath Optional file path to the Audio LoRA weights file.
 */
data class LoraConfig(val loraPath: String? = null, val audioLoraPath: String? = null)

/**
 * Configuration for a LiteRT-LM [Session].
 *
 * @property samplerConfig Configuration for the sampling process. If `null`, then uses the engine's
 *   default values.
 * @property loraConfig Configuration for LoRA weights.
 */
data class SessionConfig(
  val samplerConfig: SamplerConfig? = null,
  val loraConfig: LoraConfig? = null,
)
