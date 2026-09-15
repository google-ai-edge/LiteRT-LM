/*
 * Copyright 2026 Google LLC
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

import kotlin.jvm.Volatile

data class SupportedModalities(
  val text: Boolean,
  val vision: Boolean,
  val audio: Boolean,
  val video: Boolean,
)

// TODO: b/554164915 - Reuse SamplerConfig instead of SamplerParameters (matching Python),
// once SamplerConfig can support nullable fields to represent unset/0 values from models.
data class SamplerParameters(val type: Int, val temperature: Float, val topK: Int, val topP: Float)

/** Hardware backends supported by LiteRT-LM models. */
enum class BackendType(val value: Int) {
  CPU(1),
  GPU(2),
  NPU(3);

  companion object {
    fun fromValue(value: Int): BackendType? = entries.firstOrNull { it.value == value }
  }
}

/** NPU brand options. */
enum class NpuBrand(val value: Int) {
  UNKNOWN(0),
  QUALCOMM(1),
  GOOGLE_TENSOR(2),
  MEDIATEK(3),
  INTEL(4),
  SAMSUNG(5);

  companion object {
    fun fromValue(value: Int): NpuBrand = entries.firstOrNull { it.value == value } ?: UNKNOWN
  }
}

/** Type of the LiteRT-LM model. */
enum class ModelType(val value: Int) {
  UNKNOWN(0),
  LLM(1),
  EMBEDDING(2);

  companion object {
    fun fromValue(value: Int): ModelType = entries.firstOrNull { it.value == value } ?: UNKNOWN
  }
}

/** Input and output modalities supported by LiteRT-LM models. */
enum class Modality(val value: Int) {
  TEXT(0),
  VISION(1),
  AUDIO(2),
  VIDEO(3),
}

/** Capabilities specific to Large Language Models (LLM). */
interface LlmCapability {
  /** Checks if the loaded LiteRT-LM file supports speculative decoding. */
  fun hasSpeculativeDecodingSupport(): Boolean

  /** Checks if the loaded LiteRT-LM file supports thinking/reasoning. */
  fun supportsThinking(): Boolean

  /** Checks if the loaded LiteRT-LM file supports function calling/tool use. */
  fun supportsFunctionCalling(): Boolean

  /** Returns the default sampler parameters for the model. */
  fun defaultSamplerParams(): SamplerParameters

  /**
   * Checks if the loaded LiteRT-LM file has dynamic context. Dynamic context means the context size
   * can be configured by the caller up to the maximum limit.
   */
  fun isDynamicContext(): Boolean

  /** Returns the maximum vision token budget for the model, or -1 if not defined. */
  fun maxVisionTokenBudget(): Int

  /** Returns the list of vision signature selection choices, or null if vision is not supported. */
  fun visionSignatureSelection(): IntArray?
}

/** Capabilities specific to Embedding models. */
interface EmbeddingCapability {
  /** Returns the output embedding dimension, or null if not defined. */
  fun dimension(): Int?

  /** Returns the list of supported embedding signature lengths, or null if not defined. */
  fun signatureSelection(): IntArray?

  /**
   * Checks if the loaded LiteRT-LM file has dynamic context. Dynamic context means the context size
   * can be configured by the caller up to the maximum limit.
   */
  fun isDynamicContext(): Boolean

  /** Returns the maximum vision token budget for the model, or -1 if not defined. */
  fun maxVisionTokenBudget(): Int

  /** Returns the list of vision signature selection choices, or null if vision is not supported. */
  fun visionSignatureSelection(): IntArray?
}

/**
 * Provides information about capabilities and metadata of a LiteRT-LM file.
 *
 * The user is expected to leverage the ModelInfo API to investigate the metadata of a LiteRT-LM
 * file before using it to build an engine instance.
 *
 * ### Example Usage:
 * ```kotlin
 * try {
 *   // 1. Load the model metadata
 *   ModelInfo.from("/path/to/model.litertlm").use { modelInfo ->
 *     // 2. Pattern-match using Kotlin sealed class
 *     when (modelInfo) {
 *       is ModelInfo.Llm -> {
 *         val supportsThinking = modelInfo.supportsThinking()
 *         val supportsFunctionCall = modelInfo.supportsFunctionCalling()
 *         val hasSpeculativeDecoding = modelInfo.hasSpeculativeDecodingSupport()
 *         val sampler = modelInfo.defaultSamplerParams()
 *         println("Temp: ${sampler.temperature}, TopK: ${sampler.topK}, TopP: ${sampler.topP}")
 *       }
 *       is ModelInfo.Embedding -> {
 *         val dim = modelInfo.dimension() // e.g. 768
 *         val signatures = modelInfo.signatureSelection() // e.g. [128, 256, 512]
 *       }
 *       is ModelInfo.Unknown -> {
 *         println("Unknown model type")
 *       }
 *     }
 *
 *     // 3. Inspect shared context limits and runtime version requirements
 *     val maxContext = modelInfo.maxContextTokens()
 *     val minVersion = modelInfo.minRuntimeVersion()
 *     if (minVersion != null) {
 *       println("Minimum required LiteRT-LM runtime version: $minVersion")
 *     }
 *
 *     // 4. Inspect hardware backends (ordered by priority), NPU brand, etc.
 *     val textBackends = modelInfo.supportedBackends(Modality.TEXT)
 *     val defaultBackend = textBackends.firstOrNull()
 *     println("Default backend for text: $defaultBackend")
 *
 *     if (textBackends.contains(BackendType.NPU)) {
 *       val brand = modelInfo.npuBrand(Modality.TEXT)
 *       val socName = modelInfo.socName(Modality.TEXT)
 *       if (socName != null) {
 *         println("Target NPU SoC: $socName ($brand)")
 *       }
 *     }
 *   }
 * } catch (e: Exception) {
 *   println("Failed to load model file info: ${e.message}")
 * }
 * ```
 */
sealed class ModelInfo protected constructor(ptr: Long) : AutoCloseable {
  private val lock = Any()

  @Volatile internal var handle: Long? = ptr

  abstract val modelType: ModelType

  /** Returns the supported input modalities. */
  fun inputModalities(): SupportedModalities = withHandle {
    SupportedModalities(
      text = LiteRtLmJni.nativeSupportsInputModality(it, 0),
      vision = LiteRtLmJni.nativeSupportsInputModality(it, 1),
      audio = LiteRtLmJni.nativeSupportsInputModality(it, 2),
      video = LiteRtLmJni.nativeSupportsInputModality(it, 3),
    )
  }

  /** Returns the maximum vision token budget for the model, or -1 if not defined. */
  fun maxVisionTokenBudget(): Int = withHandle { LiteRtLmJni.nativeMaxVisionTokenBudget(it) }

  /**
   * Gets the maximum supported context tokens for the loaded LiteRT-LM file.
   * - If the model is static, this is the fixed context size.
   * - If the model is dynamic, this is the largest context size that can be set.
   */
  fun maxContextTokens(): Int = withHandle { LiteRtLmJni.nativeMaxContextTokens(it) }

  /**
   * Checks if the loaded LiteRT-LM file has dynamic context. Dynamic context means the context size
   * can be configured by the caller up to the maximum limit.
   */
  open fun isDynamicContext(): Boolean = withHandle { LiteRtLmJni.nativeIsDynamicContext(it) }

  /** Returns the list of vision signature selection choices, or null if vision is not supported. */
  open fun visionSignatureSelection(): IntArray? = withHandle {
    LiteRtLmJni.nativeVisionSignatureSelection(it)
  }

  /**
   * Returns the minimum LiteRT-LM runtime version required to run this model, or null if not
   * defined.
   */
  fun minRuntimeVersion(): String? = withHandle { LiteRtLmJni.nativeMinRuntimeVersion(it) }

  /**
   * Returns the list of supported backends for a given modality, ordered by priority (first is
   * default).
   */
  fun supportedBackends(modality: Modality): List<BackendType> = withHandle {
    val backendValues =
      LiteRtLmJni.nativeModalitySupportedBackends(it, modality.value)
        ?: return@withHandle emptyList()
    val result = mutableListOf<BackendType>()
    for (v in backendValues) {
      BackendType.fromValue(v)?.let { result.add(it) }
    }
    result
  }

  /** Returns the detected NPU brand for a given modality, or NpuBrand.UNKNOWN. */
  fun npuBrand(modality: Modality): NpuBrand = withHandle {
    val brandVal = LiteRtLmJni.nativeModalityNpuBrand(it, modality.value)
    NpuBrand.fromValue(brandVal)
  }

  /** Returns the NPU SoC name string for a given modality, or null if not set. */
  fun socName(modality: Modality): String? = withHandle {
    LiteRtLmJni.nativeModalitySocName(it, modality.value)
  }

  /** Closes the loaded model info and releases underlying resources. */
  override fun close() {
    synchronized(lock) {
      val ptr = handle ?: return
      LiteRtLmJni.nativeDeleteModelInfo(ptr)
      handle = null
    }
  }

  internal fun <T> withHandle(block: (Long) -> T): T {
    synchronized(lock) {
      val h = checkNotNull(handle) { "ModelInfo instance is already closed." }
      return block(h)
    }
  }

  /** Capabilities specific to Large Language Models (LLM). */
  class Llm internal constructor(ptr: Long) : ModelInfo(ptr), LlmCapability {
    override val modelType: ModelType = ModelType.LLM

    override fun hasSpeculativeDecodingSupport(): Boolean = withHandle {
      LiteRtLmJni.nativeHasSpeculativeDecodingSupport(it)
    }

    override fun supportsThinking(): Boolean = withHandle { LiteRtLmJni.nativeSupportsThinking(it) }

    override fun supportsFunctionCalling(): Boolean = withHandle {
      LiteRtLmJni.nativeSupportsFunctionCalling(it)
    }

    override fun defaultSamplerParams(): SamplerParameters = withHandle {
      SamplerParameters(
        type = LiteRtLmJni.nativeSamplerType(it),
        temperature = LiteRtLmJni.nativeSamplerTemp(it),
        topK = LiteRtLmJni.nativeSamplerTopK(it),
        topP = LiteRtLmJni.nativeSamplerTopP(it),
      )
    }

    override fun isDynamicContext(): Boolean = super.isDynamicContext()

    override fun visionSignatureSelection(): IntArray? = super.visionSignatureSelection()
  }

  /** Capabilities specific to Embedding models. */
  class Embedding internal constructor(ptr: Long) : ModelInfo(ptr), EmbeddingCapability {
    override val modelType: ModelType = ModelType.EMBEDDING

    override fun dimension(): Int? = withHandle {
      val dim = LiteRtLmJni.nativeEmbeddingDimension(it)
      if (dim > 0) dim else null
    }

    override fun signatureSelection(): IntArray? = withHandle {
      LiteRtLmJni.nativeEmbeddingSignatureSelection(it)
    }

    override fun isDynamicContext(): Boolean = super.isDynamicContext()

    override fun visionSignatureSelection(): IntArray? = super.visionSignatureSelection()
  }

  /** Default fallback type when model type is UNKNOWN. */
  class Unknown internal constructor(ptr: Long) : ModelInfo(ptr) {
    override val modelType: ModelType = ModelType.UNKNOWN
  }

  companion object {
    @JvmStatic
    fun from(modelPath: String): ModelInfo {
      val ptr = LiteRtLmJni.nativeCreateModelInfo(modelPath)
      if (ptr == 0L) {
        throw LiteRtLmJniException("Failed to load model info for model: $modelPath")
      }
      return when (ModelType.fromValue(LiteRtLmJni.nativeModelType(ptr))) {
        ModelType.LLM -> Llm(ptr)
        ModelType.EMBEDDING -> Embedding(ptr)
        else -> Unknown(ptr)
      }
    }
  }
}
