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

/** Input and output modalities supported by LiteRT-LM models. */
enum class Modality(val value: Int) {
  TEXT(0),
  VISION(1),
  AUDIO(2),
  VIDEO(3),
}

/**
 * Provides information about capabilities and features supported by a LiteRT-LM file.
 *
 * The user is expected to leverage the Capabilities API to investigate the capabilities of a
 * LiteRT-LM file before using it to build an engine instance.
 *
 * ### Example Usage:
 * ```kotlin
 * try {
 *   // 1. Load the model metadata
 *   Capabilities("/path/to/model.litertlm").use { capabilities ->
 *     // 2. Query basic capability flags
 *     val supportsThinking = capabilities.supportsThinking()
 *     val supportsFunctionCall = capabilities.supportsFunctionCalling()
 *     val hasSpeculativeDecoding = capabilities.hasSpeculativeDecodingSupport()
 *
 *     // 3. Inspect context limits and runtime version requirements
 *     val maxContext = capabilities.maxContextTokens()
 *     val isDynamic = capabilities.isDynamicContext()
 *     val minVersion = capabilities.minRuntimeVersion()
 *     if (minVersion != null) {
 *       println("Minimum required LiteRT-LM runtime version: $minVersion")
 *     }
 *
 *     // 4. Check supported input modalities and vision signatures
 *     if (capabilities.inputModalities().vision) {
 *       val visionBudget = capabilities.maxVisionTokenBudget()
 *       val signatures = capabilities.visionSignatureSelection()
 *       if (signatures != null) {
 *         println("Supported vision token capacities: ${signatures.contentToString()}")
 *       }
 *     }
 *
 *     // 5. Inspect hardware backends (ordered by priority), NPU brand, etc.
 *     val textBackends =
 *       capabilities.supportedBackends(Modality.TEXT) // e.g. [CPU, GPU, NPU]
 *     val defaultBackend = textBackends.firstOrNull() // e.g. BackendType.CPU
 *     println("Default backend for text: $defaultBackend")
 *
 *     if (textBackends.contains(BackendType.NPU)) {
 *       val brand = capabilities.npuBrand(Modality.TEXT) // e.g. QUALCOMM, GOOGLE_TENSOR, MEDIATEK
 *       val socName = capabilities.socName(Modality.TEXT)
 *       if (socName != null) {
 *         println("Target NPU SoC: $socName ($brand)") // e.g. "SM8750 (QUALCOMM)"
 *       }
 *     }
 *
 *     // 6. Retrieve default sampler parameters
 *     val sampler = capabilities.defaultSamplerParams()
 *     println("Temperature: ${sampler.temperature}, TopK: ${sampler.topK}, TopP: ${sampler.topP}")
 *   }
 * } catch (e: Exception) {
 *   println("Failed to load model file capabilities: ${e.message}")
 * }
 * ```
 *
 * @param modelPath The file path to the LiteRT-LM model.
 */
class Capabilities(modelPath: String) : AutoCloseable {
  private val lock = Any()

  @Volatile private var handle: Long? = null

  init {
    val ptr = LiteRtLmJni.nativeCreateCapabilities(modelPath)
    if (ptr == 0L) {
      throw LiteRtLmJniException("Failed to load capabilities for model: $modelPath")
    }
    handle = ptr
  }

  /** Checks if the loaded LiteRT-LM file supports speculative decoding. */
  fun hasSpeculativeDecodingSupport(): Boolean {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeHasSpeculativeDecodingSupport(handle!!)
    }
  }

  /** Checks if the loaded LiteRT-LM file supports thinking/reasoning. */
  fun supportsThinking(): Boolean {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeSupportsThinking(handle!!)
    }
  }

  /** Checks if the loaded LiteRT-LM file supports function calling/tool use. */
  fun supportsFunctionCalling(): Boolean {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeSupportsFunctionCalling(handle!!)
    }
  }

  /** Returns the supported input modalities. */
  fun inputModalities(): SupportedModalities {
    synchronized(lock) {
      checkInitialized()
      return SupportedModalities(
        text = LiteRtLmJni.nativeSupportsInputModality(handle!!, 0),
        vision = LiteRtLmJni.nativeSupportsInputModality(handle!!, 1),
        audio = LiteRtLmJni.nativeSupportsInputModality(handle!!, 2),
        video = LiteRtLmJni.nativeSupportsInputModality(handle!!, 3),
      )
    }
  }

  /** Returns the default sampler parameters for the model. */
  fun defaultSamplerParams(): SamplerParameters {
    synchronized(lock) {
      checkInitialized()
      return SamplerParameters(
        type = LiteRtLmJni.nativeSamplerType(handle!!),
        temperature = LiteRtLmJni.nativeSamplerTemp(handle!!),
        topK = LiteRtLmJni.nativeSamplerTopK(handle!!),
        topP = LiteRtLmJni.nativeSamplerTopP(handle!!),
      )
    }
  }

  /** Returns the maximum vision token budget for the model, or -1 if not defined. */
  fun maxVisionTokenBudget(): Int {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeMaxVisionTokenBudget(handle!!)
    }
  }

  /**
   * Gets the maximum supported context tokens for the loaded LiteRT-LM file.
   * - If the model is static ([isDynamicContext] is false), this is the fixed context size.
   * - If the model is dynamic ([isDynamicContext] is true), this is the largest context size that
   *   can be set.
   */
  fun maxContextTokens(): Int {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeMaxContextTokens(handle!!)
    }
  }

  /**
   * Checks if the loaded LiteRT-LM file has dynamic context. Dynamic context means the context size
   * can be configured by the caller up to the maximum limit.
   */
  fun isDynamicContext(): Boolean {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeIsDynamicContext(handle!!)
    }
  }

  /** Returns the list of vision signature selection choices, or null if vision is not supported. */
  fun visionSignatureSelection(): IntArray? {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeVisionSignatureSelection(handle!!)
    }
  }

  /**
   * Returns the minimum LiteRT-LM runtime version required to run this model, or null if not
   * defined.
   */
  fun minRuntimeVersion(): String? {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeMinRuntimeVersion(handle!!)
    }
  }

  /**
   * Returns the list of supported backends for a given modality, ordered by priority (first is
   * default).
   */
  fun supportedBackends(modality: Modality): List<BackendType> {
    synchronized(lock) {
      checkInitialized()
      val backendValues =
        LiteRtLmJni.nativeModalitySupportedBackends(handle!!, modality.value) ?: return emptyList()
      val result = mutableListOf<BackendType>()
      for (v in backendValues) {
        BackendType.fromValue(v)?.let { result.add(it) }
      }
      return result
    }
  }

  /** Returns the detected NPU brand for a given modality, or NpuBrand.UNKNOWN. */
  fun npuBrand(modality: Modality): NpuBrand {
    synchronized(lock) {
      checkInitialized()
      val brandVal = LiteRtLmJni.nativeModalityNpuBrand(handle!!, modality.value)
      return NpuBrand.fromValue(brandVal)
    }
  }

  /** Returns the NPU SoC name string for a given modality, or null if not set. */
  fun socName(modality: Modality): String? {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeModalitySocName(handle!!, modality.value)
    }
  }

  /** Closes the loaded capabilities and releases underlying resources. */
  override fun close() {
    synchronized(lock) {
      val ptr = handle ?: return
      LiteRtLmJni.nativeDeleteCapabilities(ptr)
      handle = null
    }
  }

  private fun checkInitialized() {
    check(handle != null) { "Capabilities instance is already closed." }
  }
}
