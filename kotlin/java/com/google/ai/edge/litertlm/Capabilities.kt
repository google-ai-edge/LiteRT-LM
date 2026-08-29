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

/**
 * Provides information about capabilities and features supported by a LiteRT-LM file.
 *
 * The users is expected to leverage the Capabilities API to investigate the capabilities of a
 * LiteRT-LM file before using it to build a LiteRtLmEngine instance.
 *
 * ### Example Usage:
 * ```kotlin
 * // 1. Load the model metadata
 * try {
 *   Capabilities("/path/to/model.litertlm").use { capabilities ->
 *     // 2. Query basic capability flags
 *     val supportsThinking = capabilities.supportsThinking()
 *     val supportsFunctionCall = capabilities.supportsFunctionCalling()
 *     val hasSpeculativeDecoding = capabilities.hasSpeculativeDecodingSupport()
 *
 *     // 3. Check supported input modalities
 *     if (capabilities.inputModalities().vision) {
 *       println("Vision input is supported!")
 *     }
 *
 *     // 4. Check vision token budget
 *     val visionBudget = capabilities.maxVisionTokenBudget()
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
