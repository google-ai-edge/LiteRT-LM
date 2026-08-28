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

/**
 * Manages the lifecycle of a LiteRT-LM Embedding Engine, providing an interface for interacting
 * with native embedding models.
 *
 * Example usage:
 * ```
 * val config = EmbeddingEngineConfig(modelPath = "...")
 * val engine = EmbeddingEngine(config)
 * engine.initialize()
 * val response = engine.computeEmbedding(listOf(InputData.Text("Hello world")))
 * engine.close()
 * ```
 *
 * @param config The configuration for the embedding engine.
 */
class EmbeddingEngine(val config: EmbeddingEngineConfig) : AutoCloseable {
  private val lock = Any()

  @Volatile private var handle: Long? = null

  /** Returns `true` if the engine is initialized and ready for use; `false` otherwise. */
  fun isInitialized(): Boolean {
    return handle != null
  }

  /**
   * Initializes the native LiteRT-LM Embedding Engine.
   *
   * @throws IllegalStateException if the engine has already been initialized.
   */
  fun initialize() {
    synchronized(lock) {
      check(!isInitialized()) { "EmbeddingEngine is already initialized." }

      val mainBackendNumThreads =
        (config.backend as? Backend.CPU)?.threadCount?.takeIf { it > 0 } ?: -1
      val audioBackendNumThreads =
        (config.audioBackend as? Backend.CPU)?.threadCount?.takeIf { it > 0 } ?: -1

      handle =
        LiteRtLmJni.nativeCreateEmbeddingEngine(
          config.modelFd ?: -1,
          config.modelPath ?: "",
          config.backend.name,
          config.visionBackend?.name ?: "",
          config.audioBackend?.name ?: "",
          config.cacheDir ?: "",
          (config.backend as? Backend.NPU)?.nativeLibraryDir ?: "",
          (config.visionBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
          (config.audioBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
          mainBackendNumThreads,
          audioBackendNumThreads,
          config.maxInputLength ?: -1,
          config.visionTokensPerImage ?: -1,
        )
    }
  }

  /**
   * Computes embedding for multimodal input contents.
   *
   * @param contents The list of [InputData] items to compute embeddings for.
   * @param options Additional options for embedding generation.
   * @return The [EmbeddingResponse] containing the calculated embedding vector(s).
   * @throws IllegalStateException if the engine is not initialized.
   */
  @JvmOverloads
  fun computeEmbedding(
    contents: List<InputData>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ): EmbeddingResponse {
    synchronized(lock) {
      checkInitialized()
      return LiteRtLmJni.nativeComputeEmbedding(
        handle!!,
        contents.toTypedArray(),
        options.normalize,
        options.insertSpecialTokens,
        options.outputSize,
        options.visionTokensPerImage,
      )
    }
  }

  /**
   * Computes embeddings for a batch of multimodal input requests.
   *
   * @param contentsBatch A list of input requests, where each request is a list of [InputData].
   * @param options Additional options for embedding generation.
   * @return A list of [EmbeddingResponse] objects for each request in the batch.
   * @throws IllegalStateException if the engine is not initialized.
   */
  @JvmOverloads
  fun computeEmbeddingBatch(
    contentsBatch: List<List<InputData>>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ): List<EmbeddingResponse> {
    synchronized(lock) {
      checkInitialized()
      val nativeBatch = contentsBatch.map { it.toTypedArray() }.toTypedArray()
      return LiteRtLmJni.nativeComputeEmbeddingBatch(
          handle!!,
          nativeBatch,
          options.normalize,
          options.insertSpecialTokens,
          options.outputSize,
          options.visionTokensPerImage,
        )
        .toList()
    }
  }

  /**
   * Closes the engine and releases the native embedding engine's resources.
   *
   * @throws IllegalStateException if the engine is not initialized.
   */
  override fun close() {
    synchronized(lock) {
      checkInitialized()

      LiteRtLmJni.nativeDeleteEmbeddingEngine(handle!!)
      handle = null
    }
  }

  /** Throws [IllegalStateException] if the engine is not initialized. */
  private fun checkInitialized() {
    check(isInitialized()) { "EmbeddingEngine is not initialized." }
  }
}
