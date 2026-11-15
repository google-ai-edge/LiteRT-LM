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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.withContext

/**
 * Manages the lifecycle of a LiteRT-LM Embedding Engine, providing an interface for interacting
 * with native embedding models.
 *
 * Example usage:
 * ```
 * suspend fun runEmbedding() {
 *   val config = EmbeddingEngineConfig(modelPath = "...")
 *   val engine = EmbeddingEngine(config)
 *   engine.initialize()
 *   val response = engine.computeEmbedding(listOf(InputData.Text("Hello world")))
 *   val asyncResponse = engine.computeEmbeddingAsync(listOf(InputData.Text("Hello world")))
 *   engine.close()
 * }
 * ```
 *
 * @param config The configuration for the embedding engine.
 */
class EmbeddingEngine(val config: EmbeddingEngineConfig) : AutoCloseable {
  private val lock = Any()
  private val asyncExecutor = Dispatchers.Default.asExecutor()

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
          config.activationDataType?.value ?: -1,
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
      val currentHandle = checkInitialized()
      return LiteRtLmJni.nativeComputeEmbedding(
        currentHandle,
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
      val currentHandle = checkInitialized()
      val nativeBatch = contentsBatch.map { it.toTypedArray() }.toTypedArray()
      return LiteRtLmJni.nativeComputeEmbeddingBatch(
          currentHandle,
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
   * Computes embedding for multimodal input contents asynchronously.
   *
   * @param contents The list of [InputData] items to compute embeddings for.
   * @param options Additional options for embedding generation.
   * @return The [EmbeddingResponse] containing the calculated embedding vector(s).
   * @throws IllegalStateException if the engine is not initialized.
   */
  suspend fun computeEmbeddingAsync(
    contents: List<InputData>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ): EmbeddingResponse {
    val unused = checkInitialized()
    val contentsCopy = contents.toList()
    return withContext(Dispatchers.Default) { computeEmbedding(contentsCopy, options) }
  }

  /**
   * Computes embedding for multimodal input contents asynchronously with a callback.
   *
   * @param contents The list of [InputData] items to compute embeddings for.
   * @param callback The callback to receive the computed [EmbeddingResponse] or error.
   * @param options Additional options for embedding generation.
   * @throws IllegalStateException if the engine is not initialized.
   */
  @JvmOverloads
  fun computeEmbeddingAsync(
    contents: List<InputData>,
    callback: EmbeddingCallback<EmbeddingResponse>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ) {
    val unused = checkInitialized()
    val contentsCopy = contents.toList()
    asyncExecutor.execute {
      val result =
        try {
          computeEmbedding(contentsCopy, options)
        } catch (t: Throwable) {
          callback.onError(t)
          return@execute
        }
      callback.onSuccess(result)
    }
  }

  /**
   * Computes embeddings for a batch of multimodal input requests asynchronously.
   *
   * @param contentsBatch A list of input requests, where each request is a list of [InputData].
   * @param options Additional options for embedding generation.
   * @return A list of [EmbeddingResponse] objects for each request in the batch.
   * @throws IllegalStateException if the engine is not initialized.
   */
  suspend fun computeEmbeddingBatchAsync(
    contentsBatch: List<List<InputData>>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ): List<EmbeddingResponse> {
    val unused = checkInitialized()
    val contentsBatchCopy = contentsBatch.map { it.toList() }
    return withContext(Dispatchers.Default) { computeEmbeddingBatch(contentsBatchCopy, options) }
  }

  /**
   * Computes embeddings for a batch of multimodal input requests asynchronously with a callback.
   *
   * @param contentsBatch A list of input requests, where each request is a list of [InputData].
   * @param callback The callback to receive the computed list of [EmbeddingResponse] objects or
   *   error.
   * @param options Additional options for embedding generation.
   * @throws IllegalStateException if the engine is not initialized.
   */
  @JvmOverloads
  fun computeEmbeddingBatchAsync(
    contentsBatch: List<List<InputData>>,
    callback: EmbeddingCallback<List<EmbeddingResponse>>,
    options: EmbeddingOptions = EmbeddingOptions(),
  ) {
    val unused = checkInitialized()
    val contentsBatchCopy = contentsBatch.map { it.toList() }
    asyncExecutor.execute {
      val result =
        try {
          computeEmbeddingBatch(contentsBatchCopy, options)
        } catch (t: Throwable) {
          callback.onError(t)
          return@execute
        }
      callback.onSuccess(result)
    }
  }

  /**
   * Closes the engine and releases the native embedding engine's resources.
   *
   * @throws IllegalStateException if the engine is not initialized.
   */
  override fun close() {
    synchronized(lock) {
      val currentHandle = checkInitialized()

      LiteRtLmJni.nativeDeleteEmbeddingEngine(currentHandle)
      handle = null
    }
  }

  /** Returns the native handle, or throws [IllegalStateException] if not initialized. */
  private fun checkInitialized(): Long {
    return checkNotNull(handle) { "EmbeddingEngine is not initialized." }
  }
}

/** A callback for receiving asynchronous embedding computation results. */
interface EmbeddingCallback<T> {
  /**
   * Called when the embedding computation completes successfully.
   *
   * @param result The computed embedding result.
   */
  fun onSuccess(result: T)

  /**
   * Called when an error occurs during the embedding computation.
   *
   * @param throwable The error that occurred.
   */
  fun onError(throwable: Throwable)
}
