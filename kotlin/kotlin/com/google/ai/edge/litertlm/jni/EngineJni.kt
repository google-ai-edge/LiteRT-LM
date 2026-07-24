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
 * Manages the lifecycle of a LiteRT-LM engine, providing an interface for interacting with the
 * underlying native library.
 *
 * Example usage:
 * ```
 * val config = EngineConfig(modelPath = "...")
 * val nativeLibraryLoader = "..."
 * val engine = EngineJni(config, nativeLibraryLoader)
 * engine.initialize()
 * ...
 * engine.close()
 * ```
 *
 * @param engineConfig The configuration for the engine.
 */
internal class EngineJni(
  engineConfig: EngineConfig,
  nativeLibraryLoader: NativeLibraryLoader
) : Engine(engineConfig, nativeLibraryLoader) {

  /**
   * Initializes the native LiteRT-LM engine.
   *
   * **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
   * the model size and device hardware. It is strongly recommended to call this method on a
   * background thread to avoid blocking the main thread.
   *
   * @throws IllegalStateException if the engine has already been initialized.
   */
  override fun initialize() {
    synchronized(lock) {
      check(!isInitialized()) { "Engine is already initialized." }

      val mainBackendNumThreads =
        (engineConfig.backend as? Backend.CPU)
          ?.let { it.threadCount ?: it.numOfThreads }
          ?.let { if (it > 0) it else -1 } ?: -1
      val audioBackendNumThreads =
        (engineConfig.audioBackend as? Backend.CPU)
          ?.let { it.threadCount ?: it.numOfThreads }
          ?.let { if (it > 0) it else -1 } ?: -1

      LiteRtLmNative.init(nativeLibraryLoader)

      handle =
        LiteRtLmNative.nativeCreateEngine(
          engineConfig.modelPath,
          engineConfig.backend.name,
          // convert the null value to "" to avoid passing nullable object in JNI.
          engineConfig.visionBackend?.name ?: "",
          engineConfig.audioBackend?.name ?: "",
          // convert the null value to -1 to avoid passing nullable object in JNI.
          engineConfig.maxNumTokens ?: -1,
          engineConfig.maxNumImages ?: -1,
          engineConfig.cacheDir ?: "",
          @OptIn(ExperimentalApi::class) ExperimentalFlags.enableBenchmark,
          @OptIn(ExperimentalApi::class) ExperimentalFlags.enableSpeculativeDecoding,
          (engineConfig.backend as? Backend.NPU)?.nativeLibraryDir ?: "",
          (engineConfig.visionBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
          (engineConfig.audioBackend as? Backend.NPU)?.nativeLibraryDir ?: "",
          mainBackendNumThreads,
          audioBackendNumThreads,
        )
    }
  }
}
