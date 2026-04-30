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
 * Provides information about capabilities and features supported by a LiteRT-LM file.
 *
 * The users is expected to leverage the Capabilities API to investigate the capabilities of a
 * LiteRT-LM file before using it to build a LiteRtLmEngine instance.
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

  /** Closes the loaded capabilities and releases underlying resources. */
  override fun close() {
    synchronized(lock) {
      checkInitialized()
      LiteRtLmJni.nativeDeleteCapabilities(handle!!)
      handle = null
    }
  }

  private fun checkInitialized() {
    check(handle != null) { "Capabilities instance is already closed." }
  }
}
