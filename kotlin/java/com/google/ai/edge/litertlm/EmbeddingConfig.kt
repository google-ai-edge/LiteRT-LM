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

/**
 * Configuration for the LiteRT-LM [EmbeddingEngine].
 *
 * @property modelPath The file path to the `.litertlm` embedding model bundle.
 * @property backend The execution backend for the main text encoder (CPU, GPU, NPU).
 * @property visionBackend The execution backend for the vision encoder (if model is multimodal).
 * @property audioBackend The execution backend for the audio encoder (if model is multimodal).
 * @property cacheDir Directory for compiled model artifacts and caching.
 * @property modelFd File descriptor for the model bundle. Supported on Android.
 * @property maxInputLength Maximum sequence length (in tokens) for text encoder signatures.
 * @property visionTokensPerImage Desired number of vision tokens generated per image.
 */
data class EmbeddingEngineConfig
@JvmOverloads
constructor(
  val modelPath: String? = null,
  val backend: Backend = Backend.CPU(),
  val visionBackend: Backend? = null,
  val audioBackend: Backend? = null,
  val cacheDir: String? = null,
  val modelFd: Int? = null,
  val maxInputLength: Int? = null,
  val visionTokensPerImage: Int? = null,
) {
  init {
    val hasPath = !modelPath.isNullOrEmpty()
    val hasFd = modelFd != null && modelFd >= 0
    require(hasPath xor hasFd) {
      "Either model path or model file descriptor must be specified, but not both."
    }
  }
}

/**
 * Configuration options for a single or batch embedding calculation.
 *
 * @property normalize Whether to L2-normalize the resulting output vectors. If `null`, uses the C++
 *   engine default.
 * @property insertSpecialTokens Whether to automatically insert special tokens (BOS, EOS, start/end
 *   of image, start/end of audio). If `null`, uses the C++ engine default.
 * @property outputSize The output embedding size to truncate to. If `null`, uses the C++ engine
 *   default.
 * @property visionTokensPerImage The number of vision soft tokens to generate per image. If `null`,
 *   uses the C++ engine default.
 */
data class EmbeddingOptions
@JvmOverloads
constructor(
  val normalize: Boolean? = null,
  val insertSpecialTokens: Boolean? = null,
  val outputSize: Int? = null,
  val visionTokensPerImage: Int? = null,
)

/**
 * Represents the embedding result for an input item.
 *
 * @property embedding Dense float vector output representation.
 */
data class EmbeddingResponse(val embedding: FloatArray) {
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is EmbeddingResponse) return false
    return embedding.contentEquals(other.embedding)
  }

  override fun hashCode(): Int {
    return embedding.contentHashCode()
  }
}
