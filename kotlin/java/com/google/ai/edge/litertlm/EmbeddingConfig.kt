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
 */
data class EmbeddingEngineConfig(
  val modelPath: String,
  val backend: Backend = Backend.CPU(),
  val visionBackend: Backend? = null,
  val audioBackend: Backend? = null,
  val cacheDir: String? = null,
)

/**
 * Configuration options for a single or batch embedding calculation.
 *
 * @property outputDimensionality Optional target dimension for Matryoshka / truncated embeddings.
 *   `null` uses the model's default output dimension.
 * @property normalize Whether to L2-normalize the resulting output vectors. Defaults to `true`.
 */
data class EmbeddingOptions
@JvmOverloads
constructor(val outputDimensionality: Int? = null, val normalize: Boolean = true) {
  init {
    require(outputDimensionality == null || outputDimensionality > 0) {
      "outputDimensionality must be positive or null, but got $outputDimensionality."
    }
  }
}

/**
 * Represents the embedding result for an input item.
 *
 * @property embedding Dense float vector output representation.
 * @property reducedEmbedding Optional secondary or layer-reduced embedding (e.g. per-layer
 *   embeddings for EmbeddingGemma v2). `null` if not supported or requested.
 */
data class EmbeddingResponse(val embedding: FloatArray, val reducedEmbedding: FloatArray? = null) {
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is EmbeddingResponse) return false
    if (!embedding.contentEquals(other.embedding)) return false
    if (reducedEmbedding != null) {
      if (other.reducedEmbedding == null) return false
      if (!reducedEmbedding.contentEquals(other.reducedEmbedding)) return false
    } else if (other.reducedEmbedding != null) return false
    return true
  }

  override fun hashCode(): Int {
    var result = embedding.contentHashCode()
    result = 31 * result + (reducedEmbedding?.contentHashCode() ?: 0)
    return result
  }
}
