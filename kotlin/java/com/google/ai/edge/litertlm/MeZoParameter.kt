/*
 * Copyright 2025 Google LLC.
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
 * Represents a named model parameter for MeZO fine-tuning.
 *
 * Each parameter points to a contiguous block of float32 model weights that can be perturbed and
 * updated by the MeZO optimizer.
 *
 * @property name Name of the parameter (e.g., "attention.query_weight_0").
 * @property dataPointer Native pointer to the mutable float32 weight data. Must remain valid for
 *   the duration of [MeZoFineTuner.step].
 * @property numElements Number of float elements in the parameter buffer.
 * @property isBiasOrLayerNorm Whether this parameter is a bias or layer normalization weight. When
 *   true, weight decay is not applied during updates.
 */
data class MeZoParameter(
  val name: String,
  val dataPointer: Long,
  val numElements: Long,
  val isBiasOrLayerNorm: Boolean = false,
) {
  init {
    require(dataPointer != 0L) { "dataPointer must be a valid non-null native pointer." }
    require(numElements > 0) { "numElements must be positive, got $numElements." }
  }
}
