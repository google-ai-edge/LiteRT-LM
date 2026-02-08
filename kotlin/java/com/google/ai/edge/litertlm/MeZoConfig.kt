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
 * Configuration for MeZO (Memory-efficient Zeroth-Order) fine-tuning.
 *
 * MeZO estimates gradients using only forward passes, achieving the same memory footprint as
 * inference. ConMeZO extends MeZO with cone-constrained momentum for faster convergence.
 *
 * @property learningRate Learning rate for parameter updates. Must be positive.
 * @property epsilon Perturbation scale for finite difference gradient estimation. Must be positive.
 * @property weightDecay Weight decay coefficient. Must be non-negative.
 * @property seed Random seed for reproducibility. 0 uses a random seed.
 * @property useConMeZo Enable ConMeZO (cone-constrained momentum). When enabled, perturbation
 *   directions are biased toward a momentum vector from past gradients.
 * @property momentumDecay EMA decay for the ConMeZO momentum vector. Must be in [0, 1].
 * @property coneAngle Half-angle of the sampling cone in radians. Must be in [0, pi/2]. Smaller
 *   values concentrate perturbations closer to the momentum direction.
 */
data class MeZoConfig(
  val learningRate: Float = 1e-6f,
  val epsilon: Float = 1e-3f,
  val weightDecay: Float = 0.0f,
  val seed: Long = 0L,
  val useConMeZo: Boolean = false,
  val momentumDecay: Float = 0.9f,
  val coneAngle: Float = 0.7854f,
) {
  init {
    require(learningRate > 0) { "learningRate must be positive, got $learningRate." }
    require(epsilon > 0) { "epsilon must be positive, got $epsilon." }
    require(weightDecay >= 0) { "weightDecay must be non-negative, got $weightDecay." }
    require(momentumDecay in 0.0f..1.0f) {
      "momentumDecay must be in [0, 1], got $momentumDecay."
    }
    require(coneAngle in 0.0f..PI_OVER_2) {
      "coneAngle must be in [0, pi/2], got $coneAngle."
    }
  }

  private companion object {
    const val PI_OVER_2 = 1.5707964f
  }
}
