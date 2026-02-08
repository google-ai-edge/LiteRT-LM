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
 * Optimizer mode for the MeZO family of zeroth-order optimizers.
 *
 * @property nativeValue The integer value passed to the native C API.
 */
enum class OptimizerMode(val nativeValue: Int) {
  /** Standard MeZO (SPSA gradient estimator). */
  VANILLA_MEZO(0),

  /** ConMeZO (cone-constrained momentum). */
  CON_MEZO(1),

  /** AGZO (random-subspace projected perturbation). Proof-of-concept. */
  AGZO(2),
}
