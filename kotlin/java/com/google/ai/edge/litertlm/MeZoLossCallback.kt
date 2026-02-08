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
 * Callback interface for computing the loss during a MeZO fine-tuning step.
 *
 * The loss function is called twice per step (once with positive perturbation, once with negative
 * perturbation). Implementations should run a forward pass over the training data and return the
 * scalar loss.
 */
fun interface MeZoLossCallback {

  /**
   * Computes the forward-pass loss for the current parameter values.
   *
   * @return The scalar loss value.
   * @throws Exception if the loss computation fails.
   */
  fun computeLoss(): Float
}
