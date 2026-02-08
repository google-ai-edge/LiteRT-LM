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
 * MeZO (Memory-efficient Zeroth-Order) fine-tuner for on-device LLMs.
 *
 * Implements the SPSA gradient estimator: perturbs model parameters with random noise, evaluates
 * the loss at two points, and estimates the gradient from the difference. When ConMeZO is enabled,
 * perturbation directions are biased toward a momentum vector for faster convergence.
 *
 * Example:
 * ```kotlin
 * val config = MeZoConfig(learningRate = 1e-7f, useConMeZo = true)
 * MeZoFineTuner(config).use { finetuner ->
 *   val loss = finetuner.step(parameters) { computeForwardPassLoss() }
 *   println("Step ${finetuner.stepCount}: loss=$loss")
 * }
 * ```
 */
class MeZoFineTuner(config: MeZoConfig) : AutoCloseable {

  private var nativeConfigPointer: Long
  private var nativePointer: Long
  private var closed = false

  init {
    nativeConfigPointer = LiteRtLmJni.nativeMeZoConfigCreate()
    require(nativeConfigPointer != 0L) { "Failed to create native MeZO config." }

    LiteRtLmJni.nativeMeZoConfigSetLearningRate(nativeConfigPointer, config.learningRate)
    LiteRtLmJni.nativeMeZoConfigSetEpsilon(nativeConfigPointer, config.epsilon)
    LiteRtLmJni.nativeMeZoConfigSetWeightDecay(nativeConfigPointer, config.weightDecay)
    LiteRtLmJni.nativeMeZoConfigSetSeed(nativeConfigPointer, config.seed)
    LiteRtLmJni.nativeMeZoConfigSetUseConMeZo(nativeConfigPointer, config.useConMeZo)
    LiteRtLmJni.nativeMeZoConfigSetMomentumDecay(nativeConfigPointer, config.momentumDecay)
    LiteRtLmJni.nativeMeZoConfigSetConeAngle(nativeConfigPointer, config.coneAngle)

    nativePointer = LiteRtLmJni.nativeMeZoFineTunerCreate(nativeConfigPointer)
    require(nativePointer != 0L) { "Failed to create native MeZO fine-tuner." }

    // Config is copied by the native side; safe to delete now.
    LiteRtLmJni.nativeMeZoConfigDelete(nativeConfigPointer)
    nativeConfigPointer = 0L
  }

  /** The number of completed optimization steps. */
  val stepCount: Long
    get() {
      check(!closed) { "MeZoFineTuner has been closed." }
      return LiteRtLmJni.nativeMeZoFineTunerGetStepCount(nativePointer)
    }

  /** The current learning rate. Can be updated for scheduling. */
  var learningRate: Float
    get() {
      check(!closed) { "MeZoFineTuner has been closed." }
      return LiteRtLmJni.nativeMeZoFineTunerGetLearningRate(nativePointer)
    }
    set(value) {
      check(!closed) { "MeZoFineTuner has been closed." }
      LiteRtLmJni.nativeMeZoFineTunerSetLearningRate(nativePointer, value)
    }

  /**
   * Performs one MeZO optimization step.
   *
   * The loss callback is invoked twice (once with positive perturbation, once with negative).
   * Parameters are restored to their original values before the gradient update is applied.
   *
   * @param parameters The named parameters to optimize.
   * @param lossCallback Callback that computes the forward-pass loss.
   * @return The loss from the positive perturbation.
   */
  fun step(parameters: List<MeZoParameter>, lossCallback: MeZoLossCallback): Float {
    check(!closed) { "MeZoFineTuner has been closed." }
    require(parameters.isNotEmpty()) { "Parameters must not be empty." }

    val names = Array(parameters.size) { parameters[it].name }
    val dataPointers = LongArray(parameters.size) { parameters[it].dataPointer }
    val numElements = LongArray(parameters.size) { parameters[it].numElements }
    val isBiasOrLayerNorm = BooleanArray(parameters.size) { parameters[it].isBiasOrLayerNorm }

    return LiteRtLmJni.nativeMeZoFineTunerStep(
      nativePointer, names, dataPointers, numElements, isBiasOrLayerNorm, lossCallback
    )
  }

  override fun close() {
    if (!closed) {
      closed = true
      if (nativeConfigPointer != 0L) {
        LiteRtLmJni.nativeMeZoConfigDelete(nativeConfigPointer)
        nativeConfigPointer = 0L
      }
      if (nativePointer != 0L) {
        LiteRtLmJni.nativeMeZoFineTunerDelete(nativePointer)
        nativePointer = 0L
      }
    }
  }
}
