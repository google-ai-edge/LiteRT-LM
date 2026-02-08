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
 * Handle to trainable model parameters (e.g., LoRA weights) extracted from a session.
 *
 * The float pointers in the returned [MeZoParameter] objects remain valid as long as this handle
 * is open and the owning session is alive. Pass the parameters to [MeZoFineTuner.step] for
 * fine-tuning.
 *
 * Example:
 * ```kotlin
 * session.getTrainableParameters().use { handle ->
 *   val params = handle.parameters
 *   finetuner.step(params) { computeLoss() }
 * }
 * ```
 */
class TrainableParameterHandle internal constructor(
  private var nativePointer: Long
) : AutoCloseable {

  private var closed = false

  /** The trainable parameters as [MeZoParameter] objects ready for [MeZoFineTuner.step]. */
  val parameters: List<MeZoParameter> by lazy {
    check(!closed) { "TrainableParameterHandle has been closed." }
    val count = LiteRtLmJni.nativeTrainableParamsCount(nativePointer)
    (0 until count).map { i ->
      MeZoParameter(
        name = LiteRtLmJni.nativeTrainableParamsGetName(nativePointer, i) ?: "",
        dataPointer = LiteRtLmJni.nativeTrainableParamsGetDataPointer(nativePointer, i),
        numElements = LiteRtLmJni.nativeTrainableParamsGetNumElements(nativePointer, i),
        isBiasOrLayerNorm = LiteRtLmJni.nativeTrainableParamsIsBiasOrLayerNorm(nativePointer, i),
      )
    }
  }

  override fun close() {
    if (!closed) {
      closed = true
      if (nativePointer != 0L) {
        LiteRtLmJni.nativeTrainableParamsDelete(nativePointer)
        nativePointer = 0L
      }
    }
  }
}
