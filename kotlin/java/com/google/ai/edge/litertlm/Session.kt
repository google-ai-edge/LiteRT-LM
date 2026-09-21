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

import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Manages the lifecycle of a LiteRT-LM session, providing an interface for interacting with the
 * native library.
 *
 * @param handle The pointer to the underlying native session object.
 */
class Session(private val handle: Long) : AutoCloseable {
  private val _isAlive = AtomicBoolean(true)

  /** Whether the session is alive and ready to be used, */
  val isAlive: Boolean
    get() = _isAlive.get()

  /**
   * Closes the session and releases the native session's resources.
   *
   * @throws IllegalStateException if the session has already been closed.
   */
  override fun close() {
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteSession(handle)
    } else {
      throw IllegalStateException("Session is closed already.")
    }
  }

  /**
   * Adds the [InputData] and starts the prefilling process.
   *
   * User can break down their [InputData] into multiple chunks and call this function multiple
   * times.
   *
   * This is a blocking call and the function will return when the prefill process is done.
   *
   * @param inputData An array of [InputData] to be processed by the model.
   * @throws IllegalStateException if the session is not alive.
   */
  fun runPrefill(inputData: List<InputData>) {
    checkIsAlive()
    return LiteRtLmJni.nativeRunPrefill(handle, inputData.toTypedArray())
  }

  /**
   * Runs the decode step for the model to predict the response based on the input data added by
   * [runPrefill].
   *
   * This is a blocking call and the function will return when the decoding process is done.
   *
   * @return The generated content as a [String].
   * @throws IllegalStateException if the session is not alive.
   */
  fun runDecode(): String {
    checkIsAlive()
    return LiteRtLmJni.nativeRunDecode(handle)
  }

  /**
   * Scores candidate continuations of the input added by [runPrefill]; no text is generated.
   *
   * The runtime scores one target text per call today and reports an error for more. Each score is
   * the target's log probability after the prefilled input (negative; higher is more likely).
   * Scoring consumes the target's tokens: to score another candidate from the same position, call
   * [saveCheckpoint] before the first scoring and [rewindToCheckpoint] between scorings.
   *
   * This is a blocking call.
   *
   * @param targetTexts The candidate text(s) to score; must not be empty.
   * @param storeTokenLengths Whether to also return the token length of each target.
   * @return One score per target, and the token lengths when requested.
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the runtime rejects the request or fails.
   */
  fun runTextScoring(
    targetTexts: List<String>,
    storeTokenLengths: Boolean = false,
  ): TextScoringResponse {
    checkIsAlive()
    require(targetTexts.isNotEmpty()) { "targetTexts must not be empty." }
    val response =
      LiteRtLmJni.nativeRunTextScoring(handle, targetTexts.toTypedArray(), storeTokenLengths)
    check(response.scores.size == targetTexts.size) {
      "Expected ${targetTexts.size} score(s), got ${response.scores.size}."
    }
    return response
  }

  /**
   * Saves the session's current position under [label], to return to with [rewindToCheckpoint].
   *
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  fun saveCheckpoint(label: String) {
    checkIsAlive()
    LiteRtLmJni.nativeSaveCheckpoint(handle, label)
  }

  /**
   * Returns the session to the position saved under [label]; later checkpoints are dropped.
   *
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the label does not exist or the native method fails.
   */
  fun rewindToCheckpoint(label: String) {
    checkIsAlive()
    LiteRtLmJni.nativeRewindToCheckpoint(handle, label)
  }

  /**
   * Returns the session to a step number (see [getCurrentStep]).
   *
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  fun rewindToStep(step: Int) {
    checkIsAlive()
    LiteRtLmJni.nativeRewindToStep(handle, step)
  }

  /**
   * Returns the session's current step: the number of tokens processed so far.
   *
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  fun getCurrentStep(): Int {
    checkIsAlive()
    return LiteRtLmJni.nativeGetCurrentStep(handle)
  }

  /**
   * Generates content from the provided [InputData] and any previous input data added by
   * [runPrefill].
   *
   * This handles both the prefilling and decoding steps.
   *
   * @param inputData An array of [InputData] to be processed by the model. If the user wants to run
   *   the decode loop only, they can pass an empty array.
   * @return The generated content as a [String].
   * @throws IllegalStateException if the session is not alive.
   */
  fun generateContent(inputData: List<InputData>): String {
    checkIsAlive()
    return LiteRtLmJni.nativeGenerateContent(handle, inputData.toTypedArray())
  }

  /**
   * Generates content from the provided [InputData] and previous input data added by [runPrefill].
   *
   * This handles both the prefilling and decoding steps.
   *
   * @param inputData An array of [InputData] to be processed by the model.
   * @param responseCallback The callback to receive the streaming responses.
   * @throws IllegalStateException if the session is not alive.
   */
  fun generateContentStream(inputData: List<InputData>, responseCallback: ResponseCallback) {
    checkIsAlive()
    val jniCallback = JniInferenceCallbackImpl(responseCallback)
    LiteRtLmJni.nativeGenerateContentStream(handle, inputData.toTypedArray(), jniCallback)
  }

  private inner class JniInferenceCallbackImpl(private val callback: ResponseCallback) :
    LiteRtLmJni.JniInferenceCallback {
    override fun onNext(response: String) {
      callback.onNext(response)
    }

    override fun onDone() {
      callback.onDone()
    }

    override fun onError(statusCode: Int, message: String) {
      if (statusCode == 1) { // StatusCode::kCancelled
        callback.onError(CancellationException(message))
      } else {
        callback.onError(LiteRtLmJniException("Status Code: $statusCode. Message: $message"))
      }
    }
  }

  /**
   * Cancels any ongoing inference process (prefill or decode).
   *
   * If there is no ongoing inference process, it is a no-op.
   *
   * @throws IllegalStateException if the session is not alive.
   */
  fun cancelProcess() {
    checkIsAlive()
    LiteRtLmJni.nativeCancelProcess(handle)
  }

  /** Throws [IllegalStateException] if the session is not alive. */
  private fun checkIsAlive() {
    check(isAlive) { "Session is not alive." }
  }
}

/**
 * A sealed class representing the input data that can be provided to the LiteRT-LM.
 *
 * This corresponds to the native `InputData` in
 * runtime/engine/io_types.h.
 */
sealed class InputData {
  /**
   * Represents text input.
   *
   * @param text The input string.
   */
  data class Text(val text: String) : InputData()

  /**
   * Represents audio input.
   *
   * Supported format: WAV.
   *
   * @param bytes The raw audio data.
   */
  // TODO(b/439003966): add Android-friendly methods to create audio input data.
  data class Audio(val bytes: ByteArray) : InputData()

  /**
   * Represents image input.
   *
   * Supported format: PNG and JPG.
   *
   * @param bytes The raw image data.
   */
  // TODO(b/439003966): add Android-friendly methods to create image input data.
  data class Image(val bytes: ByteArray) : InputData()
}

/**
 * The result of [Session.runTextScoring].
 *
 * @param scores One score per target text: the target's log probability after the prefilled input
 *   (negative; higher is more likely).
 * @param tokenLengths The token length of each target when it was requested, else null.
 */
data class TextScoringResponse(val scores: FloatArray, val tokenLengths: IntArray?) {
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is TextScoringResponse) return false
    return scores.contentEquals(other.scores) &&
      (tokenLengths?.contentEquals(other.tokenLengths) ?: (other.tokenLengths == null))
  }

  override fun hashCode(): Int {
    return 31 * scores.contentHashCode() + (tokenLengths?.contentHashCode() ?: 0)
  }
}

/** An callback for receiving streaming responses. */
interface ResponseCallback {
  /**
   * Called when a new response is available.
   *
   * @param response The response chunk.
   */
  fun onNext(response: String) {}

  /** Called when the stream is complete. */
  fun onDone() {}

  /**
   * Called when an error occurs.
   *
   * @param throwable The error that occurred. This will be a
   *   [java.util.concurrent.CancellationException] if the stream was cancelled normally, and a
   *   [LiteRtLmJniException] for other errors.
   */
  fun onError(throwable: Throwable) {}
}
