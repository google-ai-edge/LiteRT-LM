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

/** A wrapper for the native JNI methods. */
internal object LiteRtLmJni {

  init {
    NativeLibraryLoader.load()
  }

  /**
   * Creates a new LiteRT-LM engine.
   *
   * @param modelPath The path to the model file.
   * @param backend The backend to use for the engine. It should be the string of the corresponding
   *   value in `litert::lm::Backend`.
   * @param visionBackend The backend to use for the vision executor. If empty, vision executor will
   *   not be initialized. It should be the string of the corresponding value in
   *   `litert::lm::Backend`.
   * @param audioBackend The backend to use for the audio executor. If empty, audio executor will
   *   not be initialized. It should be the string of the corresponding value in
   *   `litert::lm::Backend`.
   * @param maxNumTokens The maximum number of tokens to be processed by the engine. When
   *   non-positive, use the engine's default.
   * @param enableBenchmark Whether to enable benchmark mode or not.
   * @param cacheDir The directory for cache files.
   * @param enableBenchmark Whether to enable benchmark or not.
   * @param npuLibrariesDir The directory for the NPU libraries.
   * @return A pointer to the native engine instance.
   */
  external fun nativeCreateEngine(
    modelPath: String,
    backend: String,
    visionBackend: String,
    audioBackend: String,
    maxNumTokens: Int,
    cacheDir: String,
    enableBenchmark: Boolean,
    npuLibrariesDir: String,
  ): Long

  /**
   * Delete the LiteRT-LM engine.
   *
   * @param enginePointer A pointer to the native engine instance.
   */
  external fun nativeDeleteEngine(enginePointer: Long)

  /**
   * Creates a new LiteRT-LM session.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @return A pointer to the native session instance.
   */
  external fun nativeCreateSession(enginePointer: Long, samplerConfig: SamplerConfig?): Long

  /**
   * Delete the LiteRT-LM session.
   *
   * @param sessionPointer A pointer to the native session instance.
   */
  external fun nativeDeleteSession(sessionPointer: Long)

  /**
   * Runs the prefill step for the given input data.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  external fun nativeRunPrefill(sessionPointer: Long, inputData: Array<InputData>)

  /**
   * Runs the decode step.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @return The generated content.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  external fun nativeRunDecode(sessionPointer: Long): String

  /**
   * Generates content from the given input data.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @return The generated content.
   */
  external fun nativeGenerateContent(sessionPointer: Long, inputData: Array<InputData>): String

  /**
   * Generates content from the given input data in a streaming fashion.
   *
   * <p>The [callback] will only receive callback if this method returns normally.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @param callback The callback to receive the streaming responses.
   */
  external fun nativeGenerateContentStream(
    sessionPointer: Long,
    inputData: Array<InputData>,
    callback: JniInferenceCallback,
  )

  /**
   * Callback for the nativeGenerateContentStream.
   *
   * <p>Keep the data type simple (string) to avoid constructing complex JVM object in native layer.
   */
  interface JniInferenceCallback {
    /**
     * Called when a new response is generated.
     *
     * @param response The response string.
     */
    fun onNext(response: String)

    /** Called when the inference is done and finished successfully. */
    fun onDone()

    /**
     * Called when an error occurs.
     *
     * @param statusCode The int value of the underlying Status::code returned.
     * @param message The message.
     */
    fun onError(statusCode: Int, message: String)
  }

  /**
   * Cancels the ongoing inference process.
   *
   * @param sessionPointer A pointer to the native session instance.
   */
  external fun nativeCancelProcess(sessionPointer: Long)

  /**
   * Creates a new LiteRT-LM conversation.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @param systemMessageJsonString The system instruction to be used in the conversation.
   * @param toolsDescriptionJsonString A json string of a list of tool definitions (Open API json).
   *   could be used.
   * @param enableConversationConstrainedDecoding Whether to enable conversation constrained
   *   decoding.
   * @return A pointer to the native conversation instance.
   */
  external fun nativeCreateConversation(
    enginePointer: Long,
    samplerConfig: SamplerConfig?,
    messageJsonString: String,
    toolsDescriptionJsonString: String,
    enableConversationConstrainedDecoding: Boolean,
  ): Long

  /**
   * Deletes the LiteRT-LM conversation.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   */
  external fun nativeDeleteConversation(conversationPointer: Long)

  /**
   * Send message from the given input data asynchronously.
   *
   * <p>The [callback] will only receive callback if this method returns normally.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   * @param messageJsonString The message to be processed by the native conversation instance.
   * @param callback The callback to receive the streaming responses.
   */
  external fun nativeSendMessageAsync(
    conversationPointer: Long,
    messageJsonString: String,
    callback: JniMessageCallback,
  )

  /**
   * Send message from the given input data synchronously.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   * @param messageJsonString The message to be processed by the native conversation instance.
   * @return The response message in JSON string format.
   */
  external fun nativeSendMessage(conversationPointer: Long, messageJsonString: String): String

  /**
   * Cancels the ongoing conversation process.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   */
  external fun nativeConversationCancelProcess(conversationPointer: Long)

  /**
   * Gets the benchmark info for the conversation.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   * @return The benchmark info.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  external fun nativeConversationGetBenchmarkInfo(conversationPointer: Long): BenchmarkInfo

  /**
   * Callback for the nativeSendMessageAsync.
   *
   * <p>Keep the data type simple (string) to avoid constructing complex JVM object in native layer.
   */
  interface JniMessageCallback {
    /**
     * Called when a message is received.
     *
     * @param messageJsonString The message in JSON string format.
     */
    fun onMessage(messageJsonString: String)

    /** Called when the message stream is done. */
    fun onDone()

    /**
     * Called when an error occurs.
     *
     * @param statusCode The int value of the underlying Status::code returned.
     * @param message The message.
     */
    fun onError(statusCode: Int, message: String)
  }

  /**
   * Sets the minimum log severity for the native LiteRT-LM library.
   *
   * @param logSeverity The minimum log level to set. See [LogSeverity].
   */
  external fun nativeSetMinLogSeverity(logSeverity: Int)
}
