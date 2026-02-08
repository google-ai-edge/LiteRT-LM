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
   * Creates a new LiteRT-LM session with a specific LoRA adapter.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @param loraId The LoRA adapter ID to use, or -1 for base model.
   * @return A pointer to the native session instance.
   */
  external fun nativeCreateSessionWithLora(
    enginePointer: Long,
    samplerConfig: SamplerConfig?,
    loraId: Int
  ): Long

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
   * Creates a new LiteRT-LM conversation with a specific LoRA adapter.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @param messageJsonString The system instruction to be used in the conversation.
   * @param toolsDescriptionJsonString A json string of a list of tool definitions (Open API json).
   * @param enableConversationConstrainedDecoding Whether to enable conversation constrained
   *   decoding.
   * @param loraId The LoRA adapter ID to use, or -1 for base model.
   * @return A pointer to the native conversation instance.
   */
  external fun nativeCreateConversationWithLora(
    enginePointer: Long,
    samplerConfig: SamplerConfig?,
    messageJsonString: String,
    toolsDescriptionJsonString: String,
    enableConversationConstrainedDecoding: Boolean,
    loraId: Int,
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

  // ---------------------------------------------------------------------------
  // MeZO Fine-Tuning JNI Methods
  // ---------------------------------------------------------------------------

  /** Creates a native MeZO config with default values. */
  external fun nativeMeZoConfigCreate(): Long

  /** Destroys a native MeZO config. */
  external fun nativeMeZoConfigDelete(configPointer: Long)

  /** Sets the learning rate on a native MeZO config. */
  external fun nativeMeZoConfigSetLearningRate(configPointer: Long, learningRate: Float)

  /** Sets the epsilon on a native MeZO config. */
  external fun nativeMeZoConfigSetEpsilon(configPointer: Long, epsilon: Float)

  /** Sets the weight decay on a native MeZO config. */
  external fun nativeMeZoConfigSetWeightDecay(configPointer: Long, weightDecay: Float)

  /** Sets the random seed on a native MeZO config. */
  external fun nativeMeZoConfigSetSeed(configPointer: Long, seed: Long)

  /** Enables or disables ConMeZO on a native MeZO config. */
  external fun nativeMeZoConfigSetUseConMeZo(configPointer: Long, useConMeZo: Boolean)

  /** Sets the momentum decay on a native MeZO config. */
  external fun nativeMeZoConfigSetMomentumDecay(configPointer: Long, momentumDecay: Float)

  /** Sets the cone angle on a native MeZO config. */
  external fun nativeMeZoConfigSetConeAngle(configPointer: Long, coneAngle: Float)

  /** Sets the optimizer mode on a native MeZO config (0=VanillaMeZo, 1=ConMeZo, 2=Agzo). */
  external fun nativeMeZoConfigSetOptimizerMode(configPointer: Long, mode: Int)

  /** Sets the AGZO subspace rank on a native MeZO config. */
  external fun nativeMeZoConfigSetAgzoSubspaceRank(configPointer: Long, rank: Int)

  /** Creates a native MeZO fine-tuner from a config. */
  external fun nativeMeZoFineTunerCreate(configPointer: Long): Long

  /** Destroys a native MeZO fine-tuner. */
  external fun nativeMeZoFineTunerDelete(finetunerPointer: Long)

  /**
   * Performs one MeZO optimization step.
   *
   * @param finetunerPointer Pointer to the native fine-tuner.
   * @param names Parameter names.
   * @param dataPointers Native pointers to float32 weight data.
   * @param numElements Number of elements per parameter.
   * @param isBiasOrLayerNorm Whether each parameter is bias/layernorm.
   * @param lossCallback Callback to compute the loss.
   * @return The loss from the positive perturbation.
   */
  external fun nativeMeZoFineTunerStep(
    finetunerPointer: Long,
    names: Array<String>,
    dataPointers: LongArray,
    numElements: LongArray,
    isBiasOrLayerNorm: BooleanArray,
    lossCallback: MeZoLossCallback,
  ): Float

  /** Returns the step count from a native MeZO fine-tuner. */
  external fun nativeMeZoFineTunerGetStepCount(finetunerPointer: Long): Long

  /** Sets the learning rate on a native MeZO fine-tuner. */
  external fun nativeMeZoFineTunerSetLearningRate(finetunerPointer: Long, learningRate: Float)

  /** Returns the current learning rate from a native MeZO fine-tuner. */
  external fun nativeMeZoFineTunerGetLearningRate(finetunerPointer: Long): Float
}
