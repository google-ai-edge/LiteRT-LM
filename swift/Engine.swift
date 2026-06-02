// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import OSLog
import CLiteRTLM

/// Manages the lifecycle of a LiteRT-LM engine, providing an interface for interacting with the
/// underlying native library.
///
/// Example usage:
/// ```
/// let config = try EngineConfig(modelPath: "...")
/// let engine = Engine(engineConfig: config)
/// try await engine.initialize()
/// ```
public actor Engine {
  private let logger = Logger(
    subsystem: "com.google.odml.litertlm.swift",
    category: "Engine"
  )

  /// The configuration for the engine.
  public let engineConfig: EngineConfig

  /// The native handle to the LiteRT-LM engine. A non-nil value indicates an initialized engine.
  private var handle: OpaquePointer? = nil

  /// - Parameter engineConfig: The configuration for the engine.
  public init(engineConfig: EngineConfig) {
    self.engineConfig = engineConfig
  }

  /// Returns `true` if the engine is initialized and ready for use; `false` otherwise.
  public func isInitialized() -> Bool {
    return handle != nil
  }

  /// Initializes the native LiteRT-LM engine.
  ///
  /// **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
  /// the model size and device hardware. It is strongly recommended to call this method on a
  /// background thread to avoid blocking the main thread.
  ///
  /// - Throws: A `LiteRTLMError` if the engine fails to initialize.
  public func initialize() throws {
    try initializeInternal(benchmarkPrefillTokens: nil, benchmarkDecodeTokens: nil)
  }

  /// Initializes the native LiteRT-LM engine specifically for benchmarking, avoiding global state params.
  func initializeForBenchmark(prefillTokens: Int, decodeTokens: Int) throws {
    try initializeInternal(
      benchmarkPrefillTokens: prefillTokens, benchmarkDecodeTokens: decodeTokens)
  }

  private func initializeInternal(benchmarkPrefillTokens: Int?, benchmarkDecodeTokens: Int?) throws
  {
    if isInitialized() {
      throw LiteRTLMError.engine(.alreadyInitialized)
    }

    // Convert the enums to strings for passing to the native library.
    let backendStr = engineConfig.backend.rawValue
    let visionBackendStr = engineConfig.visionBackend?.rawValue
    let audioBackendStr = engineConfig.audioBackend?.rawValue

    let settings = litert_lm_engine_settings_create(
      engineConfig.modelPath, backendStr, visionBackendStr, audioBackendStr)

    guard let settings else {
      throw LiteRTLMError.engine(.failedToCreateSettings)
    }

    defer { litert_lm_engine_settings_delete(settings) }

    if let maxNumTokens = engineConfig.maxNumTokens {
      litert_lm_engine_settings_set_max_num_tokens(settings, Int32(maxNumTokens))
    }
    if let cacheDir = engineConfig.cacheDir {
      litert_lm_engine_settings_set_cache_dir(settings, cacheDir)
    }
    if let prefill = benchmarkPrefillTokens, let decode = benchmarkDecodeTokens {
      litert_lm_engine_settings_enable_benchmark(settings)
      litert_lm_engine_settings_set_num_prefill_tokens(settings, Int32(prefill))
      litert_lm_engine_settings_set_num_decode_tokens(settings, Int32(decode))
    } else if ExperimentalFlags.enableBenchmark {
      litert_lm_engine_settings_enable_benchmark(settings)
    }
    if let enableSpeculativeDecoding = ExperimentalFlags.enableSpeculativeDecoding {
      litert_lm_engine_settings_set_enable_speculative_decoding(settings, enableSpeculativeDecoding)
    }

    guard let engine = litert_lm_engine_create(settings) else {
      throw LiteRTLMError.engine(.failedToCreateEngine)
    }

    self.handle = engine
  }

  /// Creates a new `Conversation` from the initialized engine.
  ///
  /// - Parameter ConversationConfig: The configuration for the conversation.
  /// - Returns: `Conversation` The created conversation.
  /// - Throws: A `LiteRTLMError` if the conversation creation fails.
  ///
  public func createConversation(with config: ConversationConfig? = nil) throws -> Conversation {
    guard isInitialized() else {
      throw LiteRTLMError.engine(.notInitialized)
    }

    // We can force unwrap handle here because the engine is guaranteed to be initialized, and
    // initialization will set the handle.
    let engineHandle = self.handle!

    let conversationConfig = config ?? ConversationConfig()

    let systemMessage = conversationConfig.systemMessage
    let initialSystemMessageCount = conversationConfig.initialMessages.filter { $0.role == .system }
      .count

    if systemMessage != nil && initialSystemMessageCount > 0 {
      throw LiteRTLMError.config(.multipleSystemMessages)
    }
    if initialSystemMessageCount > 1 {
      throw LiteRTLMError.config(.multipleSystemMessages)
    }

    let toolManager = ToolManager(tools: conversationConfig.tools)

    let systemMessageJsonStr = (try? conversationConfig.systemMessage?.contents.jsonString) ?? ""
    let toolDescriptionJsonStr = toolManager.toolsJsonDescription

    let initialMessagesJson = conversationConfig.initialMessages.map { $0.toJson }
    let messagesJsonStr: String
    if !initialMessagesJson.isEmpty,
      let messagesData = try? JSONSerialization.data(
        withJSONObject: initialMessagesJson, options: []),
      let serializedStr = String(data: messagesData, encoding: .utf8)
    {
      messagesJsonStr = serializedStr
    } else {
      messagesJsonStr = ""
    }

    let cSessionConfig = litert_lm_session_config_create()
    guard let cSessionConfig else {
      throw LiteRTLMError.engine(.failedToCreateSessionConfig)
    }
    defer { litert_lm_session_config_delete(cSessionConfig) }

    if let samplerParams = conversationConfig.samplerConfig {
      var params = LiteRtLmSamplerParams(
        // Based on the current engine implementation, when SamplerConfig is set, we must switch to
        // the topP sampling type.
        type: kLiteRtLmSamplerTypeTopP,
        top_k: Int32(samplerParams.topK),
        top_p: samplerParams.topP,
        temperature: samplerParams.temperature,
        seed: Int32(samplerParams.seed)
      )
      litert_lm_session_config_set_sampler_params(cSessionConfig, &params)
    }

    guard let cConversationConfig = litert_lm_conversation_config_create() else {
      throw LiteRTLMError.engine(.failedToCreateConversationConfig)
    }
    defer { litert_lm_conversation_config_delete(cConversationConfig) }

    litert_lm_conversation_config_set_session_config(cConversationConfig, cSessionConfig)
    if !systemMessageJsonStr.isEmpty {
      litert_lm_conversation_config_set_system_message(cConversationConfig, systemMessageJsonStr)
    }
    if !toolDescriptionJsonStr.isEmpty {
      litert_lm_conversation_config_set_tools(cConversationConfig, toolDescriptionJsonStr)
    }
    if !messagesJsonStr.isEmpty {
      litert_lm_conversation_config_set_messages(cConversationConfig, messagesJsonStr)
    }
    litert_lm_conversation_config_set_enable_constrained_decoding(
      cConversationConfig, ExperimentalFlags.enableConversationConstrainedDecoding)

    guard
      let conversationHandle = litert_lm_conversation_create(
        engineHandle, cConversationConfig)
    else {
      throw LiteRTLMError.engine(.failedToCreateConversation)
    }

    return Conversation(handle: conversationHandle, toolManager: toolManager)
  }

  deinit {
    if let handle = handle {
      litert_lm_engine_delete(handle)
    }
  }
}
