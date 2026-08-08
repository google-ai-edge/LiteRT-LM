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

extension ResponseFormat.FormatType {
  var cConstraintType: LiteRtLmConstraintType {
    switch self {
    case .regex:
      return kLiteRtLmConstraintTypeRegex
    case .jsonObject:
      return kLiteRtLmConstraintTypeJsonSchema
    }
  }
}

typealias CConversationHandle = OpaquePointer

private let logger = Logger(
  subsystem: "com.google.odml.litertlm.swift",
  category: "Conversation"
)

private let recurringToolCallLimit = 25

/// Represents a conversation with the LiteRT-LM model.
///
/// Example usage:
/// ```swift
/// // Assuming 'engine' is an instance of Engine
/// let conversation = try await engine.createConversation()
///
/// // Send a message and get the response.
/// let response = try conversation.sendMessage(Message("Hello world"))
///
/// // Send a message async with response chunks as AsyncThrowingStream.
/// for try await chunk in conversation.sendMessageStream(Message("Hello world")) {
///   print(chunk.text)
/// }
/// ```
///
/// This class facilitates interaction with the LiteRT-LM model by handling message sending
/// and response reception.
public class Conversation {
  private let logger = Logger(
    subsystem: "com.google.ai.edge.litertlm.swift",
    category: "Conversation"
  )

  private var handle: CConversationHandle?
  private let toolManager: ToolManager
  private let automaticToolCalling: Bool
  private let engine: Engine
  private let enableResponseFormat: Bool
  private let visualTokenBudget: Int32?

  /// Whether the conversation is alive and ready to be used.
  public var isAlive: Bool {
    return handle != nil
  }

  init(
    handle: CConversationHandle, toolManager: ToolManager, automaticToolCalling: Bool = true,
    engine: Engine, enableResponseFormat: Bool = false, visualTokenBudget: Int32? = nil
  ) {
    self.handle = handle
    self.toolManager = toolManager
    self.automaticToolCalling = automaticToolCalling
    self.engine = engine
    self.enableResponseFormat = enableResponseFormat
    self.visualTokenBudget = visualTokenBudget
  }

  deinit {
    if let handle = handle {
      self.handle = nil
      litert_lm_conversation_delete(handle)
    }
  }

  private func shouldApplyResponseFormat(
    _ responseFormat: ResponseFormat?,
    forMessageDict messageDict: [String: Any]
  ) -> Bool {
    guard responseFormat != nil else { return false }
    let isToolResponse = (messageDict["role"] as? String) == "tool"
    if automaticToolCalling && toolManager.toolsJsonDescription != "[]" && !isToolResponse {
      return false
    }
    return true
  }

  /// Sends a message to the model and returns the response. This is a synchronous call.
  ///
  /// - Parameter message: The message to send to the model.
  /// - Parameter extraContext: The extra context to send to the model.
  /// - Parameter repetitionPenaltyConfig: Optional configuration for repetition penalty.
  /// - Parameter noRepeatNgramConfig: Optional configuration for no repeat ngram penalty.
  /// - Parameter maxOutputTokens: Optional maximum number of output tokens.
  /// - Parameter thinkingConfig: Optional configuration for thinking/reasoning generation.
  /// - Parameter responseFormat: Optional response format for constrained decoding.
  /// - Returns: The model's response message.
  /// - Throws: A `LiteRTLMError` if sending the message fails or the model
  ///   returns an invalid response.
  public func sendMessage(
    _ message: Message,
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil
  ) async throws -> Message {
    let handle = try checkIsAlive()

    if responseFormat != nil && !enableResponseFormat {
      throw LiteRTLMError.conversation(.responseFormatNotEnabled)
    }

    var currentMessageJson: [String: Any] = message.toJson

    for i in 0..<recurringToolCallLimit {
      let (responseJson, responseString) = try attemptSendMessage(
        handle: handle,
        messageJson: currentMessageJson,
        extraContext: extraContext,
        repetitionPenaltyConfig: repetitionPenaltyConfig,
        noRepeatNgramConfig: noRepeatNgramConfig,
        suppressTokensConfig: suppressTokensConfig,
        maxOutputTokens: maxOutputTokens,
        thinkingConfig: i == 0 ? thinkingConfig : nil,
        responseFormat: responseFormat
      )

      guard let toolCalls = responseJson["tool_calls"] as? [[String: Any]] else {
        if responseJson["content"] != nil || responseJson["channels"] != nil {
          return try Conversation.jsonToMessage(responseString)
        } else {
          throw LiteRTLMError.conversation(.invalidResponse(responseString))
        }
      }
      if !automaticToolCalling {
        return try Conversation.jsonToMessage(responseString)
      }
      currentMessageJson = try await handleToolCalls(toolCalls)
    }
    throw LiteRTLMError.conversation(
      .recurringToolCallLimitExceeded(limit: recurringToolCallLimit))
  }

  private func attemptSendMessage(
    handle: CConversationHandle,
    messageJson: [String: Any],
    extraContext: [String: Any]?,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil
  ) throws
    -> (responseJson: [String: Any], responseString: String)
  {
    let messageData = try JSONSerialization.data(withJSONObject: messageJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    var extraContextString: String? = nil
    if let extraContext = extraContext, !extraContext.isEmpty,
      let extraData = try? JSONSerialization.data(withJSONObject: extraContext)
    {
      extraContextString = String(data: extraData, encoding: .utf8)
    }
    let optionalArgs = litert_lm_conversation_optional_args_create()
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }
    if let visualTokenBudget = self.visualTokenBudget ?? ExperimentalFlags.visualTokenBudget {
      litert_lm_conversation_optional_args_set_visual_token_budget(
        optionalArgs, Int32(visualTokenBudget))
    }
    if let repetitionPenaltyConfig = repetitionPenaltyConfig {
      guard let cRepetitionPenaltyConfig = litert_lm_repetition_penalty_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native repetition penalty config."))
      }
      defer { litert_lm_repetition_penalty_config_delete(cRepetitionPenaltyConfig) }

      if let repetitionPenalty = repetitionPenaltyConfig.repetitionPenalty {
        litert_lm_repetition_penalty_config_set_repetition_penalty(
          cRepetitionPenaltyConfig, repetitionPenalty)
      }
      if let presencePenalty = repetitionPenaltyConfig.presencePenalty {
        litert_lm_repetition_penalty_config_set_presence_penalty(
          cRepetitionPenaltyConfig, presencePenalty)
      }
      if let frequencyPenalty = repetitionPenaltyConfig.frequencyPenalty {
        litert_lm_repetition_penalty_config_set_frequency_penalty(
          cRepetitionPenaltyConfig, frequencyPenalty)
      }
      if let windowSize = repetitionPenaltyConfig.windowSize {
        litert_lm_repetition_penalty_config_set_window_size(
          cRepetitionPenaltyConfig, Int32(windowSize))
      }
      litert_lm_conversation_optional_args_set_repetition_penalty_config(
        optionalArgs, cRepetitionPenaltyConfig)
    }
    if let noRepeatNgramConfig = noRepeatNgramConfig {
      guard let cNoRepeatNgramConfig = litert_lm_no_repeat_ngram_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native no repeat ngram config."))
      }
      defer { litert_lm_no_repeat_ngram_config_delete(cNoRepeatNgramConfig) }

      if let noRepeatNgramSize = noRepeatNgramConfig.noRepeatNgramSize {
        litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
          cNoRepeatNgramConfig, Int32(noRepeatNgramSize))
      }
      if let windowSize = noRepeatNgramConfig.windowSize {
        litert_lm_no_repeat_ngram_config_set_window_size(
          cNoRepeatNgramConfig, Int32(windowSize))
      }
      litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
        optionalArgs, cNoRepeatNgramConfig)
    }
    if let suppressTokens = suppressTokensConfig?.suppressTokens, !suppressTokens.isEmpty {
      guard let cSuppressTokensConfig = litert_lm_suppress_tokens_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native suppress tokens config."))
      }
      defer { litert_lm_suppress_tokens_config_delete(cSuppressTokensConfig) }

      let cTokens = suppressTokens.map { Int32($0) }
      cTokens.withUnsafeBufferPointer { buffer in
        litert_lm_suppress_tokens_config_set_suppress_tokens(
          cSuppressTokensConfig, buffer.baseAddress, buffer.count)
      }
      litert_lm_conversation_optional_args_set_suppress_tokens_config(
        optionalArgs, cSuppressTokensConfig)
    }
    if let maxOutputTokens = maxOutputTokens {
      litert_lm_conversation_optional_args_set_max_output_tokens(
        optionalArgs, Int32(maxOutputTokens))
    }
    if let thinkingConfig = thinkingConfig {
      guard let cThinkingConfig = litert_lm_thinking_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native thinking config."))
      }
      defer { litert_lm_thinking_config_delete(cThinkingConfig) }
      litert_lm_thinking_config_set_enable_thinking(
        cThinkingConfig, thinkingConfig.enableThinking)
      litert_lm_thinking_config_set_thinking_token_budget(
        cThinkingConfig, Int32(thinkingConfig.thinkingTokenBudget))
      litert_lm_conversation_optional_args_set_thinking_config(optionalArgs, cThinkingConfig)
    }

    if let responseFormat = responseFormat,
      shouldApplyResponseFormat(responseFormat, forMessageDict: messageJson)
    {
      litert_lm_conversation_optional_args_set_constraint(
        optionalArgs, responseFormat.type.cConstraintType, responseFormat.schemaOrPattern)
    }

    guard
      let responsePtr = litert_lm_conversation_send_message(
        handle, messageString, extraContextString, optionalArgs)
    else {
      throw LiteRTLMError.conversation(.invalidResponse("Native sendMessage returned null."))

    }
    // Delete the response pointer at the end of each iteration. Handled by defer block.
    let responsePtrRef = responsePtr
    defer { litert_lm_json_response_delete(responsePtrRef) }

    guard let responseChars = litert_lm_json_response_get_string(responsePtr) else {
      throw LiteRTLMError.conversation(
        .invalidResponse("Native get string for response returned null."))
    }
    let responseString = String(cString: responseChars)

    guard let responseData = responseString.data(using: .utf8),
      let responseJson = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
    else {
      throw LiteRTLMError.conversation(.invalidJson("Failed to parse native response JSON."))
    }
    return (responseJson, responseString)
  }

  fileprivate func handleToolCalls(_ toolCalls: [[String: Any]]) async throws -> [String: Any] {
    var toolResponses: [[String: Any]] = []

    for toolCall in toolCalls {
      guard let function = toolCall["function"] as? [String: Any],
        let name = function["name"] as? String,
        let argsObject = function["arguments"] as? [String: Any]
      else {
        continue
      }
      do {
        let result = try await toolManager.execute(name: name, arguments: argsObject)

        toolResponses.append([
          "type": "tool_response",
          "name": name,
          "response": result,
        ])
      } catch {
        throw LiteRTLMError.conversation(.toolExecutionError(name: name, error: "\(error)"))
      }
    }

    return ["role": "tool", "content": toolResponses]
  }

  /// Throws an error if the conversation is not alive.
  ///
  /// - Returns: The `OpaquePointer` handle if the conversation is alive.
  /// - Throws: A `LiteRTLMError` if `handle` is nil, indicating the conversation is not alive.
  fileprivate func checkIsAlive() throws -> OpaquePointer {
    guard let handle else {
      throw LiteRTLMError.conversation(.notAlive)
    }
    return handle
  }

  /// Sends a message to the model and returns an async stream of response chunks.
  ///
  /// - Parameter message: The message to send.
  /// - Parameter extraContext: The extra context to send to the model.
  /// - Parameter repetitionPenaltyConfig: Optional configuration for repetition penalty.
  /// - Parameter maxOutputTokens: Optional maximum number of output tokens.
  /// - Parameter thinkingConfig: Optional configuration for thinking/reasoning generation.
  /// - Parameter responseFormat: Optional response format for constrained decoding.
  /// - Returns: An async throwing stream of `Message` chunks.
  public func sendMessageStream(
    _ message: Message,
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil
  ) -> AsyncThrowingStream<Message, Error> {
    return AsyncThrowingStream { continuation in
      do {
        if responseFormat != nil && !enableResponseFormat {
          throw LiteRTLMError.conversation(.responseFormatNotEnabled)
        }
        let handle = try self.checkIsAlive()
        let messageJson: [String: Any] = message.toJson
        let context = StreamContext(
          continuation: continuation,
          conversation: self,
          repetitionPenaltyConfig: repetitionPenaltyConfig,
          noRepeatNgramConfig: noRepeatNgramConfig,
          suppressTokensConfig: suppressTokensConfig,
          maxOutputTokens: maxOutputTokens,
          responseFormat: responseFormat
        )

        try self.sendToStream(
          handle: handle,
          messageJson: messageJson,
          extraContext: extraContext,
          repetitionPenaltyConfig: repetitionPenaltyConfig,
          noRepeatNgramConfig: noRepeatNgramConfig,
          suppressTokensConfig: suppressTokensConfig,
          maxOutputTokens: maxOutputTokens,
          thinkingConfig: thinkingConfig,
          responseFormat: responseFormat,
          context: context
        )
      } catch {
        continuation.finish(throwing: error)
      }
    }
  }

  /// Sends a message to the model and handles the response via a streaming callback.
  ///
  /// This function is used internally by `sendMessageStream` and for handling
  /// subsequent tool call responses within the stream.
  ///
  /// - Parameters:
  ///   - handle: The `CConversationHandle` for the current conversation.
  ///   - messageJson: The message to send, represented as a JSON dictionary.
  ///   - extraContext: The extra context to send to the model.
  ///   - repetitionPenaltyConfig: Optional configuration for repetition penalty.
  ///   - thinkingConfig: Optional configuration for thinking/reasoning generation.
  ///   - maxOutputTokens: Optional maximum number of output tokens.
  ///   - context: The `StreamContext` containing the `AsyncThrowingStream.Continuation`
  ///     and other state for the streaming process.
  /// - Throws: A `LiteRTLMError` if the message fails to send or the response is invalid.
  ///   native `send_message_stream` call fails.
  func sendToStream(
    handle: CConversationHandle,
    messageJson: [String: Any],
    extraContext: [String: Any]? = nil,
    repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
    noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
    suppressTokensConfig: SuppressTokensConfig? = nil,
    maxOutputTokens: Int? = nil,
    thinkingConfig: ThinkingConfig? = nil,
    responseFormat: ResponseFormat? = nil,
    context: StreamContext
  ) throws {
    let messageData = try JSONSerialization.data(withJSONObject: messageJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    var extraContextString: String? = nil
    if let extraContext = extraContext, !extraContext.isEmpty,
      let extraData = try? JSONSerialization.data(withJSONObject: extraContext)
    {
      extraContextString = String(data: extraData, encoding: .utf8)
    }

    let optionalArgs = litert_lm_conversation_optional_args_create()
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }
    if let visualTokenBudget = self.visualTokenBudget ?? ExperimentalFlags.visualTokenBudget {
      litert_lm_conversation_optional_args_set_visual_token_budget(
        optionalArgs, Int32(visualTokenBudget))
    }
    if let repetitionPenaltyConfig = repetitionPenaltyConfig {
      guard let cRepetitionPenaltyConfig = litert_lm_repetition_penalty_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native repetition penalty config."))
      }
      defer { litert_lm_repetition_penalty_config_delete(cRepetitionPenaltyConfig) }

      if let repetitionPenalty = repetitionPenaltyConfig.repetitionPenalty {
        litert_lm_repetition_penalty_config_set_repetition_penalty(
          cRepetitionPenaltyConfig, repetitionPenalty)
      }
      if let presencePenalty = repetitionPenaltyConfig.presencePenalty {
        litert_lm_repetition_penalty_config_set_presence_penalty(
          cRepetitionPenaltyConfig, presencePenalty)
      }
      if let frequencyPenalty = repetitionPenaltyConfig.frequencyPenalty {
        litert_lm_repetition_penalty_config_set_frequency_penalty(
          cRepetitionPenaltyConfig, frequencyPenalty)
      }
      if let windowSize = repetitionPenaltyConfig.windowSize {
        litert_lm_repetition_penalty_config_set_window_size(
          cRepetitionPenaltyConfig, Int32(windowSize))
      }
      litert_lm_conversation_optional_args_set_repetition_penalty_config(
        optionalArgs, cRepetitionPenaltyConfig)
    }
    if let noRepeatNgramConfig = noRepeatNgramConfig {
      guard let cNoRepeatNgramConfig = litert_lm_no_repeat_ngram_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native no repeat ngram config."))
      }
      defer { litert_lm_no_repeat_ngram_config_delete(cNoRepeatNgramConfig) }

      if let noRepeatNgramSize = noRepeatNgramConfig.noRepeatNgramSize {
        litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
          cNoRepeatNgramConfig, Int32(noRepeatNgramSize))
      }
      if let windowSize = noRepeatNgramConfig.windowSize {
        litert_lm_no_repeat_ngram_config_set_window_size(
          cNoRepeatNgramConfig, Int32(windowSize))
      }
      litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
        optionalArgs, cNoRepeatNgramConfig)
    }
    if let suppressTokens = suppressTokensConfig?.suppressTokens, !suppressTokens.isEmpty {
      guard let cSuppressTokensConfig = litert_lm_suppress_tokens_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native suppress tokens config."))
      }
      defer { litert_lm_suppress_tokens_config_delete(cSuppressTokensConfig) }

      let cTokens = suppressTokens.map { Int32($0) }
      cTokens.withUnsafeBufferPointer { buffer in
        litert_lm_suppress_tokens_config_set_suppress_tokens(
          cSuppressTokensConfig, buffer.baseAddress, buffer.count)
      }
      litert_lm_conversation_optional_args_set_suppress_tokens_config(
        optionalArgs, cSuppressTokensConfig)
    }
    if let maxOutputTokens = maxOutputTokens {
      litert_lm_conversation_optional_args_set_max_output_tokens(
        optionalArgs, Int32(maxOutputTokens))
    }
    if let thinkingConfig = thinkingConfig {
      guard let cThinkingConfig = litert_lm_thinking_config_create() else {
        throw LiteRTLMError.conversation(
          .invalidResponse("Failed to create native thinking config."))
      }
      defer { litert_lm_thinking_config_delete(cThinkingConfig) }
      litert_lm_thinking_config_set_enable_thinking(
        cThinkingConfig, thinkingConfig.enableThinking)
      litert_lm_thinking_config_set_thinking_token_budget(
        cThinkingConfig, Int32(thinkingConfig.thinkingTokenBudget))
      litert_lm_conversation_optional_args_set_thinking_config(optionalArgs, cThinkingConfig)
    }

    if let responseFormat = responseFormat,
      shouldApplyResponseFormat(responseFormat, forMessageDict: messageJson)
    {
      litert_lm_conversation_optional_args_set_constraint(
        optionalArgs, responseFormat.type.cConstraintType, responseFormat.schemaOrPattern)
    }

    let contextPtr = Unmanaged.passRetained(context).toOpaque()

    let status = litert_lm_conversation_send_message_stream(
      handle,
      messageString,
      extraContextString,
      optionalArgs,
      streamCallback,
      contextPtr
    )

    guard status == 0 else {
      Unmanaged<StreamContext>.fromOpaque(contextPtr).release()
      throw LiteRTLMError.conversation(.failedToStartStream(status: Int(status)))
    }
  }

  /// Cancels the ongoing asynchronous inference process.
  public func cancel() throws {
    let handle = try checkIsAlive()
    litert_lm_conversation_cancel_process(handle)
  }

  /// Renders the message into a string for testing and logging.
  ///
  /// This function does not need to be called for actual message sending, as the `sendMessage` and
  /// `sendMessageStream` functions will handle rendering internally.
  ///
  /// - Parameter message: The message to render.
  /// - Returns: The rendered message string.
  /// - Throws: A `LiteRTLMError` if the conversation is not alive, serializing fails, or rendering fails.
  public func renderMessageIntoString(_ message: Message) throws -> String {
    let handle = try checkIsAlive()
    let messageData = try JSONSerialization.data(withJSONObject: message.toJson)
    guard let messageString = String(data: messageData, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.failedToSerializeMessage)
    }

    guard let cString = litert_lm_conversation_render_message_to_string(handle, messageString)
    else {
      throw LiteRTLMError.conversation(.invalidResponse("Failed to render message into string."))
    }
    return String(cString: cString)
  }

  /// Renders the preface into a string for testing and logging.
  ///
  /// - Returns: The rendered preface string.
  /// - Throws: A `LiteRTLMError` if the conversation is not alive, or rendering fails.
  public func renderPrefaceIntoString() throws -> String {
    let handle = try checkIsAlive()
    guard let cString = litert_lm_conversation_render_preface_to_string(handle) else {
      throw LiteRTLMError.conversation(.invalidResponse("Failed to render preface into string."))
    }
    return String(cString: cString)
  }

  /// Gets the number of tokens in the conversation KV Cache (prefill + decode).
  ///
  /// - Throws: A `LiteRTLMError` if the conversation is not alive.
  public func getTokenCount() throws -> Int {
    let handle = try checkIsAlive()
    return Int(litert_lm_conversation_get_token_count(handle))
  }

  /// Retrieves the benchmark information from the conversation.
  ///
  /// - Returns: The benchmark information
  /// - Throws: A `LiteRTLMError` if the benchmark flag is not enabled or info is unavailable.
  public func getBenchmarkInfo() throws -> BenchmarkInfo {
    let handle = try checkIsAlive()

    if !ExperimentalFlags.enableBenchmark {
      throw LiteRTLMError.conversation(.benchmarkNotEnabled)
    }

    guard let benchmarkInfoPtr = litert_lm_conversation_get_benchmark_info(handle) else {
      throw LiteRTLMError.conversation(.benchmarkInfoUnavailable)
    }
    defer { litert_lm_benchmark_info_delete(benchmarkInfoPtr) }

    let numPrefillTurns = litert_lm_benchmark_info_get_num_prefill_turns(benchmarkInfoPtr)
    let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchmarkInfoPtr)

    let initTimeInSecond = litert_lm_benchmark_info_get_total_init_time_in_second(benchmarkInfoPtr)
    let timeToFirstTokenInSecond = litert_lm_benchmark_info_get_time_to_first_token(
      benchmarkInfoPtr)

    let lastPrefillTokenCount: Int =
      numPrefillTurns > 0
      ? Int(
        litert_lm_benchmark_info_get_prefill_token_count_at(
          benchmarkInfoPtr, numPrefillTurns - 1)) : 0
    let lastPrefillTokensPerSec: Double =
      numPrefillTurns > 0
      ? litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
        benchmarkInfoPtr, numPrefillTurns - 1) : 0.0

    let lastDecodeTokenCount: Int =
      numDecodeTurns > 0
      ? Int(
        litert_lm_benchmark_info_get_decode_token_count_at(
          benchmarkInfoPtr, numDecodeTurns - 1)) : 0
    let lastDecodeTokensPerSec: Double =
      numDecodeTurns > 0
      ? litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
        benchmarkInfoPtr, numDecodeTurns - 1) : 0.0

    return BenchmarkInfo(
      initTimeInSecond: initTimeInSecond,
      timeToFirstTokenInSecond: timeToFirstTokenInSecond,
      lastPrefillTokenCount: lastPrefillTokenCount,
      lastDecodeTokenCount: lastDecodeTokenCount,
      lastPrefillTokensPerSecond: lastPrefillTokensPerSec,
      lastDecodeTokensPerSecond: lastDecodeTokensPerSec
    )
  }

  /// Internal Helper Function to convert a JSON string to a `Message`.
  ///
  /// - Parameter jsonString: The JSON string to convert.
  /// - Returns: The `Message` representation of the JSON string.
  /// - Throws: `LiteRTLMError` if the JSON string is invalid.
  public static func jsonToMessage(_ jsonString: String) throws -> Message {
    guard let data = jsonString.data(using: .utf8),
      let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw LiteRTLMError.message(.failedToConvertToJson)
    }

    var contents: [Content] = []
    if let contentArray = jsonObject["content"] as? [[String: Any]] {
      for item in contentArray {
        if let type = item["type"] as? String {
          switch type {
          case "text":
            if let text = item["text"] as? String {
              contents.append(.text(text))
            }
          case "tool_response":
            if let name = item["name"] as? String, let response = item["response"] {
              let id = item["id"] as? String ?? ""
              contents.append(.toolResponse(name: name, response: response, id: id))
            }
          case "image":
            if let path = item["path"] as? String {
              contents.append(.imageFile(path))
            } else if let blob = item["blob"] as? String, let data = Data(base64Encoded: blob) {
              contents.append(.imageData(data))
            }
          case "audio":
            if let path = item["path"] as? String {
              contents.append(.audioFile(path))
            } else if let blob = item["blob"] as? String, let data = Data(base64Encoded: blob) {
              contents.append(.audioData(data))
            }
          default: break
          }
        }
      }
    }

    var channels: [String: String] = [:]
    if let channelsDict = jsonObject["channels"] as? [String: Any] {
      for (key, value) in channelsDict {
        if let strValue = value as? String {
          channels[key] = strValue
        }
      }
    }

    var toolCalls: [ToolCall] = []
    if let toolCallsArray = jsonObject["tool_calls"] as? [[String: Any]] {
      for item in toolCallsArray {
        if let function = item["function"] as? [String: Any],
          let name = function["name"] as? String,
          let args = function["arguments"] as? [String: Any]
        {
          let id = item["id"] as? String ?? ""
          toolCalls.append(ToolCall(name: name, id: id, arguments: args))
        }
      }
    }

    if contents.isEmpty && channels.isEmpty && toolCalls.isEmpty {
      throw LiteRTLMError.message(.invalidContent)
    }

    let roleRaw = jsonObject["role"] as? String ?? "user"
    let role: Role
    switch roleRaw {
    case "assistant", "model":
      role = .model
    default:
      role = Role(rawValue: roleRaw) ?? .user
    }

    return Message(
      contents: Contents(contents: contents),
      role: role,
      channels: channels,
      toolCalls: toolCalls
    )
  }

  /// Context object to bridge the C callback to the Swift AsyncThrowingStream.
  class StreamContext {
    let continuation: AsyncThrowingStream<Message, Error>.Continuation
    let conversation: Conversation
    var toolCallCount: Int = 0
    var pendingToolCalls: [[String: Any]] = []
    let repetitionPenaltyConfig: RepetitionPenaltyConfig?
    let noRepeatNgramConfig: NoRepeatNgramConfig?
    let suppressTokensConfig: SuppressTokensConfig?
    let maxOutputTokens: Int?
    let responseFormat: ResponseFormat?

    init(
      continuation: AsyncThrowingStream<Message, Error>.Continuation,
      conversation: Conversation,
      repetitionPenaltyConfig: RepetitionPenaltyConfig? = nil,
      noRepeatNgramConfig: NoRepeatNgramConfig? = nil,
      suppressTokensConfig: SuppressTokensConfig? = nil,
      maxOutputTokens: Int? = nil,
      responseFormat: ResponseFormat? = nil
    ) {
      self.continuation = continuation
      self.conversation = conversation
      self.repetitionPenaltyConfig = repetitionPenaltyConfig
      self.noRepeatNgramConfig = noRepeatNgramConfig
      self.suppressTokensConfig = suppressTokensConfig
      self.maxOutputTokens = maxOutputTokens
      self.responseFormat = responseFormat
    }
  }
}

/// A callback function to bridge the C callback to the Swift AsyncThrowingStream.
private func streamCallback(
  userData: UnsafeMutableRawPointer?,
  chunk: OpaquePointer?
) {
  guard let userData = userData else { return }

  let context = Unmanaged<Conversation.StreamContext>.fromOpaque(userData).takeUnretainedValue()

  let isFinal = litert_lm_stream_chunk_is_final(chunk)
  let errorMessage = litert_lm_stream_chunk_get_error(chunk)

  if let errorMessage = errorMessage {
    let errorString = String(cString: errorMessage)
    let error = LiteRTLMError.conversation(.invalidResponse(errorString))
    context.continuation.finish(throwing: error)

    Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
    return
  }

  if let responseJson = litert_lm_stream_chunk_get_text(chunk) {
    let responseString = String(cString: responseJson)
    do {
      guard let responseData = responseString.data(using: .utf8),
        let jsonObject = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
      else {
        throw LiteRTLMError.conversation(.invalidJson("Invalid JSON chunk"))
      }

      if let toolCalls = jsonObject["tool_calls"] as? [[String: Any]] {
        context.pendingToolCalls.append(contentsOf: toolCalls)
      }

      if jsonObject["content"] != nil || jsonObject["channels"] != nil {
        let message = try Conversation.jsonToMessage(responseString)
        context.continuation.yield(message)
      }
    } catch {
      logger.error("Failed to parse response JSON: \(error.localizedDescription)")
      context.continuation.finish(throwing: error)
      Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
      return
    }
  }

  if isFinal {
    if !context.pendingToolCalls.isEmpty {
      if context.toolCallCount >= recurringToolCallLimit {
        context.continuation.finish(
          throwing: LiteRTLMError.conversation(
            .recurringToolCallLimitExceeded(limit: recurringToolCallLimit)))
        Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
        return
      }

      context.toolCallCount += 1
      let toolCalls = context.pendingToolCalls
      context.pendingToolCalls = []

      Task {
        do {
          let toolResponseJson = try await context.conversation.handleToolCalls(toolCalls)
          try context.conversation.sendToStream(
            handle: context.conversation.checkIsAlive(),
            messageJson: toolResponseJson,
            repetitionPenaltyConfig: context.repetitionPenaltyConfig,
            noRepeatNgramConfig: context.noRepeatNgramConfig,
            suppressTokensConfig: context.suppressTokensConfig,
            maxOutputTokens: context.maxOutputTokens,
            responseFormat: context.responseFormat,
            context: context
          )
        } catch {
          context.continuation.finish(throwing: error)
        }
        // Release the reference for the current (finished) call.
        // The new call from sendToStream created its own retained reference.
        Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
      }
    } else {
      context.continuation.finish()
      Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
    }
  }
}
