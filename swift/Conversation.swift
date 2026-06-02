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

  /// Whether the conversation is alive and ready to be used.
  public var isAlive: Bool {
    return handle != nil
  }

  init(handle: CConversationHandle, toolManager: ToolManager) {
    self.handle = handle
    self.toolManager = toolManager
  }

  deinit {
    if let handle = handle {
      litert_lm_conversation_delete(handle)
    }
  }

  /// Sends a message to the model and returns the response. This is a synchronous call.
  ///
  /// - Parameter message: The message to send to the model.
  /// - Parameter extraContext: The extra context to send to the model.
  /// - Returns: The model's response message.
  /// - Throws: A `LiteRTLMError` if sending the message fails or the model
  ///   returns an invalid response.
  public func sendMessage(_ message: Message, extraContext: [String: Any]? = nil) async throws
    -> Message
  {
    let handle = try checkIsAlive()

    var currentMessageJson: [String: Any] = message.toJson

    for _ in 0..<recurringToolCallLimit {
      let (responseJson, responseString) = try attemptSendMessage(
        handle: handle, messageJson: currentMessageJson, extraContext: extraContext)

      guard let toolCalls = responseJson["tool_calls"] as? [[String: Any]] else {
        if responseJson["content"] != nil || responseJson["channels"] != nil {
          return try Conversation.jsonToMessage(responseString)
        } else {
          throw LiteRTLMError.conversation(.invalidResponse(responseString))
        }
      }
      currentMessageJson = try await handleToolCalls(toolCalls)
    }
    throw LiteRTLMError.conversation(.recurringToolCallLimitExceeded(limit: recurringToolCallLimit))
  }

  private func attemptSendMessage(
    handle: CConversationHandle, messageJson: [String: Any], extraContext: [String: Any]?
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
    if let visualTokenBudget = ExperimentalFlags.visualTokenBudget {
      litert_lm_conversation_optional_args_set_visual_token_budget(optionalArgs, Int32(visualTokenBudget))
    }
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }

    guard let responsePtr = litert_lm_conversation_send_message(
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
  /// - Returns: An async throwing stream of `Message` chunks.
  public func sendMessageStream(_ message: Message, extraContext: [String: Any]? = nil)
    -> AsyncThrowingStream<Message, Error>
  {
    return AsyncThrowingStream { continuation in
      do {
        let handle = try self.checkIsAlive()
        let messageJson: [String: Any] = message.toJson
        let context = StreamContext(continuation: continuation, conversation: self)

        try self.sendToStream(
          handle: handle, messageJson: messageJson, extraContext: extraContext, context: context)
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
  ///   - context: The `StreamContext` containing the `AsyncThrowingStream.Continuation`
  ///     and other state for the streaming process.
  /// - Throws: A `LiteRTLMError` if the message fails to send or the response is invalid.
  ///   native `send_message_stream` call fails.
  func sendToStream(
    handle: CConversationHandle,
    messageJson: [String: Any],
    extraContext: [String: Any]? = nil,
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
    if let visualTokenBudget = ExperimentalFlags.visualTokenBudget {
      litert_lm_conversation_optional_args_set_visual_token_budget(optionalArgs, Int32(visualTokenBudget))
    }
    defer { litert_lm_conversation_optional_args_delete(optionalArgs) }

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
        if let type = item["type"] as? String, type == "text", let text = item["text"] as? String {
          contents.append(.text(text))
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

    if contents.isEmpty && channels.isEmpty {
      throw LiteRTLMError.message(.invalidContent)
    }

    return Message(contents: contents, channels: channels)
  }

  /// Context object to bridge the C callback to the Swift AsyncThrowingStream.
  class StreamContext {
    let continuation: AsyncThrowingStream<Message, Error>.Continuation
    let conversation: Conversation
    var toolCallCount: Int = 0
    var pendingToolCalls: [[String: Any]] = []

    init(continuation: AsyncThrowingStream<Message, Error>.Continuation, conversation: Conversation)
    {
      self.continuation = continuation
      self.conversation = conversation
    }
  }
}

/// A callback function to bridge the C callback to the Swift AsyncThrowingStream.
private func streamCallback(
  userData: UnsafeMutableRawPointer?,
  responseJson: UnsafePointer<CChar>?,
  isFinal: Bool,
  errorMessage: UnsafePointer<CChar>?
) {
  guard let userData = userData else { return }

  let context = Unmanaged<Conversation.StreamContext>.fromOpaque(userData).takeUnretainedValue()

  if let errorMessage = errorMessage {
    let errorString = String(cString: errorMessage)
    let error = LiteRTLMError.conversation(.invalidResponse(errorString))
    context.continuation.finish(throwing: error)

    Unmanaged<Conversation.StreamContext>.fromOpaque(userData).release()
    return
  }

  if let responseJson = responseJson {
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
