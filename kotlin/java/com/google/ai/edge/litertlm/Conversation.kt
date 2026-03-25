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

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/**
 * Represents a conversation with the LiteRT-LM model.
 *
 * Example usage:
 * ```kotlin
 * // Assuming 'engine' is an instance of LiteRtLm
 * val conversation = engine.createConversation()
 *
 * // Send a message and get the response.
 * val response = conversation.sendMessage("Hello world")
 *
 * // Send a message async with response chunks as Kotlin Flow.
 * conversation.sendMessageAsync("Hello world").collect { print(it) }
 *
 * // Send a message async with response chunks as a callback.
 * conversation.sendMessageAsync(
 *   "Hello world",
 *   object : MessageCallback {
 *     override fun onMessage(message: Message) {
 *       print(message) // Handle the streaming response
 *     }
 *
 *     override fun onDone() {
 *       // Done
 *     }
 *
 *     override fun onError(error: Throwable) {
 *       // Handle any errors
 *     }
 *   },
 * )
 *
 * // Close the conversation at the end to free the underlying native resource.
 * conversation.close()
 * ```
 *
 * This class is AutoCloseable, so you can use `use` block to ensure resources are released.
 *
 * @property handle The native handle to the conversation object.
 * @property toolManager The ToolManager instance to use for this conversation.
 * @property automaticToolCalling Whether to enable automatic tool calling.
 */
class Conversation(
  private val handle: Long,
  val toolManager: ToolManager = ToolManager(),
  val automaticToolCalling: Boolean = true,
) : AutoCloseable {
  private val _isAlive = AtomicBoolean(true)

  /** Whether the conversation is alive and ready to be used, */
  val isAlive: Boolean
    get() = _isAlive.get()

  /**
   * Sends a message to the model and returns the response. This is a synchronous call.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param message The message to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return The model's response message.
   * @throws IllegalStateException if the conversation is not alive, if the native layer returns an
   *   invalid response, or if the tool call limit is exceeded.
   * @throws LiteRtLmJniException if an error occurs during the native call.
   */
  fun sendMessage(message: Message, extraContext: Map<String, Any> = emptyMap()): Message {
    checkIsAlive()

    var currentMessageJson = message.toJson()
    var extraContextJsonString = extraContext.toJsonObject().toString()

    for (i in 0..<RECURRING_TOOL_CALL_LIMIT) {
      val responseJsonString =
        LiteRtLmJni.nativeSendMessage(handle, currentMessageJson.toString(), extraContextJsonString)
      val responseJsonObject = JsonParser.parseString(responseJsonString).asJsonObject

      if (responseJsonObject.has("tool_calls")) {
        if (!automaticToolCalling) {
          return jsonToMessage(responseJsonObject)
        }
        currentMessageJson = handleToolCalls(responseJsonObject)
        // Loop back to send the tool response
      } else if (responseJsonObject.has("content")) {
        return jsonToMessage(responseJsonObject)
      } else {
        throw IllegalStateException("Invalid response from native layer: $responseJsonString")
      }
    }
    throw IllegalStateException("Exceeded recurring tool call limit of $RECURRING_TOOL_CALL_LIMIT")
  }

  /**
   * Sends a list of content to the model and returns the response. This is a synchronous call.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param contents The list of contents to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return The model's response message.
   * @throws IllegalStateException if the conversation is not alive, if the native layer returns an
   *   invalid response, or if the tool call limit is exceeded.
   * @throws LiteRtLmJniException if an error occurs during the native call.
   */
  fun sendMessage(contents: Contents, extraContext: Map<String, Any> = emptyMap()): Message {
    return sendMessage(Message.user(contents), extraContext)
  }

  /**
   * Sends a text to the model and returns the response. This is a synchronous call.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param text The text to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return The model's response message.
   * @throws IllegalStateException if the conversation is not alive, if the native layer returns an
   *   invalid response, or if the tool call limit is exceeded.
   * @throws LiteRtLmJniException if an error occurs during the native call.
   */
  fun sendMessage(text: String, extraContext: Map<String, Any> = emptyMap()): Message =
    sendMessage(Contents.of(text), extraContext)

  /**
   * Send a message to the model and returns the response async with a callback.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param message The message to send to the model.
   * @param callback The callback to receive the streaming responses.
   * @param extraContext Optional context used for prompt template rendering.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(
    message: Message,
    callback: MessageCallback,
    extraContext: Map<String, Any> = emptyMap(),
  ) {
    checkIsAlive()

    val extraContextJsonString = extraContext.toJsonObject().toString()

    val jniCallback = JniMessageCallbackImpl(callback)
    LiteRtLmJni.nativeSendMessageAsync(
      handle,
      message.toJson().toString(),
      extraContextJsonString,
      jniCallback,
    )
  }

  /**
   * Send a list of contents to the model and returns the response async with a callback.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param contents The list of contents to send to the model.
   * @param callback The callback to receive the streaming responses.
   * @param extraContext Optional context used for prompt template rendering.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(
    contents: Contents,
    callback: MessageCallback,
    extraContext: Map<String, Any> = emptyMap(),
  ) = sendMessageAsync(Message.user(contents), callback, extraContext)

  /**
   * Send a text to the model and returns the response async with a callback.
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param text The text to send to the model.
   * @param callback The callback to receive the streaming responses.
   * @param extraContext Optional context used for prompt template rendering.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(
    text: String,
    callback: MessageCallback,
    extraContext: Map<String, Any> = emptyMap(),
  ) = sendMessageAsync(Contents.of(text), callback, extraContext)

  /**
   * Sends a message to the model and returns the response async as a [Flow].
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param message The message to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return A Flow of messages representing the model's response.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(
    message: Message,
    extraContext: Map<String, Any> = emptyMap(),
  ): Flow<Message> = callbackFlow {
    sendMessageAsync(
      message,
      object : MessageCallback {
        override fun onMessage(message: Message) {
          val unused = trySend(message)
        }

        override fun onDone() {
          close()
        }

        override fun onError(throwable: Throwable) {
          close(throwable)
        }
      },
      extraContext,
    )
    awaitClose {}
  }

  /**
   * Sends a list of contents to the model and returns the response async as a [Flow].
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param contents The list of contents to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return A Flow of messages representing the model's response.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(
    contents: Contents,
    extraContext: Map<String, Any> = emptyMap(),
  ): Flow<Message> = sendMessageAsync(Message.user(contents), extraContext)

  /**
   * Sends a text to the model and returns the response async as a [Flow].
   *
   * This method handles potential tool calls returned by the model. If a tool call is detected, the
   * corresponding tool is executed, and the result is sent back to the model. This process is
   * repeated until the model returns a final response without tool calls, up to
   * [RECURRING_TOOL_CALL_LIMIT] times.
   *
   * @param text The text to send to the model.
   * @param extraContext Optional context used for prompt template rendering.
   * @return A Flow of messages representing the model's response.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(text: String, extraContext: Map<String, Any> = emptyMap()): Flow<Message> =
    sendMessageAsync(Contents.of(text), extraContext)

  private fun handleToolCalls(toolCallsJsonObject: JsonObject): JsonObject {
    val toolCallsJSONArray = toolCallsJsonObject.getAsJsonArray("tool_calls")
    val toolResponsesJSONArray = JsonArray()

    for (toolCallElement in toolCallsJSONArray) {
      val toolCallJSONObject = toolCallElement.asJsonObject
      if (!toolCallJSONObject.has("function")) {
        continue
      }
      val functionJSONObject = toolCallJSONObject.getAsJsonObject("function")
      val functionName = functionJSONObject.get("name").asString
      val arguments = functionJSONObject.getAsJsonObject("arguments")

      val result = toolManager.execute(functionName, arguments)
      val toolResponseJSONObject =
        JsonObject().apply {
          addProperty("type", "tool_response")
          addProperty("name", functionName)
          add("response", result)
        }
      toolResponsesJSONArray.add(toolResponseJSONObject)
    }
    return JsonObject().apply {
      addProperty("role", "tool")
      add("content", toolResponsesJSONArray)
    }
  }

  private inner class JniMessageCallbackImpl(private val callback: MessageCallback) :
    LiteRtLmJni.JniMessageCallback {

    /** The tool response to be returned back */
    private var pendingToolResponseJSONMessage: JsonObject? = null
    private var toolCallCount = 0

    override fun onMessage(messageJsonString: String) {
      val messageJsonObject = JsonParser.parseString(messageJsonString).asJsonObject

      if (messageJsonObject.has("tool_calls")) {
        if (!automaticToolCalling) {
          callback.onMessage(jsonToMessage(messageJsonObject))
          return
        }
        if (toolCallCount >= RECURRING_TOOL_CALL_LIMIT) {
          callback.onError(
            IllegalStateException(
              "Exceeded recurring tool call limit of $RECURRING_TOOL_CALL_LIMIT"
            )
          )
          return
        }
        toolCallCount++
        pendingToolResponseJSONMessage = handleToolCalls(messageJsonObject)
      } else if (messageJsonObject.has("content") or messageJsonObject.has("channels")) {
        callback.onMessage(jsonToMessage(messageJsonObject))
      }
    }

    override fun onDone() {
      val localToolResponse = pendingToolResponseJSONMessage
      if (localToolResponse != null) {
        // If there is pending tool response message, send the message.
        LiteRtLmJni.nativeSendMessageAsync(
          handle,
          localToolResponse.toString(),
          "{}",
          this@JniMessageCallbackImpl,
        )
        pendingToolResponseJSONMessage = null // Clear after sending
      } else {
        // If no pending action, then call onDone to the original user callback.
        callback.onDone()
      }
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
   * Cancels any ongoing inference process.
   *
   * If there is no ongoing inference process, it is a no-op.
   *
   * @throws IllegalStateException if the session is not alive.
   */
  // b/450903294 is a pending feature request to roll the internal state back.
  fun cancelProcess() {
    checkIsAlive()
    LiteRtLmJni.nativeConversationCancelProcess(handle)
  }

  /**
   * Gets the benchmark info for the conversation.
   *
   * @return The benchmark info.
   * @throws IllegalStateException if the conversation is not alive.
   * @throws LiteRtLmJniException if benchmark is not enabled in the engine config.
   */
  @ExperimentalApi
  fun getBenchmarkInfo(): BenchmarkInfo {
    checkIsAlive()
    return LiteRtLmJni.nativeConversationGetBenchmarkInfo(handle)
  }

  /**
   * Closes the conversation and releases the native conversation's resources.
   *
   * @throws IllegalStateException if the conversation has already been closed.
   */
  override fun close() {
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteConversation(handle)
    } else {
      throw IllegalStateException("Conversation is closed already.")
    }
  }

  /** Throws [IllegalStateException] if the conversation is not alive. */
  private fun checkIsAlive() {
    check(isAlive) { "Conversation is not alive." }
  }

  companion object {
    /**
     * The maximum number of times the model can call tools in a single turn before an error is
     * thrown.
     */
    private const val RECURRING_TOOL_CALL_LIMIT = 25

    private fun jsonToMessage(messageJsonObject: JsonObject): Message {
      val contents = mutableListOf<Content>()

      if (messageJsonObject.has("content")) {
        val contentsJsonArray = messageJsonObject.getAsJsonArray("content")

        for (contentElement in contentsJsonArray) {
          val contentJsonObject = contentElement.asJsonObject
          val type = contentJsonObject.get("type").asString

          if (type == "text") {
            contents.add(Content.Text(contentJsonObject.get("text").asString))
          }
        }
      }

      val toolCalls = mutableListOf<ToolCall>()
      if (messageJsonObject.has("tool_calls")) {
        val toolCallsJsonArray = messageJsonObject.getAsJsonArray("tool_calls")
        for (toolCallElement in toolCallsJsonArray) {
          val toolCallJsonObject = toolCallElement.asJsonObject
          if (toolCallJsonObject.has("function")) {
            val functionJsonObject = toolCallJsonObject.getAsJsonObject("function")
            val functionName = functionJsonObject.get("name").asString
            val arguments = functionJsonObject.getAsJsonObject("arguments")
            toolCalls.add(ToolCall(functionName, arguments.toMap()))
          }
        }
      }

      val channels = mutableMapOf<String, String>()
      if (messageJsonObject.has("channels")) {
        val channelsJsonObject = messageJsonObject.getAsJsonObject("channels")
        for (entry in channelsJsonObject.entrySet()) {
          channels[entry.key] = entry.value.asString
        }
      }

      // Note: consider to parse the "role" from the messageJsonObject.
      // It seems that models can return "assistant" or "model".
      return Message.model(Contents.of(contents), toolCalls, channels)
    }
  }
}

/** A callback for receiving streaming message responses. */
interface MessageCallback {
  /**
   * Called when a new message chunk is available from the model.
   *
   * This method may be called multiple times for a single `sendMessageAsync` call as the model
   * streams its response.
   *
   * @param contents The message chunk.
   */
  fun onMessage(message: Message)

  /**
   * Called when all message chunks are sent for a given `sendMessageAsync` call.
   *
   * If the model response includes tool calls, this method is called *after* the tool calls have
   * been executed and their responses have been sent back to the model.
   */
  fun onDone()

  /**
   * Called when an error occurs during the response streaming process.
   *
   * @param throwable The error that occurred.
   */
  fun onError(throwable: Throwable)
}
