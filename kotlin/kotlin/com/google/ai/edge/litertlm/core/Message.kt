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
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

/** The role of the message in a conversation. */
enum class Role(val value: String) {
  SYSTEM("system"), // represent the system
  USER("user"), // represent the user
  MODEL("model"), // represent the model
  TOOL("tool"), // represent a tool response
}

/**
 * Represents a message in the conversation. A message contains a [Role], a [Content] list, and an
 * optional list of [ToolCall].
 */
class Message
internal constructor(
  val role: Role,
  val contents: Contents = Contents.empty(),
  val toolCalls: List<ToolCall> = emptyList(),
  val channels: Map<String, String> = emptyMap(),
) {

  /** Convert to [JsonObject]. Used internally. */
  internal fun toJson() =
    JsonObject().apply {
      addProperty("role", role.value)
      if (contents.contents.isNotEmpty()) {
        add("content", contents.toJson())
      }
      if (toolCalls.isNotEmpty()) {
        val toolCallsJson = JsonArray()
        for (toolCall in toolCalls) {
          toolCallsJson.add(toolCall.toJson())
        }
        add("tool_calls", toolCallsJson)
      }
      if (channels.isNotEmpty()) {
        val channelsJson = JsonObject()
        for ((key, value) in channels) {
          channelsJson.addProperty(key, value)
        }
        add("channels", channelsJson)
      }
    }

  /** Convert the message to a string. */
  override fun toString() = contents.toString()

  companion object {

    /** Creates a system [Message] from the given text. */
    fun system(text: String) = system(Contents.of(text))

    /** Creates a system [Message] from the given contents. */
    fun system(contents: Contents) = Message(Role.SYSTEM, contents)

    /** Creates a user [Message] from the given text. */
    fun user(text: String) = user(Contents.of(text))

    /** Creates a user [Message] from the given contents. */
    fun user(contents: Contents) = Message(Role.USER, contents)

    /** Creates a model [Message] from the given text. */
    fun model(text: String) = model(Contents.of(text))

    /** Creates a model [Message] from the given contents, tool calls, and channels. */
    fun model(
      contents: Contents = Contents.empty(),
      toolCalls: List<ToolCall> = emptyList(),
      channels: Map<String, String> = emptyMap(),
    ) = Message(Role.MODEL, contents, toolCalls, channels)

    /** Creates a tool [Message] from the given contents. */
    fun tool(contents: Contents) = Message(Role.TOOL, contents)

    /** Creates a user [Message] from a text string. */
    @Deprecated("Use factory methods like user(), model() or Contents.of().")
    fun of(text: String) = user(text)

    /** Creates a user [Message] from the array of [Content]. */
    @Deprecated("Use factory methods like user(), model() or Contents.of().")
    fun of(vararg contents: Content) = user(Contents.of(contents.toList()))

    /** Creates a user [Message] from a list of [Content]. */
    @Deprecated("Use factory methods like user(), model() or Contents.of().")
    fun of(contents: List<Content>) = user(Contents.of(contents))
  }
}

class Contents private constructor(val contents: List<Content>) {
  /** Convert to [JsonObject]. Used internally. */
  internal fun toJson() =
    JsonArray().apply {
      for (content in contents) {
        this.add(content.toJson())
      }
    }

  /** Convert the Contents to a string. */
  override fun toString() = contents.joinToString("")

  companion object {

    /** Creates an empty [Contents] list. */
    internal fun empty() = Contents(emptyList())

    /** Creates a [Contents] from a text string. */
    fun of(text: String) = Contents.of(Content.Text(text))

    /** Creates a [Contents] from the array of [Content]. */
    fun of(vararg contents: Content) = Contents.of(contents.toList())

    /** Creates a [Contents] from a list of [Content]. */
    fun of(contents: List<Content>) = Contents(contents)
  }
}

/** Tool call returned by the model. ToolCalls are stored in [Message.toolCalls]. */
data class ToolCall(val name: String, val arguments: Map<String, Any?>) {
  internal fun toJson() =
    JsonObject().apply {
      addProperty("type", "function")
      val functionObj =
        JsonObject().apply {
          addProperty("name", name)
          add("arguments", arguments.toJsonObject())
        }
      add("function", functionObj)
    }
}

/** Represents a content in the [Message] of the conversation. */
sealed class Content {
  /** Convert to [JsonObject]. Used internally. */
  internal abstract fun toJson(): JsonObject

  /** Text. */
  data class Text(val text: String) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "text")
        addProperty("text", text)
      }

    override fun toString() = text
  }

  /** Image provided as raw bytes. */
  @OptIn(ExperimentalEncodingApi::class)
  data class ImageBytes(val bytes: ByteArray) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "image")
        addProperty("blob", Base64.encode(bytes))
      }
  }

  /** Image provided by a file. */
  data class ImageFile(val absolutePath: String) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "image")
        addProperty("path", absolutePath)
      }
  }

  /** Audio provided as raw bytes. */
  @OptIn(ExperimentalEncodingApi::class)
  data class AudioBytes(val bytes: ByteArray) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "audio")
        addProperty("blob", Base64.encode(bytes))
      }
  }

  /** Audio provided by a file. */
  data class AudioFile(val absolutePath: String) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "audio")
        addProperty("path", absolutePath)
      }
  }

  /** Tool response provided by the user when automatic tool calling is disabled. */
  data class ToolResponse(val name: String, val response: Any?) : Content() {
    override fun toJson() =
      JsonObject().apply {
        addProperty("type", "tool_response")
        addProperty("name", name)
        add("response", response.toJsonElement())
      }
  }
}
