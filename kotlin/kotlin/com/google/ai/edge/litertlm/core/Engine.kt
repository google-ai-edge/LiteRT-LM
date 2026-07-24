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
import kotlin.concurrent.Volatile

/**
 * Abstract class for managing the lifecycle of a LiteRT-LM engine.
 */
abstract class Engine(
  val engineConfig: EngineConfig,
  protected val nativeLibraryLoader: NativeLibraryLoader
) : AutoCloseable {
  // A lock to protect access to the engine's state and native handle.
  protected val lock = Any()

  /**
   * The native handle to the LiteRT-LM engine. A non-null value indicates an initialized engine.
   *
   * `@Volatile` ensures that changes to the handle are immediately visible across all threads.
   */
  @Volatile protected var handle: Long? = null

  /** Returns `true` if the engine is initialized and ready for use; `false` otherwise. */
  fun isInitialized(): Boolean {
    return handle != null
  }

  /**
   * Initializes the native LiteRT-LM engine.
   */
  abstract fun initialize()

  /**
   * Closes the engine and releases the native LiteRT-LM engine's resources.
   *
   * @throws IllegalStateException if the engine is not initialized.
   */
  override fun close() {
    synchronized(lock) {
      checkInitialized()

      // Using !! is okay. Checked initialization already.
      LiteRtLmNative.nativeDeleteEngine(handle!!)
      handle = null // Reset the handle to indicate the engine is released.
    }
  }

  /**
   * Creates a new [Conversation] from the initialized engine.
   *
   * @param conversationConfig The configuration for the conversation.
   * @return A new [Conversation] instance.
   * @throws IllegalStateException
   */
  fun createConversation(
    conversationConfig: ConversationConfig = ConversationConfig()
  ): Conversation {
    synchronized(lock) {
      checkInitialized()

      val toolManager = ToolManager(conversationConfig.tools)
      val messagesJson: JsonArray =
        JsonArray().apply {
          conversationConfig.systemInstruction?.let { this.add(Message(Role.SYSTEM, it).toJson()) }

          for (message in conversationConfig.initialMessages) {
            this.add(message.toJson())
          }
        }

      // Convert the channels to a JSON array, if provided.
      // If `channels` is null, the `Conversation` uses the default from the
      // `LlmMetadata` or the model type.
      // If channels is empty, channels will be disabled.
      val channelsJson: JsonArray? =
        conversationConfig.channels?.let { channels ->
          JsonArray().apply {
            for (channel in channels) {
              this.add(channel.toJson())
            }
          }
        }

      @OptIn(ExperimentalApi::class) // opt-in experimental flags
      return Conversation(
        LiteRtLmNative.nativeCreateConversation(
          handle!!, // Using !! is okay. Checked initialization already.
          conversationConfig.samplerConfig,
          messagesJson.toString(),
          toolManager.getToolsDescription().toString(),
          channelsJson?.toString(),
          conversationConfig.extraContext.toJsonObject().toString(),
          ExperimentalFlags.enableConversationConstrainedDecoding,
          ExperimentalFlags.filterChannelContentFromKvCache,
          ExperimentalFlags.overwritePromptTemplate,
          conversationConfig.loraConfig?.loraPath,
          conversationConfig.loraConfig?.audioLoraPath,
          conversationConfig.prefillPrefaceOnInit,
          conversationConfig.maxOutputToken ?: -1,
          conversationConfig.thinkingConfig,
          conversationConfig.enableResponseFormat,
        ),
        toolManager,
        conversationConfig.automaticToolCalling,
        conversationConfig.enableResponseFormat,
      )
    }
  }

  /**
   * Creates a new [Session] from the initialized engine.
   *
   * @param sessionConfig The configuration for the session.
   * @return A new [Session] instance.
   * @throws IllegalStateException if the engine is not initialized.
   */
  fun createSession(sessionConfig: SessionConfig = SessionConfig()): Session {
    synchronized(lock) {
      checkInitialized()

      // Using !! is okay. Checked initialization already.
      return Session(
        LiteRtLmNative.nativeCreateSession(
          handle!!,
          sessionConfig.samplerConfig,
          sessionConfig.loraConfig?.loraPath,
          sessionConfig.loraConfig?.audioLoraPath,
        )
      )
    }
  }

  /** Throws [IllegalStateException] if the engine is not initialized. */
  protected fun checkInitialized() {
    check(isInitialized()) { "Engine is not initialized." }
  }

  companion object {
    /**
     * Sets the minimum log severity for the native libraries. This affects global logging for all
     * engine instances. If not set, it uses the native libraries' default.
     */
    fun setNativeMinLogSeverity(level: LogSeverity) {
      LiteRtLmNative.nativeSetMinLogSeverity(level.severity)
    }
  }
}

enum class LogSeverity(val severity: Int) {
  VERBOSE(0),
  DEBUG(1),
  INFO(2),
  WARNING(3),
  ERROR(4),
  FATAL(5),
  INFINITY(1000),
}
