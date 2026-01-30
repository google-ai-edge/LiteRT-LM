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
import kotlin.jvm.Volatile

/**
 * Manages the lifecycle of a LiteRT-LM engine, providing an interface for interacting with the
 * underlying native library.
 *
 * Example usage:
 * ```
 * val config = EngineConfig(modelPath = "...")
 * val engine = Engine(config)
 * engine.initialize()
 * ...
 * engine.close()
 * ```
 *
 * @param engineConfig The configuration for the engine.
 */
class Engine(val engineConfig: EngineConfig) : AutoCloseable {
  // A lock to protect access to the engine's state and native handle.
  private val lock = Any()

  /**
   * The native handle to the LiteRT-LM engine. A non-null value indicates an initialized engine.
   *
   * `@Volatile` ensures that changes to the handle are immediately visible across all threads.
   */
  @Volatile private var handle: Long? = null

  /** Returns `true` if the engine is initialized and ready for use; `false` otherwise. */
  fun isInitialized(): Boolean {
    return handle != null
  }

  /**
   * Initializes the native LiteRT-LM engine.
   *
   * **Note:** This operation can take a significant amount of time (e.g., 10 seconds) depending on
   * the model size and device hardware. It is strongly recommended to call this method on a
   * background thread to avoid blocking the main thread.
   *
   * @throws IllegalStateException if the engine has already been initialized.
   */
  fun initialize() {
    synchronized(lock) {
      check(!isInitialized()) { "Engine is already initialized." }

      handle =
        LiteRtLmJni.nativeCreateEngine(
          engineConfig.modelPath,
          engineConfig.backend.name,
          // convert the null value to "" to avoid passing nullable object in JNI.
          engineConfig.visionBackend?.name ?: "",
          engineConfig.audioBackend?.name ?: "",
          // convert the null value to -1 to avoid passing nullable object in JNI.
          engineConfig.maxNumTokens ?: -1,
          engineConfig.cacheDir ?: "",
          @OptIn(ExperimentalApi::class) ExperimentalFlags.enableBenchmark,
          @OptIn(ExperimentalApi::class) ExperimentalFlags.npuLibrariesDir,
        )
    }
  }

  /**
   * Closes the engine and releases the native LiteRT-LM engine's resources.
   *
   * @throws IllegalStateException if the engine is not initialized.
   */
  override fun close() {
    synchronized(lock) {
      checkInitialized()

      // Using !! is okay. Checked initialization already.
      LiteRtLmJni.nativeDeleteEngine(handle!!)
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
          conversationConfig.systemMessage?.let {
            // Convert the message's role to Role.SYSTEM for backward compatibility.
            this.add(Message(it.contents, Role.SYSTEM).toJson())
          }

          conversationConfig.systemInstruction?.let { this.add(Message(it, Role.SYSTEM).toJson()) }

          for (message in conversationConfig.initialMessages) {
            this.add(message.toJson())
          }
        }

      @OptIn(ExperimentalApi::class) // opt-in experimental flags
      return Conversation(
        LiteRtLmJni.nativeCreateConversation(
          handle!!, // Using !! is okay. Checked initialization already.
          conversationConfig.samplerConfig,
          messagesJson.toString(),
          toolManager.getToolsDescription().toString(),
          ExperimentalFlags.enableConversationConstrainedDecoding,
        ),
        toolManager,
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
      return Session(LiteRtLmJni.nativeCreateSession(handle!!, sessionConfig.samplerConfig))
    }
  }

  /** Throws [IllegalStateException] if the engine is not initialized. */
  private fun checkInitialized() {
    check(isInitialized()) { "Engine is not initialized." }
  }

  companion object {
    /**
     * Sets the minimum log severity for the native libraries. This affects global logging for all
     * engine instances. If not set, it uses the native libraries' default.
     */
    fun setNativeMinLogSeverity(level: LogSeverity) {
      LiteRtLmJni.nativeSetMinLogSeverity(level.severity)
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
