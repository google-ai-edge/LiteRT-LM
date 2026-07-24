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
package com.google.ai.edge.litertlm.example

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.LogSeverity

suspend fun main(args: Array<String>) {
  val modelPath =
    requireNotNull(args.getOrNull(0)) { "Model path must be provided as the first argument." }

  Engine.setNativeMinLogSeverity(LogSeverity.ERROR) // silence noisy log for the TUI.

  val engineConfig = EngineConfig(modelPath = modelPath, backend = Backend.CPU())
  Engine(engineConfig).use { engine ->
    engine.initialize()

    val conversationConfig =
      ConversationConfig(systemInstruction = Contents.of("You are a helpful assistant."))
    engine.createConversation(conversationConfig).use { conversation ->
      while (true) {
        print("\n>>> ")
        conversation.sendMessageAsync(readln()).collect { print(YELLOW + it + RESET) }
      }
    }
  }
}

// ANSI color codes
const val RESET = "\u001B[0m"
const val YELLOW = "\u001B[33m"
