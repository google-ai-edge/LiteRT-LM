/*
 * Copyright 2026 Google LLC
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

/**
 * Whether the model may, must, or must not call the conversation's tools for a message, and which
 * of them it may call.
 *
 * It is only used if the conversation has tools and was created with
 * [ExperimentalFlags.enableConversationConstrainedDecoding] set to true. If no tool choice is
 * given, the model's default tool call constraint is used.
 *
 * Function names are the names in the tool descriptions, e.g. `get_weather` for a Kotlin function
 * `getWeather` when [ExperimentalFlags.convertCamelToSnakeCaseInToolDescription] is true.
 */
sealed class ToolChoice {
  /** The model decides whether to answer in text, call tools, or both ("auto"). */
  data object Auto : ToolChoice()

  /** The model must answer in text only, with no tool calls ("none"). */
  data object None : ToolChoice()

  /** The model must call at least one tool ("required"). */
  data object Required : ToolChoice()

  /**
   * The model must call the function [name], which must be one of the conversation's tools.
   *
   * @property name The name of the function to call.
   */
  data class ForcedFunction(val name: String) : ToolChoice() {
    init {
      require(name.isNotEmpty()) { "name must not be empty." }
    }
  }

  /**
   * The model may only call the functions in [names], which must be among the conversation's tools.
   *
   * @property names The names of the functions that the model may call.
   * @property required If true, the model must call at least one of them. Otherwise, it may also
   *   answer in text.
   */
  data class AllowedTools
  @JvmOverloads
  constructor(val names: List<String>, val required: Boolean = false) : ToolChoice() {
    init {
      require(names.isNotEmpty()) { "names must not be empty." }
      require(names.none { it.isEmpty() }) { "names must not contain an empty name." }
    }
  }
}

/**
 * Returns the tool choice type passed to the native layer, or 0 if there is no tool choice.
 *
 * Keep in sync with `CreateToolChoiceFromJni` in `jni/litertlm.cc`.
 */
internal fun ToolChoice?.toNativeType(): Int =
  when (this) {
    null -> 0
    ToolChoice.Auto -> 1
    ToolChoice.None -> 2
    ToolChoice.Required -> 3
    is ToolChoice.ForcedFunction -> 4
    is ToolChoice.AllowedTools -> if (required) 6 else 5
  }

/** Returns the function names passed to the native layer, or null if the choice has none. */
internal fun ToolChoice?.toNativeNames(): Array<String>? =
  when (this) {
    is ToolChoice.ForcedFunction -> arrayOf(name)
    is ToolChoice.AllowedTools -> names.toTypedArray()
    else -> null
  }
