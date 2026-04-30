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

/**
 * Experimental flags for the LiteRT-LM.
 *
 * This class is experimental, may change or be removed without notice.
 *
 * To access this APi, the caller need annotation `@OptIn(ExperimentalApi::class)`.
 */
@ExperimentalApi
object ExperimentalFlags {

  /**
   * Whether to enable benchmark.
   *
   * Note: This flag is read only when a new [Engine] is created. Changing this value will not
   * affect any existing [Engine] or [Conversation] instances.
   */
  var enableBenchmark: Boolean = false

  /**
   * Whether to enable speculative decoding.
   *
   * If null, use the model's default. If true, enable speculative decoding; an error will be thrown
   * if the model does not support it. If false, disable it.
   *
   * Note: This flag is read only when a new [Engine] is created. Changing this value will not
   * affect any existing [Engine] or [Conversation] instances.
   */
  var enableSpeculativeDecoding: Boolean? = null

  /**
   * Whether to enable conversation constrained decoding. This is primarily used for function
   * calling.
   *
   * Note: This flag is read only when a new [Conversation] is created. Changing this value will not
   * affect any existing [Conversation] instances.
   */
  var enableConversationConstrainedDecoding: Boolean = false

  /**
   * Whether to convert the function and parameter names in to snake case for tool calling.
   *
   * Kotlin idiomatic style uses camelCase for names. However, many large language models are
   * predominantly trained on datasets where snake_case is common.
   *
   * By default, this API converts Kotlin camelCase names to snake_case to potentially improve tool
   * calling performance with models more familiar with snake_case.
   *
   * While the choice between snake_case and camelCase often has minimal impact, it can be
   * significant for smaller, fine-tuned models that were trained exclusively with one convention.
   *
   * Set this flag to `false` if your model is specifically trained with camelCase tool descriptions
   * to skip the conversion. Otherwise, the default of `true` (using snake_case) is recommended.
   *
   * Note: This flag is read only when a new [Conversation] is created. Changing this value will not
   * affect any existing [Conversation] instances.
   */
  var convertCamelToSnakeCaseInToolDescription: Boolean = true

  /**
   * Whether to filter channel content from the KV cache.
   *
   * If true, channel content (e.g. reasoning) will be filtered from the KV cache.
   *
   * Note: This flag is read only when a new [Conversation] is created. Changing this value will not
   * affect any existing [Conversation] instances.
   */
  var filterChannelContentFromKvCache: Boolean = false

  /**
   * A prompt template to overwrite the default one.
   *
   * Note: This flag is read only when a new [Conversation] is created. Changing this value will not
   * affect any existing [Conversation] instances.
   */
  var overwritePromptTemplate: String? = null
}

// Mark this annotation itself as requiring opt-in
@RequiresOptIn(
  message = "This API is experimental and temporary. It may change or be removed without notice.",
  level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
annotation class ExperimentalApi
