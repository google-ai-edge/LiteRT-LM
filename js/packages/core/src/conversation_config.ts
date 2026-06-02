/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {SessionConfig} from './session_config.js';
import {JsonValue} from './types.js';

/**
 * Parameters for a tool, following JSON Schema format.
 */
export declare interface ToolParameters {
  type?: string;
  properties?: Record<string, JsonValue>;
  required?: string[];
  [key: string]: JsonValue|undefined;
}

/**
 * A tool is a function that can be called by the LLM.
 */
export declare interface Tool {
  name: string;
  description?: string;
  parameters?: ToolParameters;
  [key: string]: JsonValue|undefined;
}

/**
 * Value of a tool response.
 */
export declare interface ToolResponseValue {
  name?: string;
  tool_name?: string;
  response?: Record<string, JsonValue>;
  value?: JsonValue;
  [key: string]: JsonValue|undefined;
}

/**
 * Item in the content array of a Message.
 */
export declare interface MessageContentItem {
  type: string;
  text?: string;
  path?: string;
  tool_response?: ToolResponseValue;
  [key: string]: JsonValue|undefined;
}

/**
 * A function call predicted by the model.
 */
export declare interface ToolCallFunction {
  name: string;
  arguments: Record<string, JsonValue>;
  [key: string]: JsonValue|undefined;
}

/**
 * A tool call in the message.
 */
export declare interface ToolCall {
  type: string;
  function: ToolCallFunction;
  [key: string]: JsonValue|undefined;
}

/**
 * A message in the conversation.
 */
export declare interface Message {
  role: string;
  content?: string|MessageContentItem[];
  channels?: Record<string, string>;
  tool_calls?: ToolCall[];
  [key: string]: JsonValue|undefined;
}

/**
 * Types that can be used as a Message in the conversation.
 */
export type MessageLike = string|Message;

/**
 * Preface is the initial messages, tools and extra context for the
 * conversation to begin with. It provides the initial background for the
 * conversation.
 */
export declare interface Preface {
  // The messages in the preface. The messages provided the initial background
  // for the conversation. For example, the messages can be the conversation
  // history, prompt engineering instructions, few-shot examples, etc.
  messages?: Message[];
  // The tools able to be used by the model in the conversation.
  tools?: Tool[];
  // The extra context that is not part of the messages or tools. This is can be
  // extended by the model to support other features. For example, configurable
  // template rendering or other model-specific features.
  extra_context?: Record<string, JsonValue>;
  [key: string]: JsonValue|undefined;
}

/**
 * Configuration for the Conversation instance.
 */
export declare interface ConversationConfig {
  sessionConfig?: SessionConfig;
  preface?: Preface;
  enableConstrainedDecoding?: boolean;
  prefillPrefaceOnInit?: boolean;
  filterChannelContentFromKvCache?: boolean;
  // TODO: b/477709280 - Add support for custom channel arrays.
}
