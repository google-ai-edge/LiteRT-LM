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

import {Message, MessageLike} from '../conversation_config.js';
import {BenchmarkInfo} from '../wasm_binding_types.js';

/**
 * Interface representing standard chat interaction methods.
 */
export interface ChatInterface {
  sendMessage(message: MessageLike): Promise<Message>;
  sendMessageStreaming(message: MessageLike):
      ReadableStream<Message>;
  cancel(): void;
  getHistory(): Message[]|Promise<Message[]>;
  getTokenCount(): Promise<number>;
  getBenchmarkInfo(): Promise<BenchmarkInfo>;
  delete(): Promise<void>;
}
