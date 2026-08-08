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

import {Message, MessageLike} from './conversation_config.js';
import {Mutex} from './mutex.js';
import {ChatInterface} from './orchestration/chat_interface.js';
import {BenchmarkInfo, Conversation as WasmConversation, Engine as WasmEngine} from './wasm_binding_types.js';

const BUSY_MESSAGE =
    'Conversation is busy. A generation is already in progress.';

/**
 * LiteRT-LM Conversation
 */
export class Conversation implements ChatInterface {
  private isBusy = false;

  constructor(
      private readonly conversation: WasmConversation,
      private readonly engine: WasmEngine,
      private readonly mutexes: {executor: Mutex}) {}

  async sendMessage(message: MessageLike|MessageLike[]): Promise<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      return await this.mutexes.executor.acquireAndRun(async () => {
        const currentMessageJson = messageToJsonString(message);
        const resultStr =
            await this.conversation.sendMessage(currentMessageJson);
        return JSON.parse(resultStr) as Message;
      });
    } finally {
      this.isBusy = false;
    }
  }

  /**
   * Sends a message to the LLM and returns a ReadableStream that yields
   * message chunks as they are generated.
   */
  sendMessageStreaming(message: MessageLike|MessageLike[]):
      ReadableStream<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;

    let isCancelled = false;
    let hasErrored = false;
    return new ReadableStream<Message>({
      start: (controller) => {
        const currentMessageJson = messageToJsonString(message);

        const executeGeneration = async(): Promise<void> => {
          await this.mutexes.executor.acquireAndRun(async () => {
            await this.conversation.sendMessageAsync(
                currentMessageJson,
                (chunk: string|null, isDone: boolean, error: string|null) => {
                  if (isCancelled || hasErrored) return;
                  if (error) {
                    hasErrored = true;
                    this.isBusy = false;
                    controller.error(new Error(error));
                    return;
                  }
                  if (chunk) {
                    try {
                      const msg = JSON.parse(chunk) as Message;
                      controller.enqueue(msg);
                    } catch (e) {
                      hasErrored = true;
                      this.isBusy = false;
                      controller.error(e);
                    }
                  }
                });
            // Since we're using the synchronus execution manager in C++, which
            // lazily executes tasks, we must queue the message and call
            // waitUntilDone to start the execution concurrently while holding
            // the mutex lock.
            await this.engine.waitUntilDone();
          });

          if (isCancelled || hasErrored) return;
          this.isBusy = false;
          controller.close();
        };

        executeGeneration().catch((e) => {
          if (isCancelled || hasErrored) return;
          this.isBusy = false;
          controller.error(e);
        });
      },
      cancel: () => {
        isCancelled = true;
        this.isBusy = false;
        this.cancel();
      }
    });
  }

  /**
   * Sends a signal to cancel any current generation.
   */
  cancel() {
    this.conversation.cancelProcess();
  }

  async getHistory(): Promise<Message[]> {
    return this.mutexes.executor.acquireAndRun(() => {
      const historyStr = this.conversation.getHistory();
      return JSON.parse(historyStr) as Message[];
    });
  }

  async getTokenCount(): Promise<number> {
    return this.mutexes.executor.acquireAndRun(() => {
      return this.conversation.getTokenCount();
    });
  }

  async getBenchmarkInfo(): Promise<BenchmarkInfo> {
    return this.mutexes.executor.acquireAndRun(() => {
      return this.conversation.getBenchmarkInfo();
    });
  }

  /**
   * Clones the conversation. The cloned conversation will be independent of the
   * original conversation, including the history, state, etc.
   */
  async clone(): Promise<Conversation> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      return await this.mutexes.executor.acquireAndRun(() => {
        const wasmClonedConversation = this.conversation.clone();
        return new Conversation(
            wasmClonedConversation, this.engine, this.mutexes);
      });
    } finally {
      this.isBusy = false;
    }
  }

  async delete(): Promise<void> {
    await this.mutexes.executor.acquireAndRun(() => {
      this.conversation.delete();
    });
  }
}

function messageToJsonString(messageLike: MessageLike|MessageLike[]): string {
  let message: Message|Message[];
  if (Array.isArray(messageLike)) {
    message = messageLike.map(toMessage);
  } else {
    message = toMessage(messageLike);
  }
  return JSON.stringify(message);
}

function toMessage(messageLike: MessageLike): Message {
  if (typeof messageLike === 'string') {
    return {role: 'user', content: messageLike};
  }
  return messageLike;
}
