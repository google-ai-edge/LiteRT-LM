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

import {AudioContentPart, ContentPart, ImageContentPart, Message, MessageLike} from './conversation_config.js';
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
      private readonly mutexes: {executor: Mutex},
      /**
       * Max vision tokens per image, passed to the image preprocessor. Should
       * match the engine's `maxVisionTokensPerImage` so images fit the
       * selected vision encoder signatures. 0 means no budget.
       */
      private readonly visualTokenBudget = 0) {}

  async sendMessage(message: MessageLike): Promise<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    try {
      const {json, blobs} = await serializeMessage(message);
      return await this.mutexes.executor.acquireAndRun(async () => {
        const resultStr = await this.conversation.sendMessage(
            json, blobs, this.visualTokenBudget);
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
  sendMessageStreaming(message: MessageLike):
      ReadableStream<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;

    let isCancelled = false;
    let hasErrored = false;
    return new ReadableStream<Message>({
      start: (controller) => {
        const executeGeneration = async(): Promise<void> => {
          const {json, blobs} = await serializeMessage(message);
          await this.mutexes.executor.acquireAndRun(async () => {
            await this.conversation.sendMessageAsync(
                json, blobs,
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
                },
                this.visualTokenBudget);
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
            wasmClonedConversation, this.engine, this.mutexes,
            this.visualTokenBudget);
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

interface SerializedMessage {
  json: string;
  blobs: Uint8Array[];
}

async function serializeMessage(messageLike: MessageLike):
    Promise<SerializedMessage> {
  const blobs: Uint8Array[] = [];
  const message = await toWireMessage(messageLike, blobs);
  return {json: JSON.stringify(message), blobs};
}

function toMessage(messageLike: MessageLike): Message {
  if (typeof messageLike === 'string') {
    return {role: 'user', content: messageLike};
  }
  return messageLike;
}

/**
 * Converts a message to the JSON-serializable form expected by the C++
 * Conversation. Binary image and audio `data` is extracted into `blobs` and
 * referenced by `blob_index`, while string `data` (already base64-encoded) is
 * passed as `blob`.
 */
async function toWireMessage(
    messageLike: MessageLike, blobs: Uint8Array[]): Promise<Message> {
  const message = toMessage(messageLike);
  if (!Array.isArray(message.content)) {
    return message;
  }
  const content: ContentPart[] = [];
  for (const part of message.content) {
    if ((part.type !== 'image' && part.type !== 'audio') ||
        !('data' in part) || part.data === undefined) {
      content.push(part);
      continue;
    }
    const {data, ...rest} = part as ImageContentPart | AudioContentPart;
    if (typeof data === 'string') {
      content.push({...rest, blob: data} as unknown as ContentPart);
      continue;
    }
    const bytes = await toUint8Array(data);
    const blobIndex = blobs.length;
    blobs.push(bytes);
    content.push({...rest, blob_index: blobIndex} as unknown as ContentPart);
  }
  return {...message, content};
}

async function toUint8Array(data: Uint8Array|ArrayBuffer|
                            Blob): Promise<Uint8Array> {
  if (data instanceof Blob) {
    return new Uint8Array(await data.arrayBuffer());
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return data;
}
