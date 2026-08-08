/**
 * Copyright 2026 Google LLC
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

import type {ChatInterface} from '../orchestration/chat_interface.js';
import type {Conversation} from '../conversation.js';
import type {ConversationConfig, Message, MessageLike} from '../conversation_config.js';
import type {Engine} from '../engine.js';
import type {EngineSettings} from '../engine_settings.js';
import type {Session} from '../session.js';
import type {SessionConfig} from '../session_config.js';
import type {RecursiveRequired} from '../types.js';
import type {BenchmarkInfo} from '../wasm_binding_types.js';

/** Fake implementation of Conversation for testing. */
export class ConversationFake implements ChatInterface {
  history: Message[] = [];
  nextResponses: Message[] = [];

  constructor(public initialHistory: Message[] = []) {
    this.history = [...initialHistory];
  }

  /**
   * Enqueues a message that the fake will return on the next call to
   * sendMessage or sendMessageStreaming. Can be called multiple times.
   */
  queueResponse(message: Message) {
    this.nextResponses.push(message);
  }

  private popNextResponse(): Message {
    if (this.nextResponses.length > 0) {
      return this.nextResponses.shift()!;
    }
    return {role: 'model', content: 'hello from fake'};
  }

  private appendInput(message: MessageLike|MessageLike[]) {
    if (Array.isArray(message)) {
      this.history.push(...message.map(
          m => typeof m === 'string' ? {role: 'user', content: m} as Message :
                                       m));
    } else {
      this.history.push(
          typeof message === 'string' ?
              {role: 'user', content: message} as Message :
              message);
    }
  }

  async sendMessage(message: MessageLike|MessageLike[]): Promise<Message> {
    this.appendInput(message);
    const newMessage = this.popNextResponse();
    this.history.push(newMessage);
    return newMessage;
  }

  sendMessageStreaming(message: MessageLike|MessageLike[]):
      ReadableStream<Message> {
    this.appendInput(message);
    return new ReadableStream<Message>({
      start: (controller) => {
        const newMessage = this.popNextResponse();
        this.history.push(newMessage);
        controller.enqueue(newMessage);
        controller.close();
      }
    });
  }

  getHistory(): Message[] {
    return [...this.history];
  }

  cancel() {}

  async delete(): Promise<void> {}

  async getTokenCount(): Promise<number> {
    return this.history.length * 10;
  }

  async getBenchmarkInfo(): Promise<BenchmarkInfo> {
    return {
      lastPrefillTokensPerSecond: 0,
      lastPrefillTokenCount: 0,
      lastDecodeTokensPerSecond: 0,
      lastDecodeTokenCount: 0,
      timeToFirstTokenInSecond: 0,
    };
  }

  clone(): Conversation {
    const cloned = new ConversationFake([...this.history]);
    cloned.nextResponses = [...this.nextResponses];
    return cloned as unknown as Conversation;
  }
}

/** Fake implementation of Session for testing. */
export class SessionFake {
  inputsPrefilled: string[] = [];

  async runPrefill(inputs: string[]): Promise<void> {
    this.inputsPrefilled.push(...inputs);
  }

  async runDecode(): ReturnType<Session['runDecode']> {
    return {getTexts: () => ['decoded response'], delete: () => {}} as
        Awaited<ReturnType<Session['runDecode']>>;
  }

  cancel() {}

  clone(): Session {
    const cloned = new SessionFake();
    cloned.inputsPrefilled = [...this.inputsPrefilled];
    return cloned as unknown as Session;
  }

  async delete(): Promise<void> {}
}

/** Fake implementation of Engine for testing. */
export class EngineFake {
  readonly settings: RecursiveRequired<EngineSettings>;
  cachedSession = new SessionFake();
  cachedConversation = new ConversationFake();

  constructor(settings: EngineSettings) {
    this.settings = settings as RecursiveRequired<EngineSettings>;
  }

  static async create(engineSettings: EngineSettings, inputPromptAsHint = ''):
      Promise<Engine> {
    return new EngineFake(engineSettings) as unknown as Engine;
  }

  async createSession(sessionConfig: SessionConfig = {}): Promise<Session> {
    return this.cachedSession as unknown as Session;
  }

  async createConversation(config?: ConversationConfig): Promise<Conversation> {
    return this.cachedConversation as unknown as Conversation;
  }

  async delete(): Promise<void> {}
}
