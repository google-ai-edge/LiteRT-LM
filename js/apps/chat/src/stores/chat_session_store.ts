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

import {AutoToolChat, ChatInterface, JsonValue, SamplerType, ToolProgressEvent, ToolWithImplementation} from '@litert-lm/core';
import {z} from 'zod';

import {CodeSandbox} from '../components/code_sandbox.js';

import {ModelLoaderService} from './model_loader_service.js';
import {SettingsStore} from './settings_store.js';

const ConversationMetaSchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.number(),
  modelPath: z.string(),
});

const ConversationMetaListSchema = z.array(ConversationMetaSchema);

const StoredMessageSchema = z.object({
  role: z.union([z.literal('user'), z.literal('assistant')]),
  text: z.string(),
  senderName: z.string(),
  thoughtText: z.string().optional(),
  prefillSpeed: z.string().optional(),
  decodeSpeed: z.string().optional(),
  tokensCount: z.string().optional(),
});

const StoredMessageListSchema = z.array(StoredMessageSchema);

/** Represents an individual chat message in the UI history. */
export type StoredMessage = z.infer<typeof StoredMessageSchema>;

/** Represents a saved conversation in the sidebar history index. */
export type ConversationMeta = z.infer<typeof ConversationMetaSchema>;

/** State store for managing individual chat sessions and history arrays. */
export class ChatSessionStore {
  isGenerating = false;
  isCancelled = false;

  activeSavedConvId: string|null = null;
  conversationsList: ConversationMeta[] = [];
  messages: StoredMessage[] = [];
  activeConversation: ChatInterface|null = null;
  codeSandbox = new CodeSandbox();

  private readonly CONVS_LIST_KEY = 'litertlm-conversations-list';

  constructor(
      private readonly updateCallback: () => void,
      private readonly settings: SettingsStore,
      private readonly modelLoader: ModelLoaderService,
      private readonly updateStatus: (msg: string) => void) {
    this.loadSavedConversationsIndex();
  }

  loadSavedConversationsIndex() {
    try {
      const data = window.localStorage.getItem(this.CONVS_LIST_KEY);
      if (data) {
        const parsed = JSON.parse(data);
        const result = ConversationMetaListSchema.safeParse(parsed);
        if (result.success) {
          this.conversationsList = result.data;
          return;
        } else {
          console.warn(
              '[LiteRT-LM] Invalid conversations index in LocalStorage, ignoring:',
              result.error);
        }
      }
      this.conversationsList = [];
    } catch (e) {
      console.error('[LiteRT-LM] Failed to load conversations index:', e);
      this.conversationsList = [];
    }
  }

  private saveSavedConversationsIndex() {
    try {
      window.localStorage.setItem(
          this.CONVS_LIST_KEY, JSON.stringify(this.conversationsList));
    } catch (e) {
      console.error('[LiteRT-LM] Failed to save conversations index:', e);
    }
  }

  async startNewConversation() {
    if (this.isGenerating) return;
    this.activeSavedConvId = null;
    window.localStorage.removeItem('litertlm-active-conv-id');
    this.messages = [];
    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }
    this.updateStatus('Ready for a new conversation.');
    this.updateCallback();
  }

  async selectConversation(id: string) {
    if (this.isGenerating) return;
    this.activeSavedConvId = id;
    window.localStorage.setItem('litertlm-active-conv-id', id);

    const historyKey = `litertlm-chat-history-${id}`;
    try {
      const historyData = window.localStorage.getItem(historyKey);
      if (!historyData) return;

      const parsed = JSON.parse(historyData);
      const result = StoredMessageListSchema.safeParse(parsed);
      if (!result.success) {
        console.warn(
            '[LiteRT-LM] Invalid conversation history in LocalStorage, ignoring:',
            result.error);
        return;
      }

      this.messages = result.data;
      this.updateStatus('Restored conversation from history.');

      const matchMeta = this.conversationsList.find(c => c.id === id);
      if (matchMeta) {
        this.settings.selectedModelPath = matchMeta.modelPath;
        this.settings.saveSettings();
      }

      if (this.activeConversation) {
        await this.activeConversation.delete();
        this.activeConversation = null;
      }

      this.updateCallback();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to load conversation history:', e);
    }
  }

  deleteConversation(id: string) {
    if (this.isGenerating) return;

    this.conversationsList = this.conversationsList.filter(c => c.id !== id);
    this.saveSavedConversationsIndex();

    window.localStorage.removeItem(`litertlm-chat-history-${id}`);

    if (this.activeSavedConvId === id) {
      void this.startNewConversation();
    } else {
      this.updateCallback();
    }
  }

  private commitActiveChatHistory() {
    if (!this.activeSavedConvId) {
      const id = Date.now().toString();
      const snippet = this.messages[0]?.text || 'Untitled Conversation';
      const title =
          snippet.length > 26 ? snippet.substring(0, 26) + '...' : snippet;

      this.activeSavedConvId = id;
      window.localStorage.setItem('litertlm-active-conv-id', id);

      this.conversationsList.unshift({
        id,
        title,
        createdAt: Date.now(),
        modelPath: this.settings.selectedModelPath
      });
      this.saveSavedConversationsIndex();
    }

    const historyKey = `litertlm-chat-history-${this.activeSavedConvId}`;
    window.localStorage.setItem(historyKey, JSON.stringify(this.messages));
    this.updateCallback();
  }

  private getSamplerTypeEnum(type: string): SamplerType {
    if (type === 'top_k') return SamplerType.TOP_K;
    if (type === 'top_p') return SamplerType.TOP_P;
    return SamplerType.GREEDY;
  }

  async createConversationSession() {
    if (!this.modelLoader.engine) return;
    this.updateStatus('Creating conversation session...');

    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }

    const jsTool: ToolWithImplementation = {
      type: 'function',
      function: {
        name: 'run_javascript',
        description:
            'Executes Javascript code in a sandbox and returns the result and console output. VERY useful for doing math, text manipulation, and other algorithmic operations.',
        parameters: {
          type: 'object',
          properties: {
            code: {
              type: 'string',
              description:
                  'The Javascript code to execute. Can use console.log to output data, or return a value.'
            }
          },
          required: ['code']
        }
      },
      execute: async (args: Record<string, JsonValue>) => {
        const code = args['code'] as string;
        if (typeof code !== 'string') {
          return {error: 'Expected string parameter "code"'};
        }
        return await this.codeSandbox.run(code);
      }
    };

    this.activeConversation = new AutoToolChat({
      engine: this.modelLoader.engine,
      config: {
        sessionConfig: {
          maxOutputTokens: this.settings.maxOutputTokens,
          samplerParams: {
            type: this.getSamplerTypeEnum(this.settings.samplerType),
            temperature: this.settings.temperature,
            p: this.settings.topP,
            k: this.settings.topK,
          }
        },
        preface: {
          messages: this.messages.length > 0 ?
              this.messages.map(msg => ({role: msg.role, content: msg.text})) :
              undefined,
          extra_context: {
            'enable_thinking': this.settings.enableThinking,
          }
        },
      },
      tools: [jsTool],
      onToolProgress: (e: ToolProgressEvent) => {
        if (e.status === 'started') {
          this.updateStatus(`Running ${e.name}...`);
        } else if (e.status === 'completed') {
          this.updateStatus(`Finished ${e.name}.`);
        } else if (e.status === 'error') {
          this.updateStatus(`Error running ${e.name}.`);
        }
      }
    });

    this.updateStatus('Model loaded and ready.');
  }

  async sendMessage(promptText: string) {
    if (this.isGenerating || !promptText.trim()) return;

    if (!this.modelLoader.engine) {
      await this.modelLoader.loadModelWeights(async () => {
        await this.createConversationSession();
      });
    } else if (!this.activeConversation) {
      await this.createConversationSession();
    }

    if (!this.modelLoader.engine || !this.activeConversation) return;

    this.isGenerating = true;
    this.isCancelled = false;
    this.updateStatus('Thinking...');

    const userMsg: StoredMessage =
        {role: 'user', text: promptText, senderName: 'User', tokensCount: '-'};
    this.messages.push(userMsg);
    this.commitActiveChatHistory();

    const modelName =
        this.settings.selectedModelPath.split('/').pop() || 'Assistant';
    const assistantMsg: StoredMessage =
        {role: 'assistant', text: '', senderName: modelName, thoughtText: ''};
    this.messages.push(assistantMsg);
    this.updateCallback();

    const activeMsgIndex = this.messages.length - 1;

    let fullResponseText = '';
    let fullThoughtText = '';
    let tokenCount = 0;

    try {
      const responseStream =
          await this.activeConversation!.sendMessageStreaming(promptText);
      const reader = responseStream.getReader();

      while (true) {
        if (this.isCancelled) {
          console.log('[LiteRT-LM] Inference stream loop cancelled.');
          await reader.cancel('User cancelled');
          break;
        }

        const {done, value} = await reader.read();
        if (done) break;

        if (value && value.channels &&
            typeof value.channels['thought'] === 'string') {
          fullThoughtText += value.channels['thought'];
          this.messages[activeMsgIndex] = {
            ...this.messages[activeMsgIndex]!,
            thoughtText: fullThoughtText
          };
        }

        if (value && value.content) {
          const newChunkText = typeof value.content === 'string' ?
              value.content :
              (value.content[0]?.['text'] || '');

          fullResponseText += newChunkText;
          this.messages[activeMsgIndex] = {
            ...this.messages[activeMsgIndex]!,
            text: fullResponseText
          };
        }

        tokenCount++;
        this.updateCallback();
      }

      let promptTokens = 0;
      let finalDecodeTokens = tokenCount;
      let prefillSpeedVal = '0.0';
      let decodeSpeedVal = '0.0';

      try {
        const benchmark = await this.activeConversation!.getBenchmarkInfo();

        if (benchmark.lastPrefillTokenCount > 0) {
          promptTokens = benchmark.lastPrefillTokenCount;
          prefillSpeedVal = benchmark.lastPrefillTokensPerSecond.toFixed(1);
        }

        if (benchmark.lastDecodeTokenCount > 0) {
          finalDecodeTokens = benchmark.lastDecodeTokenCount;
          decodeSpeedVal = benchmark.lastDecodeTokensPerSecond.toFixed(1);
        }
      } catch (e) {
        console.warn('Benchmark metrics not available:', e);
      }

      this.messages[activeMsgIndex - 1] = {
        ...this.messages[activeMsgIndex - 1]!,
        tokensCount: promptTokens.toString()
      };

      this.messages[activeMsgIndex] = {
        ...this.messages[activeMsgIndex]!,
        prefillSpeed: `${prefillSpeedVal} tk/s`,
        decodeSpeed: `${decodeSpeedVal} tk/s`,
        tokensCount: finalDecodeTokens.toString()
      };

      this.updateStatus('Generation completed.');
    } catch (err: unknown) {
      console.error(err);
      const error = err as Error;
      this.updateStatus(`Generation failed: ${error.message || error}`);

      if (this.isCancelled) {
        fullResponseText += '\n\n*[Generation stopped by user]*';
        this.messages[activeMsgIndex] = {
          ...this.messages[activeMsgIndex]!,
          text: fullResponseText
        };
      }
    } finally {
      if (this.isCancelled) {
        try {
          if (this.activeConversation) {
            await this.activeConversation.delete();
          }
          this.activeConversation = null;
          await this.createConversationSession();
        } catch (e) {
          console.error(
              '[LiteRT-LM] Failed KV Cache rebuild on cancellation:', e);
        }
      }

      this.commitActiveChatHistory();
      this.isGenerating = false;
    }
  }

  cancelGeneration() {
    this.isCancelled = true;
  }

  async redoResponse(targetMessageIndex: number) {
    if (this.isGenerating || targetMessageIndex < 1) return;

    if (this.messages[targetMessageIndex]?.role !== 'assistant') {
      console.warn('Cannot redo a non-assistant message');
      return;
    }

    const previousUserMsg = this.messages[targetMessageIndex - 1];
    if (!previousUserMsg || previousUserMsg.role !== 'user') {
      console.warn('Cannot find corresponding user prompt to redo');
      return;
    }

    this.messages = this.messages.slice(0, targetMessageIndex - 1);
    this.commitActiveChatHistory();

    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }

    await this.sendMessage(previousUserMsg.text);
  }

  async rewindAndEdit(targetMessageIndex: number): Promise<string|null> {
    if (this.isGenerating || targetMessageIndex < 0) return null;

    let previousUserMsgText = '';

    if (this.messages[targetMessageIndex]?.role === 'user') {
      previousUserMsgText = this.messages[targetMessageIndex]!.text;
      this.messages = this.messages.slice(0, targetMessageIndex);
    } else {
      const userMsgIdx = targetMessageIndex - 1;
      previousUserMsgText = this.messages[userMsgIdx]?.text || '';
      this.messages = this.messages.slice(0, userMsgIdx);
    }

    this.commitActiveChatHistory();

    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }

    return previousUserMsgText;
  }
}
