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

import {AutoToolChat, ChatInterface, ContentPart, Message, MessageLike, SamplerType, ToolProgressEvent} from '@litert-lm/core';
import {z} from 'zod';

import {CodeSandbox} from '../components/code_sandbox.js';
import {createObjectUrl} from '../components/util.js';

import {ModelLoaderService} from './model_loader_service.js';
import {findModel, migrateModelPath, ModelSettings, SettingsStore} from './settings_store.js';
import {createEnabledTools} from './tools.js';

const ConversationMetaSchema = z.object({
  id: z.string(),
  title: z.string(),
  createdAt: z.number(),
  modelPath: z.string(),
});

const ConversationMetaListSchema = z.array(ConversationMetaSchema);

/**
 * A persisted part of a user message. Only an image's or audio clip's id and
 * name are saved; its bytes are kept in memory for the lifetime of the page.
 */
const StoredPartSchema = z.union([
  z.object({type: z.literal('text'), text: z.string()}),
  z.object({
    type: z.union([z.literal('image'), z.literal('audio')]),
    id: z.string(),
    name: z.string(),
  }),
]);

const StoredToolCallSchema = z.object({
  id: z.number(),
  name: z.string(),
  argsText: z.string(),
  status: z.union([
    z.literal('started'),
    z.literal('completed'),
    z.literal('error'),
  ]),
  outputText: z.string().optional(),
});

/** A persisted tool call execution record on an assistant message. */
export type StoredToolCall = z.infer<typeof StoredToolCallSchema>;

const StoredMessageSchema = z.object({
  role: z.union([z.literal('user'), z.literal('assistant')]),
  // Plain-text form of the message. Media appear as e.g. `[image: name]`.
  text: z.string(),
  // Ordered text and media parts, present only for messages with media.
  parts: z.array(StoredPartSchema).optional(),
  senderName: z.string(),
  thoughtText: z.string().optional(),
  toolCalls: z.array(StoredToolCallSchema).optional(),
  stopped: z.boolean().optional(),
  prefillSpeed: z.string().optional(),
  decodeSpeed: z.string().optional(),
  tokensCount: z.string().optional(),
});

const StoredMessageListSchema = z.array(StoredMessageSchema);

/** Represents an individual chat message in the UI history. */
export type StoredMessage = z.infer<typeof StoredMessageSchema>;

/** A persisted part of a user message. */
export type StoredPart = z.infer<typeof StoredPartSchema>;

/** The kinds of media that can be attached to a prompt. */
export type MediaKind = 'image'|'audio';

/** An image or audio clip attached to a prompt. */
export interface PromptMedia {
  id: string;
  kind: MediaKind;
  name: string;
  blob: Blob;
  /** Object URL of `blob`, for previews and playback. */
  url: string;
}

/**
 * A single element of a prompt, in the order the user arranged it. Media are
 * interleaved with text so the user controls where each image or audio clip
 * sits relative to the surrounding words.
 */
export type PromptPart = {
  type: 'text'; text: string;
}|{
  type: 'media';
  media: PromptMedia;
};

/** Returns the placeholder text used for media the model cannot see. */
export function mediaPlaceholder(kind: MediaKind, name: string): string {
  return `[${kind}: ${name}]`;
}

/** Returns the plain-text form of a prompt, with media as placeholders. */
export function promptPartsToText(parts: PromptPart[]): string {
  return parts
      .map(
          p => p.type === 'text' ? p.text :
                                   mediaPlaceholder(p.media.kind, p.media.name))
      .join('');
}

/** Whether a prompt has anything worth sending. */
export function promptHasContent(parts: PromptPart[]): boolean {
  return parts.some(p => p.type === 'media' || p.text.trim().length > 0);
}

/** Trims leading and trailing whitespace from a prompt. */
function trimPromptParts(parts: PromptPart[]): PromptPart[] {
  const trimmed = parts.filter(p => p.type === 'media' || p.text.length > 0)
                      .map(p => ({...p}));
  const first = trimmed[0];
  if (first?.type === 'text') first.text = first.text.trimStart();
  const last = trimmed[trimmed.length - 1];
  if (last?.type === 'text') last.text = last.text.trimEnd();
  return trimmed.filter(p => p.type === 'media' || p.text.length > 0);
}

/** Whether two arrays have the same elements, in any order. */
function haveSameElements(a: readonly string[], b: readonly string[]): boolean {
  const setA = new Set(a);
  const setB = new Set(b);
  return setA.size === setB.size && [...setA].every(item => setB.has(item));
}

/** Represents a saved conversation in the sidebar history index. */
export type ConversationMeta = z.infer<typeof ConversationMetaSchema>;

/** Snapshot of a recently deleted conversation that can be restored via Undo. */
export interface DeletedConversationSnapshot {
  meta: ConversationMeta;
  index: number;
  messages: StoredMessage[];
  wasActive: boolean;
}

function formatToolCallArgs(e: ToolProgressEvent): string {
  if (e.name === 'run_javascript' && typeof e.arguments['code'] === 'string') {
    return e.arguments['code'];
  }
  return JSON.stringify(e.arguments, null, 2);
}

function formatToolCallOutput(e: ToolProgressEvent): string {
  if (e.status === 'error' && e.error) {
    return `Error: ${e.error}`;
  }
  const res = e.result as {
    resultAsString?: string;
    error?: string;
    consoleMessages?: Array<{type: string; text: string}>;
  } | undefined;
  if (res && typeof res === 'object') {
    const parts: string[] = [];
    if (Array.isArray(res.consoleMessages) && res.consoleMessages.length > 0) {
      parts.push(
          res.consoleMessages
              .map(c => c.type === 'log' ? c.text : `[${c.type}] ${c.text}`)
              .join('\n'));
    }
    if (typeof res.error === 'string' && res.error) {
      parts.push(`Error: ${res.error}`);
    } else if (
        typeof res.resultAsString === 'string' &&
        res.resultAsString !== 'undefined') {
      parts.push(`→ ${res.resultAsString}`);
    }
    if (parts.length > 0) return parts.join('\n');
    if (res.resultAsString === 'undefined') {
      return 'Completed (no return value)';
    }
  }
  if (e.result === undefined) return '';
  return typeof e.result === 'string' ? e.result :
                                        JSON.stringify(e.result, null, 2);
}

/** Returns the status to show when nothing is in progress. */
export function getIdleStatus(modelLoaded: boolean): string {
  return modelLoaded ? 'Model loaded and ready.' :
                       'Idle · Weights load on first message';
}

/** State store for managing individual chat sessions and history arrays. */
export class ChatSessionStore {
  isGenerating = false;
  isCancelled = false;
  activeToolName: string|null = null;
  deletedConversation: DeletedConversationSnapshot|null = null;

  activeSavedConvId: string|null = null;
  activeModelSettings: ModelSettings|null = null;
  conversationsList: ConversationMeta[] = [];
  messages: StoredMessage[] = [];
  activeConversation: ChatInterface|null = null;
  codeSandbox = new CodeSandbox();

  private readonly CONVS_LIST_KEY = 'litertlm-conversations-list';
  private deleteToastTimer: ReturnType<typeof setTimeout>|null = null;
  private activeMsgIndex: number|null = null;
  private pendingToolCalls: StoredToolCall[] = [];

  /**
   * Media attached to prompts during this page's lifetime, by id. Saved
   * history only references media by id, so media from a previous page load
   * are not found here.
   */
  private readonly media = new Map<string, PromptMedia>();

  constructor(
      private readonly updateCallback: () => void,
      private readonly settings: SettingsStore,
      private readonly modelLoader: ModelLoaderService,
      private readonly updateStatus: (msg: string) => void,
      private readonly storage: Storage = window.localStorage) {
    this.loadSavedConversationsIndex();
  }

  /** Creates an image or audio clip that can be placed in a prompt. */
  createMedia(kind: MediaKind, name: string, blob: Blob): PromptMedia {
    const media: PromptMedia = {
      id: Math.random().toString(36).substring(2) + Date.now().toString(36),
      kind,
      name,
      blob,
      url: createObjectUrl(blob),
    };
    this.media.set(media.id, media);
    return media;
  }

  /** Returns media created during this page's lifetime, if available. */
  getMedia(id: string): PromptMedia|undefined {
    return this.media.get(id);
  }

  /**
   * Returns the prompt parts of a stored message. Media that are no longer in
   * memory become text placeholders.
   */
  getMessageParts(msg: StoredMessage): PromptPart[] {
    if (!msg.parts) return [{type: 'text', text: msg.text}];
    return msg.parts.map((p): PromptPart => {
      if (p.type === 'text') return p;
      const media = this.media.get(p.id);
      return media ? {type: 'media', media} :
                     {type: 'text', text: mediaPlaceholder(p.type, p.name)};
    });
  }

  loadSavedConversationsIndex() {
    try {
      const data = this.storage.getItem(this.CONVS_LIST_KEY);
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
      this.storage.setItem(
          this.CONVS_LIST_KEY, JSON.stringify(this.conversationsList));
    } catch (e) {
      console.error('[LiteRT-LM] Failed to save conversations index:', e);
    }
  }

  private readConversationHistory(id: string): StoredMessage[]|null {
    const historyData = this.storage.getItem(`litertlm-chat-history-${id}`);
    if (!historyData) return null;
    const parsed = JSON.parse(historyData);
    const result = StoredMessageListSchema.safeParse(parsed);
    if (!result.success) {
      console.warn(
          '[LiteRT-LM] Invalid conversation history in LocalStorage, ignoring:',
          result.error);
      return null;
    }
    return result.data;
  }

  private writeConversationHistory(id: string, messages: StoredMessage[]) {
    this.storage.setItem(
        `litertlm-chat-history-${id}`, JSON.stringify(messages));
  }

  async startNewConversation() {
    if (this.isGenerating) return;
    this.activeSavedConvId = null;
    this.storage.removeItem('litertlm-active-conv-id');
    this.messages = [];
    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }
    this.updateStatus(getIdleStatus(Boolean(this.modelLoader.engine)));
    this.updateCallback();
  }

  async selectConversation(id: string) {
    if (this.isGenerating) return;
    this.activeSavedConvId = id;
    this.storage.setItem('litertlm-active-conv-id', id);

    try {
      const restored = this.readConversationHistory(id);
      if (!restored) return;

      this.messages = restored;
      this.updateStatus('Restored conversation from history.');

      const matchMeta = this.conversationsList.find(c => c.id === id);
      if (matchMeta) {
        this.settings.selectedModelPath = migrateModelPath(matchMeta.modelPath);
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

    const index = this.conversationsList.findIndex(c => c.id === id);
    const meta = index >= 0 ? this.conversationsList[index] : undefined;
    let savedMessages: StoredMessage[] = [];
    if (this.activeSavedConvId === id) {
      savedMessages = [...this.messages];
    } else {
      try {
        savedMessages = this.readConversationHistory(id) ?? [];
      } catch {
        savedMessages = [];
      }
    }

    if (meta) {
      this.clearDeleteToastTimer();
      this.deletedConversation = {
        meta,
        index,
        messages: savedMessages,
        wasActive: this.activeSavedConvId === id,
      };
      this.deleteToastTimer = setTimeout(() => {
        this.dismissDeletedConversation();
      }, 7000);
    }

    this.conversationsList = this.conversationsList.filter(c => c.id !== id);
    this.saveSavedConversationsIndex();

    this.storage.removeItem(`litertlm-chat-history-${id}`);

    if (this.activeSavedConvId === id) {
      void this.startNewConversation();
    } else {
      this.updateCallback();
    }
  }

  private clearDeleteToastTimer() {
    if (this.deleteToastTimer !== null) {
      clearTimeout(this.deleteToastTimer);
      this.deleteToastTimer = null;
    }
  }

  dismissDeletedConversation() {
    this.clearDeleteToastTimer();
    if (!this.deletedConversation) return;
    this.deletedConversation = null;
    this.updateCallback();
  }

  async undoDeleteConversation() {
    const snap = this.deletedConversation;
    if (!snap) return;
    this.clearDeleteToastTimer();
    this.deletedConversation = null;

    if (!this.conversationsList.some(c => c.id === snap.meta.id)) {
      const insertAt = Math.min(snap.index, this.conversationsList.length);
      const next = [...this.conversationsList];
      next.splice(insertAt, 0, snap.meta);
      this.conversationsList = next;
    }
    this.saveSavedConversationsIndex();
    this.writeConversationHistory(snap.meta.id, snap.messages);

    if (snap.wasActive && this.messages.length === 0 && !this.isGenerating) {
      await this.selectConversation(snap.meta.id);
    } else {
      this.updateCallback();
    }
  }

  private commitActiveChatHistory() {
    if (!this.activeSavedConvId) {
      const id = Date.now().toString();
      const rawSnippet = this.messages[0]?.text || 'Untitled Conversation';
      const snippet = rawSnippet.replace(/\s+/g, ' ').trim();
      const title =
          snippet.length > 44 ? snippet.substring(0, 44) + '...' : snippet;

      this.activeSavedConvId = id;
      this.storage.setItem('litertlm-active-conv-id', id);

      this.conversationsList.unshift({
        id,
        title,
        createdAt: Date.now(),
        modelPath: this.settings.selectedModelPath
      });
      this.saveSavedConversationsIndex();
    }

    this.writeConversationHistory(this.activeSavedConvId, this.messages);
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

    const prefaceMessages: Message[] = [];
    const systemPrompt = this.settings.systemPrompt.trim();
    if (systemPrompt) {
      prefaceMessages.push({role: 'system', content: systemPrompt});
    }
    // Media in the history are replayed as text placeholders (e.g.
    // `[image: name]`); only media in new messages reach the model.
    prefaceMessages.push(
        ...this.messages.map(msg => ({role: msg.role, content: msg.text})));

    this.activeConversation = new AutoToolChat({
      engine: this.modelLoader.engine,
      config: {
        sessionConfig: {
          maxOutputTokens: this.settings.maxOutputTokens,
          samplerParams: {
            type: this.getSamplerTypeEnum(this.settings.samplerType),
            temperature: this.settings.temperature,
            p: this.settings.topP,
            // Greedy sampling skips the reload that raises the engine's max
            // Top-K (see `ModelLoaderService.needsModelReload`).
            k: this.settings.samplerType === 'greedy' ? 1 : this.settings.topK,
          }
        },
        preface: {
          messages: prefaceMessages.length > 0 ? prefaceMessages : undefined,
          extra_context: {
            'enable_thinking': this.settings.enableThinking,
          }
        },
      },
      tools: createEnabledTools(
          this.settings.disabledTools, {codeSandbox: this.codeSandbox}),
      onToolProgress: (e: ToolProgressEvent) => {
        const record: StoredToolCall = {
          id: e.id,
          name: e.name,
          argsText: formatToolCallArgs(e),
          status: e.status,
          outputText: e.status === 'started' ? undefined :
                                               formatToolCallOutput(e),
        };
        const existingIdx =
            this.pendingToolCalls.findIndex(tc => tc.id === e.id);
        if (existingIdx >= 0) {
          this.pendingToolCalls[existingIdx] = record;
        } else {
          this.pendingToolCalls.push(record);
        }
        if (this.activeMsgIndex !== null &&
            this.messages[this.activeMsgIndex]?.role === 'assistant') {
          this.messages[this.activeMsgIndex] = {
            ...this.messages[this.activeMsgIndex]!,
            toolCalls: [...this.pendingToolCalls],
          };
        }
        if (e.status === 'started') {
          this.activeToolName = e.name;
          this.updateStatus(`Running ${e.name}...`);
        } else if (e.status === 'completed') {
          this.activeToolName = null;
          this.updateStatus(`Finished ${e.name}.`);
        } else if (e.status === 'error') {
          this.activeToolName = null;
          this.updateStatus(`Error running ${e.name}.`);
        }
        this.updateCallback();
      }
    });

    this.activeModelSettings = structuredClone(this.settings.modelSettings);
    this.updateStatus('Model loaded and ready.');
  }

  private needsNewSession(settings: ModelSettings): boolean {
    if (!this.activeConversation || !this.activeModelSettings) return true;
    return this.activeModelSettings.selectedModelPath !== settings.selectedModelPath ||
           this.activeModelSettings.contextLength !== settings.contextLength ||
           this.activeModelSettings.maxOutputTokens !== settings.maxOutputTokens ||
           this.activeModelSettings.samplerType !== settings.samplerType ||
           this.activeModelSettings.temperature !== settings.temperature ||
           this.activeModelSettings.topP !== settings.topP ||
           (settings.samplerType !== 'greedy' &&
            this.activeModelSettings.topK !== settings.topK) ||
           this.activeModelSettings.enableThinking !== settings.enableThinking ||
           this.activeModelSettings.systemPrompt !== settings.systemPrompt ||
           !haveSameElements(
               this.activeModelSettings.disabledTools, settings.disabledTools);
  }

  /**
   * Sends a prompt to the model. A prompt is either plain text or an ordered
   * list of text, image, and audio parts.
   */
  async sendMessage(prompt: string|PromptPart[]) {
    const parts = trimPromptParts(
        typeof prompt === 'string' ? [{type: 'text', text: prompt}] : prompt);
    if (this.isGenerating || !promptHasContent(parts)) return;

    const settings = this.settings.modelSettings;
    if (this.modelLoader.needsModelReload(settings)) {
      await this.modelLoader.loadModelWeights(settings, async () => {
        await this.createConversationSession();
      });
    } else if (this.needsNewSession(settings)) {
      await this.createConversationSession();
    }

    if (!this.modelLoader.engine || !this.activeConversation) return;

    this.isGenerating = true;
    this.isCancelled = false;
    this.updateStatus('Thinking...');

    const promptText = promptPartsToText(parts);
    const hasMedia = parts.some(p => p.type === 'media');
    const userMsg: StoredMessage = {
      role: 'user',
      text: promptText,
      senderName: 'User',
      tokensCount: '-'
    };
    if (hasMedia) {
      userMsg.parts = parts.map(
          (p): StoredPart => p.type === 'text' ?
              p :
              {type: p.media.kind, id: p.media.id, name: p.media.name});
    }
    this.messages.push(userMsg);
    this.commitActiveChatHistory();

    // Keep plain-text prompts as strings; send prompts with media as an
    // ordered content array so each image or audio clip sits where the user
    // placed it.
    const modelMessage: MessageLike = hasMedia ? {
      role: 'user',
      content: parts.map(
          (p): ContentPart => p.type === 'text' ?
              {type: 'text', text: p.text} :
              {type: p.media.kind, data: p.media.blob}),
    } :
                                                 promptText;

    const modelName =
        findModel(this.settings, this.settings.selectedModelPath)?.name ||
        this.settings.selectedModelPath.split('/').pop() || 'Assistant';
    const assistantMsg: StoredMessage =
        {role: 'assistant', text: '', senderName: modelName, thoughtText: ''};
    this.messages.push(assistantMsg);
    this.updateCallback();

    const activeMsgIndex = this.messages.length - 1;
    this.activeMsgIndex = activeMsgIndex;
    this.pendingToolCalls = [];

    let fullResponseText = '';
    let fullThoughtText = '';
    let tokenCount = 0;

    try {
      const responseStream =
          await this.activeConversation!.sendMessageStreaming(modelMessage);
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
          const firstPart = Array.isArray(value.content) ? value.content[0] : undefined;
          const newChunkText = typeof value.content === 'string' ?
              value.content :
              (firstPart?.type === 'text' ? firstPart.text : '');

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
      let prefillSpeed: string|undefined;
      let decodeSpeed: string|undefined;

      try {
        const benchmark = await this.activeConversation!.getBenchmarkInfo();

        if (benchmark.lastPrefillTokenCount > 0) {
          promptTokens = benchmark.lastPrefillTokenCount;
          prefillSpeed =
              `${benchmark.lastPrefillTokensPerSecond.toFixed(1)} tk/s`;
        }

        if (benchmark.lastDecodeTokenCount > 0) {
          finalDecodeTokens = benchmark.lastDecodeTokenCount;
          decodeSpeed = `${benchmark.lastDecodeTokensPerSecond.toFixed(1)} tk/s`;
        }
      } catch (e) {
        console.warn('Benchmark metrics not available:', e);
      }

      if (this.isCancelled) {
        // Keep the partial response with whichever metrics were measured.
        if (promptTokens > 0) {
          this.messages[activeMsgIndex - 1] = {
            ...this.messages[activeMsgIndex - 1]!,
            tokensCount: promptTokens.toString(),
          };
        }
        this.messages[activeMsgIndex] = {
          ...this.messages[activeMsgIndex]!,
          text: fullResponseText,
          stopped: true,
          ...(prefillSpeed ? {prefillSpeed} : {}),
          ...(decodeSpeed ? {decodeSpeed} : {}),
          ...(finalDecodeTokens > 0 ?
                  {tokensCount: finalDecodeTokens.toString()} :
                  {}),
        };
      } else {
        this.messages[activeMsgIndex - 1] = {
          ...this.messages[activeMsgIndex - 1]!,
          tokensCount: promptTokens.toString()
        };

        this.messages[activeMsgIndex] = {
          ...this.messages[activeMsgIndex]!,
          prefillSpeed: prefillSpeed ?? '0.0 tk/s',
          decodeSpeed: decodeSpeed ?? '0.0 tk/s',
          tokensCount: finalDecodeTokens.toString()
        };

        this.updateStatus('Generation completed.');
      }
    } catch (err: unknown) {
      if (this.isCancelled) {
        this.messages[activeMsgIndex] = {
          ...this.messages[activeMsgIndex]!,
          text: fullResponseText,
          stopped: true,
          ...(tokenCount > 0 ? {tokensCount: tokenCount.toString()} : {}),
        };
      } else {
        console.error(err);
        const error = err as Error;
        const activeMsg = this.messages[activeMsgIndex];
        const hasToolCalls = Boolean(activeMsg?.toolCalls?.length);
        if (!fullResponseText && !fullThoughtText && !hasToolCalls &&
            activeMsg?.role === 'assistant' && !activeMsg.text &&
            !activeMsg.thoughtText) {
          this.messages.splice(activeMsgIndex, 1);
        } else if (
            !fullResponseText && (fullThoughtText || hasToolCalls) &&
            activeMsg) {
          this.messages[activeMsgIndex] = {
            ...activeMsg,
            text: '*[Generation interrupted by an error]*',
          };
        }
        this.updateStatus(`Generation failed: ${error.message || error}`);
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
        this.updateStatus('Generation stopped.');
      }

      this.activeMsgIndex = null;
      this.activeToolName = null;
      this.commitActiveChatHistory();
      this.isGenerating = false;
      this.updateCallback();
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

    await this.sendMessage(
        previousUserMsg.parts ? this.getMessageParts(previousUserMsg) :
                                previousUserMsg.text);
  }

  /**
   * Removes the given message, the messages after it, and its user prompt
   * from the history. Returns that prompt so it can be edited and re-sent.
   */
  async rewindAndEdit(targetMessageIndex: number):
      Promise<PromptPart[]|null> {
    if (this.isGenerating || targetMessageIndex < 0) return null;

    const userMsgIdx = this.messages[targetMessageIndex]?.role === 'user' ?
        targetMessageIndex :
        targetMessageIndex - 1;
    const userMsg = this.messages[userMsgIdx];
    this.messages = this.messages.slice(0, Math.max(0, userMsgIdx));

    this.commitActiveChatHistory();

    if (this.activeConversation) {
      await this.activeConversation.delete();
      this.activeConversation = null;
    }

    return userMsg ? this.getMessageParts(userMsg) : null;
  }
}
