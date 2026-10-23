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

import {Conversation} from '../conversation.js';
import {ContentPart, ConversationConfig, FunctionTool, Message, MessageLike, Tool, ToolCall} from '../conversation_config.js';
import {Engine} from '../engine.js';
import {JsonValue} from '../types.js';
import {BenchmarkInfo} from '../wasm_binding_types.js';

import {ChatInterface} from './chat_interface.js';
import {isWebMcpTool, toTool, WebMcpTool} from './webmcp_tool.js';

const BUSY_MESSAGE =
    'Conversation is busy. A generation is already in progress.';


type ToolExecute = (args: Record<string, JsonValue>) =>
    JsonValue|Promise<JsonValue>;

/**
 * A tool with an `execute` function implementation.
 */
export type ToolWithImplementation = Tool&{execute: ToolExecute};

/**
 * Event detail representing the execution progress of a single tool.
 */
export interface ToolProgressEvent {
  id: number;
  name: string;
  arguments: Record<string, JsonValue>;
  status: 'started'|'completed'|'error';
  result?: unknown;
  error?: string;
}

/**
 * Options for configuring an AutoToolChat.
 */
export interface AutoToolChatOptions {
  engine: Engine;
  config?: ConversationConfig;
  tools?: Array<ToolWithImplementation|WebMcpTool>;
  recurringToolCallLimit?: number;
  onToolProgress?: (event: ToolProgressEvent) => void;
}

/**
 * Wraps a conversation and automatically executes tools for the model.
 *
 * Tools are called in parallel, but the model waits for all of them to
 * finish before being woken up again.
 */
export class AutoToolChat implements ChatInterface {
  private baseConversation: Conversation|null = null;
  private isBusy = false;
  private isCancelled = false;
  private readonly tools = new Map<string, ToolWithImplementation>();
  private nextToolCallId = 1;
  private currentBatch: ToolCallBatch|null = null;

  constructor(private readonly options: AutoToolChatOptions) {
    if (options.config?.preface?.tools &&
        options.config.preface.tools.length > 0) {
      throw new Error(
          'AutoToolChat handles tools itself. Do not provide tools via ConversationConfig.preface.tools.');
    }

    const activeTools = options.tools || [];
    for (const t of activeTools) {
      let tool: FunctionTool&{execute: ToolExecute};
      if (isWebMcpTool(t)) {
        tool = {...toTool(t), execute: t.execute!};
      } else if (!('function' in t)) {
        const {execute, ...declaration} = t;
        tool = {type: 'function', function: declaration, execute};
      } else {
        tool = t;
      }

      const name = tool.function.name;
      if (!name) throw new Error('Tool must have a name.');
      if (!tool.execute) {
        throw new Error(`Tool ${name} must have an execute function.`);
      }
      if (this.tools.has(name)) {
        throw new Error(`Duplicate tool name ${name} provided.`);
      }
      this.tools.set(name, tool);
    }
  }

  private async getConversation(): Promise<Conversation> {
    if (!this.baseConversation) {
      const config: ConversationConfig =
          this.options.config ? {...this.options.config} : {};
      if (this.tools.size > 0) {
        if (!config.preface) {
          config.preface = {};
        }
        // Extract the explicit tool definitions (e.g. name, description,
        // parameters) to pass to createConversation without the execute
        // implementation.
        config.preface.tools = Array.from(this.tools.values()).map(t => {
          const {execute, ...rest} = t;
          return rest;
        });
      }
      this.baseConversation =
          await this.options.engine.createConversation(config);
    }
    return this.baseConversation;
  }

  async sendMessage(message: MessageLike): Promise<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    this.isCancelled = false;

    try {
      const conversation = await this.getConversation();
      let currentMessage = message;
      const recurringToolCallLimit = this.options.recurringToolCallLimit ?? 25;
      let iteration = 0;

      while (iteration < recurringToolCallLimit) {
        iteration++;
        const responseMsg = await conversation.sendMessage(currentMessage);

        if (responseMsg.tool_calls && responseMsg.tool_calls.length > 0) {
          const toolResponses =
              await this.executeToolCalls(responseMsg.tool_calls);
          if (this.isCancelled) {
            throw new Error('Conversation cancelled');
          }
          // Feed back as a tool message
          currentMessage = {role: 'tool', content: toolResponses};
          continue;
        }
        return responseMsg;
      }
      throw new Error('Tool calling exceeded the recurring limit');
    } finally {
      this.isBusy = false;
    }
  }

  sendMessageStreaming(message: MessageLike):
      ReadableStream<Message> {
    if (this.isBusy) {
      throw new Error(BUSY_MESSAGE);
    }
    this.isBusy = true;
    this.isCancelled = false;

    let isStreamCancelled = false;
    return new ReadableStream<Message>({
      start: (controller) => {
        let currentMessage = message;
        let iteration = 0;
        const recurringToolCallLimit =
            this.options.recurringToolCallLimit ?? 25;


        const executeIteration = async(): Promise<void> => {
          if (iteration >= recurringToolCallLimit) {
            this.isBusy = false;
            controller.error(
                new Error('Tool calling exceeded the recurring limit'));
            return;
          }
          iteration++;
          let toolCalls: ToolCall[] = [];

          try {
            const conversation = await this.getConversation();
            const stream = conversation.sendMessageStreaming(currentMessage);

            for await (const chunk of stream) {
              if (isStreamCancelled || this.isCancelled) {
                // Because we break out of the loop early, the underlying
                // iterator automatically invokes stream.cancel()
                break;
              }

              if (chunk.tool_calls && chunk.tool_calls.length > 0) {
                toolCalls = toolCalls.concat(chunk.tool_calls);
                // Omit the tool_calls property before enqueuing to the user
                const cleanValue = {...chunk};
                delete cleanValue.tool_calls;
                if (cleanValue.content) {  // only enqueue if there's content
                                           // besides tool_calls
                  controller.enqueue(cleanValue);
                }
              } else {
                controller.enqueue(chunk);
              }
            }
          } catch (e) {
            this.isBusy = false;
            controller.error(e);
            return;
          }

          if (isStreamCancelled || this.isCancelled) return;

          if (toolCalls.length > 0) {
            const toolResponses = await this.executeToolCalls(toolCalls);
            if (isStreamCancelled || this.isCancelled) return;
            currentMessage = {role: 'tool', content: toolResponses};
            await executeIteration();
            return;
          }

          this.isBusy = false;
          controller.close();
        };

        executeIteration().catch((e) => {
          if (isStreamCancelled || this.isCancelled) return;
          this.isBusy = false;
          controller.error(e);
        });
      },
      cancel: () => {
        isStreamCancelled = true;
        this.cancel();
      }
    });
  }

  /**
   * Cancels the current generation.
   * Note: This does not abort any in-progress tool calls natively. Tools that
   * affect state may still complete in the background eventually, but their
   * results will be discarded and not reported back to the model or the user.
   */
  cancel() {
    this.isCancelled = true;
    if (this.currentBatch) {
      this.currentBatch.cancel();
    }
    if (this.baseConversation) {
      this.baseConversation.cancel();
    }
  }

  async getHistory(): Promise<Message[]> {
    const conversation = await this.getConversation();
    return conversation.getHistory();
  }

  async getTokenCount(): Promise<number> {
    const conversation = await this.getConversation();
    return conversation.getTokenCount();
  }

  async getBenchmarkInfo(): Promise<BenchmarkInfo> {
    const conversation = await this.getConversation();
    return conversation.getBenchmarkInfo();
  }

  async delete(): Promise<void> {
    if (this.baseConversation) {
      await this.baseConversation.delete();
      this.baseConversation = null;
    }
  }

  private async executeToolCalls(toolCalls: ToolCall[]):
      Promise<ContentPart[]> {
    this.currentBatch = new ToolCallBatch(
        this.tools, this.options.onToolProgress, () => this.nextToolCallId++);
    try {
      return await this.currentBatch.executeAll(toolCalls);
    } finally {
      this.currentBatch = null;
    }
  }
}

class ToolCallBatch {
  private isCancelled = false;

  constructor(
      private readonly tools: Map<string, ToolWithImplementation>,
      private readonly onToolProgress: ((event: ToolProgressEvent) => void)|
      undefined,
      private readonly getNextId: () => number) {}

  cancel() {
    this.isCancelled = true;
  }

  async executeAll(toolCalls: ToolCall[]): Promise<ContentPart[]> {
    return Promise.all(toolCalls.map(async (tc) => {
      const tool = this.tools.get(tc.function.name);
      const callId = this.getNextId();
      let result: ReturnType<ToolExecute>;

      if (this.onToolProgress) {
        this.onToolProgress({
          id: callId,
          name: tc.function.name,
          arguments: tc.function.arguments,
          status: 'started'
        });
      }

      if (!tool) {
        const errorMsg = `Tool ${tc.function.name} not found.`;
        result = {error: errorMsg};
        if (!this.isCancelled && this.onToolProgress) {
          this.onToolProgress({
            id: callId,
            name: tc.function.name,
            arguments: tc.function.arguments,
            status: 'error',
            error: errorMsg
          });
        }
      } else {
        try {
          result = await tool.execute(tc.function.arguments);
          if (!this.isCancelled && this.onToolProgress) {
            this.onToolProgress({
              id: callId,
              name: tc.function.name,
              arguments: tc.function.arguments,
              status: 'completed',
              result
            });
          }
        } catch (e) {
          const errorMsg = String(e);
          result = {
            error: `Error executing tool ${tc.function.name}: ${errorMsg}`
          };
          if (!this.isCancelled && this.onToolProgress) {
            this.onToolProgress({
              id: callId,
              name: tc.function.name,
              arguments: tc.function.arguments,
              status: 'error',
              error: errorMsg
            });
          }
        }
      }

      return {type: 'tool_response', name: tc.function.name, response: result};
    }));
  }
}
