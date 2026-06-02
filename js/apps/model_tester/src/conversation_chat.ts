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

import {Conversation, Engine, type Message, type SessionConfig} from '@litert-lm/core';
import {css, html, LitElement} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';

// tslint:disable:no-new-decorators

/**
 * Runs inference in a LiteRT LM conversation and displays the output.
 */
@customElement('conversation-chat')
export class ConversationChat extends LitElement {
  static override styles = css`
    :host {
      display: flex;
      flex-direction: column;
      gap: 16px;
      height: 100%;
      width: 100%;
    }
    .panel {
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 16px;
      background: #fff;
    }
    .row {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      align-items: center;
    }
    textarea {
      width: 100%;
      min-height: 100px;
      font-family: monospace;
      padding: 8px;
      box-sizing: border-box;
      margin-top: 8px;
      border-radius: 4px;
      border: 1px solid #aaa;
    }
    button {
      padding: 8px 16px;
      background: #1a73e8;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    button.stop {
      background: #d93025;
    }
    button:disabled {
      background: #ccc;
      cursor: not-allowed;
    }
    .status {
      font-style: italic;
      color: #666;
    }
  `;

  @property({attribute: false}) engine?: Engine;
  private _sessionConfig?: SessionConfig;
  @property({attribute: false})
  get sessionConfig(): SessionConfig|undefined {
    return this._sessionConfig;
  }
  set sessionConfig(val: SessionConfig|undefined) {
    this._sessionConfig = val;
    if (this.usedSessionConfig === undefined) {
      this.usedSessionConfig = val;
      this.dispatchEvent(new CustomEvent('session-config-used', {
        detail: val,
        bubbles: true,
        composed: true,
      }));
    }
  }

  @property({attribute: false}) usedSessionConfig?: SessionConfig;

  @state() private isGenerating = false;
  @state() private statusMessage = '';
  @state() private outputText = '';

  @query('#prompt') private promptInput!: HTMLTextAreaElement;

  private currentConversation?: Conversation;

  private async onRun() {
    if (!this.engine) return;

    const prompt = this.promptInput.value.trim();
    if (!prompt) return;

    this.isGenerating = true;
    this.statusMessage = 'Generating...';
    this.outputText = '';

    try {
      this.usedSessionConfig = this.sessionConfig;
      this.dispatchEvent(new CustomEvent('session-config-used', {
        detail: this.sessionConfig,
        bubbles: true,
        composed: true,
      }));
      this.currentConversation = await this.engine.createConversation({
        sessionConfig: this.sessionConfig,
      });

      const message: Message = {role: 'user', content: prompt};
      const stream = this.currentConversation.sendMessageStreaming(message);

      const reader = stream.getReader();
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        if (value && value.content) {
          if (typeof value.content === 'string') {
            this.outputText += value.content;
          } else {
            for (const item of value.content) {
              if (item.type === 'text' && item.text) {
                this.outputText += item.text;
              }
            }
          }
        }
      }
      this.statusMessage = 'Done.';
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      if (message.toLowerCase().includes('cancel') ||
          message.toLowerCase().includes('abort')) {
        this.statusMessage = 'Cancelled by user.';
      } else {
        this.statusMessage = `Error during inference: ${message}`;
        console.error(e);
      }
    } finally {
      if (this.currentConversation) {
        await this.currentConversation.delete();
        this.currentConversation = undefined;
      }
      this.isGenerating = false;
    }
  }

  private onStop() {
    this.currentConversation?.cancel();
  }

  override render() {
    return html`
      <div class="panel">
        <label for="prompt">Prompt:</label>
        <textarea id="prompt" ?disabled=${
    !this.engine ||
        this.isGenerating} placeholder="Enter your prompt here..."></textarea>
        <div class="row">
          ${
        this.isGenerating ?
        html`<button class="stop" @click=${this.onStop}>Stop</button>` :
        html`<button @click=${this.onRun} ?disabled=${
    !this.engine}>Run</button>`}
        </div>
        <div class="status" style="margin-top: 8px; font-size: 13px;">${
        this.statusMessage}</div>
      </div>

      <div class="panel" style="flex: 1; display: flex; flex-direction: column;">
        <label>Output:</label>
        <textarea readonly .value=${
        this.outputText} style="flex: 1;"></textarea>
      </div>
    `;
  }
}
