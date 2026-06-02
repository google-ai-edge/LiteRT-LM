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

import './conversation_chat';
import './session_chat';
import './litert_lm_settings_editor';

import {Engine, EngineSettings, loadLiteRtLm, type SessionConfig} from '@litert-lm/core';
import {css, html, LitElement} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';
// Placeholder for internal dependency on trusted resource url

import {LitertLmSettingsEditor} from './litert_lm_settings_editor.js';

// tslint:disable:no-new-decorators

/**
 * A component for testing LiteRT LM models.
 */
@customElement('litert-lm-model-tester')
export class LitertLmModelTester extends LitElement {
  static override styles = css`
    :host {
      display: block;
      font-family: sans-serif;
      margin: 0;
      padding: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .header {
      padding: 16px 24px;
      background: #f1f3f4;
      border-bottom: 1px solid #ddd;
    }
    .header h1 {
      margin: 0;
      font-size: 24px;
    }
    .layout {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .main-area {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .sidebar {
      width: 400px;
      padding: 24px;
      border-left: 1px solid #ddd;
      background: #fafafa;
      overflow-y: auto;
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
    button {
      padding: 8px 16px;
      background: #1a73e8;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    button:disabled {
      background: #ccc;
      cursor: not-allowed;
    }
    .status {
      font-style: italic;
      color: #666;
    }
    .tabs {
      display: flex;
      gap: 16px;
    }
    .tab {
      padding: 8px 16px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      font-weight: 500;
    }
    .tab.active {
      border-bottom: 2px solid #1a73e8;
      color: #1a73e8;
    }
    [hidden] {
      display: none !important;
    }
  `;

  @state() private modelLoaded = false;
  @state() private activeTab: 'conversation'|'session' = 'conversation';
  @state() private statusMessage = 'Waiting for model...';
  @state() private loadedSettings?: Partial<EngineSettings>;
  @state() private loadedSessionConfig?: SessionConfig;
  @state() private selectedFile?: File;

  @query('#settings-editor') private settingsEditor!: LitertLmSettingsEditor;

  @state() private engine?: Engine;

  override async connectedCallback() {
    super.connectedCallback();
    try {
      this.statusMessage = 'Loading WASM module...';
      await loadLiteRtLm('/wasm');
      this.statusMessage = 'WASM loaded. Please select a model file.';
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      this.statusMessage = `Error loading WASM: ${message}`;
    }
  }

  private async onFileChanged(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    this.selectedFile = file;
    await this.loadModel();
  }

  private async onReloadModel() {
    await this.loadModel();
  }

  @state() private activeConversationConfig?: SessionConfig;
  @state() private activeSessionChatConfig?: SessionConfig;
  @state() private liveSessionConfig?: SessionConfig;

  private async loadModel() {
    if (!this.selectedFile) return;

    this.statusMessage = 'Loading model...';
    try {
      if (this.engine) {
        await this.engine.delete();
        this.engine = undefined;
      }

      this.loadedSettings = this.settingsEditor.getSettings();
      const settings: EngineSettings = {
        ...this.loadedSettings,
        model: this.selectedFile.stream(),
      };

      this.engine = await Engine.create(settings);
      const initConfig = this.settingsEditor.getSessionConfig();
      this.loadedSessionConfig = initConfig;
      this.liveSessionConfig = initConfig;
      this.activeConversationConfig = initConfig;
      this.activeSessionChatConfig = initConfig;

      this.modelLoaded = true;
      this.statusMessage = 'Model loaded and ready.';
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e);
      this.statusMessage = `Error loading model: ${message}`;
      console.error(e);
    }
  }

  private onSessionConfigChanged(e: CustomEvent<SessionConfig>) {
    this.liveSessionConfig = e.detail;
  }

  private onSessionConfigUsed(e: CustomEvent<SessionConfig>) {
    if (this.activeTab === 'conversation') {
      this.activeConversationConfig = e.detail;
    } else {
      this.activeSessionChatConfig = e.detail;
    }
  }

  override render() {
    const comparedConfig = this.activeTab === 'conversation' ?
        this.activeConversationConfig :
        this.activeSessionChatConfig;

    return html`
      <div class="header">
        <h1>LiteRT LM Model Tester</h1>
      </div>

      <div class="layout" @session-config-changed=${
        this.onSessionConfigChanged} @session-config-used=${
        this.onSessionConfigUsed}>
        <div class="main-area">
          <div class="panel">
            <div class="row">
              <label>Model File (.task or .litertlm):</label>
              <input type="file" accept=".litertlm,.task" @change=${
        this.onFileChanged}>
              <button @click=${this.onReloadModel} ?disabled=${
    !this.selectedFile} style="background: #5f6368; font-size: 13px; padding: 4px 8px;">Reload</button>
            </div>
            <div class="status">${this.statusMessage}</div>
            <p style="font-size: 13px; color: #555;">(Note: You must reload the model to apply setting changes)</p>
          </div>

          <hr style="width: 100%; border: 0; border-top: 1px solid #ddd; margin: 0;" />

          <div class="tabs">
            <div
              class="tab ${
        this.activeTab ===
        'conversation' ? 'active' : ''}"
              @click=${() => {
      this.activeTab = 'conversation';
    }}
            >
              Conversation
            </div>
            <div
              class="tab ${
        this.activeTab ===
        'session' ? 'active' : ''}"
              @click=${() => {
      this.activeTab = 'session';
    }}
            >
              Session
            </div>
          </div>
          <div style="flex: 1; display: flex; flex-direction: column; overflow: hidden;">
            <conversation-chat
              ?hidden=${this.activeTab !== 'conversation'}
              .engine=${this.engine}
              .sessionConfig=${this.liveSessionConfig}
            ></conversation-chat>
            <session-chat
              ?hidden=${this.activeTab !== 'session'}
              .engine=${this.engine}
              .sessionConfig=${this.liveSessionConfig}
            ></session-chat>
          </div>
        </div>

        <div class="sidebar">
          <h2>Model Settings</h2>
          <litert-lm-settings-editor id="settings-editor" .loadedSettings=${
        this.loadedSettings} .loadedSessionConfig=${
        comparedConfig}></litert-lm-settings-editor>
        </div>
      </div>
    `;
  }
}
