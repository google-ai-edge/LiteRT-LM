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

import './custom_dropdown';

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {MODELS, PartialSettingsSchema, Settings} from '../stores/settings_store.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Component representing the sidebar drawer. */
@customElement('litert-sidebar')
export class LitertSidebar extends LitElement {
  @property({ type: Object })
  state!: LlmChatStateController;

  static override styles = [
    sharedStyles, css`
      .section-title {
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--teal);
        margin: 0 0 12px 0;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
      }

      .control-group {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .status-card {
        background-color: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        font-family: ui-monospace, monospace;
        font-size: 0.75rem;
      }

      .status-header {
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .status-indicator {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: #64748b; 
      }

      .status-indicator.loading { background-color: #eab308; animation: pulse 1.5s infinite; }
      .status-indicator.ready { background-color: var(--teal); }
      .status-indicator.error { background-color: #ef4444; }

      @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
      }

      .status-text {
        color: var(--text-muted);
        word-break: break-all;
        max-height: 100px;
        overflow-y: auto;
      }

      .metrics-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 0.75rem;
        font-family: ui-monospace, monospace;
        color: var(--text-muted);
        margin-top: 12px;
      }

      .metric-row {
        display: flex;
        justify-content: space-between;
      }

      .metric-val {
        color: var(--text);
        font-weight: 700;
      }

      .conversations-list {
        display: flex;
        flex-direction: column;
        gap: 4px;
        flex: 1;
        overflow-y: auto;
        overscroll-behavior: contain;
        margin-top: 4px;
        padding: 0;
        background: none;
        border: none;
      }

      .sidebar-status-footer {
        margin-top: auto;
        border-top: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 8px;
        position: relative;
        z-index: 10;
        box-shadow: 0 -8px 16px rgba(0, 0, 0, 0.45);
        background-color: var(--bg-card);
        margin-left: -16px;
        margin-right: -16px;
        margin-bottom: -16px;
        padding-top: 12px;
        padding-left: 16px;
        padding-right: 16px;
        padding-bottom: 16px;
      }

      .conv-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.78rem;
        color: var(--text-muted);
        transition: background-color 0.15s, color 0.15s, border-color 0.15s;
        user-select: none;
        border: 1px solid transparent;
      }
      .conv-item:not(.active):hover {
        background-color: var(--bg-input);
        color: var(--text);
      }
      .conv-item.active {
        background-color: var(--teal);
        color: var(--bg-dark) !important;
        font-weight: bold;
      }
      .conv-item.active:hover {
        background-color: var(--teal) !important;
        color: var(--bg-dark) !important;
        cursor: default;
      }
      .new-chat-item {
        color: var(--teal) !important;
        border: 1px dashed var(--teal);
        margin-bottom: 6px;
        font-weight: bold;
      }
      .new-chat-item:not(.active):hover {
        background-color: rgba(45, 212, 191, 0.08) !important;
        color: var(--teal) !important;
      }
      .conv-title {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 6px;
      }
      .btn-delete-conv {
        background: none;
        border: none;
        color: inherit;
        cursor: pointer;
        font-size: 0.7rem;
        padding: 2px 4px;
        border-radius: 4px;
        opacity: 0.5;
        transition: opacity 0.15s, color 0.15s;
      }
      .btn-delete-conv:hover {
        opacity: 1;
        color: #ef4444 !important;
      }

      .btn-dismiss-sidebar {
        display: none; 
      }

      .clear-all-btn {
        background: none;
        border: none;
        color: #ef4444;
        font-size: 0.65rem;
        font-weight: bold;
        cursor: pointer;
        padding: 2px 6px;
        border-radius: 4px;
        transition: background-color 0.15s;
      }

      .clear-all-btn:hover {
        background-color: rgba(239, 68, 68, 0.1);
      }

      @media (max-width: 768px) {
        .btn-dismiss-sidebar {
          display: block !important; 
          background: none;
          border: 1px solid rgba(0, 201, 158, 0.3);
          color: var(--teal);
          border-radius: 4px;
          font-size: 0.68rem;
          font-weight: bold;
          padding: 4px 10px;
          cursor: pointer;
          transition: background-color 0.15s, border-color 0.15s, color 0.15s;
        }
        .btn-dismiss-sidebar:hover {
          background-color: rgba(0, 201, 158, 0.08);
          border-color: var(--teal);
          color: #ffffff;
        }
      }
    `
  ];

  override connectedCallback() {
    super.connectedCallback();

    // Receive state updates from the centralized state controller.
    this.state.addHost(this);
  }

  override disconnectedCallback() {
    this.state.removeHost(this);
    super.disconnectedCallback();
  }

  private getTotalCacheSize(): string {
    let totalBytes = 0;
    for (const [_, size] of this.state.modelLoader.cachedModels) {
      totalBytes += size;
    }
    const sizeInGB = totalBytes / 1e9;
    return `${sizeInGB.toFixed(2)} GB`;
  }

  private handleModelChange(e: CustomEvent<string>) {
    const path = e.detail;
    if (this.state.settings.selectedModelPath !== path) {
      this.state.settings.selectedModelPath = path;
      this.state.settings.saveSettings();

      // Auto-trigger model loading compilation in background on settings change
      this.state.modelLoader.loadModelWeights(async () => {
        await this.state.chatSession.createConversationSession();
      });
    }
  }

  private handleRemoveCached(e: Event, modelPath: string) {
    e.stopPropagation();  // Stop click propagation to prevent selecting model
    this.state.modelLoader.deleteModelFromCache(modelPath);
  }

  private handleSliderInput(e: Event, prop: keyof Settings) {
    const target = e.target as HTMLInputElement | HTMLSelectElement;
    let val: unknown;

    if (target instanceof HTMLInputElement) {
      if (target.type === 'checkbox') {
        val = target.checked;
      } else {
        val = target.type === 'number' ? Number(target.value) : target.value;
      }
    } else {
      val = target.value;
    }

    const parseResult = PartialSettingsSchema.safeParse({[prop]: val});
    if (parseResult.success) {
      Object.assign(this.state.settings, parseResult.data);
      this.state.settings.saveSettings();
    } else {
      console.warn('[LiteRT-LM] Invalid settings input:', parseResult.error);
    }
  }

  private dismissSidebar() {
    this.dispatchEvent(new CustomEvent('close', {
      bubbles: true,
      composed: true,
    }));
  }

  override render() {
    const activeModelFilename =
        this.state.settings.selectedModelPath.split('/').pop() || '';

    return html`
      <!-- Model Selection Group -->
      <div class="control-group">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h2 class="section-title" style="margin: 0; border: none; padding: 0;">Model Selection</h2>
          <button id="btn-dismiss-sidebar" class="btn-dismiss-sidebar" aria-label="Dismiss Configurations" @click=${
        this.dismissSidebar}>Done</button>
        </div>

        <custom-dropdown
          .value=${this.state.settings.selectedModelPath}
          @change=${this.handleModelChange}
        >
          ${MODELS.map(model => {
      const cachedSizeBytes =
          this.state.modelLoader.cachedModels.get(model.filename);
      const downloadProgress =
          this.state.modelLoader.downloadProgresses.get(model.filename);

      return html`
              <div class="dropdown-item" data-value="${model.path}">
                <span class="model-name">${model.name}</span>

                ${
          cachedSizeBytes ? html`
                  <span class="cached-info" style="display: flex; align-items: center; gap: 6px;">
                    <span class="size-badge" style="font-size: 0.62rem; background-color: var(--border); padding: 2px 6px; border-radius: 4px; color: var(--text-muted);">${
                                (cachedSizeBytes / 1e9).toFixed(1)} GB</span>
                    <button class="delete-cache-btn" title="Remove from cache" style="background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.78rem; padding: 2px 4px;" @click=${
                                (e: Event) => this.handleRemoveCached(
                                    e, model.path)}>✕</button>
                  </span>
                ` :
              downloadProgress !== undefined ? html`
                  <span class="downloading-info" style="font-size: 0.62rem; color: var(--teal); font-weight: bold;">
                    ${downloadProgress}%
                  </span>
                ` :
                                               html`
                  <span class="download-badge" style="font-size: 0.6rem; color: var(--text-muted); opacity: 0.65;">Download (${
                                                   model.size})</span>
                `}
              </div>
            `;
    })}
        </custom-dropdown>

        <!-- Compact inline Caching size display & Clear All link -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 0.65rem; color: var(--text-muted);">
          <span>Total Cached: <b style="color: var(--teal); font-family: ui-monospace, monospace;">${
        this.getTotalCacheSize()}</b></span>
          <button class="btn-clear-cache" style="background: none; border: none; color: #ef4444; font-size: 0.65rem; cursor: pointer; text-decoration: underline;" @click=${
        () => this.state.modelLoader.clearAllCache()}>Clear All</button>
        </div>
      </div>

      <!-- Static Inference Config Group -->
      <div class="control-group" style="border-top: 1px solid var(--border); padding-top: 8px;">
        <!-- Side-by-side Context & Max Gen Row -->
        <div style="display: flex; gap: 10px; margin-bottom: 6px;">
          <div style="flex: 1; display: flex; flex-direction: column; gap: 4px;">
            <label for="context-length" style="font-size: 0.65rem;">Context Length</label>
            <input type="number" id="context-length" .value=${
        String(
            this.state.settings
                .contextLength)} min="256" step="256" style="padding: 8px 4px; font-size: 0.72rem; width: 100%; box-sizing: border-box; text-align: center;" @input=${
        (e: Event) => this.handleSliderInput(e, 'contextLength')}>
          </div>
          <div style="flex: 1; display: flex; flex-direction: column; gap: 4px;">
            <label for="max-output-tokens" style="font-size: 0.65rem;">Max Gen</label>
            <input type="number" id="max-output-tokens" .value=${
        String(
            this.state.settings
                .maxOutputTokens)} min="128" step="64" style="padding: 8px 4px; font-size: 0.72rem; width: 100%; box-sizing: border-box; text-align: center;" @input=${
        (e: Event) => this.handleSliderInput(e, 'maxOutputTokens')}>
          </div>
        </div>

        <!-- Side-by-side Samplers Row -->
        <div style="display: flex; gap: 6px; align-items: flex-end;">
          <div style="flex: 1.2; display: flex; flex-direction: column; gap: 4px;">
            <label for="sampler-type" style="font-size: 0.65rem;">Sampler</label>
            <select id="sampler-type" .value=${
        this.state.settings
            .samplerType} style="padding: 8px 2px; font-size: 0.7rem; width: 100%; box-sizing: border-box;" @change=${
        (e: Event) => this.handleSliderInput(e, 'samplerType')}>
              <option value="greedy">Greedy</option>
              <option value="top_k">Top-K</option>
              <option value="top_p">Top-P</option>
            </select>
          </div>
          <div style="flex: 0.9; display: flex; flex-direction: column; gap: 4px;">
            <label for="temperature" style="font-size: 0.65rem;">Temp</label>
            <input type="number" id="temperature" .value=${
        String(
            this.state.settings
                .temperature)} min="0.0" max="2.0" step="0.1" style="padding: 8px 2px; font-size: 0.7rem; width: 100%; box-sizing: border-box; text-align: center;" @input=${
        (e: Event) => this.handleSliderInput(e, 'temperature')}>
          </div>
          <div style="flex: 0.9; display: flex; flex-direction: column; gap: 4px;">
            <label for="top-p" style="font-size: 0.65rem;">Top-P</label>
            <input type="number" id="top-p" .value=${
        String(
            this.state.settings
                .topP)} min="0.0" max="1.0" step="0.05" style="padding: 8px 2px; font-size: 0.7rem; width: 100%; box-sizing: border-box; text-align: center;" @input=${
        (e: Event) => this.handleSliderInput(e, 'topP')}>
          </div>
          <div style="flex: 0.9; display: flex; flex-direction: column; gap: 4px;">
            <label for="top-k" style="font-size: 0.65rem;">Top-K</label>
            <input type="number" id="top-k" .value=${
        String(
            this.state.settings
                .topK)} min="1" max="128" step="1" style="padding: 8px 2px; font-size: 0.7rem; width: 100%; box-sizing: border-box; text-align: center;" @input=${
        (e: Event) => this.handleSliderInput(e, 'topK')}>
          </div>
        </div>

        <div style="margin-top: 12px; display: flex; justify-content: space-between; align-items: center; width: 100%;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="enable-thinking" ?checked=${
        this.state.settings
            .enableThinking} style="width: 14px; height: 14px; accent-color: var(--teal); cursor: pointer;" @change=${
        (e: Event) => this.handleSliderInput(e, 'enableThinking')}>
            <label for="enable-thinking" style="margin: 0; cursor: pointer; text-transform: none; font-size: 0.78rem; color: var(--text);">Thinking</label>
          </div>
          <button id="btn-reset-settings" class="btn btn-secondary" style="font-size: 0.7rem; padding: 4px 8px; height: 22px; line-height: 1;" @click=${
        () => this.state.settings.resetDefaults()}>Reset</button>
        </div>
      </div>

      <!-- Saved Conversations List -->
      <div id="conversations-list" class="conversations-list" style="border-top: 1px solid var(--border); padding-top: 10px;">
        <!-- ➕ New Chat dashed list header -->
        <div class="conv-item new-chat-item ${
        this.state.chatSession.activeSavedConvId === null ?
            'active' :
            ''}" @click=${() => this.state.chatSession.startNewConversation()}>
          <!-- SVG large plus icon for new chat -->
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span class="conv-title">New Chat</span>
        </div>

        <!-- Saved index loop -->
        ${
        this.state.chatSession.conversationsList.map(
            conv => html`
          <div class="conv-item ${
                this.state.chatSession.activeSavedConvId === conv.id ?
                    'active' :
                    ''}" @click=${
                () => this.state.chatSession.selectConversation(conv.id)}>
            <span class="conv-title" title="${conv.title}">${conv.title}</span>
            <button class="btn-delete-conv" aria-label="Delete Conversation" style="background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.8rem; padding: 4px;" @click=${
                (e: Event) => {
                  e.stopPropagation();
                  this.state.chatSession.deleteConversation(conv.id);
                }}>✕</button>
          </div>
        `)}
      </div>

      <!-- Anchored Sidebar Status Footer -->
      <div class="sidebar-status-footer">
        <div class="status-card" style="background-color: var(--slate-bubble-bot); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 6px;">
          <div class="status-header" style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: bold; margin: 0;">
            <span id="status-dot" class="status-indicator ${
        this.state.modelLoader.isModelLoading ? 'loading' :
            this.state.modelLoader.engine     ? 'ready' :
                                                ''}"></span>
            <span>Status:</span>
          </div>
          <div id="status-text" class="status-text" style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4; word-break: break-word;">${
        this.state.statusText}</div>
          ${
        false ?
            html`<div id="status-cache-text" style="margin-top: 2px; font-size: 0.65rem; color: #ef4444; display: block;">${
                this.state.statusText}</div>` :
            ''}

          <!-- Loading Progress Bar Container -->
          ${
        this.state.modelLoader.isModelLoading &&
                this.state.modelLoader.downloadProgresses.has(
                    activeModelFilename) ?
            html`
            <div id="progress-container" style="display: block; margin-top: 6px; border-top: 1px dashed var(--border); padding-top: 8px;">
              <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-muted); margin-bottom: 4px;">
                <span id="progress-label">${
                this.state.statusText.includes('Downloading') ?
                    'Downloading...' :
                    'Compiling...'}</span>
                <span id="progress-percent">${
                this.state.modelLoader.downloadProgresses.get(
                    activeModelFilename)}%</span>
              </div>
              <div style="width: 100%; height: 6px; background-color: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; overflow: hidden;">
                <div id="progress-bar" style="width: ${
                this.state.modelLoader.downloadProgresses.get(
                    activeModelFilename)}%; height: 100%; background-color: var(--teal); transition: width 0.1s ease;"></div>
              </div>
              <div id="progress-speed" style="font-size: 0.62rem; color: var(--text-muted); font-family: ui-monospace, monospace; margin-top: 4px; text-align: right;">
                ${
                this.state.modelLoader.downloadSpeeds.get(activeModelFilename)}
              </div>
              
              <!-- Warning-red Cancel Download button -->
              ${
                this.state.modelLoader.downloadAbortController ?
                    html`
                <button id="btn-cancel-download" class="btn btn-secondary" style="margin-top: 8px; width: 100%; font-size: 0.7rem; padding: 4px 8px; height: 22px; line-height: 1; color: #ef4444; border-color: rgba(239, 68, 68, 0.35);" @click=${
                        () => this.state.modelLoader.cancelDownload()}>
                  Cancel Download
                </button>
              ` :
                    ''}
            </div>
          ` :
            ''}

          <!-- Metrics container showing only Load Time -->
          <div class="metrics-container" style="margin-top: 6px; display: flex; flex-direction: column; gap: 6px; border-top: 1px solid var(--border); padding-top: 8px;">
            ${
        this.state.modelLoader.metricLoadTime !== '-' ?
            html`
            <div class="metric-row" style="display: flex; justify-content: space-between; font-size: 0.75rem; font-family: inherit;">
              <span style="color: var(--text-muted);">Load Time:</span>
              <span style="display: block; font-size: 0.6rem; color: var(--teal); margin-top: 4px; font-weight: 500;">Load Time: ${
                this.state.modelLoader.metricLoadTime}</span>
            </div>
            ` :
            ''}
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-sidebar': LitertSidebar;
  }
}
