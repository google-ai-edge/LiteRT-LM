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

import {css, html} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';

import {deepActiveElement, restoreFocus, StatefulElement} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Dialog for uploading, downloading and managing the embedding model. */
@customElement('model-picker-modal')
export class ModelPickerModal extends StatefulElement {

  static override styles = [
    sharedStyles,
    css`
      :host { display: block; }

      .modal-body {
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        max-height: 70vh;
        overflow-y: auto;
      }

      .info-box {
        background: rgba(91, 227, 227, 0.08);
        border: 1px solid rgba(91, 227, 227, 0.2);
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.85rem;
        color: #a5f0f0;
        line-height: 1.4;
      }

      .model-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid var(--teal);
        background: rgba(0, 201, 158, 0.08);
      }

      .model-name {
        font-weight: 600;
        color: var(--text);
        font-size: 0.9rem;
        word-break: break-all;
      }

      .model-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 2px;
      }

      .source-options {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .upload-zone {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 22px 16px;
        border: 2px dashed var(--border-light);
        border-radius: 10px;
        background: var(--bg-input);
        color: var(--text-muted);
        font-size: 0.85rem;
        cursor: pointer;
        text-align: center;
        transition: border-color 0.15s, background 0.15s;
      }

      .upload-zone:hover,
      .upload-zone.drag-over {
        border-color: var(--teal);
        background: rgba(0, 201, 158, 0.06);
      }

      .upload-zone strong {
        color: var(--text);
        font-size: 0.95rem;
      }

      .or-divider {
        text-align: center;
        font-size: 0.75rem;
        color: var(--text-muted);
      }

      .progress-track { height: 6px; border-radius: 3px; }
      .progress-fill.indeterminate {
        width: 30% !important;
        animation: slide 1.2s ease-in-out infinite;
      }

      @keyframes slide {
        from { transform: translateX(-100%); }
        to { transform: translateX(333%); }
      }
    `,
  ];

  @state() private isDragOver = false;
  @query('#model-file-input') private fileInputEl?: HTMLInputElement;
  @query('.close-btn') private closeBtnEl?: HTMLButtonElement;

  private opener: HTMLElement|null = null;

  private readonly onWindowKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && this.hasState && this.state.isModelModalOpen) {
      e.preventDefault();
      this.closeModal();
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    this.opener = deepActiveElement();
    window.addEventListener('keydown', this.onWindowKeyDown);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('keydown', this.onWindowKeyDown);
    restoreFocus(this.opener);
    this.opener = null;
  }

  override firstUpdated() {
    this.closeBtnEl?.focus();
  }

  private closeModal() {
    this.state.isModelModalOpen = false;
    this.state.requestUpdate();
  }

  private async afterLoad() {
    if (!this.state.models.engine) return;
    this.closeModal();
    if (this.state.indexer.isAuthorized &&
        this.state.indexer.countPending(this.state.promptTemplate) > 0) {
      await this.state.startIndexing();
    }
  }

  private async loadFile(file: File) {
    if (!file.name.endsWith('.litertlm')) {
      this.state.setStatus(`"${file.name}" is not a .litertlm model file.`);
      return;
    }
    try {
      await this.state.models.loadFromFile(file);
      await this.afterLoad();
    } catch (_) {}
  }

  private async download() {
    const url = this.state.models.modelUrl;
    if (!url) return;
    try {
      await this.state.models.loadFromUrl(url);
      await this.afterLoad();
    } catch (_) {}
  }

  private handleFileChange(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) void this.loadFile(file);
  }

  private handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.isDragOver = false;
    const file = e.dataTransfer?.files?.[0];
    if (file) void this.loadFile(file);
  }

  private async forgetModel() {
    if (!confirm('Remove this model from browser storage? You will need to upload or download it again.')) {
      return;
    }
    await this.state.models.forgetModel();
  }

  private renderProgress() {
    const {models} = this.state;
    if (!models.isLoading) return '';
    const percent = models.loadingPercent;
    return html`
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 10px; color: var(--teal); font-size: 0.85rem;">
          <div class="spinner"></div>
          <span style="flex: 1;" role="status">${models.loadingProgressText}</span>
          ${models.canCancelLoad ? html`
            <button
              class="btn btn-secondary"
              style="padding: 4px 10px; font-size: 0.8rem;"
              @click=${() => models.cancelLoad()}
            >
              Cancel
            </button>
          ` : ''}
        </div>
        <div class="progress-track">
          <div
            class="progress-fill ${percent === null ? 'indeterminate' : ''}"
            style="width: ${percent ?? 0}%;"
          ></div>
        </div>
      </div>
    `;
  }

  private modelHost(): string {
    const url = this.state.models.modelUrl;
    if (!url) return '';
    try {
      return new URL(url, window.location.href).host;
    } catch {
      return '';
    }
  }

  private renderSourceOptions() {
    const {models} = this.state;
    const host = this.modelHost();
    return html`
      <div class="source-options">
        ${models.modelUrl ? html`
          <button
            class="btn btn-primary"
            ?disabled=${models.isLoading}
            @click=${this.download}
          >
            <span class="icon">download</span>
            ${host ? `Download from ${host}` : 'Download the model'}
          </button>
          <div class="or-divider">or use a local copy</div>
        ` : ''}
        <input
          id="model-file-input"
          type="file"
          accept=".litertlm"
          style="display: none;"
          @change=${this.handleFileChange}
        />
        <div
          class="upload-zone ${this.isDragOver ? 'drag-over' : ''}"
          role="button"
          tabindex="0"
          @click=${() => this.fileInputEl?.click()}
          @keydown=${(e: KeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') this.fileInputEl?.click();
          }}
          @dragover=${(e: DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            this.isDragOver = true;
          }}
          @dragleave=${() => {
            this.isDragOver = false;
          }}
          @drop=${this.handleDrop}
        >
          <span class="icon" style="font-size: 1.6rem;">inventory_2</span>
          <strong>Upload a .litertlm model file</strong>
          <span>Click to browse, or drop the file here. It is kept in browser storage for next time.</span>
        </div>
      </div>
    `;
  }

  override render() {
    if (!this.state.isModelModalOpen) return html``;

    const {models} = this.state;
    const isLoaded = !!models.engine;

    return html`
      <div class="modal-backdrop" @click=${(e: Event) => {
        if (e.target === e.currentTarget) this.closeModal();
      }}>
        <div
          class="modal-card"
          role="dialog"
          aria-modal="true"
          aria-labelledby="model-modal-title"
        >
          <div class="modal-header">
            <div class="modal-title" id="model-modal-title">EmbeddingGemma Model</div>
            <button
              type="button"
              class="close-btn"
              aria-label="Close"
              @click=${this.closeModal}
            ><span class="icon" aria-hidden="true">close</span></button>
          </div>

          <div class="modal-body">
            <div class="info-box">
              Search runs entirely on your device: the model is compiled and executed on
              your GPU with WebGPU, and your files never leave the browser.
            </div>

            ${isLoaded ? html`
              <div class="model-card">
                <div>
                  <div class="model-name">${models.loadedModelName}</div>
                  <div class="model-meta">
                    ${[models.loadedModelSize,
                       models.loadTimeSec !== null ?
                           `loaded in ${models.loadTimeSec.toFixed(1)}s` :
                           ''].filter((s) => s).join(' · ')}
                  </div>
                </div>
                <span class="badge badge-teal">Active</span>
              </div>
            ` : ''}

            ${this.renderSourceOptions()}
            ${this.renderProgress()}
          </div>

          <div class="modal-footer">
            ${isLoaded ? html`
              <button class="btn btn-danger" @click=${this.forgetModel}>
                Remove from storage
              </button>
            ` : html`<div></div>`}

            <button class="btn btn-secondary" @click=${this.closeModal}>
              ${isLoaded ? 'Done' : 'Close'}
            </button>
          </div>
        </div>
      </div>
    `;
  }
}
