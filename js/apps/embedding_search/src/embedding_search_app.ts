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

import './components/app_header';
import './components/model_picker_modal';
import './components/index_manager_panel';
import './components/search_bar';
import './components/results_grid';

import {css, html, LitElement} from 'lit';
import {customElement, state} from 'lit/decorators.js';
import {EmbeddingSearchStateController} from './state_controller.js';
import {sharedStyles} from './styles/shared_styles.js';
import {classifyMediaFile, extractDroppedFiles, toSupportedMediaBlob} from './utils/media_types.js';

/* tslint:disable:no-new-decorators */

/** Root element of the embedding search demo. */
@customElement('embedding-search-app')
export class EmbeddingSearchApp extends LitElement {
  private state = new EmbeddingSearchStateController(this);

  @state() private isDragging = false;
  private dragCounter = 0;

  static override styles = [
    sharedStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100vh;
        overflow: hidden;
        position: relative;
      }

      .main-content {
        flex: 1;
        min-height: 0;
        overflow: hidden;
        padding: 12px 24px 0 24px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-width: 1400px;
        width: 100%;
        margin: 0 auto;
      }

      .bottom-bar-area {
        flex-shrink: 0;
        padding: 10px 24px 14px 24px;
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
      }

      .drag-drop-overlay {
        position: fixed;
        inset: 0;
        background: rgba(11, 15, 25, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2000;
        pointer-events: none;
      }

      .drag-drop-modal {
        background: var(--bg-card);
        border: 2px dashed var(--teal);
        border-radius: 16px;
        padding: 40px 60px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
      }

      .drag-drop-icon { font-size: 3.5rem; }
      .drag-drop-title { font-size: 1.25rem; font-weight: 700; color: #ffffff; }
      .drag-drop-subtitle { font-size: 0.9rem; color: var(--text-muted); }
    `,
  ];

  override connectedCallback() {
    super.connectedCallback();
    window.addEventListener('dragover', this.preventWindowDrop);
    window.addEventListener('drop', this.preventWindowDrop);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('dragover', this.preventWindowDrop);
    window.removeEventListener('drop', this.preventWindowDrop);
  }

  private readonly preventWindowDrop = (e: DragEvent) => {
    e.preventDefault();
  };

  private handleDragEnter(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.dragCounter++;
    this.isDragging = true;
  }

  private handleDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  }

  private handleDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (--this.dragCounter <= 0) {
      this.dragCounter = 0;
      this.isDragging = false;
    }
  }

  private async handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.dragCounter = 0;
    this.isDragging = false;
    if (!e.dataTransfer) return;

    let files: File[];
    try {
      files = await extractDroppedFiles(e.dataTransfer);
    } catch {
      this.state.setStatus(
          'Could not fetch that image directly from the website (CORS) — try right-clicking "Copy image" and pasting into the search box.');
      return;
    }
    for (const file of files) {
      const kind = classifyMediaFile(file);
      if (kind) {
        const blob = await toSupportedMediaBlob(file, kind);
        this.state.search.appendChip(kind, file.name, blob);
        this.state.setStatus(`Added ${kind} "${file.name}" to query.`);
      } else {
        this.state.setStatus(`Unsupported file type for search query: ${file.name}`);
      }
    }
  }

  override render() {
    return html`
      <div
        style="display: flex; flex-direction: column; width: 100%; height: 100%;"
        @dragenter=${this.handleDragEnter}
        @dragover=${this.handleDragOver}
        @dragleave=${this.handleDragLeave}
        @drop=${this.handleDrop}
      >
        <embedding-search-header .state=${this.state}></embedding-search-header>

        <main class="main-content">
          <index-manager-panel .state=${this.state}></index-manager-panel>
          <embedding-search-results .state=${this.state}></embedding-search-results>
        </main>

        <div class="bottom-bar-area">
          <embedding-search-bar .state=${this.state}></embedding-search-bar>
        </div>

        ${this.isDragging ? html`
          <div class="drag-drop-overlay">
            <div class="drag-drop-modal">
              <span class="drag-drop-icon icon">upload_file</span>
              <div class="drag-drop-title">Drop images or audio files to add to search</div>
              <div class="drag-drop-subtitle">Files will be added as query chips</div>
            </div>
          </div>
        ` : ''}

        ${this.state.isModelModalOpen ? html`<model-picker-modal .state=${this.state}></model-picker-modal>` : ''}
      </div>
    `;
  }
}
