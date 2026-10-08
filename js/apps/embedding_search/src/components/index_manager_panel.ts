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
import {customElement, state} from 'lit/decorators.js';

import {StatefulElement} from '../state_controller.js';
import {SAMPLE_ALBUM_DIR_ID} from '../stores/sample_album_store.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Panel for mounting directories and running and monitoring indexing. */
@customElement('index-manager-panel')
export class IndexManagerPanel extends StatefulElement {

  static override styles = [
    sharedStyles,
    css`
      :host { display: block; width: 100%; }

      .toolbar {
        display: flex;
        flex-direction: column;
        border-bottom: 1px solid var(--border);
        padding-bottom: 10px;
      }

      .toolbar-row {
        display: flex;
        align-items: center;
        gap: 14px;
        min-height: 40px;
      }

      .folder-picker { position: relative; flex-shrink: 0; }

      .folder-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        max-width: 280px;
        padding: 7px 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--bg-input);
        color: var(--text);
        font: inherit;
        font-size: 0.875rem;
        cursor: pointer;
        transition: border-color 0.15s, background 0.15s;
      }

      .folder-btn:hover, .folder-btn.open {
        border-color: var(--teal);
        background: var(--bg-card);
      }

      .folder-btn.empty { border-style: dashed; color: var(--text-muted); }

      .folder-name {
        font-weight: 600;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .folder-btn.empty .folder-name { color: var(--text-muted); font-weight: 500; }

      .caret {
        font-size: 1.1rem;
        color: var(--text-muted);
        flex-shrink: 0;
        transition: transform 0.15s;
      }

      .folder-btn.open .caret { transform: rotate(180deg); }

      .folder-menu {
        position: absolute;
        top: calc(100% + 6px);
        left: 0;
        min-width: 230px;
        max-width: 340px;
        max-height: 320px;
        overflow-y: auto;
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 4px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .folder-menu-row {
        display: flex;
        align-items: center;
        border-radius: 6px;
        transition: background 0.12s;
      }

      .folder-menu-row:hover { background: rgba(255, 255, 255, 0.06); }
      .folder-menu-row.active { background: rgba(0, 201, 158, 0.12); }

      .folder-menu-item {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        min-width: 0;
        padding: 8px 10px;
        border: none;
        border-radius: 6px;
        background: transparent;
        color: var(--text);
        font: inherit;
        font-size: 0.85rem;
        text-align: left;
        cursor: pointer;
      }

      .folder-menu-row:hover .folder-menu-item:not(:disabled) { color: #ffffff; }
      .folder-menu-row.active .folder-menu-item { color: #ffffff; font-weight: 600; }
      .folder-menu-row.active .item-icon { color: var(--teal); }

      .folder-menu-item.add-new { width: 100%; color: var(--cyan); font-weight: 500; }
      .folder-menu-item.add-new:hover:not(:disabled) {
        background: rgba(91, 227, 227, 0.1);
      }

      .folder-menu-item:disabled, .folder-delete-btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }

      .item-icon { font-size: 1.1rem; color: var(--text-muted); flex-shrink: 0; }
      .folder-menu-item.add-new .item-icon { color: var(--cyan); }

      .item-label {
        flex: 1;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .folder-delete-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        margin-right: 4px;
        padding: 0;
        border: none;
        border-radius: 5px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        flex-shrink: 0;
      }

      .folder-delete-btn .icon { font-size: 1rem; }
      .folder-delete-btn:hover:not(:disabled) {
        background: rgba(239, 68, 68, 0.18);
        color: var(--red);
      }

      .folder-menu-divider {
        height: 1px;
        background: var(--border);
        margin: 3px 4px;
      }

      .status {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .status-primary, .status-secondary {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .status-primary { font-size: 0.875rem; color: #ffffff; }
      .status-primary .dim { color: var(--text-muted); font-weight: 400; }
      .status-primary.complete { color: var(--teal); }

      .status-secondary { font-size: 0.75rem; color: var(--text-muted); }
      .status-secondary.processing {
        color: var(--teal);
        font-family: ui-monospace, monospace;
      }

      .actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
      .actions .btn { padding: 7px 14px; font-size: 0.825rem; }

      .progress-track { height: 2px; margin-top: 10px; }
      .progress-fill { background: var(--teal); opacity: 0.5; }
      .progress-fill.active {
        opacity: 1;
        background: linear-gradient(90deg, var(--teal), var(--cyan));
      }

      .timing { color: var(--cyan); font-variant-numeric: tabular-nums; }
      .warn { color: var(--amber); }
    `,
  ];

  @state() private isMenuOpen = false;
  private tickTimer: number|null = null;

  private readonly onDocumentClick = (e: MouseEvent) => {
    if (!this.isMenuOpen) return;
    const picker = this.renderRoot.querySelector('.folder-picker');
    if (picker && !e.composedPath().includes(picker)) {
      this.isMenuOpen = false;
    }
  };

  private readonly onDocumentKeyDown = (e: KeyboardEvent) => {
    if (this.isMenuOpen && e.key === 'Escape') {
      this.isMenuOpen = false;
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    window.addEventListener('click', this.onDocumentClick);
    window.addEventListener('keydown', this.onDocumentKeyDown);
  }

  override updated(changed: Map<string, unknown>) {
    super.updated(changed);
    const indexing = this.state.indexer.progress.isIndexing;
    if (indexing && this.tickTimer === null) {
      this.tickTimer = window.setInterval(() => this.requestUpdate(), 1000);
    } else if (!indexing && this.tickTimer !== null) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('click', this.onDocumentClick);
    window.removeEventListener('keydown', this.onDocumentKeyDown);
    if (this.tickTimer !== null) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
  }

  private async reauthorize() {
    if ((await this.state.reauthorize()) && this.state.models.engine &&
        this.state.indexer.countPending(this.state.promptTemplate) > 0) {
      await this.state.startIndexing();
    }
  }

  private async clearIndex() {
    if (!confirm('Are you sure you want to clear the indexed embeddings? You will need to re-index files.')) {
      return;
    }
    await this.state.indexer.clearIndex();
  }

  private renderStatusPrimary(
      pendingCount: number, totalFiles: number, breakdown: string) {
    const {indexer, models} = this.state;
    const {progress} = indexer;

    if (progress.isIndexing) {
      const elapsedMs = performance.now() - progress.startedAt;
      const perFile = progress.embeddedCount > 0 ?
          progress.embedMs / progress.embeddedCount : 0;
      return html`
        <div class="status-primary">
          Indexing <b>${progress.current}</b> of <b>${progress.total}</b>
          <span class="dim">· ${progress.percentage}%</span>
          <span class="timing">
            · ${formatDuration(elapsedMs)} elapsed
            ${perFile > 0 ? html`· ${Math.round(perFile)} ms/file` : ''}
          </span>
          ${progress.failedCount > 0 ? html`<span class="warn">· ${progress.failedCount} failed</span>` : ''}
        </div>
      `;
    }

    if (!indexer.targetDirName) {
      return html`
        <div class="status-primary">
          <span class="dim">Load the sample album, or select a folder of text, image, and audio files.</span>
        </div>
      `;
    }

    if (this.state.needsReauthorization) {
      return html`
        <div class="status-primary">
          <span class="icon warn">lock</span>
          Access to <b>${indexer.targetDirName}</b> must be re-granted
          <span class="dim">· ${indexer.indexedDocuments.size} file(s) indexed earlier</span>
        </div>
      `;
    }

    if (totalFiles === 0) {
      return html`
        <div class="status-primary">
          <span class="dim">No compatible files found in this folder.</span>
        </div>
      `;
    }

    const run = indexer.lastRun;
    const failedNote = run && run.failedCount > 0 ?
        html`<span class="warn">· ${run.failedCount} failed last run</span>` : '';

    if (pendingCount === 0) {
      return html`
        <div class="status-primary complete">
          <span class="icon">check_circle</span> All ${totalFiles} files indexed
          ${run ? html`
            <span class="timing" title=${`${formatDuration(run.embedMs)} of that inside the model; the rest is file I/O. Measured ${new Date(run.finishedAt).toLocaleString()}.`}>
              · embedded ${run.embeddedCount} in ${formatDuration(run.wallMs)}
              ${run.embeddedCount > 0 ? html`(${Math.round(run.wallMs / run.embeddedCount)} ms/file)` : ''}
            </span>
          ` : ''}
          ${failedNote}
          <span class="dim">· ${breakdown}</span>
        </div>
      `;
    }

    return html`
      <div class="status-primary">
        <b>${totalFiles - pendingCount}</b> of <b>${totalFiles}</b> files indexed
        <span class="dim">
          · ${pendingCount} pending ${models.engine ? '' : '· load a model to index'}
        </span>
        ${failedNote}
      </div>
    `;
  }

  override render() {
    const {indexer, samples} = this.state;
    const {progress, discoveredFiles} = indexer;
    const totalFiles = discoveredFiles.length;
    const pendingCount = indexer.countPending(this.state.promptTemplate);
    const indexedCount = totalFiles - pendingCount;

    const counts = {text: 0, image: 0, audio: 0};
    for (const f of discoveredFiles) counts[f.category]++;
    const breakdown = (['text', 'image', 'audio'] as const)
        .filter((k) => counts[k] > 0)
        .map((k) => `${counts[k]} ${k}`)
        .join(' · ');

    const percentage = progress.isIndexing ?
        progress.percentage :
        (totalFiles > 0 ? Math.round((indexedCount / totalFiles) * 100) : 0);
    const isSampleAlbum = indexer.targetDirId === SAMPLE_ALBUM_DIR_ID;

    return html`
      <div class="toolbar">
        <div class="toolbar-row">
          <div class="folder-picker">
            <button
              class="folder-btn ${indexer.targetDirName ? '' : 'empty'} ${this.isMenuOpen ? 'open' : ''}"
              @click=${() => { this.isMenuOpen = !this.isMenuOpen; }}
              title=${indexer.targetDirName ?
                  `${indexer.targetDirName} — click to switch or add a folder` :
                  'Choose a folder to index'}
            >
              <span class="icon">${isSampleAlbum ? 'photo_library' : 'folder'}</span>
              <span class="folder-name">
                ${samples.isLoading ? 'Loading samples…' : (indexer.targetDirName || 'Choose folder…')}
              </span>
              <span class="caret icon">expand_more</span>
            </button>

            ${this.isMenuOpen ? html`
              <div class="folder-menu" role="listbox">
                ${this.state.availableDirectories.map((dir) => {
                  const isActive = dir.id === indexer.targetDirId;
                  const isSample = dir.id === SAMPLE_ALBUM_DIR_ID;
                  return html`
                    <div class="folder-menu-row ${isActive ? 'active' : ''}">
                      <button
                        class="folder-menu-item"
                        role="option"
                        aria-selected=${isActive}
                        ?disabled=${samples.isLoading}
                        @click=${() => {
                          this.isMenuOpen = false;
                          void this.state.selectDirectory(dir.id);
                        }}
                      >
                        <span class="item-icon icon">${isSample ? 'photo_library' : 'folder'}</span>
                        <span class="item-label">${dir.name}</span>
                      </button>
                      ${isSample ? '' : html`
                        <button
                          class="folder-delete-btn"
                          title="Remove ${dir.name}"
                          ?disabled=${samples.isLoading}
                          @click=${(e: Event) => {
                            e.stopPropagation();
                            void this.state.removeDirectory(dir.id);
                          }}
                        >
                          <span class="icon">close</span>
                        </button>
                      `}
                    </div>
                  `;
                })}
                <div class="folder-menu-divider"></div>
                <button
                  class="folder-menu-item add-new"
                  ?disabled=${samples.isLoading}
                  @click=${() => {
                    this.isMenuOpen = false;
                    void this.state.addNewDirectory();
                  }}
                >
                  <span class="item-icon icon">create_new_folder</span>
                  <span class="item-label">Add new…</span>
                </button>
              </div>
            ` : ''}
          </div>

          <div class="status">
            ${this.renderStatusPrimary(pendingCount, totalFiles, breakdown)}
            <div
              class="status-secondary ${progress.isIndexing ? 'processing' : ''}"
              title=${progress.isIndexing ? progress.currentFile : this.state.statusMessage}
            >
              ${progress.isIndexing ? progress.currentFile : this.state.statusMessage}
            </div>
          </div>

          <div class="actions">
            ${progress.isIndexing ? html`
              <button class="btn btn-danger" @click=${() => indexer.cancelIndexing()}>
                Stop
              </button>
            ` : this.state.needsReauthorization ? html`
              <button
                class="btn btn-primary"
                @click=${() => void this.reauthorize()}
                title="The browser forgot its permission for this folder when the page was reloaded"
              >
                <span class="icon">lock_open</span>
                Re-grant access to ${indexer.targetDirName}
              </button>
            ` : html`
              <button
                class="btn btn-primary"
                ?disabled=${totalFiles === 0 && (!indexer.targetDirName || indexer.isAuthorized)}
                @click=${() => void this.state.startIndexing()}
                title="Compute embeddings for new and changed files"
              >
                ${pendingCount === 0 && totalFiles > 0 ? 'Re-index' : 'Index files'}
              </button>
              <button
                class="btn btn-secondary"
                ?disabled=${indexer.indexedDocuments.size === 0}
                @click=${this.clearIndex}
                title="Clear cached embeddings for this folder"
              >
                Clear
              </button>
            `}
          </div>
        </div>

        ${totalFiles > 0 ? html`
          <div class="progress-track">
            <div
              class="progress-fill ${progress.isIndexing ? 'active' : ''}"
              style="width: ${percentage}%;"
            ></div>
          </div>
        ` : ''}
      </div>
    `;
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${String(rest).padStart(2, '0')}s`;
}
