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

import {EXAMPLE_QUERIES, ExampleQuery} from '../app_config.js';
import {StatefulElement} from '../state_controller.js';
import {FileCategory, IndexedDocument} from '../stores/index_store.js';
import {SAMPLE_ALBUM_DIR_ID} from '../stores/sample_album_store.js';
import {sharedStyles} from '../styles/shared_styles.js';
import {codeHighlightStyles, renderFileText} from '../utils/code_highlight.js';
import {createObjectUrl} from '../utils/hljs_util.js';

/* tslint:disable:no-new-decorators */

/** Thumbnails are rendered at 32–38 CSS px; this covers 2–3× displays. */
const THUMBNAIL_WIDTH = 96;

/** How many thumbnail object URLs to keep alive (results show at most 50). */
const THUMBNAIL_CACHE_SIZE = 200;

/** Category filter pills, in display order. */
const FILTER_OPTIONS: ReadonlyArray<readonly['all' | FileCategory, string]> = [
  ['all', 'All'],
  ['text', 'Text'],
  ['image', 'Images'],
  ['audio', 'Audio'],
];

/**
 * Makes an object URL for a small rendition of `file`, so the list does not
 * decode every full-size photo. Falls back to the original bytes when the
 * browser cannot downscale.
 */
async function createThumbnailUrl(file: File): Promise<string> {
  try {
    const bitmap = await createImageBitmap(
        file, {resizeWidth: THUMBNAIL_WIDTH, resizeQuality: 'low'});
    try {
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('No 2D canvas context.');
      ctx.drawImage(bitmap, 0, 0);
      return createObjectUrl(await canvas.convertToBlob({type: 'image/png'}));
    } finally {
      bitmap.close();
    }
  } catch {
    return createObjectUrl(file);
  }
}

/** Grid of search results, example queries and search timings. */
@customElement('embedding-search-results')
export class ResultsGrid extends StatefulElement {
  @state() private activeDocText = '';
  @state() private isLoadingDocText = false;
  @state() private activeMediaUrl = '';
  @state() private previewError = '';
  private currentDocPath = '';
  private loadToken = 0;
  private playNextSelection = false;
  private autoplayMedia = false;
  private thumbnailDirId = '';
  private readonly thumbnails = new Map<string, string>();
  private readonly pendingThumbnails = new Set<string>();

  static override styles = [
    sharedStyles,
    codeHighlightStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        width: 100%;
        flex: 1;
        min-height: 0;
        overflow: hidden;
      }

      .split-container {
        display: flex;
        flex: 1;
        min-height: 0;
        overflow: hidden;
      }

      .list-pane {
        width: 360px;
        flex-shrink: 0;
        border-right: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .list-header {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .list-header-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .list-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .list-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
      }

      .filter-row {
        display: flex;
        gap: 6px;
      }

      .filter-pill {
        padding: 3px 8px;
        border-radius: 12px;
        font: inherit;
        font-size: 0.7rem;
        font-weight: 600;
        line-height: normal;
        background: var(--bg-dark);
        border: 1px solid var(--border);
        color: var(--text-muted);
        cursor: pointer;
        transition: all 0.15s;
      }

      .filter-pill:focus-visible {
        outline: none;
        box-shadow: 0 0 0 2px rgba(0, 201, 158, 0.35);
      }

      .filter-pill.active {
        background: rgba(0, 201, 158, 0.15);
        border-color: var(--teal);
        color: var(--teal);
      }

      .files-list {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
      }

      .file-item {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 10px 14px;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        border-left: 3px solid transparent;
        background: transparent;
        color: inherit;
        font: inherit;
        text-align: left;
        cursor: pointer;
        transition: background 0.12s;
      }

      .file-item:hover {
        background: rgba(255, 255, 255, 0.03);
      }

      .file-item:focus-visible {
        outline: none;
      }

      .file-item:focus-visible:not(.selected) {
        background: rgba(255, 255, 255, 0.06);
        box-shadow: inset 0 0 0 2px var(--teal);
      }

      .file-item.selected {
        background: rgba(0, 201, 158, 0.08);
        border-left-color: var(--teal);
      }

      .file-icon {
        font-size: 1.25rem;
        flex-shrink: 0;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .file-thumb {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        object-fit: cover;
        border: 1px solid var(--border);
        flex-shrink: 0;
        background: var(--bg-input);
      }

      .file-info {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .file-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .file-path {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: ui-monospace, monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .score-tag {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        flex-shrink: 0;
      }

      .score-high {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
      }

      .score-mid {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.4);
      }

      .score-low {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
      }

      .preview-pane {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .preview-header {
        padding: 10px 18px;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
      }

      .preview-header-left {
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 0;
      }

      .preview-filename {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        word-break: break-all;
      }

      .preview-path {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: ui-monospace, monospace;
        word-break: break-all;
      }

      .preview-header-right {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
      }

      .add-to-search-btn {
        padding: 4px 10px;
        font-size: 0.8rem;
      }

      .preview-body {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .image-viewer,
      .audio-viewer {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 20px;
        height: 100%;
        min-height: 250px;
        padding: 20px;
      }

      .image-viewer img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      }

      .audio-icon-large {
        font-size: 4rem;
      }

      .preview-error {
        color: var(--amber);
        text-align: center;
        max-width: 40ch;
      }

      .text-viewer {
        padding: 16px 20px;
        font-family: ui-monospace, monospace;
        font-size: 0.85rem;
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--text);
        overflow-y: auto;
        flex: 1;
      }

      .empty-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        height: 100%;
        padding: 48px 20px;
        text-align: center;
        color: var(--text-muted);
        gap: 12px;
        overflow-y: auto;
      }

      .empty-container h3 {
        color: #ffffff;
      }

      .empty-container .pulse {
        color: var(--teal);
        animation: pulse 1.2s ease-in-out infinite;
      }

      @keyframes pulse {
        50% { opacity: 0.35; }
      }

      .timing {
        color: var(--cyan);
        font-variant-numeric: tabular-nums;
        font-weight: 600;
      }

      .timing-breakdown {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
      }

      .timing-breakdown b {
        color: var(--text);
        font-weight: 600;
      }

      .examples {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        margin-top: 14px;
        max-width: 900px;
      }

      .examples-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-top: 8px;
      }

      .examples-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
      }

      .example-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 18px;
        border-radius: 999px;
        border: 1px solid var(--border-light);
        background: var(--bg-card);
        color: var(--text);
        font-family: inherit;
        font-size: 0.95rem;
        cursor: pointer;
        transition: border-color 0.15s, background 0.15s, transform 0.15s;
      }

      .example-pill.multimodal {
        padding: 6px 18px 6px 8px;
      }

      .example-pill:hover:not(:disabled) {
        border-color: var(--teal);
        background: rgba(0, 201, 158, 0.1);
        transform: translateY(-1px);
      }

      .example-pill:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }

      .example-thumb {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid var(--cyan);
        flex-shrink: 0;
      }

      .example-thumb.placeholder {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        background: var(--bg-input);
      }
    `,
  ];

  override connectedCallback() {
    super.connectedCallback();
    window.addEventListener('keydown', this.onKeyDown);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('keydown', this.onKeyDown);
    this.loadToken++;
    this.releaseActiveMedia();
    this.clearThumbnails();
  }

  private releaseActiveMedia() {
    if (this.activeMediaUrl) {
      URL.revokeObjectURL(this.activeMediaUrl);
      this.activeMediaUrl = '';
    }
  }

  private clearThumbnails() {
    for (const url of this.thumbnails.values()) URL.revokeObjectURL(url);
    this.thumbnails.clear();
    this.pendingThumbnails.clear();
  }

  private syncThumbnailDirectory() {
    const dirId = this.state.indexer.targetDirId;
    if (dirId !== this.thumbnailDirId) {
      this.clearThumbnails();
      this.thumbnailDirId = dirId;
    }
  }

  /** Moves the selection through the results with the Up/Down arrow keys. */
  private readonly onKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.defaultPrevented || !this.hasState) return;
    if (this.state.isModelModalOpen || this.state.isCameraModalOpen) return;

    const target = e.composedPath()[0];
    if (target instanceof HTMLElement &&
        (target.isContentEditable || target.tagName === 'INPUT' ||
         target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) {
      return;
    }
    if (this.state.search.selectAdjacent(e.key === 'ArrowDown' ? 1 : -1)) {
      e.preventDefault();
    }
  };

  private getImageThumbnail(doc: {path: string}): string|null {
    this.syncThumbnailDirectory();
    const cached = this.thumbnails.get(doc.path);
    if (cached !== undefined) {
      this.thumbnails.delete(doc.path);
      this.thumbnails.set(doc.path, cached);
      return cached;
    }
    if (!this.pendingThumbnails.has(doc.path)) {
      this.pendingThumbnails.add(doc.path);
      void this.loadThumbnail(doc, this.thumbnailDirId);
    }
    return null;
  }

  private async loadThumbnail(doc: {path: string}, dirId: string):
      Promise<void> {
    try {
      const handle = this.state.indexer.handleFor(doc.path);
      if (!handle) return;
      const file = await handle.getFile();
      if (!this.isConnected || this.thumbnailDirId !== dirId) return;
      const url = await createThumbnailUrl(file);
      if (!this.isConnected || this.thumbnailDirId !== dirId) {
        URL.revokeObjectURL(url);
        return;
      }
      while (this.thumbnails.size >= THUMBNAIL_CACHE_SIZE) {
        const oldest = this.thumbnails.keys().next().value!;
        URL.revokeObjectURL(this.thumbnails.get(oldest)!);
        this.thumbnails.delete(oldest);
      }
      this.thumbnails.set(doc.path, url);
      this.requestUpdate();
    } catch (e) {
      console.warn(`[EmbeddingSearch] Failed to load thumbnail for ${doc.path}:`, e);
    } finally {
      if (this.thumbnailDirId === dirId) {
        this.pendingThumbnails.delete(doc.path);
      }
    }
  }

  override updated(changedProperties: Map<string, unknown>) {
    super.updated(changedProperties);
    this.syncThumbnailDirectory();
    const selectedDoc = this.state.search.selectedDoc;
    if (selectedDoc && selectedDoc.path !== this.currentDocPath) {
      this.currentDocPath = selectedDoc.path;
      this.autoplayMedia = this.playNextSelection;
      this.playNextSelection = false;
      const selectedEl =
          this.renderRoot.querySelector<HTMLElement>('.file-item.selected');
      selectedEl?.scrollIntoView({block: 'nearest'});
      if (this.shadowRoot?.activeElement?.classList.contains('file-item')) {
        selectedEl?.focus({preventScroll: true});
      }
      void this.loadDocContent(selectedDoc);
    } else if (!selectedDoc && this.currentDocPath) {
      this.currentDocPath = '';
      this.loadToken++;
      this.activeDocText = '';
      this.previewError = '';
      this.isLoadingDocText = false;
      this.releaseActiveMedia();
    }
  }

  /**
   * Loads the selected file for the preview pane. Every step checks that this
   * load is still the latest one before updating state.
   */
  private async loadDocContent(doc: IndexedDocument) {
    const token = ++this.loadToken;
    this.releaseActiveMedia();
    this.activeDocText = '';
    this.previewError = '';
    this.isLoadingDocText = doc.type === 'text';

    try {
      const handle = this.state.indexer.handleFor(doc.path);
      if (!handle) throw new Error('File not found in the folder.');

      const file = await handle.getFile();
      if (token !== this.loadToken) return;

      if (doc.type === 'text') {
        const text = await file.text();
        if (token !== this.loadToken) return;
        this.activeDocText = text;
      } else {
        this.activeMediaUrl = createObjectUrl(file);
      }
    } catch (e) {
      if (token !== this.loadToken) return;
      console.warn(`[EmbeddingSearch] Failed to load preview for ${doc.path}:`, e);
      if (doc.type === 'text') {
        this.activeDocText = doc.previewText || 'Failed to read file content.';
      } else {
        this.previewError = `Could not load this ${doc.type}: ${
            e instanceof Error ? e.message : String(e)}`;
      }
    } finally {
      if (token === this.loadToken) this.isLoadingDocText = false;
    }
  }

  private selectDoc(doc: IndexedDocument, explicit = false) {
    this.playNextSelection = explicit;
    this.state.search.setSelectedDoc(doc);
  }

  private async addDocToQuery(doc: IndexedDocument) {
    if (doc.type !== 'image' && doc.type !== 'audio') return;
    try {
      const handle = this.state.indexer.handleFor(doc.path);
      if (!handle) return;
      const file = await handle.getFile();
      this.state.search.appendChip(doc.type, doc.name, file);
    } catch (e) {
      console.warn(`[EmbeddingSearch] Failed to add ${doc.path} to query:`, e);
    }
  }

  private renderExample(example: ExampleQuery) {
    const enabled = this.state.canRunExample(example) &&
        !this.state.search.isSearching;
    const isMultimodal = example.parts.some((p) => 'image' in p);
    return html`
      <button
        class="example-pill ${isMultimodal ? 'multimodal' : ''}"
        ?disabled=${!enabled}
        title=${enabled ? 'Run this example' : 'Index more images to try this example'}
        @click=${() => void this.state.runExample(example)}
      >
        ${example.parts.map((part) => {
      if ('text' in part) return html`<span>${part.text}</span>`;
      const doc = this.state.resolveExampleImage(part.image);
      const url = doc ? this.getImageThumbnail(doc) : null;
      return url ? html`<img class="example-thumb" src=${url} alt=${doc!.name} loading="lazy" decoding="async" />` :
                   html`<span class="example-thumb placeholder icon">image</span>`;
    })}
      </button>
    `;
  }

  private renderEmptyState() {
    const {indexer, samples} = this.state;
    const {progress} = indexer;
    const discovered = indexer.discoveredFiles.length;
    if (progress.isIndexing || indexer.isScanning ||
        (indexer.targetDirId && discovered > 0)) {
      let title: string;
      let detail: string;
      if (progress.isIndexing) {
        title = `Indexing ${indexer.targetDirName}…`;
        detail = progress.total > 0 ?
            `${progress.current} of ${progress.total} files` +
                (progress.currentFile ? ` · ${progress.currentFile}` : '') :
            'Preparing…';
      } else if (indexer.isScanning) {
        title = `Scanning ${indexer.targetDirName}…`;
        detail = 'Looking for text, image, and audio files.';
      } else {
        title = `Ready to index ${indexer.targetDirName}`;
        detail = `${discovered} file(s) found. ` +
            (this.state.models.engine ?
                 'Click "Index files" above to start.' :
                 'Load a model to start indexing.');
      }
      return html`
        <div class="split-container">
          <div class="empty-container" aria-live="polite">
            <span class="icon pulse" style="font-size: 3rem;">folder</span>
            <h3>${title}</h3>
            <p>${detail}</p>
          </div>
        </div>
      `;
    }

    return html`
      <div class="split-container">
        <div class="empty-container">
          <span class="icon" style="font-size: 3rem;">photo_library</span>
          <h3>Nothing indexed yet</h3>
          <p>
            Start with the AI Edge Gallery sample photos, or choose your own
            folder of text, image, and audio files above.
          </p>
          ${indexer.targetDirId === SAMPLE_ALBUM_DIR_ID ? '' : html`
            <button
              class="btn btn-primary"
              ?disabled=${samples.isLoading}
              @click=${() => void this.state.loadSampleAlbum()}
            >
              <span class="icon">auto_awesome</span>
              ${samples.isLoading ? samples.progressText || 'Loading…' : 'Load the sample album'}
            </button>
          `}
        </div>
      </div>
    `;
  }

  private renderPrompt() {
    const textOnly = EXAMPLE_QUERIES.filter((q) => q.parts.every((p) => 'text' in p));
    const multimodal = EXAMPLE_QUERIES.filter((q) => q.parts.some((p) => 'image' in p));
    return html`
      <div class="split-container">
        <div class="empty-container">
          <span class="icon" style="font-size: 3rem;">search</span>
          <h3>Search Across ${this.state.indexer.indexedDocuments.size} Files</h3>
          <p>Type below, drop in images or audio, or start from an example.</p>
          <div class="examples">
            <div class="examples-label">Try a text query</div>
            <div class="examples-row">${textOnly.map((q) => this.renderExample(q))}</div>
            <div class="examples-label">…or mix images and text, in any order</div>
            <div class="examples-row">${multimodal.map((q) => this.renderExample(q))}</div>
          </div>
        </div>
      </div>
    `;
  }

  private renderResultList() {
    const {search} = this.state;
    const {results, selectedDoc} = search;
    return html`
      <div class="list-pane">
        <div class="list-header">
          <div class="list-header-top">
            <span class="list-title">Matches (${results.length})</span>
            <span
              class="list-meta timing"
              title="Time to embed the query on the GPU, then to score and sort every indexed file."
            >
              ${search.searchLatencyMs} ms
            </span>
          </div>
          <div class="timing-breakdown">
            embed query <b>${Math.round(search.queryEmbedMs)} ms</b>
            · rank ${search.comparedCount} files
            <b>${search.rankMs < 1 ? search.rankMs.toFixed(2) : Math.round(search.rankMs)} ms</b>
          </div>
          <div class="filter-row" role="group" aria-label="Filter by type">
            ${FILTER_OPTIONS.map(([cat, label]) => {
              const active = search.selectedCategory === cat;
              return html`
                <button
                  type="button"
                  class="filter-pill ${active ? 'active' : ''}"
                  aria-pressed=${active ? 'true' : 'false'}
                  @click=${() => this.state.setCategoryFilter(cat)}
                >
                  ${label}
                </button>
              `;
            })}
          </div>
        </div>

        <div class="files-list">
          ${results.length === 0 ? html`
            <div class="empty-container">
              <p>No matching files.</p>
            </div>
          ` : results.map(({doc, score, scorePercent}) => {
            const isSelected = selectedDoc?.path === doc.path;
            const scoreClass = score >= 0.75 ? 'score-high' :
                               score >= 0.50 ? 'score-mid' : 'score-low';
            const thumbUrl = doc.type === 'image' ? this.getImageThumbnail(doc) : null;
            const icon = doc.type === 'image' ? 'image' :
                         doc.type === 'audio' ? 'music_note' : 'description';

            return html`
              <button
                type="button"
                class="file-item ${isSelected ? 'selected' : ''}"
                aria-current=${isSelected ? 'true' : 'false'}
                @click=${() => this.selectDoc(doc, /* explicit= */ true)}
                title="${doc.path}"
              >
                ${thumbUrl ? html`
                  <img
                    class="file-thumb"
                    src=${thumbUrl}
                    alt="${doc.name}"
                    loading="lazy"
                    decoding="async"
                  />
                ` : html`
                  <span class="file-icon icon">${icon}</span>
                `}
                <div class="file-info">
                  <span class="file-name">${doc.name}</span>
                  <span class="file-path">${doc.path}</span>
                </div>
                <span class="score-tag ${scoreClass}">${scorePercent}</span>
              </button>
            `;
          })}
        </div>
      </div>
    `;
  }

  private renderPreview() {
    const {results, selectedDoc} = this.state.search;
    if (!selectedDoc) {
      return results.length === 0 ? html`
        <div class="empty-container">
          <span class="icon" style="font-size: 3rem;">search_off</span>
          <h3>No Matching Files Found</h3>
          <p>Try searching with different keywords, alternative media, or adjusting filters.</p>
        </div>
      ` : html`
        <div class="empty-container">
          <span class="icon" style="font-size: 2.5rem;">arrow_back</span>
          <p>Select a file from the list on the left to view its preview.</p>
        </div>
      `;
    }

    return html`
      <div class="preview-header">
        <div class="preview-header-left">
          <div class="preview-filename">${selectedDoc.name}</div>
          <div class="preview-path">${selectedDoc.path}</div>
        </div>
        <div class="preview-header-right">
          <span class="badge ${
            selectedDoc.type === 'image' ? 'badge-teal' :
            selectedDoc.type === 'audio' ? 'badge-purple' : 'badge-blue'}">
            ${selectedDoc.type.toUpperCase()}
          </span>
          <span class="badge badge-muted">${(selectedDoc.size / 1024).toFixed(1)} KB</span>
          <span class="badge badge-muted">${new Date(selectedDoc.lastModified).toLocaleDateString()}</span>
          ${selectedDoc.type === 'image' || selectedDoc.type === 'audio' ? html`
            <button
              class="btn btn-secondary add-to-search-btn"
              title="Insert this ${selectedDoc.type} into the search box"
              @click=${() => void this.addDocToQuery(selectedDoc)}
            >
              <span class="icon">add</span> Add to search
            </button>
          ` : ''}
        </div>
      </div>

      <div class="preview-body">
        ${selectedDoc.type === 'image' ? html`
          <div class="image-viewer">
            ${this.activeMediaUrl ? html`
              <img src=${this.activeMediaUrl} alt="${selectedDoc.name}" />
            ` : this.previewError ? html`
              <span class="preview-error">${this.previewError}</span>
            ` : html`<span>Loading image...</span>`}
          </div>
        ` : selectedDoc.type === 'audio' ? html`
          <div class="audio-viewer">
            <span class="audio-icon-large icon">music_note</span>
            ${this.activeMediaUrl ? html`
              <audio
                controls
                ?autoplay=${this.autoplayMedia}
                src=${this.activeMediaUrl}
              ></audio>
            ` : this.previewError ? html`
              <span class="preview-error">${this.previewError}</span>
            ` : html`<span>Loading audio...</span>`}
          </div>
        ` : html`
          <!-- No whitespace around the text: .text-viewer is pre-wrap. -->
          <div class="text-viewer">${
            this.isLoadingDocText ?
              'Loading document content...' :
              renderFileText(this.activeDocText || selectedDoc.previewText || 'Empty file.', selectedDoc.path)}</div>
        `}
      </div>
    `;
  }

  override render() {
    if (this.state.indexer.indexedDocuments.size === 0) {
      return this.renderEmptyState();
    }
    if (!this.state.search.hasSearched) {
      return this.renderPrompt();
    }
    return html`
      <div class="split-container">
        ${this.renderResultList()}
        <div class="preview-pane">${this.renderPreview()}</div>
      </div>
    `;
  }
}
