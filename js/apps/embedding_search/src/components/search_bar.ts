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

import {css, html, render} from 'lit';
import {customElement, query, state} from 'lit/decorators.js';

import {deepActiveElement, restoreFocus, StatefulElement} from '../state_controller.js';
import {QueryChip, QueryPart} from '../stores/search_store.js';
import {sharedStyles} from '../styles/shared_styles.js';
import {classifyMediaFile, toSupportedMediaBlob} from '../utils/media_types.js';

/* tslint:disable:no-new-decorators */

interface ChipElement extends HTMLSpanElement {
  chip?: QueryChip;
}

/** Multimodal query composer (text, image, camera and audio chips). */
@customElement('embedding-search-bar')
export class SearchBar extends StatefulElement {
  @state() private isRecordingMic = false;
  @state() private recordTimer = 0;
  private timerInterval: number|null = null;

  @state() private isCameraModalOpen = false;
  @state() private isCameraFrontFacing = false;
  private cameraStream: MediaStream|null = null;
  private cameraOpener: HTMLElement|null = null;

  /**
   * The parts array this component last wrote to (or read from) the editor
   * DOM. Used to tell our own edits apart from external query changes.
   */
  private lastSyncedParts: QueryPart[]|null = null;

  /** Last known caret position, so toolbar actions can insert in place. */
  private savedRange: Range|null = null;

  @query('#query-editor') private editorEl?: HTMLDivElement;
  @query('#file-input') private fileInputEl?: HTMLInputElement;
  @query('#camera-fallback-input') private cameraFallbackInputEl?: HTMLInputElement;
  @query('#camera-video') private cameraVideoEl?: HTMLVideoElement;
  @query('#camera-snap-btn') private cameraSnapBtnEl?: HTMLButtonElement;

  private readonly onWindowKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && this.isCameraModalOpen && !e.defaultPrevented) {
      e.preventDefault();
      this.closeCamera();
    }
  };

  static override styles = [
    sharedStyles,
    css`
      :host {
        display: block;
        width: 100%;
      }

      .search-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-width: 1100px;
        margin: 0 auto;
        width: 100%;
      }

      .search-box {
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 6px 12px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
      }

      .search-box:focus-within {
        border-color: var(--teal);
        box-shadow: 0 0 0 2px rgba(0, 201, 158, 0.25), 0 8px 28px rgba(0, 0, 0, 0.5);
      }

      /*
       * The query editor is a contenteditable so that media chips can be
       * placed inline between words. Its children are managed imperatively
       * (see syncEditorFromParts) because re-rendering them with Lit on every
       * keystroke would reset the caret.
       */
      .editor {
        position: relative;
        flex: 1;
        min-width: 160px;
        max-height: 140px;
        overflow-y: auto;
        padding: 8px 4px;
        color: #ffffff;
        font-size: 0.95rem;
        font-family: inherit;
        line-height: 1.9;
        outline: none;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }

      .editor.is-empty::before {
        content: attr(data-placeholder);
        position: absolute;
        left: 4px;
        top: 8px;
        color: var(--text-muted);
        pointer-events: none;
        user-select: none;
      }

      .query-chip {
        display: inline-flex;
        align-items: center;
        vertical-align: middle;
        gap: 6px;
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2px 8px 2px 4px;
        margin: 0 2px;
        font-size: 0.8rem;
        color: #ffffff;
        max-width: 220px;
      }

      .query-chip::selection,
      .query-chip *::selection {
        background: rgba(0, 201, 158, 0.35);
      }

      .chip-thumb {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid var(--border);
        flex-shrink: 0;
      }

      .chip-icon {
        font-size: 0.9rem;
        flex-shrink: 0;
      }

      .chip-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.75rem;
        user-select: none;
      }

      .chip-del-btn {
        background: transparent;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        padding: 0 2px;
        font-size: 0.75rem;
        line-height: 1;
        display: flex;
        align-items: center;
        border-radius: 50%;
        user-select: none;
      }

      .chip-del-btn:hover {
        color: var(--red);
      }

      .btn-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        border: 1px solid transparent;
        color: var(--text-muted);
        font-size: 1.15rem;
        cursor: pointer;
        transition: all 0.15s;
        flex-shrink: 0;
      }

      .btn-icon:hover {
        background: var(--bg-dark);
        color: #ffffff;
        border-color: var(--border);
      }

      .btn-icon.plus {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--teal);
      }

      .btn-icon.plus:hover {
        background: rgba(0, 201, 158, 0.15);
        border-color: rgba(0, 201, 158, 0.4);
      }

      .btn-icon.recording {
        background: rgba(239, 68, 68, 0.2);
        border-color: var(--red);
        color: #fca5a5;
        animation: pulse 1.5s infinite;
        width: auto;
        border-radius: 18px;
        padding: 0 10px;
        font-size: 0.85rem;
        font-weight: 600;
        gap: 4px;
      }

      @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
      }

      .search-submit-btn {
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        gap: 6px;
        flex-shrink: 0;
      }

      .modal-card {
        max-width: 500px;
      }

      .camera-preview-container {
        position: relative;
        background: #000000;
        width: 100%;
        aspect-ratio: 4 / 3;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }

      .camera-stream {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

      .camera-stream.mirrored {
        transform: scaleX(-1);
      }
    `,
  ];

  override connectedCallback() {
    super.connectedCallback();
    window.addEventListener('keydown', this.onWindowKeyDown);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('keydown', this.onWindowKeyDown);
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    if (this.isRecordingMic) {
      this.isRecordingMic = false;
      if (this.hasState) void this.state.recorder.cancel();
    }
    this.closeCamera();
  }

  override updated(changed: Map<string, unknown>) {
    super.updated(changed);
    // Rebuild the editor only when the query changed from outside this
    // component (page-level drop, clear button, folder switch).
    if (this.editorEl && this.state.search.parts !== this.lastSyncedParts) {
      this.syncEditorFromParts();
    }
  }

  /** Replaces the editor's contents with the query parts from the store. */
  private syncEditorFromParts() {
    const editor = this.editorEl;
    if (!editor) return;

    const parts = this.state.search.parts;
    this.lastSyncedParts = parts;
    editor.replaceChildren();
    const isOnlyNbsp = parts.length > 0 &&
        parts.every((p) => p.type === 'text' && /^[\u00a0]+$/.test(p.text));
    if (!isOnlyNbsp) {
      for (const part of parts) {
        editor.appendChild(
            part.type === 'text' ? document.createTextNode(part.text) :
                                   this.createChipNode(part.chip));
      }
    }
    this.updateEmptyState();
  }

  /** Reads the editor DOM back into ordered query parts. */
  private serializeEditor(): QueryPart[] {
    const editor = this.editorEl;
    if (!editor) return [];

    const parts: QueryPart[] = [];
    const pushText = (rawText: string) => {
      const text = rawText.replace(/\u00a0/g, ' ');
      if (!text) return;
      const last = parts[parts.length - 1];
      if (last && last.type === 'text') {
        last.text += text;
      } else {
        parts.push({type: 'text', text});
      }
    };

    for (const node of Array.from(editor.childNodes)) {
      const chip = (node as ChipElement).chip;
      if (chip) {
        parts.push({type: 'chip', chip});
      } else if (
          node.nodeType === Node.TEXT_NODE || node instanceof HTMLElement) {
        pushText(node.textContent ?? '');
      }
    }
    return parts;
  }

  /** Pushes the current editor contents into the store. */
  private commitEditor() {
    const hadChip = this.lastSyncedParts?.some((p) => p.type === 'chip') ?? false;
    const parts = this.serializeEditor();
    if (hadChip &&
        parts.every((p) => p.type === 'text' && /^\s*$/.test(p.text))) {
      parts.length = 0;
      this.editorEl?.replaceChildren();
    }
    this.lastSyncedParts = parts;
    this.state.search.setParts(parts);
    this.updateEmptyState();
  }

  private updateEmptyState() {
    const editor = this.editorEl;
    if (!editor) return;
    editor.classList.toggle(
        'is-empty',
        (editor.textContent ?? '') === '' &&
            editor.querySelector('.query-chip') === null);
  }

  private createChipNode(chip: QueryChip): HTMLElement {
    const span = document.createElement('span') as ChipElement;
    span.className = 'query-chip';
    span.contentEditable = 'false';
    span.chip = chip;
    render(
        html`
          ${chip.type === 'image' ? html`
            <img class="chip-thumb" src=${chip.previewUrl} alt="${chip.name}" />
          ` : html`
            <span class="chip-icon icon">music_note</span>
          `}
          <span class="chip-name" title="${chip.name}">${chip.name}</span>
          <button
            class="chip-del-btn"
            title="Remove"
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${(e: Event) => {
          e.stopPropagation();
          span.remove();
          this.commitEditor();
        }}
          ><span class="icon">close</span></button>
        `,
        span);
    return span;
  }

  private shadowSelection(): Selection|null {
    const root =
        this.shadowRoot as ShadowRoot & {getSelection?: () => Selection | null};
    return root?.getSelection?.() ?? document.getSelection();
  }

  /** The live caret range, if it currently sits inside the editor. */
  private getEditorRange(): Range|null {
    const editor = this.editorEl;
    const selection = this.shadowSelection();
    if (!editor || !selection || selection.rangeCount === 0) return null;
    const range = selection.getRangeAt(0);
    return editor.contains(range.commonAncestorContainer) ? range : null;
  }

  /** Whether the caret is on the first or last visual line of the editor. */
  private caretOnEdgeLine(edge: 'first'|'last'): boolean {
    const editor = this.editorEl;
    const range = this.getEditorRange();
    if (!editor || !range) return true;
    const caret = range.cloneRange();
    caret.collapse(edge === 'first');
    const caretRect = caret.getBoundingClientRect();
    if (caretRect.height === 0) return true;
    const box = editor.getBoundingClientRect();
    return edge === 'first' ?
        caretRect.top - box.top < caretRect.height * 1.5 :
        box.bottom - caretRect.bottom < caretRect.height * 1.5;
  }

  /**
   * Inserts a chip where the caret is, so the user controls where the media
   * sits relative to the surrounding text. Falls back to appending.
   */
  private insertChipAtCaret(chip: QueryChip) {
    const editor = this.editorEl;
    if (!editor) {
      const {search} = this.state;
      search.setParts([...search.parts, {type: 'chip', chip}]);
      return;
    }

    const node = this.createChipNode(chip);
    let range = this.getEditorRange() ?? this.savedRange;
    if (!range || !editor.contains(range.commonAncestorContainer)) {
      range = document.createRange();
      range.selectNodeContents(editor);
      range.collapse(false);
    }

    range.deleteContents();
    range.insertNode(node);

    // Leave a space after the chip so the user can keep typing immediately.
    const spacer = document.createTextNode('\u00a0');
    node.after(spacer);

    const after = document.createRange();
    after.setStart(spacer, spacer.length);
    after.collapse(true);
    this.savedRange = after.cloneRange();
    const selection = this.shadowSelection();
    if (selection) {
      selection.removeAllRanges();
      selection.addRange(after);
      editor.focus();
    }

    this.commitEditor();
  }

  private handleEditorSelect() {
    const range = this.getEditorRange();
    if (range) this.savedRange = range.cloneRange();
  }

  /** Strips formatting so pasted rich text does not smuggle markup in. */
  private handleEditorPaste(e: ClipboardEvent) {
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      e.preventDefault();
      void this.processFiles(Array.from(files));
      return;
    }
    e.preventDefault();
    const text = e.clipboardData?.getData('text/plain') ?? '';
    if (text) {
      document.execCommand('insertText', false, text.replace(/\s+/g, ' '));
    }
  }

  private handleFileUpload(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      void this.processFiles(Array.from(input.files));
      input.value = '';
    }
  }

  private async processFiles(files: File[]) {
    const {search} = this.state;
    for (const file of files) {
      const category = classifyMediaFile(file);
      if (category) {
        const blob = await toSupportedMediaBlob(file, category);
        this.insertChipAtCaret(search.createChip(category, file.name, blob));
      } else {
        this.state.setStatus(`Unsupported file type for search: ${file.name}`);
      }
    }
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void this.state.runSearch();
      return;
    }
    if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') &&
        !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey &&
        this.caretOnEdgeLine(e.key === 'ArrowUp' ? 'first' : 'last') &&
        this.state.search.selectAdjacent(e.key === 'ArrowDown' ? 1 : -1)) {
      e.preventDefault();
    }
  }

  private async toggleMicRecording() {
    if (this.isRecordingMic) {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
      this.isRecordingMic = false;
      const duration = this.recordTimer;
      try {
        const wavBlob = await this.state.recorder.stop();
        this.insertChipAtCaret(this.state.search.createChip(
            'audio', `Voice query (${duration}s)`, wavBlob));
      } catch (err) {
        this.state.setStatus(
            `Could not finish the recording: ${(err as Error).message}`);
      }
    } else {
      try {
        await this.state.recorder.start();
        this.isRecordingMic = true;
        this.recordTimer = 0;
        this.timerInterval = window.setInterval(() => {
          this.recordTimer++;
        }, 1000);
      } catch (err) {
        this.state.setStatus(
            `Microphone unavailable: ${(err as Error).message}`);
      }
    }
  }

  private async openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.cameraFallbackInputEl?.click();
      return;
    }

    try {
      this.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 720}},
      });
      const facingMode =
          this.cameraStream.getVideoTracks()[0]?.getSettings?.().facingMode;
      this.isCameraFrontFacing = !facingMode || facingMode === 'user';
      this.cameraOpener = deepActiveElement();
      this.setCameraModalOpen(true);
      void this.updateComplete.then(() => {
        this.cameraSnapBtnEl?.focus();
        if (this.cameraVideoEl && this.cameraStream) {
          this.cameraVideoEl.srcObject = this.cameraStream;
        }
      });
    } catch (err) {
      console.warn('Direct camera stream failed, falling back to file picker:', err);
      this.cameraFallbackInputEl?.click();
    }
  }

  private closeCamera() {
    if (this.cameraStream) {
      for (const track of this.cameraStream.getTracks()) track.stop();
      this.cameraStream = null;
    }
    if (!this.isCameraModalOpen) return;
    this.setCameraModalOpen(false);
    restoreFocus(this.cameraOpener);
    this.cameraOpener = null;
  }

  private setCameraModalOpen(open: boolean) {
    this.isCameraModalOpen = open;
    if (this.hasState && this.state.isCameraModalOpen !== open) {
      this.state.isCameraModalOpen = open;
      this.state.requestUpdate();
    }
  }

  private takeSnapshot() {
    const video = this.cameraVideoEl;
    if (!video) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      this.closeCamera();
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob) {
        this.insertChipAtCaret(this.state.search.createChip(
            'image', `Camera photo (${Date.now().toString().slice(-4)})`, blob));
      }
      this.closeCamera();
    }, 'image/jpeg', 0.92);
  }

  override render() {
    const {search} = this.state;
    const hasContent = search.hasContent;

    return html`
      <div class="search-container">
        <input
          type="file"
          id="file-input"
          accept="image/*,audio/*"
          multiple
          style="display: none;"
          @change=${this.handleFileUpload}
        />
        <input
          type="file"
          id="camera-fallback-input"
          accept="image/*"
          capture="environment"
          style="display: none;"
          @change=${this.handleFileUpload}
        />

        <div class="search-box">
          <button
            class="btn-icon plus"
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${() => this.fileInputEl?.click()}
            title="Upload image or audio file"
          >
            +
          </button>

          <div
            id="query-editor"
            class="editor is-empty"
            contenteditable="true"
            role="textbox"
            aria-multiline="false"
            aria-label="Search query"
            data-placeholder="Search files with text, or place images and audio anywhere in your query…"
            @input=${() => this.commitEditor()}
            @keyup=${this.handleEditorSelect}
            @mouseup=${this.handleEditorSelect}
            @blur=${this.handleEditorSelect}
            @keydown=${this.handleKeyDown}
            @paste=${this.handleEditorPaste}
            @drop=${(e: DragEvent) => e.preventDefault()}
          ></div>

          <button
            class="btn-icon ${this.isRecordingMic ? 'recording' : ''}"
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${this.toggleMicRecording}
            title="${this.isRecordingMic ? 'Stop recording voice query' : 'Record voice query'}"
          >
            ${this.isRecordingMic ? html`<span class="icon">stop</span> ${this.recordTimer}s` : html`<span class="icon">mic</span>`}
          </button>

          <button
            class="btn-icon"
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${this.openCamera}
            title="Take a photo with camera"
          >
            <span class="icon">photo_camera</span>
          </button>

          ${hasContent ? html`
            <button
              class="btn-icon"
              style="font-size: 0.9rem;"
              @click=${() => search.clearQuery()}
              title="Clear search query"
            >
              <span class="icon">close</span>
            </button>
          ` : ''}

          <button
            class="btn btn-primary search-submit-btn"
            ?disabled=${search.isSearching || !hasContent}
            @click=${() => this.state.runSearch()}
          >
            ${search.isSearching ? html`<div class="spinner"></div> Searching` : html`<span class="icon">search</span> Search`}
          </button>
        </div>
      </div>

      ${this.isCameraModalOpen ? html`
        <div class="modal-backdrop" @click=${(e: Event) => {
          if (e.target === e.currentTarget) this.closeCamera();
        }}>
          <div
            class="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="camera-modal-title"
          >
            <div class="modal-header">
              <span id="camera-modal-title">Take a Snapshot</span>
              <button
                type="button"
                class="close-btn"
                aria-label="Close"
                @click=${this.closeCamera}
              >
                <span class="icon" aria-hidden="true">close</span>
              </button>
            </div>
            <div class="camera-preview-container">
              <video
                id="camera-video"
                class="camera-stream ${this.isCameraFrontFacing ? 'mirrored' : ''}"
                autoplay
                playsinline
              ></video>
            </div>
            <div class="modal-footer">
              <button type="button" class="btn btn-secondary" @click=${this.closeCamera}>
                Cancel
              </button>
              <button
                type="button"
                id="camera-snap-btn"
                class="btn btn-primary"
                @click=${this.takeSnapshot}
              >
                <span class="icon">photo_camera</span> Snap Photo
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    `;
  }
}
