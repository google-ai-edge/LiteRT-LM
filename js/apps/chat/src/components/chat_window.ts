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

import './chat_bubble';

import {SUPPORTED_IMAGE_MIME_TYPES} from '@litert-lm/core';
import {css, html, LitElement, render} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {promptHasContent, PromptImage, PromptPart, promptPartsToText} from '../stores/chat_session_store.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** A shadow root, which has `getSelection` in Chrome. */
type ShadowRootWithSelection = ShadowRoot&{
  getSelection?: () => Selection | null;
};

/**
 * Returns `blob` if the model can decode it, or a PNG re-encoding of it
 * otherwise (e.g. for WebP or SVG images).
 */
async function toSupportedImage(blob: Blob): Promise<Blob> {
  if (SUPPORTED_IMAGE_MIME_TYPES.has(blob.type)) return blob;
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas is not available.');
    ctx.drawImage(bitmap, 0, 0);
    return await canvas.convertToBlob({type: 'image/png'});
  } finally {
    bitmap.close();
  }
}

/** Main chat window component managing messages and input. */
@customElement('litert-chat-window')
export class LitertChatWindow extends LitElement {
  @property({ type: Object })
  state!: LlmChatStateController;

  @state()
  private shouldAutoScroll = true;

  @state() private isDragOver = false;
  @state() private isCameraModalOpen = false;
  private cameraStream: MediaStream|null = null;

  private wasGenerating = false;
  private isProgrammaticScroll = false;

  /** Last known caret position, so toolbar actions can insert in place. */
  private savedRange: Range|null = null;

  @query('#chat-messages') private msgBox!: HTMLDivElement;

  @query('#chat-input') private chatInput!: HTMLDivElement;
  @query('#image-input') private imageInput?: HTMLInputElement;
  @query('#camera-fallback-input')
  private cameraFallbackInput?: HTMLInputElement;
  @query('#camera-video') private cameraVideo?: HTMLVideoElement;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        overflow: hidden;
      }

      .chat-messages {
        flex: 1;
        overflow-y: auto;
        overscroll-behavior: contain; 
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        background-color: var(--bg-chat);
      }

      .starters-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 16px 24px 8px 24px;
        background-color: var(--bg-chat);
        flex-shrink: 0;
      }

      .btn-starter {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid var(--border);
        color: var(--text-muted);
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.15s;
      }

      .btn-starter:hover:not(:disabled) {
        background-color: var(--border);
        color: var(--text);
      }

      .chat-input-container {
        padding: 16px 24px 24px 24px;
        background-color: var(--bg-card);
        border-top: 1px solid var(--border);
        display: flex;
        gap: 12px;
        align-items: flex-end;
        flex-shrink: 0;
      }

      .input-wrapper {
        display: flex;
        flex: 1;
        width: 100%;
        gap: 8px;
        position: relative;
        align-items: center;
        border-radius: 10px;
        border: 1px dashed transparent;
      }

      .input-wrapper.drag-over {
        border-color: var(--teal);
        background: rgba(0, 201, 158, 0.08);
      }

      /*
       * The prompt editor is a contenteditable so that images can be placed
       * inline between words. Its children are managed imperatively (see
       * setEditorParts) because re-rendering them with Lit on every keystroke
       * would reset the caret.
       */
      .editor {
        position: relative;
        flex: 1;
        min-height: 48px;
        max-height: 160px;
        overflow-y: auto;
        background-color: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        padding: 12px 16px;
        font-size: 0.85rem;
        font-family: inherit;
        line-height: 1.5;
        outline: none;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        transition: border-color 0.15s;
      }

      .editor:focus {
        border-color: var(--teal);
      }

      .editor[contenteditable='false'] {
        opacity: 0.6;
        cursor: not-allowed;
      }

      /*
       * Positioned out of flow: as in-flow generated content it would push
       * the caret after the placeholder text. Offsets match the padding.
       */
      .editor.is-empty::before {
        content: attr(data-placeholder);
        position: absolute;
        left: 16px;
        top: 12px;
        color: var(--text-muted);
        pointer-events: none;
        user-select: none;
      }

      .image-chip {
        display: inline-flex;
        align-items: center;
        vertical-align: middle;
        gap: 6px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 2px 6px 2px 2px;
        margin: 0 2px;
        font-size: 0.75rem;
        max-width: 220px;
      }

      .image-chip::selection,
      .image-chip *::selection {
        background: rgba(0, 201, 158, 0.35);
      }

      .chip-thumb {
        width: 36px;
        height: 36px;
        border-radius: 6px;
        object-fit: cover;
        flex-shrink: 0;
      }

      .chip-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
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
        user-select: none;
      }

      .chip-del-btn:hover {
        color: #ef4444;
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
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.15s;
        flex-shrink: 0;
      }

      .btn-icon:hover:not(:disabled) {
        background: var(--bg-input);
        color: var(--text);
        border-color: var(--border);
      }

      .btn-icon:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }

      .btn-icon.plus {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--teal);
      }

      .btn-input {
        height: 38px;
        padding: 0 16px;
        font-size: 0.85rem;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.15s;
      }

      /* Camera Modal */
      .modal-backdrop {
        position: fixed;
        inset: 0;
        background-color: rgba(2, 8, 36, 0.85);
        backdrop-filter: blur(6px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1200;
        padding: 20px;
      }

      .modal-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        width: 100%;
        max-width: 500px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .modal-header {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        color: var(--text);
      }

      .camera-preview-container {
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

      .modal-actions {
        padding: 14px 18px;
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        border-top: 1px solid var(--border);
      }
    `
  ];

  override connectedCallback() {
    super.connectedCallback();
    // Register this chat window as a host of our centralized state
    this.state.addHost(this);
  }

  override disconnectedCallback() {
    this.closeCamera();
    this.state.removeHost(this);
    super.disconnectedCallback();
  }

  override firstUpdated() {
    this.updateEmptyState();

    // Bind scroll event listener to manage user-driven auto-scroll anchoring
    if (this.msgBox) {
      this.msgBox.addEventListener('scroll', () => {
        // Ignore our own programmatic scrolls that follow the message stream.
        if (this.isProgrammaticScroll) {
          this.isProgrammaticScroll = false;
          return;
        }

        // Otherwise, it was a manual user scroll. Un-anchor if they scroll up.
        const threshold = 6;
        const atBottom = this.msgBox.scrollHeight - this.msgBox.scrollTop -
                this.msgBox.clientHeight <=
            threshold;

        if (!atBottom) {
          this.shouldAutoScroll = false;  // Manual scroll up, pause auto-scroll
        } else {
          this.shouldAutoScroll = true;  // Manual scroll down to bottom, resume
        }
      });
    }
  }

  protected override updated() {
    // Auto-scroll on new message.
    if (this.state.chatSession.isGenerating && !this.wasGenerating) {
      this.shouldAutoScroll = true;
    }
    this.wasGenerating = this.state.chatSession.isGenerating;

    if (this.state.chatSession.isGenerating && this.shouldAutoScroll) {
      requestAnimationFrame(() => {
        if (this.msgBox) {
          this.isProgrammaticScroll = true;
          this.msgBox.scrollTop = this.msgBox.scrollHeight;
        }
      });
    }
  }

  /** Replaces the editor's contents and moves the caret to the end. */
  private setEditorParts(parts: PromptPart[]) {
    const editor = this.chatInput;
    if (!editor) return;
    editor.replaceChildren();
    for (const part of parts) {
      editor.appendChild(
          part.type === 'text' ? document.createTextNode(part.text) :
                                 this.createImageChip(part.image));
    }
    this.updateEmptyState();

    const end = document.createRange();
    end.selectNodeContents(editor);
    end.collapse(false);
    this.savedRange = end.cloneRange();
    this.applyRange(end);
  }

  /** Reads the editor DOM back into ordered prompt parts. */
  private serializeEditor(): PromptPart[] {
    const editor = this.chatInput;
    if (!editor) return [];

    const parts: PromptPart[] = [];
    const pushText = (text: string) => {
      if (!text) return;
      // Chip spacers are non-breaking spaces; send them as normal spaces.
      text = text.replace(/\u00a0/g, ' ');
      const last = parts[parts.length - 1];
      if (last && last.type === 'text') {
        last.text += text;
      } else {
        parts.push({type: 'text', text});
      }
    };
    const endsWithNewline = () => {
      const last = parts[parts.length - 1];
      return last?.type === 'text' && last.text.endsWith('\n');
    };

    const walk = (node: Node) => {
      for (const child of Array.from(node.childNodes)) {
        if (child.nodeType === Node.TEXT_NODE) {
          pushText(child.textContent ?? '');
          continue;
        }
        if (!(child instanceof HTMLElement)) continue;

        const imageId = child.dataset['imageId'];
        const image =
            imageId ? this.state.chatSession.getImage(imageId) : undefined;
        if (image) {
          parts.push({type: 'image', image});
        } else if (child.tagName === 'BR') {
          pushText('\n');
        } else {
          // Chrome wraps new lines in <div>s when editing and pasting.
          const isBlock = child.tagName === 'DIV' || child.tagName === 'P';
          if (isBlock && parts.length > 0 && !endsWithNewline()) {
            pushText('\n');
          }
          walk(child);
        }
      }
    };
    walk(editor);
    return parts;
  }

  private updateEmptyState() {
    const editor = this.chatInput;
    if (!editor) return;
    editor.classList.toggle(
        'is-empty',
        (editor.textContent ?? '') === '' &&
            editor.querySelector('.image-chip') === null);
  }

  private createImageChip(image: PromptImage): HTMLElement {
    const span = document.createElement('span');
    span.className = 'image-chip';
    // Makes the chip atomic: the caret steps over it and Backspace deletes
    // the whole chip rather than editing its innards.
    span.contentEditable = 'false';
    span.dataset['imageId'] = image.id;
    render(
        html`
          <img class="chip-thumb" src=${image.url} alt=${image.name} />
          <span class="chip-name" title=${image.name}>${image.name}</span>
          <button
            class="chip-del-btn"
            title="Remove image"
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${(e: Event) => {
          e.stopPropagation();
          span.remove();
          this.updateEmptyState();
        }}
          >✕</button>
        `,
        span);
    return span;
  }

  private getSelection(): Selection|null {
    const root = this.shadowRoot as ShadowRootWithSelection | null;
    return root?.getSelection?.() ?? document.getSelection();
  }

  /** The live caret range, if it currently sits inside the editor. */
  private getEditorRange(): Range|null {
    const editor = this.chatInput;
    const selection = this.getSelection();
    if (!editor || !selection || selection.rangeCount === 0) return null;
    const range = selection.getRangeAt(0);
    return editor.contains(range.commonAncestorContainer) ? range : null;
  }

  private applyRange(range: Range) {
    const selection = this.getSelection();
    this.chatInput?.focus();
    if (!selection) return;
    selection.removeAllRanges();
    selection.addRange(range);
  }

  /**
   * Inserts an image where the caret is, so the user controls where the image
   * sits relative to the surrounding text. Falls back to appending.
   */
  private insertImageAtCaret(image: PromptImage) {
    const editor = this.chatInput;
    if (!editor) return;

    const node = this.createImageChip(image);
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
    this.applyRange(after);
    this.updateEmptyState();
  }

  private handleEditorSelect() {
    const range = this.getEditorRange();
    if (range) this.savedRange = range.cloneRange();
  }

  /** Inserts pasted images as chips and pasted rich text as plain text. */
  private handleEditorPaste(e: ClipboardEvent) {
    e.preventDefault();
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      this.handleEditorSelect();
      void this.addImageFiles(Array.from(files));
      return;
    }
    const text = e.clipboardData?.getData('text/plain') ?? '';
    if (text) {
      document.execCommand('insertText', false, text);
    }
  }

  private handleDragOver(e: DragEvent) {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
    this.isDragOver = true;
  }

  private handleDragLeave() {
    this.isDragOver = false;
  }

  private handleDrop(e: DragEvent) {
    e.preventDefault();
    this.isDragOver = false;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      // Drop position wins over the last caret position.
      const dropRange = this.rangeFromPoint(e.clientX, e.clientY);
      if (dropRange) this.savedRange = dropRange;
      void this.addImageFiles(Array.from(files));
    }
  }

  /** Maps a drop point to a caret position inside the editor, if possible. */
  private rangeFromPoint(x: number, y: number): Range|null {
    const editor = this.chatInput;
    const doc = document as Document & {
      caretRangeFromPoint?: (x: number, y: number) => Range | null,
    };
    if (!editor || !doc.caretRangeFromPoint) return null;
    const range = doc.caretRangeFromPoint(x, y);
    return range && editor.contains(range.commonAncestorContainer) ? range :
                                                                     null;
  }

  private handleImageInputChange(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      void this.addImageFiles(Array.from(input.files));
      input.value = '';
    }
  }

  private setStatus(msg: string) {
    this.state.statusText = msg;
    this.state.requestUpdate();
  }

  /** Adds image files to the prompt at the saved caret position. */
  private async addImageFiles(files: File[]) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        this.setStatus(`Only images can be added to a prompt: ${file.name}`);
        continue;
      }
      try {
        const blob = await toSupportedImage(file);
        this.insertImageAtCaret(
            this.state.chatSession.createImage(file.name, blob));
      } catch (e) {
        console.error('[LiteRT-LM] Failed to read image:', e);
        this.setStatus(`Could not read image ${file.name}.`);
      }
    }
  }

  private async openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.cameraFallbackInput?.click();
      return;
    }

    try {
      this.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 720}},
      });
      this.isCameraModalOpen = true;
      await this.updateComplete;
      if (this.cameraVideo && this.cameraStream) {
        this.cameraVideo.srcObject = this.cameraStream;
      }
    } catch (err) {
      console.warn(
          '[LiteRT-LM] Camera stream failed, falling back to file picker:',
          err);
      this.cameraFallbackInput?.click();
    }
  }

  private closeCamera() {
    if (this.cameraStream) {
      for (const track of this.cameraStream.getTracks()) {
        track.stop();
      }
      this.cameraStream = null;
    }
    this.isCameraModalOpen = false;
  }

  private takeSnapshot() {
    const video = this.cameraVideo;
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
        const name = `Camera photo ${new Date().toLocaleTimeString()}.jpg`;
        this.insertImageAtCaret(this.state.chatSession.createImage(name, blob));
      }
      this.closeCamera();
    }, 'image/jpeg', 0.92);
  }

  private handleEditPrompt(e: CustomEvent<{parts: PromptPart[]}>) {
    this.setEditorParts(e.detail.parts);
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.isComposing) return;
    if (e.key === 'j' && e.ctrlKey) {
      // Ctrl+J for a newline (like shift + enter) without sending the message.
      // because CLI apps use Ctrl+J to insert newlines.
      e.preventDefault();
      document.execCommand('insertLineBreak');
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      // Shift+Enter for a newline, Enter to send the message.
      e.preventDefault();
      this.triggerSendMessage();
    }
  }

  private triggerSendMessage() {
    if (!this.chatInput || this.state.chatSession.isGenerating) return;
    const parts = this.serializeEditor();
    if (!promptHasContent(parts)) return;

    this.chatInput.replaceChildren();
    this.savedRange = null;
    this.updateEmptyState();

    // Keep plain-text prompts as strings.
    const isTextOnly = parts.every(p => p.type === 'text');
    void this.state.chatSession.sendMessage(
        isTextOnly ? promptPartsToText(parts) : parts);
  }

  private useStarter(text: string) {
    this.setEditorParts([{type: 'text', text}]);
  }

  private renderCameraModal() {
    if (!this.isCameraModalOpen) return '';
    return html`
      <div class="modal-backdrop" @click=${(e: Event) => {
      if (e.target === e.currentTarget) this.closeCamera();
    }}>
        <div class="modal-card">
          <div class="modal-header">
            <span>Take a photo</span>
            <button class="btn-icon" title="Close" @click=${this.closeCamera}>
              ✕
            </button>
          </div>
          <div class="camera-preview-container">
            <video
              id="camera-video"
              class="camera-stream"
              autoplay
              playsinline
            ></video>
          </div>
          <div class="modal-actions">
            <button class="btn btn-secondary" @click=${this.closeCamera}>
              Cancel
            </button>
            <button class="btn btn-primary" @click=${this.takeSnapshot}>
              📸 Take photo
            </button>
          </div>
        </div>
      </div>
    `;
  }

  override render() {
    const isMessagesEmpty = this.state.chatSession.messages.length === 0;
    const activeModelFilename =
        this.state.settings.selectedModelPath.split('/').pop() || '';
    const isCached =
        this.state.modelLoader.cachedModels.has(activeModelFilename);
    const isLoading = this.state.modelLoader.isModelLoading;

    return html`
      <!-- Scrollable Messages Box -->
      <div id="chat-messages" class="chat-messages">

        <!-- Active bubbles mapping -->
        ${this.state.chatSession.messages.map((msg, idx) => html`
          <litert-chat-bubble
            .message=${msg}
            .index=${idx}
            .state=${this.state}
            @edit-prompt=${this.handleEditPrompt}
          ></litert-chat-bubble>
        `)}
      </div>

      <!-- Quick Starters suggestions container (only visible if history is empty) -->
      ${
        isMessagesEmpty ?
            html`
        <div class="starters-container">
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Explain WebGPU in simple terms.')}>Explain WebGPU</button>
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Write a cool particle effect in Javascript in a single HTML page.')}>Particle Effect</button>
          <button class="btn-starter" @click=${
                () => this.useStarter(
                    'Create a modern dark-themed HTML button with hovering glow transitions.')}>Glow Button HTML</button>
        </div>
      ` :
            ''}


      <!-- Input Area -->
      <div class="chat-input-container">
        <input
          type="file"
          id="image-input"
          accept="image/*"
          multiple
          hidden
          @change=${this.handleImageInputChange}
        />
        <input
          type="file"
          id="camera-fallback-input"
          accept="image/*"
          capture="environment"
          hidden
          @change=${this.handleImageInputChange}
        />
        <div
          class="input-wrapper ${this.isDragOver ? 'drag-over' : ''}"
          @dragover=${this.handleDragOver}
          @dragleave=${this.handleDragLeave}
          @drop=${this.handleDrop}
        >
          <button
            id="btn-add-image"
            class="btn-icon plus"
            title="Add images"
            ?disabled=${isLoading}
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${() => this.imageInput?.click()}
          >+</button>
          <button
            id="btn-camera"
            class="btn-icon"
            title="Take a photo"
            ?disabled=${isLoading}
            @mousedown=${(e: Event) => e.preventDefault()}
            @click=${this.openCamera}
          >📷</button>

          <!--
            Lit intentionally renders no children here; setEditorParts and the
            browser own the contents so the caret is never clobbered by a
            re-render.
          -->
          <div
            id="chat-input"
            class="editor"
            contenteditable=${isLoading ? 'false' : 'true'}
            role="textbox"
            aria-multiline="true"
            aria-label="Message"
            aria-disabled=${isLoading ? 'true' : 'false'}
            data-placeholder=${
        isLoading ? 'Model is preparing, please wait...' :
                    'Message LiteRT-LM... Add images anywhere in your message.'}
            @input=${this.updateEmptyState}
            @keyup=${this.handleEditorSelect}
            @mouseup=${this.handleEditorSelect}
            @blur=${this.handleEditorSelect}
            @keydown=${this.handleKeyDown}
            @paste=${this.handleEditorPaste}
          ></div>

          <!-- Dynamic Send / Stop action button -->
          ${
        this.state.chatSession.isGenerating ? html`
            <button
              class="btn btn-stop btn-input"
              @click=${() => this.state.chatSession.cancelGeneration()}
            >
              Stop
            </button>
          ` :
                                              html`
            <button
              id="btn-send"
              class="btn btn-primary btn-input"
              ?disabled=${isLoading}
              @click=${this.triggerSendMessage}
            >
              ${isCached ? 'Send' : 'Download & Send'}
            </button>
          `}
        </div>
      </div>

      ${this.renderCameraModal()}
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-window': LitertChatWindow;
  }
}
