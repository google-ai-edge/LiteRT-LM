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

import './camera_capture';
import './chat_bubble';
import './custom_dropdown';

import {SUPPORTED_IMAGE_MIME_TYPES} from '@litert-lm/core';
import {html, LitElement, nothing, render, svg} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {MediaKind, promptHasContent, PromptMedia, PromptPart, promptPartsToText} from '../stores/chat_session_store.js';
import {getModelKey} from '../stores/model_loader_service.js';
import {getModalityConfigs} from '../stores/settings_store.js';
import {sharedStyles} from '../styles/shared_styles.js';
import {AudioRecorder, toSupportedAudio} from '../utils/audio.js';
import type {LitertCameraCapture} from './camera_capture.js';
import {chatWindowStyles} from './chat_window_styles.js';
import type {CustomDropdown} from './custom_dropdown.js';
import {focusNowAndNextFrame} from './focus_util.js';
import {addRemoteModelFromInput, getActiveModelSummary, modelSelectorStyles, renderModelDropdownItems, selectModelWithLargeModelCheck, triggerLitertModelUpload} from './model_selector_shared.js';

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

/** Example prompts shown in the empty state. */
const STARTERS = [
  {
    title: 'Explain WebGPU',
    subtitle: 'How browser GPU shaders run LLMs locally',
    prompt: 'Explain WebGPU in simple terms.',
    icon: svg`<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>`,
  },
  {
    title: 'Particle Effect',
    subtitle: 'Interactive HTML5 canvas animation snippet',
    prompt: 'Write a cool particle effect in Javascript in a single HTML page.',
    icon: svg`<circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.6.77 1.02 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>`,
  },
  {
    title: 'Glow Button HTML',
    subtitle: 'Dark-mode CSS button with live preview',
    prompt: 'Create a modern dark-themed HTML button with hovering glow transitions.',
    icon: svg`<polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline>`,
  },
];

/** Main chat window component managing messages and input. */
@customElement('litert-chat-window')
export class LitertChatWindow extends LitElement {
  @property({ type: Object })
  state!: LlmChatStateController;

  @property({type: String})
  theme: 'dark'|'light' = 'dark';

  @property({type: Boolean})
  isOffline = false;

  @state()
  private shouldAutoScroll = true;

  @state() private hasScrollTop = false;

  @state() private hasPromptContent = false;

  @state() private isDragOver = false;
  @state() private isCameraModalOpen = false;
  @state() private isModelMenuOpen = false;
  @state() private isAddingModel = false;
  @state() private isLookingUpModel = false;
  @state()
  private isCompactViewport =
      typeof window !== 'undefined' ? window.innerWidth <= 600 : false;
  private readonly compactMediaQuery = typeof window !== 'undefined' ?
      window.matchMedia('(max-width: 600px)') :
      null;
  private containerResizeObserver: ResizeObserver|null = null;
  private readonly syncCompactContainerState = () => {
    const width = this.clientWidth;
    if (width > 0) {
      this.isCompactViewport = width <= 600;
    } else if (typeof window !== 'undefined') {
      this.isCompactViewport = window.innerWidth <= 600;
    }
  };
  private readonly handleCompactMediaChange = () => {
    this.syncCompactContainerState();
  };

  private readonly audioRecorder = new AudioRecorder();
  @state() private isRecording = false;
  @state() private recordingSeconds = 0;
  private recordingTimer: number|undefined;

  /** Audio element for previewing an audio chip, if one is playing. */
  private chipAudio: HTMLAudioElement|null = null;

  private wasGenerating = false;
  private pendingScrollOnConvSwitch = false;
  private generationJustEnded = false;
  private restoreFocusAfterStop = false;
  private lastScrollTop = 0;
  private lastUserScrollUpAt = 0;
  private touchStartY: number|null = null;
  private lastScrolledConvKey = '';

  /** The bubble being streamed into; its height only grows while streaming. */
  private streamingBubble: HTMLElement|null = null;

  /** Last known caret position, so toolbar actions can insert in place. */
  private savedRange: Range|null = null;

  @query('#chat-messages') private msgBox!: HTMLDivElement;

  @query('#chat-input') private chatInput!: HTMLDivElement;
  @query('#composer-model-dropdown')
  private modelDropdown?: CustomDropdown;
  @query('#media-input') private mediaInput?: HTMLInputElement;
  @query('#camera-fallback-input')
  private cameraFallbackInput?: HTMLInputElement;
  @query('litert-camera-capture') private camera?: LitertCameraCapture;

  static override styles =
      [sharedStyles, modelSelectorStyles, chatWindowStyles];

  private readonly handleGlobalScrollKeyDown = (e: KeyboardEvent) => {
    if (!this.shouldAutoScroll || !this.msgBox) return;
    const isUpScrollKey = e.key === 'PageUp' || e.key === 'ArrowUp' ||
        e.key === 'Home' || (e.key === ' ' && e.shiftKey);
    if (!isUpScrollKey) return;
    for (const node of e.composedPath()) {
      if (!(node instanceof HTMLElement)) continue;
      const tag = node.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
          node.isContentEditable) {
        return;
      }
    }
    if (this.msgBox.scrollHeight > this.msgBox.clientHeight + 1) {
      this.lastUserScrollUpAt = performance.now();
      this.shouldAutoScroll = false;
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    // Register this chat window as a host of our centralized state
    this.state.addHost(this);
    this.syncCompactContainerState();
    this.compactMediaQuery?.addEventListener(
        'change', this.handleCompactMediaChange);
    if (typeof ResizeObserver !== 'undefined') {
      this.containerResizeObserver = new ResizeObserver(() => {
        this.syncCompactContainerState();
      });
      this.containerResizeObserver.observe(this);
    }
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', this.handleGlobalScrollKeyDown);
    }
  }

  override disconnectedCallback() {
    document.removeEventListener('click', this.handleAddModelOutsideClick);
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', this.handleGlobalScrollKeyDown);
    }
    this.containerResizeObserver?.disconnect();
    this.containerResizeObserver = null;
    this.compactMediaQuery?.removeEventListener(
        'change', this.handleCompactMediaChange);
    this.cancelRecording();
    this.chipAudio?.pause();
    this.chipAudio = null;
    this.state.removeHost(this);
    super.disconnectedCallback();
  }

  private hasInnerScrollableUpTarget(e: WheelEvent): boolean {
    for (const node of e.composedPath()) {
      if (node === this.msgBox) break;
      if (!(node instanceof HTMLElement)) continue;
      if (node.scrollHeight > node.clientHeight + 1 && node.scrollTop > 0) {
        const overflowY = window.getComputedStyle(node).overflowY;
        if (overflowY === 'auto' || overflowY === 'scroll') {
          return true;
        }
      }
    }
    return false;
  }

  private handleMessagesWheel(e: WheelEvent) {
    if (!this.msgBox || e.deltaY >= 0) return;
    if (this.hasInnerScrollableUpTarget(e)) return;
    if (this.msgBox.scrollHeight > this.msgBox.clientHeight + 1 ||
        this.msgBox.scrollTop > 0) {
      this.lastUserScrollUpAt = performance.now();
      if (this.shouldAutoScroll) {
        this.shouldAutoScroll = false;
      }
    }
  }

  private handleMessagesTouchStart(e: TouchEvent) {
    this.touchStartY = e.touches[0]?.clientY ?? null;
  }

  private handleMessagesTouchMove(e: TouchEvent) {
    if (this.touchStartY === null || !this.msgBox) return;
    const currentY = e.touches[0]?.clientY;
    if (currentY === undefined) return;
    if (currentY - this.touchStartY > 3 &&
        (this.msgBox.scrollHeight > this.msgBox.clientHeight + 1 ||
         this.msgBox.scrollTop > 0)) {
      this.lastUserScrollUpAt = performance.now();
      if (this.shouldAutoScroll) {
        this.shouldAutoScroll = false;
      }
    }
  }

  private readonly handleMessagesScroll = () => {
    if (!this.msgBox) return;
    const currentTop = this.msgBox.scrollTop;
    const nextHasScrollTop = currentTop > 8;
    if (this.hasScrollTop !== nextHasScrollTop) {
      this.hasScrollTop = nextHasScrollTop;
    }

    const distFromBottom =
        this.msgBox.scrollHeight - currentTop - this.msgBox.clientHeight;
    const deltaTop = currentTop - this.lastScrollTop;
    this.lastScrollTop = currentTop;

    // Any upward scroll movement away from bottom immediately releases
    // auto-follow (e.g. scrollbar thumb drag, trackpad momentum, keyboard up).
    if (deltaTop < -0.5 && distFromBottom > 1) {
      this.lastUserScrollUpAt = performance.now();
      if (this.shouldAutoScroll) {
        this.shouldAutoScroll = false;
      }
      return;
    }

    // Re-pin ONLY when auto-scroll was off, the user actively scrolled DOWN,
    // outside trackpad bounce window, and genuinely reached the bottom.
    if (!this.shouldAutoScroll && deltaTop > 0 &&
        performance.now() - this.lastUserScrollUpAt > 180 &&
        distFromBottom <= 12) {
      this.shouldAutoScroll = true;
    }
  };

  override firstUpdated() {
    this.updateEmptyState();
    this.msgBox?.addEventListener('scroll', this.handleMessagesScroll);
  }

  private scrollMessagesToBottomInstant() {
    if (!this.msgBox || !this.shouldAutoScroll) return;
    this.msgBox.scrollTop = this.msgBox.scrollHeight;
    this.lastScrollTop = this.msgBox.scrollTop;
    const nextHasScrollTop = this.msgBox.scrollTop > 8;
    if (this.hasScrollTop !== nextHasScrollTop) {
      this.hasScrollTop = nextHasScrollTop;
    }
  }

  protected override willUpdate() {
    if (this.state?.settings) {
      const modalities = getModalityConfigs(this.state.settings);
      if (modalities.audio.backend === 'off' && this.isRecording) {
        this.cancelRecording();
      }
      if (modalities.vision.backend === 'off' && this.isCameraModalOpen) {
        this.camera?.close();
      }
    }
    if (!this.state?.chatSession) return;
    const isGenerating = this.state.chatSession.isGenerating;
    const messageCount = this.state.chatSession.messages.length;
    const convKey = `${this.state.chatSession.activeSavedConvId ?? 'new'}:${
        messageCount > 0 ? 'nonempty' : 'empty'}`;

    if (!isGenerating && messageCount > 0 &&
        convKey !== this.lastScrolledConvKey) {
      this.lastScrolledConvKey = convKey;
      this.shouldAutoScroll = true;
      this.lastUserScrollUpAt = 0;
      this.pendingScrollOnConvSwitch = true;
    } else if (messageCount === 0) {
      this.lastScrolledConvKey = convKey;
      this.shouldAutoScroll = true;
      this.hasScrollTop = false;
      this.lastUserScrollUpAt = 0;
    }

    if (isGenerating && !this.wasGenerating) {
      this.shouldAutoScroll = true;
      this.lastUserScrollUpAt = 0;
    } else if (!isGenerating && this.wasGenerating) {
      this.generationJustEnded = true;
    }
    this.wasGenerating = isGenerating;
  }

  protected override updated() {
    if (this.isModelMenuOpen) {
      this.modelDropdown?.requestUpdate();
    }
    if (!this.state?.chatSession) return;
    const isGenerating = this.state.chatSession.isGenerating;

    if (this.pendingScrollOnConvSwitch) {
      this.pendingScrollOnConvSwitch = false;
      requestAnimationFrame(() => {
        this.scrollMessagesToBottomInstant();
      });
    }

    if (this.generationJustEnded) {
      this.generationJustEnded = false;
      this.releaseStreamingHeight();
      if (this.restoreFocusAfterStop) {
        this.restoreFocusAfterStop = false;
        this.chatInput?.focus();
      }
      if (this.shouldAutoScroll) {
        void this.syncScrollAfterStreamEnd();
      }
    }

    if (isGenerating) {
      // Bubbles re-render in a microtask after this, so measure next frame.
      requestAnimationFrame(() => {
        // A stale frame may land after generation ended and the hold was
        // released; don't re-pin the height then.
        if (!this.msgBox || !this.state.chatSession.isGenerating) return;
        this.holdStreamingHeight();
        if (this.shouldAutoScroll) {
          this.scrollMessagesToBottomInstant();
        }
      });
    }
  }

  private async syncScrollAfterStreamEnd() {
    const bubbles = this.msgBox?.querySelectorAll('litert-chat-bubble');
    const lastBubble =
        (bubbles?.[bubbles.length - 1] as LitElement | undefined) ?? null;
    await lastBubble?.updateComplete;
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    if (!this.msgBox || !this.shouldAutoScroll) return;
    this.scrollMessagesToBottomInstant();
    requestAnimationFrame(() => {
      if (!this.msgBox || !this.shouldAutoScroll) return;
      this.scrollMessagesToBottomInstant();
    });
  }

  /**
   * Re-rendering partial markdown can shrink the streaming bubble (e.g. when a
   * code fence closes or rows collapse into a table). Pinning its min-height to
   * its tallest height so far keeps the scroll height from shrinking
   * mid-stream, so auto-scroll moves monotonically down instead of jittering
   * upwards.
   */
  private holdStreamingHeight() {
    const bubbles = this.msgBox.querySelectorAll('litert-chat-bubble');
    const bubble =
        (bubbles[bubbles.length - 1] as HTMLElement | undefined) ?? null;
    if (!bubble) return;
    if (bubble !== this.streamingBubble) {
      this.releaseStreamingHeight();
      this.streamingBubble = bubble;
    }
    // offsetHeight already includes any previous min-height, so this is a max.
    bubble.style.minHeight = `${bubble.offsetHeight}px`;
  }

  private scrollToBottom() {
    if (!this.msgBox) return;
    this.shouldAutoScroll = true;
    this.lastUserScrollUpAt = 0;
    if (this.state?.chatSession?.isGenerating) {
      this.scrollMessagesToBottomInstant();
    } else {
      this.msgBox.scrollTo({top: this.msgBox.scrollHeight, behavior: 'smooth'});
    }
  }

  /** Lets the streaming bubble shrink to its natural height again. */
  private releaseStreamingHeight() {
    this.streamingBubble?.style.removeProperty('min-height');
    this.streamingBubble = null;
  }

  /** Replaces the editor's contents and moves the caret to the end. */
  private setEditorParts(parts: PromptPart[]) {
    const editor = this.chatInput;
    if (!editor) return;
    editor.replaceChildren();
    for (const part of parts) {
      editor.appendChild(
          part.type === 'text' ? document.createTextNode(part.text) :
                                 this.createMediaChip(part.media));
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

        const mediaId = child.dataset['mediaId'];
        const media =
            mediaId ? this.state.chatSession.getMedia(mediaId) : undefined;
        if (media) {
          parts.push({type: 'media', media});
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
    const rawText = editor.textContent ?? '';
    const hasChip = editor.querySelector('.media-chip') !== null;
    editor.classList.toggle('is-empty', rawText === '' && !hasChip);
    const nextHasContent =
        hasChip || rawText.replace(/\u00a0/g, ' ').trim().length > 0;
    if (this.hasPromptContent !== nextHasContent) {
      this.hasPromptContent = nextHasContent;
    }
  }

  /** Plays or stops the audio clip of a chip. */
  private toggleChipAudio(media: PromptMedia, button: HTMLButtonElement) {
    const wasPlayingThis =
        this.chipAudio?.src === media.url && !this.chipAudio.paused;
    this.chipAudio?.pause();
    this.chipAudio = null;
    for (const b of Array.from(
             this.chatInput?.querySelectorAll('.chip-play-btn') ?? [])) {
      b.textContent = '▶';
    }
    if (wasPlayingThis) return;

    const audio = new Audio(media.url);
    audio.addEventListener('ended', () => {
      button.textContent = '▶';
    });
    this.chipAudio = audio;
    button.textContent = '⏸';
    void audio.play();
  }

  private createMediaChip(media: PromptMedia): HTMLElement {
    const span = document.createElement('span');
    span.className = 'media-chip';
    // Makes the chip atomic: the caret steps over it and Backspace deletes
    // the whole chip rather than editing its innards.
    span.contentEditable = 'false';
    span.dataset['mediaId'] = media.id;
    const preview = media.kind === 'image' ?
        html`<img class="chip-thumb" src=${media.url} alt=${media.name} />` :
        html`<button
          class="chip-play-btn"
          title="Play audio"
          @mousedown=${(e: Event) => e.preventDefault()}
          @click=${(e: Event) => {
          e.stopPropagation();
          this.toggleChipAudio(media, e.currentTarget as HTMLButtonElement);
        }}
        >▶</button>`;
    render(
        html`
          ${preview}
          <span class="chip-name" title=${media.name}>${media.name}</span>
          <button
            class="chip-del-btn"
            title="Remove ${media.kind}"
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
   * Inserts media where the caret is, so the user controls where it sits
   * relative to the surrounding text. Falls back to appending.
   */
  private insertMediaAtCaret(media: PromptMedia) {
    const editor = this.chatInput;
    if (!editor) return;

    const node = this.createMediaChip(media);
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

  /** Inserts pasted media as chips and pasted rich text as plain text. */
  private handleEditorPaste(e: ClipboardEvent) {
    e.preventDefault();
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      this.handleEditorSelect();
      void this.addMediaFiles(Array.from(files));
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
      void this.addMediaFiles(Array.from(files));
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

  private handleMediaInputChange(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      void this.addMediaFiles(Array.from(input.files));
      input.value = '';
    }
  }

  private setStatus(msg: string) {
    this.state.statusText = msg;
    this.state.requestUpdate();
  }

  /** Adds image and audio files to the prompt at the saved caret position. */
  private async addMediaFiles(files: File[]) {
    const modalities = getModalityConfigs(this.state.settings);
    const visionOff = modalities.vision.backend === 'off';
    const audioOff = modalities.audio.backend === 'off';
    for (const file of files) {
      let kind: MediaKind;
      if (file.type.startsWith('image/')) {
        if (visionOff) {
          this.setStatus('Vision is turned off in Experimental settings.');
          continue;
        }
        kind = 'image';
      } else if (file.type.startsWith('audio/')) {
        if (audioOff) {
          this.setStatus('Audio is turned off in Experimental settings.');
          continue;
        }
        kind = 'audio';
      } else {
        this.setStatus(
            `Only images and audio can be added to a prompt: ${file.name}`);
        continue;
      }
      try {
        const blob = kind === 'image' ? await toSupportedImage(file) :
                                        await toSupportedAudio(file);
        this.insertMediaAtCaret(
            this.state.chatSession.createMedia(kind, file.name, blob));
      } catch (e) {
        console.error(`[LiteRT-LM] Failed to read ${kind}:`, e);
        this.setStatus(`Could not read ${kind} ${file.name}.`);
      }
    }
  }

  private async toggleRecording() {
    if (this.isRecording) {
      await this.finishRecording();
      return;
    }
    if (getModalityConfigs(this.state.settings).audio.backend === 'off') {
      this.setStatus('Audio is turned off in Experimental settings.');
      return;
    }
    try {
      await this.audioRecorder.start();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to start recording:', e);
      this.setStatus('Could not access the microphone.');
      return;
    }
    this.isRecording = true;
    this.recordingSeconds = 0;
    this.recordingTimer = window.setInterval(() => {
      this.recordingSeconds = Math.floor(this.audioRecorder.durationSeconds);
    }, 250);
  }

  private stopRecordingTimer() {
    window.clearInterval(this.recordingTimer);
    this.recordingTimer = undefined;
    this.isRecording = false;
  }

  /** Stops recording and adds the recording to the prompt. */
  private async finishRecording() {
    this.stopRecordingTimer();
    try {
      const blob = await this.audioRecorder.stop();
      const name = `Recording ${new Date().toLocaleTimeString()}.wav`;
      this.insertMediaAtCaret(
          this.state.chatSession.createMedia('audio', name, blob));
    } catch (e) {
      console.error('[LiteRT-LM] Failed to record audio:', e);
      this.setStatus('Could not record audio.');
    }
  }

  private cancelRecording() {
    if (!this.isRecording) return;
    this.stopRecordingTimer();
    this.audioRecorder.cancel();
  }

  /** Closes the camera modal if open, returning true when handled. */
  dismissCameraModal(): boolean {
    if (!this.isCameraModalOpen) return false;
    this.camera?.close();
    return true;
  }

  private async openCamera() {
    if (getModalityConfigs(this.state.settings).vision.backend === 'off') {
      this.setStatus('Vision is turned off in Experimental settings.');
      return;
    }
    if (!(await this.camera?.open())) {
      this.cameraFallbackInput?.click();
    }
  }

  private handleCameraOpenChange(e: CustomEvent<{open: boolean}>) {
    this.isCameraModalOpen = e.detail.open;
    this.dispatchEvent(new CustomEvent('camera-open-change', {
      detail: e.detail,
      bubbles: true,
      composed: true,
    }));
    if (!e.detail.open && this.isConnected) {
      void this.updateComplete.then(() => {
        this.renderRoot.querySelector<HTMLElement>('#btn-camera')?.focus();
      });
    }
  }

  private handlePhotoCaptured(e: CustomEvent<{photo: Blob}>) {
    const name = `Camera photo ${new Date().toLocaleTimeString()}.jpg`;
    this.insertMediaAtCaret(
        this.state.chatSession.createMedia('image', name, e.detail.photo));
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

  /** Whether the selected model's weights can load without a download. */
  private getSelectedModelWeights() {
    const selectedPath = this.state.settings.selectedModelPath || '';
    const isCached = Boolean(
        this.state.modelLoader.cachedModels?.has(getModelKey(selectedPath)));
    const isLocalDir = selectedPath.startsWith('local-dir://');
    const isLocalUploaded = selectedPath.startsWith('https://local-model/');
    const isCurrentlyLoaded = Boolean(
        this.state.modelLoader.engine &&
        !this.state.modelLoader.needsModelReload?.(
            this.state.settings.modelSettings));
    const isReady = isCurrentlyLoaded || isCached || isLocalUploaded;
    return {isLocalDir, isReady, isAvailable: isReady || isLocalDir};
  }

  private isSelectedModelOfflineUncached(): boolean {
    return this.isOffline && !this.state.modelLoader.isModelLoading &&
        !this.getSelectedModelWeights().isAvailable;
  }

  private triggerSendMessage() {
    if (!this.chatInput || this.state.chatSession.isGenerating ||
        this.state.modelLoader.isModelLoading ||
        this.isSelectedModelOfflineUncached()) {
      return;
    }
    const parts = this.serializeEditor();
    if (!promptHasContent(parts)) return;

    const modalities = getModalityConfigs(this.state.settings);
    const hasDisabledImage = modalities.vision.backend === 'off' &&
        parts.some(p => p.type === 'media' && p.media.kind === 'image');
    const hasDisabledAudio = modalities.audio.backend === 'off' &&
        parts.some(p => p.type === 'media' && p.media.kind === 'audio');
    if (hasDisabledImage || hasDisabledAudio) {
      const which = hasDisabledImage && hasDisabledAudio ?
          'Vision and Audio are' :
          (hasDisabledImage ? 'Vision is' : 'Audio is');
      this.setStatus(
          `${which} turned off in Experimental settings — remove the attachment or re-enable it before sending.`);
      return;
    }

    this.chatInput.replaceChildren();
    this.savedRange = null;
    this.updateEmptyState();

    // Keep plain-text prompts as strings.
    const isTextOnly = parts.every(p => p.type === 'text');
    void this.state.chatSession.sendMessage(
        isTextOnly ? promptPartsToText(parts) : parts);
  }

  private handleStopGeneration() {
    this.restoreFocusAfterStop = true;
    this.state.chatSession.cancelGeneration();
    this.chatInput?.focus();
    void this.updateComplete.then(() => {
      if (this.restoreFocusAfterStop) {
        this.chatInput?.focus();
      }
    });
  }

  private async handleRetryFromAlert() {
    if (this.state.modelLoader.isModelLoading ||
        this.state.chatSession.isGenerating) {
      return;
    }
    this.state.dismissError?.();
    const msgs = this.state.chatSession.messages;
    const lastIdx = msgs.length - 1;
    if (lastIdx >= 0 && msgs[lastIdx]?.role === 'user' &&
        this.state.chatSession.rewindAndEdit) {
      const parts = await this.state.chatSession.rewindAndEdit(lastIdx);
      if (parts && promptHasContent(parts)) {
        const isTextOnly = parts.every(p => p.type === 'text');
        await this.state.chatSession.sendMessage(
            isTextOnly ? promptPartsToText(parts) : parts);
        return;
      }
    } else if (
        lastIdx >= 0 && msgs[lastIdx]?.role === 'assistant' &&
        this.state.chatSession.redoResponse) {
      await this.state.chatSession.redoResponse(lastIdx);
      return;
    }
    await this.state.modelLoader.loadModelWeights(
        this.state.settings.modelSettings, async () => {
          await this.state.chatSession.createConversationSession?.();
        });
  }

  /** Opens and focuses the unified model picker in the composer. */
  openModelPicker() {
    if (this.isAddingModel) {
      this.closeComposerAddModel(false);
    }
    this.modelDropdown?.open();
    this.modelDropdown?.focusButton();
  }

  /** Focuses the prompt composer input (or active composer control if loading). */
  focusInput() {
    const editor = this.chatInput;
    if (editor && editor.getAttribute('contenteditable') !== 'false') {
      editor.focus();
      return;
    }
    const fallback = this.renderRoot.querySelector<HTMLElement>(
        '#btn-composer-cancel-download, #btn-composer-model');
    fallback?.focus();
  }

  /**
   * Dismisses any open model picker menu or Add-URL popover in the composer,
   * returning true when an open control was closed.
   */
  dismissTransientMenus(): boolean {
    if (this.modelDropdown?.isOpen) {
      this.modelDropdown.close();
      this.modelDropdown.focusButton();
      return true;
    }
    if (this.isAddingModel) {
      this.closeComposerAddModel(true);
      return true;
    }
    return false;
  }

  /** Closes open composer popovers without stealing focus. */
  closeTransientMenus() {
    this.modelDropdown?.close();
    if (this.isAddingModel) {
      this.closeComposerAddModel(false);
    }
  }

  private readonly handleAddModelOutsideClick = (e: MouseEvent) => {
    if (!this.isAddingModel) return;
    const anchor =
        this.renderRoot.querySelector<HTMLElement>('.composer-model-anchor');
    const popover = this.renderRoot.querySelector<HTMLElement>(
        '#composer-add-model-popover');
    const path = typeof e.composedPath === 'function' ? e.composedPath() : [];
    const target = e.target as Node | null;
    const insideAnchor = Boolean(
        anchor &&
        (path.includes(anchor) || (target && anchor.contains(target))));
    const insidePopover = Boolean(
        popover &&
        (path.includes(popover) || (target && popover.contains(target))));
    if (!insideAnchor && !insidePopover) {
      this.closeComposerAddModel(false);
    }
  };

  private openComposerAddModel() {
    this.isAddingModel = true;
    setTimeout(() => {
      if (this.isConnected && this.isAddingModel) {
        document.addEventListener('click', this.handleAddModelOutsideClick);
      }
    }, 0);
    void this.updateComplete.then(() => {
      const focusInput = () => {
        if (!this.isAddingModel) return;
        this.renderRoot
            .querySelector<HTMLInputElement>('#composer-add-model-input')
            ?.focus();
      };
      focusNowAndNextFrame(focusInput);
    });
  }

  private closeComposerAddModel(restoreFocus = false) {
    this.isAddingModel = false;
    document.removeEventListener('click', this.handleAddModelOutsideClick);
    if (restoreFocus) {
      void this.updateComplete.then(() => {
        this.modelDropdown?.focusButton();
      });
    }
  }

  private async handleComposerModelChange(e: CustomEvent<string>) {
    const path = e.detail;
    const dropdown = this.modelDropdown;
    if (path !== 'add-url' && this.isAddingModel) {
      this.closeComposerAddModel(false);
    }
    if (path === 'upload' || path === 'select-dir' || path === 'add-url') {
      if (dropdown) {
        dropdown.value = this.state.settings.selectedModelPath;
      }
      if (path === 'upload') {
        triggerLitertModelUpload(this.state, () => {
          this.requestUpdate();
        });
        return;
      }
      if (path === 'select-dir') {
        void this.state.localDirService.mountDirectory();
        return;
      }
      this.openComposerAddModel();
      return;
    }

    const selected = await selectModelWithLargeModelCheck(this.state, path);
    if (!selected && dropdown) {
      dropdown.value = this.state.settings.selectedModelPath;
    }
    this.requestUpdate();
  }

  private async handleComposerAddModel(e: SubmitEvent) {
    e.preventDefault();
    const input = this.renderRoot.querySelector<HTMLInputElement>(
        '#composer-add-model-input');
    const text = input?.value.trim() ?? '';
    if (!text || this.isLookingUpModel ||
        this.state.modelLoader.isModelLoading) {
      return;
    }

    this.isLookingUpModel = true;
    try {
      const added = await addRemoteModelFromInput(this.state, text, () => {
        this.closeComposerAddModel(true);
        this.requestUpdate();
      });
      if (!added) return;
    } finally {
      this.isLookingUpModel = false;
    }
  }

  private handleChooseModelFromAlert() {
    const event = new CustomEvent('choose-model', {
      bubbles: true,
      composed: true,
      cancelable: true,
    });
    this.dispatchEvent(event);
    if (!event.defaultPrevented) {
      this.openModelPicker();
    }
  }

  private useStarter(text: string) {
    if (this.state.modelLoader.isModelLoading ||
        this.state.chatSession.isGenerating ||
        this.isSelectedModelOfflineUncached()) {
      return;
    }
    this.setEditorParts([{type: 'text', text}]);
    requestAnimationFrame(() => {
      const editor = this.chatInput;
      if (!editor) return;
      const end = document.createRange();
      end.selectNodeContents(editor);
      end.collapse(false);
      this.savedRange = end.cloneRange();
      this.applyRange(end);
    });
  }

  override render() {
    const isMessagesEmpty = this.state.chatSession.messages.length === 0;
    const selectedPath = this.state.settings.selectedModelPath || '';
    const activeModelKey = getModelKey(selectedPath);
    const {isLocalDir, isReady, isAvailable} = this.getSelectedModelWeights();
    const isLoading = this.state.modelLoader.isModelLoading;
    const isGenerating = this.state.chatSession.isGenerating;
    const downloadProgress =
        this.state.modelLoader.downloadProgresses?.get(activeModelKey);
    const downloadSpeed =
        this.state.modelLoader.downloadSpeeds?.get(activeModelKey);
    const isOfflineUncached = this.isOffline && !isLoading && !isAvailable;
    const needsWeightDownload = !isLoading && !isAvailable && !isOfflineUncached;
    const sendLabel = isLoading ?
        (downloadProgress !== undefined ? `Loading ${downloadProgress}%…` :
                                          'Loading…') :
        isOfflineUncached ? 'Offline · Not cached' :
        isReady           ? 'Send' :
        isLocalDir        ? 'Load & Send' :
                            'Download & Send';
    const offlineSendTitle =
        'Offline — connect to download weights, or choose a cached or local model';
    const modelSummary = getActiveModelSummary(this.state);
    const cleanModelName = modelSummary.name;
    const isAddModelDisabled = this.isLookingUpModel || isLoading;
    const showScrollBottom = !this.shouldAutoScroll && !isMessagesEmpty;
    const modalities = getModalityConfigs(this.state.settings);
    const visionOff = modalities.vision.backend === 'off';
    const audioOff = modalities.audio.backend === 'off';
    const allMediaOff = visionOff && audioOff;
    const mediaAccept = allMediaOff ? '' :
        visionOff                   ? 'audio/*' :
        audioOff                    ? 'image/*' :
                                      'image/*,audio/*';
    const addMediaTitle = allMediaOff ?
        'Vision and Audio are turned off in Experimental settings' :
        visionOff ? 'Add audio (Vision is turned off in Experimental settings)' :
        audioOff ? 'Add images (Audio is turned off in Experimental settings)' :
                   'Add images or audio';
    const addMediaAriaLabel = allMediaOff ?
        'Add images or audio (disabled: Vision and Audio are turned off in Experimental settings)' :
        addMediaTitle;
    const cameraTitle = visionOff ?
        'Vision is turned off in Experimental settings' :
        'Take a photo';
    const cameraAriaLabel = visionOff ?
        'Take a photo (disabled: Vision is turned off in Experimental settings)' :
        'Take a photo';
    const recordTitle = audioOff ?
        'Audio is turned off in Experimental settings' :
        (this.isRecording ? 'Stop and attach recording' : 'Record audio');
    const recordAriaLabel = audioOff ?
        'Record audio (disabled: Audio is turned off in Experimental settings)' :
        recordTitle;
    const editorPlaceholder = isLoading ?
        'Model is preparing, please wait...' :
        (this.isCompactViewport || allMediaOff ?
             'Message LiteRT-LM...' :
             visionOff ? 'Message LiteRT-LM... Drop or attach audio.' :
             audioOff  ? 'Message LiteRT-LM... Drop or attach images.' :
                         'Message LiteRT-LM... Drop or attach images & audio.');
    const capabilityModalityLabel = allMediaOff ? 'Text-Only Mode' :
        visionOff                               ? 'Audio' :
        audioOff                                ? 'Vision' :
                                                  'Vision & Audio';
    const capabilityModalityIcon = allMediaOff ?
        svg`<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>` :
        visionOff ?
        svg`<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line>` :
        svg`<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline>`;
    const runtimeLead =
        modalities.text.backend === 'cpu' ? 'in WASM' : 'with WebGPU';
    const heroCapabilitiesSentence = allMediaOff ?
        'Verify math in the JS sandbox and preview generated HTML with zero server round-trips.' :
        visionOff ?
        'Attach audio inline, verify math in the JS sandbox, and preview generated HTML with zero server round-trips.' :
        audioOff ?
        'Attach images inline, verify math in the JS sandbox, and preview generated HTML with zero server round-trips.' :
        'Attach images or audio inline, verify math in the JS sandbox, and preview generated HTML with zero server round-trips.';

    return html`
      <!-- Top Error Banner -->
      ${this.state.errorMessage ? html`
        <div class="banner-stack" ?inert=${this.isCameraModalOpen}>
          <div class="alert-banner" role="alert">
            <div class="alert-banner-left">
              <svg class="alert-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <div class="alert-banner-body">
                <span>${this.state.errorMessage}</span>
                <div class="alert-actions">
                  <button
                    id="btn-retry-error"
                    type="button"
                    class="btn-alert-action primary"
                    ?disabled=${isLoading || isGenerating || isOfflineUncached}
                    @click=${this.handleRetryFromAlert}
                  >Try again</button>
                  <button
                    id="btn-choose-model-error"
                    type="button"
                    class="btn-alert-action"
                    @click=${this.handleChooseModelFromAlert}
                  >Choose another model</button>
                </div>
              </div>
            </div>
            <button
              type="button"
              class="btn-dismiss-alert"
              aria-label="Dismiss error"
              title="Dismiss"
              @click=${() => this.state.dismissError?.()}
            >✕</button>
          </div>
        </div>
      ` : ''}

      <!-- Scrollable Messages Box with Subtle Top Fade Scrim -->
      <div
        class="messages-viewport ${this.hasScrollTop && !isMessagesEmpty ? 'has-scroll-top' : ''}"
        ?inert=${this.isCameraModalOpen}
      >
      <div
        id="chat-messages"
        class="chat-messages"
        role="region"
        aria-label="Chat messages"
        tabindex="0"
        @wheel=${this.handleMessagesWheel}
        @touchstart=${this.handleMessagesTouchStart}
        @touchmove=${this.handleMessagesTouchMove}
      >
        ${isMessagesEmpty ? html`
          <div class="empty-hero">
            <div class="hero-mark" aria-hidden="true">
              <img src=${this.theme === 'light' ? './assets/LiteRT_Logo_Symbol-only_RGB_Color_Blue.svg' : './assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg'} alt="" />
            </div>
            <h2 class="hero-heading">Private, on-device AI in your browser</h2>
            <p class="hero-subtitle">
              Run <strong>${cleanModelName}</strong> locally ${runtimeLead}${allMediaOff ? ' in text-only mode' : ''}. ${heroCapabilitiesSentence}
            </p>

            <ul class="capability-row" aria-label="Supported capabilities">
              <li class="capability-tag">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  ${capabilityModalityIcon}
                </svg>
                ${capabilityModalityLabel}
              </li>
              <li class="capability-tag">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <polyline points="16 18 22 12 16 6"></polyline>
                  <polyline points="8 6 2 12 8 18"></polyline>
                </svg>
                JS Tool Sandbox
              </li>
              <li class="capability-tag">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
                100% Local &amp; Cached
              </li>
            </ul>

            ${!this.hasPromptContent ? html`<div class="starters-container">
              ${STARTERS.map(({title, subtitle, prompt, icon}) => html`
                <button
                  type="button"
                  class="btn-starter"
                  title=${isOfflineUncached ? offlineSendTitle : nothing}
                  ?disabled=${isLoading || isGenerating || isOfflineUncached}
                  @click=${() => this.useStarter(prompt)}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icon}</svg>
                  <span class="starter-copy">
                    <span class="starter-title">${title}</span>
                    <span class="starter-subtitle">${subtitle}</span>
                  </span>
                </button>
              `)}
            </div>` : nothing}
          </div>
        ` : ''}

        <!-- Active bubbles mapping -->
        ${this.state.chatSession.messages.map((msg, idx) => html`
          <litert-chat-bubble
            .message=${msg}
            .index=${idx}
            .state=${this.state}
            .disabled=${isGenerating}
            .streaming=${
            isGenerating && idx === this.state.chatSession.messages.length - 1}
            @edit-prompt=${this.handleEditPrompt}
          ></litert-chat-bubble>
        `)}
      </div>
      </div>

      <!-- 2-Row Rounded Card Composer with Unified Bottom-Left Model Picker -->
      <div
        class="chat-input-container ${showScrollBottom ? 'has-scroll-above' : ''}"
        ?inert=${this.isCameraModalOpen}
      >
        ${showScrollBottom ? html`
          <button
            type="button"
            class="btn-scroll-bottom"
            aria-label="Scroll to latest message"
            @click=${this.scrollToBottom}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <polyline points="19 12 12 19 5 12"></polyline>
            </svg>
            <span>Latest</span>
          </button>
        ` : ''}
        <input
          type="file"
          id="media-input"
          accept=${mediaAccept}
          multiple
          hidden
          @change=${this.handleMediaInputChange}
        />
        <input
          type="file"
          id="camera-fallback-input"
          accept="image/*"
          capture="environment"
          hidden
          @change=${this.handleMediaInputChange}
        />
        <div
          class="input-wrapper ${this.isDragOver ? 'drag-over' : ''}"
          @dragover=${this.handleDragOver}
          @dragleave=${this.handleDragLeave}
          @drop=${this.handleDrop}
        >
          ${isLoading ? html`
            <div id="composer-loading-bar" class="composer-loading-bar" role="status" aria-live="polite">
              <div class="composer-loading-top">
                <div class="composer-loading-title">
                  <span class="spinner" aria-hidden="true"></span>
                  <span>${this.state.statusText || `Preparing ${cleanModelName}...`}</span>
                </div>
                <div class="composer-loading-right">
                  ${downloadProgress !== undefined ? html`
                    <span class="composer-loading-stats">${
                        downloadSpeed ? html`<span class="composer-loading-speed-inline">${downloadSpeed} · </span>` : nothing}${downloadProgress}%</span>
                  ` : nothing}
                  ${this.state.modelLoader.downloadAbortController ? html`
                    <button
                      id="btn-composer-cancel-download"
                      type="button"
                      class="btn-composer-cancel"
                      @click=${() => this.state.modelLoader.cancelDownload?.()}
                    >Cancel</button>
                  ` : nothing}
                </div>
              </div>
              ${downloadProgress !== undefined ? html`
                <div class="loading-progress-bar-wrap">
                  <div class="loading-progress-bar-fill" style="width: ${downloadProgress}%;"></div>
                </div>
              ` : nothing}
              ${downloadSpeed ? html`
                <div class="composer-loading-sub">${downloadSpeed}</div>
              ` : nothing}
            </div>
          ` : isOfflineUncached ? html`
            <div id="composer-offline-hint" class="composer-offline-hint" role="status" aria-live="polite">
              <div class="composer-offline-left">
                <svg class="composer-offline-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <line x1="1" y1="1" x2="23" y2="23"></line>
                  <path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"></path>
                  <path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"></path>
                  <path d="M10.71 5.05A16 16 0 0 1 22.58 9"></path>
                  <path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"></path>
                  <path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path>
                  <line x1="12" y1="20" x2="12.01" y2="20"></line>
                </svg>
                <span class="offline-hint-full">Offline · <strong>${cleanModelName}</strong> isn't cached — choose a cached or local model.</span>
                <span class="offline-hint-compact">Offline · model not cached</span>
              </div>
              <button
                id="btn-offline-choose-model"
                type="button"
                class="btn-offline-choose"
                @click=${this.handleChooseModelFromAlert}
              >Choose model</button>
            </div>
          ` : nothing}

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
            data-placeholder=${editorPlaceholder}
            @input=${this.updateEmptyState}
            @keyup=${this.handleEditorSelect}
            @mouseup=${this.handleEditorSelect}
            @blur=${this.handleEditorSelect}
            @keydown=${this.handleKeyDown}
            @paste=${this.handleEditorPaste}
          ></div>

          <div class="composer-toolbar">
            <div class="composer-tools-left">
              <div class="composer-model-anchor">
                <custom-dropdown
                  id="composer-model-dropdown"
                  .value=${selectedPath}
                  @toggle-open=${(e: CustomEvent<{open: boolean}>) => {
                    this.isModelMenuOpen = e.detail.open;
                    if (e.detail.open && this.isAddingModel) {
                      this.closeComposerAddModel(false);
                    }
                  }}
                  @change=${this.handleComposerModelChange}
                >
                  <button
                    slot="trigger"
                    id="btn-composer-model"
                    type="button"
                    class="btn-composer-model ${
                        this.isModelMenuOpen || this.isAddingModel ? 'open' : ''}"
                    title="Active model (${modelSummary.name})${
                        modelSummary.backendOverride ?
                            ` · ${modelSummary.backendOverride.detail}` :
                            ''} — click to switch model"
                    aria-label="Active model: ${modelSummary.name}, status: ${modelSummary.stateLabel}${
                        modelSummary.backendOverride ?
                            `, ${modelSummary.backendOverride.detail}` :
                            ''}. Click to switch model"
                  >
                    <span class="model-pill-dot ${modelSummary.dotClass}"></span>
                    <span class="model-pill-name">${
                        this.isCompactViewport && this.isRecording ?
                            modelSummary.name.replace(/^Gemma 4\s+/i, '') :
                            modelSummary.name}</span>
                    ${modelSummary.isActiveState ? html`
                      <span class="model-pill-state active-state ${
                        modelSummary.compactStateLabel ? 'state-full' : ''}">${
                        modelSummary.stateLabel}</span>
                      ${modelSummary.compactStateLabel ? html`
                        <span class="model-pill-state active-state state-compact">${
                          modelSummary.compactStateLabel}</span>
                      ` : nothing}
                    ` : html`
                      <span class="model-pill-state">${modelSummary.stateLabel}</span>
                    `}
                    <svg class="model-pill-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <polyline points="18 15 12 9 6 15"></polyline>
                    </svg>
                  </button>
                  ${renderModelDropdownItems(this.state, 'composer-model')}
                </custom-dropdown>

                ${this.isAddingModel ? html`
                  <div
                    id="composer-add-model-popover"
                    class="composer-add-model-popover"
                    role="group"
                    aria-label="Add model by URL or Hugging Face repo"
                  >
                    <div class="composer-add-model-label">
                      Add model from URL or Hugging Face
                    </div>
                    <form
                      id="composer-add-model-form"
                      class="add-model-form"
                      @submit=${this.handleComposerAddModel}
                      @keydown=${(e: KeyboardEvent) => {
                        if (e.key === 'Escape') {
                          e.preventDefault();
                          e.stopPropagation();
                          this.closeComposerAddModel(true);
                        }
                      }}
                    >
                      <input
                        id="composer-add-model-input"
                        type="text"
                        placeholder="org/repo or URL…"
                        aria-label="Model URL or Hugging Face repo"
                        ?disabled=${isAddModelDisabled}
                      />
                      <button
                        type="submit"
                        class="btn btn-primary btn-add-model"
                        ?disabled=${isAddModelDisabled}
                        style="padding: 7px 12px; font-size: 0.76rem;"
                      >${this.isLookingUpModel ? 'Adding...' : 'Add'}</button>
                      <button
                        type="button"
                        class="btn-cancel-add"
                        title="Cancel"
                        aria-label="Cancel adding model"
                        @click=${() => {
                          this.closeComposerAddModel(true);
                        }}
                      >✕</button>
                    </form>
                  </div>
                ` : nothing}
              </div>
            </div>

            <div class="composer-tools-right">
              <div class="composer-media-tools ${this.isRecording ? 'is-recording' : ''}">
                <button
                  id="btn-add-media"
                  type="button"
                  class="btn-icon"
                  title=${addMediaTitle}
                  aria-label=${addMediaAriaLabel}
                  ?disabled=${isLoading || allMediaOff}
                  @mousedown=${(e: Event) => e.preventDefault()}
                  @click=${() => this.mediaInput?.click()}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                  </svg>
                </button>
                <button
                  id="btn-camera"
                  type="button"
                  class="btn-icon"
                  title=${cameraTitle}
                  aria-label=${cameraAriaLabel}
                  ?disabled=${isLoading || visionOff}
                  @mousedown=${(e: Event) => e.preventDefault()}
                  @click=${this.openCamera}
                >
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                  </svg>
                </button>
                <button
                  id="btn-record"
                  type="button"
                  class="btn-icon ${this.isRecording ? 'recording' : ''}"
                  title=${recordTitle}
                  aria-label=${recordAriaLabel}
                  ?disabled=${isLoading || audioOff}
                  @mousedown=${(e: Event) => e.preventDefault()}
                  @click=${this.toggleRecording}
                >${this.isRecording ? html`⏹ ${Math.floor(this.recordingSeconds / 60)}:${
                    String(this.recordingSeconds % 60).padStart(2, '0')}` : html`
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <line x1="12" y1="19" x2="12" y2="23"></line>
                    <line x1="8" y1="23" x2="16" y2="23"></line>
                  </svg>
                `}</button>
                ${this.isRecording ? html`
                  <button
                    type="button"
                    class="btn-cancel-record"
                    @mousedown=${(e: Event) => e.preventDefault()}
                    @click=${this.cancelRecording}
                  >Cancel</button>
                ` : ''}
              </div>
              ${isGenerating ? html`
                <button
                  type="button"
                  class="btn btn-stop btn-input"
                  aria-label="Stop generating"
                  @click=${this.handleStopGeneration}
                ><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg>Stop</button>
              ` : html`
                <button
                  id="btn-send"
                  type="button"
                  class="btn btn-primary btn-input ${
                      needsWeightDownload ? 'needs-download' : ''}"
                  aria-label=${sendLabel}
                  title=${isOfflineUncached ? offlineSendTitle : sendLabel}
                  ?disabled=${isLoading || !this.hasPromptContent || isOfflineUncached}
                  @click=${this.triggerSendMessage}
                >${isLoading ? html`<span class="btn-spinner" aria-hidden="true"></span>` : isOfflineUncached ? html`<svg class="send-icon-offline" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"></path><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"></path><path d="M10.71 5.05A16 16 0 0 1 22.58 9"></path><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"></path><path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path><line x1="12" y1="20" x2="12.01" y2="20"></line></svg>` : html`<svg class="send-icon-up" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>${needsWeightDownload ? html`<svg class="send-icon-download" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>` : nothing}`}<span class="send-btn-label">${sendLabel}</span></button>
              `}
            </div>
          </div>
        </div>
      </div>

      <litert-camera-capture
        @open-change=${this.handleCameraOpenChange}
        @photo-captured=${this.handlePhotoCaptured}
      ></litert-camera-capture>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-window': LitertChatWindow;
  }
}
