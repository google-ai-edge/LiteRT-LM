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

import {css, html, LitElement, PropertyValues} from 'lit';
import {customElement, property, query} from 'lit/decorators.js';

import {copyToClipboard} from '../utils/clipboard.js';

/* tslint:disable:no-new-decorators */

interface CodeScrollState {
  scrollTop: number;
  scrollLeft: number;
  autoFollow: boolean;
  wasStreaming: boolean;
}

const codeBlockScrollRegistry =
    new WeakMap<Node, Map<number, CodeScrollState>>();

/** Clears persisted code-block scroll states for a bubble shadow root. */
export function clearCodeBlockScrollState(root: Node|null|undefined) {
  if (root) {
    codeBlockScrollRegistry.delete(root);
  }
}

/**
 * Component representing a code block with copy and interactive preview
 * capabilities.
 */
@customElement('litert-code-block')
export class LitertCodeBlock extends LitElement {
  @property({type: String, attribute: 'base-64-code'}) base64Code = '';

  @property({type: String}) language = '';

  /** Whether this code block is currently receiving streamed tokens. */
  @property({type: Boolean}) streaming = false;

  @query('.code-scroll-region') private readonly scrollRegion?: HTMLDivElement;

  private autoFollow = true;
  private savedScrollTop = 0;
  private savedScrollLeft = 0;
  private lastScrollTop = 0;
  private isProgrammaticScroll = false;
  private wasStreamingState = false;
  private touchStartY = 0;
  private lightDomObserver?: MutationObserver;

  static override styles = css`
    :host {
      display: block;
    }

    .code-container {
      margin-top: 12px;
      margin-bottom: 12px;
      border: 1px solid var(--border, #262f40);
      border-radius: 12px;
      background-color: var(--code-bg, #080b10);
      overflow: hidden;
    }

    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      background-color: var(--code-header-bg, #141923);
      padding: 8px 14px;
      border-bottom: 1px solid var(--border, #262f40);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.74rem;
      color: var(--text-muted, #94a3b8);
      line-height: 1.2;
    }

    .code-lang {
      font-weight: 600;
      text-transform: lowercase;
      letter-spacing: 0.02em;
      color: var(--text-secondary, #cbd5e1);
    }

    .code-actions {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .btn-preview-code {
      background-color: var(--teal-btn-bg, #14d4aa);
      color: var(--teal-btn-text, #041f18);
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      font-family: ui-sans-serif, system-ui, sans-serif;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: background-color 0.15s ease;
      line-height: 1;
    }

    .btn-preview-code:hover {
      background-color: var(--teal-btn-hover, #2ee0b8);
    }

    .btn-copy-code {
      background-color: var(--bg-card, #171c26);
      border: 1px solid var(--border, #262f40);
      color: var(--text, #f1f5f9);
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      font-family: ui-sans-serif, system-ui, sans-serif;
      cursor: pointer;
      transition: background-color 0.15s ease, border-color 0.15s ease;
      line-height: 1;
    }

    .btn-copy-code:hover {
      background-color: var(--bg-elevated, #1e2432);
      border-color: var(--border-strong, #364156);
    }

    button:focus-visible {
      outline: 2px solid var(--teal, #14d4aa);
      outline-offset: 2px;
    }

    .code-scroll-region {
      overflow: auto;
      max-width: 100%;
      max-height: 334px;
      overscroll-behavior: contain;
      scrollbar-width: thin;
      scrollbar-color: var(--border-strong, #364156) transparent;
    }

    .code-scroll-region::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }

    @media (max-height: 720px) {
      .code-scroll-region {
        max-height: 294px;
      }
    }

    @media (max-width: 768px) {
      .code-scroll-region {
        max-height: 254px;
      }
    }

    .code-scroll-region::-webkit-scrollbar-track {
      background: transparent;
    }

    .code-scroll-region::-webkit-scrollbar-thumb {
      background-color: var(--border-strong, #364156);
      border-radius: 999px;
    }

    .code-scroll-region:focus-visible {
      outline: 2px solid var(--teal, #14d4aa);
      outline-offset: -2px;
    }
  `;

  override connectedCallback() {
    super.connectedCallback();
    this.lightDomObserver = new MutationObserver(() => {
      this.syncScrollPosition(false);
    });
    this.lightDomObserver.observe(
        this, {childList: true, subtree: true, characterData: true});
  }

  override disconnectedCallback() {
    this.lightDomObserver?.disconnect();
    this.lightDomObserver = undefined;
    super.disconnectedCallback();
  }

  private getRegistrySlot(): {map: Map<number, CodeScrollState>; idx: number}|
      null {
    const root = this.getRootNode();
    if (!root || !('querySelectorAll' in root)) return null;
    const blocks =
        Array.from((root as ParentNode).querySelectorAll('litert-code-block'));
    const idx = blocks.indexOf(this);
    if (idx < 0) return null;
    let map = codeBlockScrollRegistry.get(root);
    if (!map) {
      map = new Map<number, CodeScrollState>();
      codeBlockScrollRegistry.set(root, map);
    }
    return {map, idx};
  }

  private persistState(isActivelyStreaming = this.isActivelyStreaming()) {
    const slot = this.getRegistrySlot();
    if (!slot) return;
    slot.map.set(slot.idx, {
      scrollTop: this.savedScrollTop,
      scrollLeft: this.savedScrollLeft,
      autoFollow: this.autoFollow,
      wasStreaming: isActivelyStreaming,
    });
  }

  private isActivelyStreaming(): boolean {
    return this.streaming || Boolean(this.closest('.streaming-markdown'));
  }

  override firstUpdated() {
    const slot = this.getRegistrySlot();
    const existing = slot?.map.get(slot.idx);
    if (existing) {
      this.savedScrollTop = existing.scrollTop;
      this.savedScrollLeft = existing.scrollLeft;
      this.autoFollow = existing.autoFollow;
      this.wasStreamingState = existing.wasStreaming;
    } else {
      this.wasStreamingState = this.isActivelyStreaming();
    }
    this.syncScrollPosition(
        this.wasStreamingState && !this.isActivelyStreaming());
  }

  protected override updated(changedProperties: PropertyValues<this>) {
    const nowStreaming = this.isActivelyStreaming();
    const justFinishedStreaming =
        (changedProperties.get('streaming') === true && !nowStreaming) ||
        (this.wasStreamingState && !nowStreaming);
    this.wasStreamingState = nowStreaming;
    if (changedProperties.has('base64Code') ||
        changedProperties.has('streaming') || justFinishedStreaming) {
      this.syncScrollPosition(justFinishedStreaming);
    }
  }

  private syncScrollPosition(justFinishedStreaming: boolean) {
    const el = this.scrollRegion;
    if (!el) return;
    const nowStreaming = this.isActivelyStreaming();
    if (this.savedScrollLeft > 0 && el.scrollLeft !== this.savedScrollLeft) {
      el.scrollLeft = this.savedScrollLeft;
    }
    if ((nowStreaming || justFinishedStreaming) && this.autoFollow) {
      if (el.scrollHeight > el.clientHeight) {
        this.isProgrammaticScroll = true;
        el.scrollTop = el.scrollHeight;
        this.savedScrollTop = el.scrollTop;
        this.lastScrollTop = el.scrollTop;
        queueMicrotask(() => {
          this.isProgrammaticScroll = false;
        });
      }
    } else if (
        this.savedScrollTop > 0 &&
        Math.abs(el.scrollTop - this.savedScrollTop) > 1) {
      this.isProgrammaticScroll = true;
      el.scrollTop = this.savedScrollTop;
      this.lastScrollTop = el.scrollTop;
      queueMicrotask(() => {
        this.isProgrammaticScroll = false;
      });
    }
    this.persistState(nowStreaming);
  }

  private setAutoFollowOff() {
    if (!this.autoFollow) return;
    this.autoFollow = false;
    if (this.scrollRegion) {
      this.savedScrollTop = this.scrollRegion.scrollTop;
    }
    this.persistState();
  }

  private handleWheel(e: WheelEvent) {
    if (e.deltaY < 0) {
      this.setAutoFollowOff();
    }
  }

  private handleTouchStart(e: TouchEvent) {
    if (e.touches.length > 0) {
      this.touchStartY = e.touches[0]!.clientY;
    }
  }

  private handleTouchMove(e: TouchEvent) {
    if (e.touches.length > 0 && e.touches[0]!.clientY - this.touchStartY > 4) {
      this.setAutoFollowOff();
    }
  }

  private handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'PageUp' || e.key === 'ArrowUp' || e.key === 'Home' ||
        (e.key === ' ' && e.shiftKey)) {
      this.setAutoFollowOff();
    }
  }

  private handleScroll() {
    const el = this.scrollRegion;
    if (!el) return;
    const currentTop = el.scrollTop;
    this.savedScrollLeft = el.scrollLeft;

    if (this.isProgrammaticScroll) {
      this.savedScrollTop = currentTop;
      this.lastScrollTop = currentTop;
      this.persistState();
      return;
    }

    if (currentTop < this.lastScrollTop - 1) {
      this.autoFollow = false;
    } else if (!this.autoFollow && currentTop > this.lastScrollTop) {
      const distFromBottom = el.scrollHeight - currentTop - el.clientHeight;
      if (distFromBottom <= 12) {
        this.autoFollow = true;
      }
    }

    this.savedScrollTop = currentTop;
    this.lastScrollTop = currentTop;
    this.persistState();
  }

  private handlePreview() {
    this.dispatchEvent(new CustomEvent('preview-html', {
      detail: {base64Code: this.base64Code},
      bubbles: true,
      composed: true,
    }));
  }

  private handleCopy(e: Event) {
    const btn = e.currentTarget as HTMLButtonElement;
    if (!this.base64Code) return;
    try {
      const binString = atob(this.base64Code);
      const bytes = Uint8Array.from(binString, (m) => m.codePointAt(0)!);
      const code = new TextDecoder().decode(bytes);

      copyToClipboard(code)
          .then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => {
              btn.textContent = 'Copy';
            }, 2000);
          })
          .catch(
              err =>
                  console.error('[LiteRT-LM] Failed to copy code block:', err));
    } catch (e) {
      console.error('[LiteRT-LM] Failed to decode code block content:', e);
    }
  }

  override render() {
    const isHtml = this.language === 'html' || this.language === 'xml';
    return html`
      <div class="code-container">
        <div class="code-header">
          <span class="code-lang">${this.language}</span>
          <div class="code-actions">
            ${isHtml && this.base64Code ? html`
              <button type="button" class="btn-preview-code" @click=${this.handlePreview}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                Preview HTML
              </button>
            ` : ''}
            <button type="button" class="btn-copy-code" @click=${this.handleCopy}>Copy</button>
          </div>
        </div>
        <div
          class="code-scroll-region"
          tabindex="0"
          role="region"
          aria-label="${this.language || 'code'} snippet"
          @wheel=${this.handleWheel}
          @touchstart=${this.handleTouchStart}
          @touchmove=${this.handleTouchMove}
          @keydown=${this.handleKeyDown}
          @scroll=${this.handleScroll}
        >
          <slot></slot>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-code-block': LitertCodeBlock;
  }
}
