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

import {css, html, LitElement, nothing, PropertyValues} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {buildModelCodeSnippet, type CodeSnippetVariant, getNpmInstallCommand} from '../stores/code_snippet.js';
import {sharedStyles} from '../styles/shared_styles.js';
import {copyToClipboard} from '../utils/clipboard.js';
import {focusNowAndNextFrame} from './focus_util.js';
import {getLanguage, highlight} from './hljs_util.js';
import {getActiveModelSummary} from './model_selector_shared.js';
import {renderHtml} from './util.js';

/* tslint:disable:no-new-decorators */

/** `localStorage` key remembering the selected code snippet tab. */
export const CODE_VARIANT_STORAGE_KEY = 'litertlm-code-snippet-variant';

/**
 * Modal dialog displaying runnable `@litert-lm/core` code snippets (npm module
 * or standalone HTML file) reflecting the currently active model and settings.
 */
@customElement('litert-code-dialog')
export class LitertCodeDialog extends LitElement {
  @property({type: Object}) state!: LlmChatStateController;

  @property({type: Boolean}) open = false;

  @state() private codeVariant: CodeSnippetVariant = 'npm';
  @state() private codeCopied = false;
  private codeCopyResetTimer: number|undefined;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        width: min(720px, calc(100vw - 32px));
        max-height: min(640px, 100%);
      }

      @keyframes codeDialogEnter {
        from {
          opacity: 0;
          transform: translateY(6px) scale(0.985);
        }
        to {
          opacity: 1;
          transform: none;
        }
      }

      .code-snippet-dialog {
        width: 100%;
        max-height: 100%;
        display: flex;
        flex-direction: column;
        border-radius: 16px;
        border: 1px solid var(--border-strong, #364156);
        background-color: var(--bg-card, #171c26);
        box-shadow: var(--shadow-md, 0 12px 32px rgba(0, 0, 0, 0.5));
        overflow: hidden;
        box-sizing: border-box;
        animation: codeDialogEnter 0.16s cubic-bezier(0.16, 1, 0.3, 1);
      }

      @media (prefers-reduced-motion: reduce) {
        .code-snippet-dialog {
          animation: none;
        }
      }

      .code-popover-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 16px;
        background-color: var(--code-header-bg, #141923);
        border-bottom: 1px solid var(--border, #262f40);
        flex-shrink: 0;
      }

      .code-popover-heading-block {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-width: 0;
        flex: 1;
      }

      .code-popover-title-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }

      .code-popover-title-wrap > svg {
        color: var(--teal, #14d4aa);
        flex-shrink: 0;
      }

      .code-popover-title {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text, #f1f5f9);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
      }

      .code-popover-subtitle {
        margin: 0;
        font-size: 0.74rem;
        line-height: 1.35;
        color: var(--text-muted, #94a3b8);
      }

      .code-popover-model-pill {
        font-size: 0.7rem;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: var(--text-muted, #94a3b8);
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 999px;
        padding: 2px 8px;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .code-popover-model-pill-subbar {
        display: none;
      }

      .code-popover-actions {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-shrink: 0;
      }

      .btn-copy-snippet {
        height: 30px;
        padding: 0 12px;
        border-radius: 8px;
        border: 1px solid transparent;
        background-color: var(--teal-btn-bg, #14d4aa);
        color: var(--teal-btn-text, #041f18);
        font-family: inherit;
        font-size: 0.75rem;
        font-weight: 700;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        white-space: nowrap;
        transition: background-color 0.14s ease, border-color 0.14s ease;
      }

      .btn-copy-snippet:hover {
        background-color: var(--teal-btn-hover, #2ee0b8);
      }

      .btn-close-code-popover {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-muted, #94a3b8);
        font-size: 0.84rem;
        line-height: 1;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        transition: background-color 0.14s ease, color 0.14s ease;
      }

      .btn-close-code-popover:hover {
        background-color: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
      }

      .code-popover-subbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px 12px;
        padding: 8px 16px;
        background-color: var(--bg-card, #171c26);
        border-bottom: 1px solid var(--border, #262f40);
        flex-shrink: 0;
      }

      .code-variant-tabs {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        padding: 2px;
        border-radius: 8px;
        background-color: var(--bg-input, #10141c);
        border: 1px solid var(--border, #262f40);
        flex-shrink: 0;
      }

      .code-variant-tab {
        height: 26px;
        padding: 0 11px;
        border-radius: 6px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-muted, #94a3b8);
        font-family: inherit;
        font-size: 0.74rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        transition: background-color 0.12s ease, color 0.12s ease;
      }

      .code-variant-tab:hover {
        color: var(--text, #f1f5f9);
      }

      .code-variant-tab.active {
        background-color: var(--teal-btn-bg, #14d4aa);
        color: var(--teal-btn-text, #041f18);
        font-weight: 700;
      }

      .code-variant-tab:focus-visible {
        outline: 2px solid var(--teal, #14d4aa);
        outline-offset: 1px;
      }

      .code-popover-pre {
        margin: 0;
        padding: 14px 16px;
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
        overscroll-behavior: contain;
        scrollbar-width: thin;
        scrollbar-color: var(--border-strong, #364156) transparent;
        background-color: var(--code-bg, light-dark(#f1f5f9, #080b10));
        color: var(--hljs-fg, light-dark(#1f2328, #e2e8f0));
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        line-height: 1.55;
        tab-size: 2;
      }

      .code-popover-pre:focus-visible {
        outline: 2px solid var(--teal, #14d4aa);
        outline-offset: -2px;
      }

      .code-popover-pre code {
        font-family: inherit;
        font-size: inherit;
        background: transparent;
        padding: 0;
      }

      .code-popover-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px 12px;
        padding: 9px 16px;
        background-color: var(--code-header-bg, #141923);
        border-top: 1px solid var(--border, #262f40);
        font-size: 0.73rem;
        color: var(--text-muted, #94a3b8);
        flex-shrink: 0;
      }

      .code-popover-install code {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        color: var(--text-secondary, #cbd5e1);
      }

      .code-popover-docs-link {
        color: light-dark(var(--teal-dark, #115e59), var(--teal, #14d4aa));
        text-decoration: none;
        font-weight: 600;
        white-space: nowrap;
      }

      .code-popover-docs-link:hover {
        text-decoration: underline;
      }

      @media (max-width: 600px) {
        :host {
          width: calc(100vw - 20px);
        }

        .code-popover-header {
          padding: 10px 12px;
          gap: 8px;
        }

        .code-popover-title {
          font-size: 0.78rem;
        }

        .code-popover-subtitle {
          font-size: 0.69rem;
        }

        .code-popover-model-pill-header {
          display: none;
        }

        .code-popover-model-pill-subbar {
          display: inline-block;
          max-width: 55%;
          background-color: var(--bg-input, #10141c);
        }

        .btn-copy-snippet {
          height: 28px;
          padding: 0 9px;
          font-size: 0.71rem;
        }

        .btn-close-code-popover {
          width: 28px;
          height: 28px;
        }

        .code-popover-subbar {
          padding: 7px 12px;
        }

        .code-popover-pre {
          padding: 12px;
          font-size: 0.75rem;
          line-height: 1.48;
        }

        .code-popover-footer {
          padding: 8px 12px;
          font-size: 0.7rem;
        }
      }
    `
  ];

  override connectedCallback() {
    super.connectedCallback();
    this.state?.addHost?.(this);
  }

  override disconnectedCallback() {
    window.clearTimeout(this.codeCopyResetTimer);
    this.codeCopyResetTimer = undefined;
    this.state?.removeHost?.(this);
    super.disconnectedCallback();
  }

  protected override willUpdate(changedProperties: PropertyValues<this>) {
    if (changedProperties.has('open')) {
      window.clearTimeout(this.codeCopyResetTimer);
      this.codeCopyResetTimer = undefined;
      this.codeCopied = false;
      if (this.open) {
        this.codeVariant = this.loadSavedCodeVariant();
      }
    }
  }

  /** Moves keyboard focus to the primary Copy action when the dialog opens. */
  focusInitialElement() {
    void this.updateComplete.then(() => {
      focusNowAndNextFrame(() => {
        if (!this.open) return;
        this.renderRoot
            .querySelector<HTMLButtonElement>('#btn-copy-model-code')
            ?.focus();
      });
    });
  }

  private getCodeVariantStorage(): Storage|undefined {
    try {
      return this.state?.storage ?? window.localStorage;
    } catch {
      return undefined;
    }
  }

  private loadSavedCodeVariant(): CodeSnippetVariant {
    try {
      const saved =
          this.getCodeVariantStorage()?.getItem(CODE_VARIANT_STORAGE_KEY);
      return saved === 'html' ? 'html' : 'npm';
    } catch {
      return 'npm';
    }
  }

  private selectCodeVariant(variant: CodeSnippetVariant, moveFocus = false) {
    if (this.codeVariant !== variant) {
      this.codeVariant = variant;
      this.codeCopied = false;
      window.clearTimeout(this.codeCopyResetTimer);
      this.codeCopyResetTimer = undefined;
    }
    try {
      this.getCodeVariantStorage()?.setItem(CODE_VARIANT_STORAGE_KEY, variant);
    } catch {
      // Ignore storage quota / privacy-mode errors.
    }
    if (moveFocus) {
      void this.updateComplete.then(() => {
        this.renderRoot
            .querySelector<HTMLButtonElement>(`#tab-code-${variant}`)
            ?.focus();
      });
    }
  }

  private readonly handleCodeVariantTabKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const next: CodeSnippetVariant =
          this.codeVariant === 'npm' ? 'html' : 'npm';
      this.selectCodeVariant(next, true);
    } else if (e.key === 'Home') {
      e.preventDefault();
      this.selectCodeVariant('npm', true);
    } else if (e.key === 'End') {
      e.preventDefault();
      this.selectCodeVariant('html', true);
    }
  };

  private requestClose() {
    this.dispatchEvent(new CustomEvent('close', {
      bubbles: true,
      composed: true,
    }));
  }

  private handleDialogKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      this.requestClose();
    }
  }

  private async copyModelCode(snippet: string) {
    try {
      await copyToClipboard(snippet);
      this.codeCopied = true;
      window.clearTimeout(this.codeCopyResetTimer);
      this.codeCopyResetTimer = window.setTimeout(() => {
        this.codeCopied = false;
        this.codeCopyResetTimer = undefined;
      }, 2000);
    } catch (err) {
      console.error('[LiteRT-LM] Failed to copy code snippet:', err);
    }
  }

  override render() {
    if (!this.open || !this.state) return nothing;
    const cleanModelName = getActiveModelSummary(this.state).name;
    const codeSnippet =
        buildModelCodeSnippet(this.state.settings, this.codeVariant);
    const codeHighlightLang = this.codeVariant === 'html' ?
        (getLanguage('html') ? 'html' : 'xml') :
        'typescript';

    return html`
      <div
        id="code-snippet-dialog"
        class="code-snippet-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="code-dialog-title"
        aria-describedby="code-dialog-desc"
        @keydown=${this.handleDialogKeyDown}
      >
        <div class="code-popover-header">
          <div class="code-popover-heading-block">
            <div class="code-popover-title-wrap">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
              </svg>
              <h2 id="code-dialog-title" class="code-popover-title">Use with @litert-lm/core</h2>
              <span class="code-popover-model-pill code-popover-model-pill-header" title=${cleanModelName}>${cleanModelName}</span>
            </div>
            <p id="code-dialog-desc" class="code-popover-subtitle">Generated from your current model and chat settings.</p>
          </div>
          <div class="code-popover-actions">
            <button
              id="btn-copy-model-code"
              type="button"
              class="btn-copy-snippet ${this.codeCopied ? 'copied' : ''}"
              @click=${() => this.copyModelCode(codeSnippet)}
            >
              ${this.codeCopied ? html`
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span>Copied</span>
              ` : html`
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copy code</span>
              `}
            </button>
            <button
              id="btn-close-code-popover"
              type="button"
              class="btn-close-code-popover"
              aria-label="Close code snippet"
              title="Close"
              @click=${this.requestClose}
            >✕</button>
          </div>
        </div>
        <div class="code-popover-subbar">
          <div
            class="code-variant-tabs"
            role="tablist"
            aria-label="Code sample format"
          >
            <button
              id="tab-code-npm"
              type="button"
              role="tab"
              class="code-variant-tab ${
                  this.codeVariant === 'npm' ? 'active' : ''}"
              aria-selected=${this.codeVariant === 'npm' ? 'true' : 'false'}
              aria-controls="code-popover-panel"
              tabindex=${this.codeVariant === 'npm' ? '0' : '-1'}
              @click=${() => this.selectCodeVariant('npm')}
              @keydown=${this.handleCodeVariantTabKeyDown}
            >npm</button>
            <button
              id="tab-code-html"
              type="button"
              role="tab"
              class="code-variant-tab ${
                  this.codeVariant === 'html' ? 'active' : ''}"
              aria-selected=${this.codeVariant === 'html' ? 'true' : 'false'}
              aria-controls="code-popover-panel"
              tabindex=${this.codeVariant === 'html' ? '0' : '-1'}
              @click=${() => this.selectCodeVariant('html')}
              @keydown=${this.handleCodeVariantTabKeyDown}
            >HTML</button>
          </div>
          <span
            class="code-popover-model-pill code-popover-model-pill-subbar"
            title=${cleanModelName}
          >${cleanModelName}</span>
        </div>
        <pre
          id="code-popover-panel"
          class="code-popover-pre"
          tabindex="0"
          role="tabpanel"
          aria-labelledby=${
              this.codeVariant === 'html' ? 'tab-code-html' : 'tab-code-npm'}
        ><code class="hljs language-${
            this.codeVariant === 'html' ? 'html' : 'typescript'}">${
            renderHtml(highlight(codeSnippet, codeHighlightLang))}</code></pre>
        <div class="code-popover-footer">
          ${this.codeVariant === 'html' ? html`
            <span class="code-popover-install">Save as <code>.html</code> and open in a WebGPU browser (use localhost/https for local files &amp; caching).</span>
          ` : html`
            <span class="code-popover-install"><code>${getNpmInstallCommand()}</code></span>
          `}
          <span class="sr-only" role="status" aria-live="polite">${this.codeCopied ? 'Copied code to clipboard' : ''}</span>
          <a
            class="code-popover-docs-link"
            href="https://ai.google.dev/edge/litert-lm/js"
            target="_blank"
            rel="noopener noreferrer"
          >JS API docs ↗</a>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-code-dialog': LitertCodeDialog;
  }
}
