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

import './code_block';

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {marked, type Tokens} from 'marked';

import {LlmChatStateController, type StoredMessage} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

import {getLanguage, highlight, highlightAuto, hljsStyles} from './hljs_util.js';
import {katexStyles, renderHtml} from './util.js';
import * as katex from 'katex';

const MATH_START = 'LitertLmMathStart';
const MATH_END = 'LitertLmMathEnd';
const MATH_RESTORE_RE = /LitertLmMathStart(\d+)LitertLmMathEnd/g;



marked.use({
  renderer: {
    code(token: Tokens.Code) {
      const originalCode = token.text;
      // Extract first word of the lang string (e.g., "python info" -> "python")
      let language = (token.lang || '').match(/\S*/)?.[0]?.toLowerCase() || '';
      let highlightedCode: string;

      try {
        if (language && getLanguage(language)) {
          highlightedCode = highlight(originalCode, language);
        } else {
          const result = highlightAuto(originalCode);
          highlightedCode = result.value;
          if (!language && result.language) {
            language = result.language;
          }
        }
      } catch (e) {
        highlightedCode = originalCode.replace(/&/g, '&amp;')
                              .replace(/</g, '&lt;')
                              .replace(/>/g, '&gt;');
      }

      if (!language) {
        language = 'code';
      }

      language = language.replace(/"/g, '&quot;').replace(/'/g, '&#39;');

      const bytes = new TextEncoder().encode(originalCode);
      const base64Code =
          btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(''));
      return `<litert-code-block base-64-code="${base64Code}" language="${
          language}">
<pre class="code-content-pre"><code class="hljs language-${language}">${
          highlightedCode}</code></pre>
</litert-code-block>`;
    }
  }
});

/* tslint:disable:no-new-decorators */

/** Component representing a single chat message bubble. */
@customElement('litert-chat-bubble')
export class LitertChatBubble extends LitElement {
  @property({ type: Object })
  message!: StoredMessage;

  @property({ type: Number })
  index!: number;

  @property({ type: Object })
  state!: LlmChatStateController;

  static override styles = [
    sharedStyles, hljsStyles, katexStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        width: 100%;
      }

      .message-bubble {
        display: flex;
        flex-direction: column;
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 0.9375rem;
        line-height: 1.5;
        word-wrap: break-word;
        animation: fadeIn 0.2s ease-out;
        position: relative;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .message-bubble.user {
        align-self: flex-end;
        background-color: var(--slate-bubble-user);
        border: 1px solid var(--border);
        border-bottom-right-radius: 2px;
      }

      .message-bubble.assistant {
        align-self: flex-start;
        background-color: var(--slate-bubble-bot);
        border: 1px solid var(--border);
        border-bottom-left-radius: 2px;
      }

      .message-sender {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
      }

      .message-sender.user { color: var(--blue); }
      .message-sender.assistant { color: var(--teal); }

      /* CoT Thought Process Blocks */
      .thought-details {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid var(--teal);
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-sizing: border-box;
      }

      .thought-summary {
        font-size: 0.7rem;
        font-weight: bold;
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        cursor: pointer;
        outline: none;
        user-select: none;
      }

      .thought-content {
        font-size: 0.78rem;
        line-height: 1.5;
        color: var(--text-muted);
        font-style: italic;
        margin-top: 8px;
        padding: 0;
      }

      .thought-content p {
        margin: 4px 0;
        font-style: italic;
      }

      .message-user-text {
        white-space: pre-wrap; 
      }
      
      .message-content {
        white-space: normal; 
      }

      .message-content code {
        font-family: ui-monospace, monospace;
        background-color: rgba(0, 0, 0, 0.3);
        padding: 2px 4px;
        border-radius: 4px;
        font-size: 0.875em;
      }

      .message-content pre {
        font-family: ui-monospace, monospace;
        background-color: rgba(0, 0, 0, 0.4);
        padding: 12px;
        border-radius: 8px;
        overflow-x: auto;
        font-size: 0.875rem;
        margin: 8px 0;
      }

      /* Styled Code Blocks & Previews */


      .code-content-pre {
        margin: 0;
        padding: 16px;
        overflow-x: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.85rem;
        line-height: 1.4;
        color: #cbd5e1;
        background-color: transparent !important;
        border-radius: 0 !important;
      }

      .message-actions {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        border-top: 1px dashed rgba(255, 255, 255, 0.1);
        padding-top: 6px;
        justify-content: flex-end;
        align-items: center; 
      }

      .btn-action {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 0.7rem;
        font-family: inherit;
        display: inline-flex;
        align-items: center;
        justify-content: center; 
        width: 58px; 
        height: 20px;
        box-sizing: border-box;
        padding: 0;
        border-radius: 4px;
        transition: color 0.15s, background-color 0.15s;
      }

      .btn-action:hover {
        color: var(--teal);
        background-color: var(--bg-input);
      }

      .message-stats {
        font-size: 0.68rem;
        color: var(--text-muted);
        font-family: ui-monospace, monospace;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin-right: auto;
        opacity: 0.65;
        user-select: none;
      }

      .message-stats b {
        color: var(--teal);
        white-space: nowrap; 
      }
    `
  ];

  private renderMath(latex: string, displayMode: boolean): string {
    try {
      return katex.renderToString(latex, {
        displayMode,
        throwOnError: false,
        output: 'html',
      });
    } catch (e) {
      console.error('[LiteRT-LM] Failed to render math:', e);
      return `<code>${latex}</code>`;
    }
  }

  private renderMarkdown(text: string): string {
    // Note: This is called in the render function for each token we receive
    // from the model.
    // It's not efficient to re-render the entire markdown from scratch for
    // each token, but it's fast enough for now.
    // A better solution would incrementally render new chunks of the markdown.
    if (!text) return '';

    const mathStore: string[] = [];

    // Split by code blocks and inline code to avoid processing math inside them
    const codePattern = /(```[\s\S]*?```|`[^`]*`)/g;
    const parts = text.split(codePattern);

    const processedParts = parts.map((part, i) => {
      // Odd indices are code blocks/inline code - leave them alone
      if (i % 2 === 1) return part;

      // Even indices are regular text - process math
      // 1. Extract and render display math: $$...$$
      let processed = part.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex: string) => {
        const idx = mathStore.length;
        mathStore.push(this.renderMath(tex.trim(), true));
        return `${MATH_START}${idx}${MATH_END}`;
      });

      // 2. Extract and render inline math: $...$
      processed = processed.replace(/\$([^\s$](?:[^$]*[^\s$])?)\$/g, (_, tex: string) => {
        const idx = mathStore.length;
        mathStore.push(this.renderMath(tex.trim(), false));
        return `${MATH_START}${idx}${MATH_END}`;
      });

      return processed;
    });

    const processedText = processedParts.join('');

    try {
      let htmlResult = marked.parse(processedText, {async: false}) as string;

      // 3. Restore rendered math blocks
      htmlResult = htmlResult.replace(MATH_RESTORE_RE, (_, idx: string) => {
        return mathStore[Number(idx)] || '';
      });

      return htmlResult;
    } catch (e) {
      console.error('[LiteRT-LM] Failed to parse markdown:', e);
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');
    }
  }

  private handleCopyMessage(e: Event) {
    const targetBtn = e.target as HTMLButtonElement;
    navigator.clipboard.writeText(this.message.text)
      .then(() => {
        targetBtn.innerText = 'Copied!';
        setTimeout(() => { targetBtn.innerText = 'Copy'; }, 2000);
      })
      .catch(err => console.error('[LiteRT-LM] Failed to copy message:', err));
  }

  private async handleRewindEdit() {
    const originalPrompt =
        await this.state.chatSession.rewindAndEdit(this.index);
    if (originalPrompt) {
      this.dispatchEvent(new CustomEvent('edit-prompt', {
        detail: {prompt: originalPrompt},
        bubbles: true,
        composed: true,
      }));
    }
  }

  override render() {
    const msg = this.message;
    const isUser = msg.role === 'user';

    // Clean size-free model names for bubble headers
    let senderName = msg.senderName;
    if (!isUser) {
      senderName = senderName
        .replace(/,\s*\d+(\.\d+)?\s*GB\)$/, ')')
        .replace(/\s*\(\d+(\.\d+)?\s*GB\)$/, '');
    }

    return html`
      <div class="message-bubble ${msg.role}" data-raw-text="${msg.text}">
        <span class="message-sender ${msg.role}">${
        isUser ? 'User' : senderName}</span>
        
        <div class="message-content">
          <!-- Thought block rendering -->
          ${
        msg.thoughtText ? html`
            <details class="thought-details" open>
              <summary class="thought-summary">Thought Process</summary>
              <div class="thought-content">${
                              renderHtml(this.renderMarkdown(
                                  msg.thoughtText || ''))}</div>
            </details>
          ` :
                          ''}
          
          <!-- Main message bubble content -->
          ${
        isUser ? html`<div class="message-user-text">${msg.text}</div>` :
                 html`<div>${
                     renderHtml(this.renderMarkdown(msg.text || ''))}</div>`}
        </div>

        <!-- Message actions bar (Edit / Copy & Retry) -->
        <div class="message-actions">
          ${
        isUser ? html`
            <!-- User stats telemetry (Token count only!) -->
            ${
                     msg.tokensCount ? html`
              <div class="message-stats">
                <span>Tokens: <b>${msg.tokensCount}</b></span>
              </div>
            ` :
                                       ''}
            <!-- User Edit trigger -->
            <button class="btn-action" style="width: 58px;" @click=${
                     this.handleRewindEdit}>✎ Edit</button>
          ` :
                 html`
            <!-- Assistant stats telemetry (prefilled auto-left!) -->
            ${
                     msg.prefillSpeed || msg.decodeSpeed ? html`
              <div class="message-stats">
                <span>Prefill: <b>${msg.prefillSpeed || '-'}</b></span>
                <span>Decode: <b>${msg.decodeSpeed || '-'}</b></span>
                <span>Tokens: <b>${msg.tokensCount || '-'}</b></span>
              </div>
            ` :
                                                           ''}
            
            <!-- Copy & Retry triggers -->
            <button class="btn-action" style="width: 58px;" @click=${
                     this.handleCopyMessage}>Copy</button>
            <button class="btn-action" style="width: 58px;" @click=${
                     () => this.state.chatSession.redoResponse(
                         this.index)}>⟲ Retry</button>
          `}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-bubble': LitertChatBubble;
  }
}
