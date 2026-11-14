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

import {clearCodeBlockScrollState} from './code_block.js';

import {css, html, LitElement, PropertyValues} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {repeat} from 'lit/directives/repeat.js';
import {marked, type Tokens} from 'marked';

import {LlmChatStateController, type StoredMessage} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';
import {copyToClipboard} from '../utils/clipboard.js';

import {getLanguage, highlight, highlightAuto, hljsStyles} from './hljs_util.js';
import {katexStyles, renderHtml} from './util.js';
import * as katex from 'katex';

const MATH_START = 'LitertLmMathStart';
const MATH_END = 'LitertLmMathEnd';
const MATH_RESTORE_RE = /LitertLmMathStart(\d+)LitertLmMathEnd/g;

/**
 * Matches a final line consisting only of markdown punctuation. While a reply
 * is streaming, such a line usually renders as something other than what it
 * is about to become: `-` under a paragraph is a setext heading until the
 * bullet's text arrives, `***` is a divider until `***Bold***` completes, `#`
 * is an empty heading, and a partial closing fence is an extra code line. A
 * bare number is included too: otherwise `1` renders as a paragraph, vanishes
 * at `1.`, and reappears as a list item.
 */
const AMBIGUOUS_TAIL_RE =
    /(^|\n)[ \t]*(?:[-*_=+ \t]*|#{1,6}|>+|\d{1,9}[.)]?|`{1,3}|~{1,3})[ \t]*$/;

function trimPendingPipeTableHeader(text: string): string {
  if (!text.includes('|')) return text;
  // Strip trailing horizontal whitespace after a newline when inspecting lines
  // so `"| Col |\n"` right before line 2's `|` doesn't flash for 1 token.
  const inspect = text.replace(/\n[ \t]*$/, '');
  const lastNl = inspect.lastIndexOf('\n');
  const lastLine = inspect.slice(lastNl + 1);
  if (!/^[ \t]*\|/.test(lastLine)) return text;

  const beforeLast = lastNl >= 0 ? inspect.slice(0, lastNl) : '';
  const prevNl = beforeLast.lastIndexOf('\n');
  const prevLine = lastNl >= 0 ? beforeLast.slice(prevNl + 1) : null;

  // Case 1: Standalone single pipe-prefixed header row with no delimiter row yet
  if (prevLine === null || !/^[ \t]*\|/.test(prevLine)) {
    return lastNl >= 0 ? inspect.slice(0, lastNl + 1) : '';
  }

  // Case 2: Header row + incomplete GFM delimiter row (`| :--- | ...`)
  const beforePrev = prevNl >= 0 ? beforeLast.slice(0, prevNl) : '';
  const twoBackLine =
      prevNl >= 0 ? beforePrev.slice(beforePrev.lastIndexOf('\n') + 1) : null;
  if ((twoBackLine === null || !/^[ \t]*\|/.test(twoBackLine)) &&
      /^[ \t]*\|[\s|:-]*$/.test(lastLine)) {
    const headerCells = prevLine.trim()
                            .replace(/^\|/, '')
                            .replace(/\|$/, '')
                            .split('|')
                            .filter(c => c.trim().length > 0)
                            .length;
    const delimCells =
        lastLine.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
    const validDelim = headerCells > 0 && delimCells.length >= headerCells &&
        delimCells.slice(0, headerCells)
            .every(c => /^:?-+:?$/.test(c.trim()));
    if (!validDelim) {
      return prevNl >= 0 ? beforeLast.slice(0, prevNl + 1) : '';
    }
  }
  return text;
}

/**
 * Drops the last line of streaming text if it's only markdown punctuation (or
 * an unterminated GFM pipe-table header before its delimiter row arrives), so
 * it doesn't flash as a heading, divider, empty block, or raw pipe text.
 */
export function trimAmbiguousTail(text: string): string {
  return trimPendingPipeTableHeader(text).replace(AMBIGUOUS_TAIL_RE, '$1');
}



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

  /** Whether this message is the reply currently being generated. */
  @property({type: Boolean}) streaming = false;

  /** Whether any generation is active in the session (disables Edit/Retry). */
  @property({type: Boolean}) disabled = false;

  @state() private isCopied = false;

  /** Tracks explicit user expand/collapse override on the CoT thought block. */
  private userToggledThought: boolean | null = null;
  private lastMessageRoleAndIndex = '';
  private copyResetTimer?: number;
  private registeredState: LlmChatStateController|null = null;

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
        font-size: 0.9375rem;
        line-height: 1.62;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        animation: fadeIn 0.18s ease-out;
        position: relative;
        color: var(--text, #f1f5f9);
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .message-bubble.user {
        align-self: flex-end;
        max-width: 78%;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        background-color: var(--slate-bubble-user, #1d2433);
        border: 1px solid var(--border, #262f40);
      }

      .user-meta-row {
        align-self: flex-end;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 4px;
        padding-right: 2px;
      }

      .user-token-badge {
        font-size: 0.68rem;
        color: var(--text-muted, #94a3b8);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        user-select: none;
      }

      .message-bubble.assistant {
        align-self: stretch;
        width: 100%;
        max-width: 100%;
        padding: 4px 2px;
        border-radius: 0;
        background-color: transparent;
        border: none;
      }

      .bubble-header {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
      }

      .message-sender {
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: 0.01em;
      }

      .message-sender.assistant {
        color: var(--text-muted, #94a3b8);
      }

      /* Tool Execution & Pre-first-token Streaming Indicators */
      .tool-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 4px 10px;
        margin-bottom: 10px;
        border-radius: 999px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        font-size: 0.74rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: var(--teal, #14d4aa);
      }

      .streaming-placeholder {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.84rem;
        line-height: 1.4;
        color: var(--text-muted, #94a3b8);
      }

      .pulse-dots {
        display: inline-flex;
        gap: 3px;
      }

      .pulse-dots span {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background-color: var(--teal, #14d4aa);
        animation: dotBounce 1.2s infinite ease-in-out;
      }

      .pulse-dots span:nth-child(2) { animation-delay: 0.16s; }
      .pulse-dots span:nth-child(3) { animation-delay: 0.32s; }

      @keyframes dotBounce {
        0%, 80%, 100% { opacity: 0.3; transform: scale(0.85); }
        40% { opacity: 1; transform: scale(1.1); }
      }

      /* CoT Thought & Completed Tool Call Disclosure Blocks */
      .thought-details,
      .tool-details {
        display: block;
        margin: 0 0 10px 0;
        padding: 0;
        border: none;
        background: none;
      }

      .thought-summary,
      .tool-summary {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        height: 28px;
        padding: 0 12px;
        border-radius: 999px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        box-sizing: border-box;
        font-size: 0.74rem;
        font-weight: 600;
        line-height: 1;
        color: var(--text-muted, #94a3b8);
        cursor: pointer;
        outline: none;
        user-select: none;
        list-style: none;
        vertical-align: middle;
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
      }

      .tool-summary {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }

      .thought-summary-label {
        display: inline-flex;
        align-items: center;
        line-height: 1;
      }

      .thought-summary:hover,
      .tool-summary:hover {
        border-color: var(--teal, #14d4aa);
        color: var(--text, #f1f5f9);
      }

      .thought-details[open] > .thought-summary,
      .tool-details[open] > .tool-summary {
        color: var(--text-secondary, #cbd5e1);
        border-color: var(--border-strong, #364156);
      }

      .tool-details.tool-error > .tool-summary {
        border-color: var(--danger, #f87171);
      }

      .thought-summary:focus-visible,
      .tool-summary:focus-visible {
        outline: 2px solid var(--teal, #14d4aa);
        outline-offset: 2px;
        border-radius: 999px;
      }

      .thought-summary::-webkit-details-marker,
      .tool-summary::-webkit-details-marker {
        display: none;
      }

      .thought-chevron {
        display: block;
        width: 13px;
        height: 13px;
        color: var(--teal, #14d4aa);
        transition: transform 0.16s ease;
        flex-shrink: 0;
      }

      .tool-icon {
        display: block;
        width: 13px;
        height: 13px;
        color: var(--teal, #14d4aa);
        flex-shrink: 0;
      }

      .tool-details.tool-error .tool-icon,
      .tool-details.tool-error .thought-chevron {
        color: var(--danger, #f87171);
      }

      .thought-details[open] > .thought-summary .thought-chevron,
      .tool-details[open] > .tool-summary .thought-chevron {
        transform: rotate(90deg);
      }

      .thought-content {
        font-size: 0.8125rem;
        line-height: 1.55;
        color: var(--text-muted, #94a3b8);
        margin-top: 8px;
        margin-left: 6px;
        padding: 2px 0 2px 14px;
        border-left: 2px solid var(--border, #262f40);
      }

      .thought-content p {
        margin: 6px 0;
      }

      .thought-content > :first-child {
        margin-top: 0;
      }

      .thought-content > :last-child {
        margin-bottom: 0;
      }

      .tool-details-body {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 8px;
        padding: 10px 12px;
        border-radius: 10px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
      }

      .tool-io-section {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .tool-io-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted, #94a3b8);
      }

      .tool-io-pre {
        margin: 0 !important;
        padding: 8px 10px !important;
        border-radius: 8px !important;
        background-color: var(--code-bg, #080b10) !important;
        border: 1px solid var(--border, #262f40) !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.76rem !important;
        line-height: 1.45;
        color: var(--hljs-fg, #e2e8f0);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        max-height: 180px;
        overflow-y: auto;
      }

      /* Streaming Token Caret */
      @keyframes streamCaretBlink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
      }

      .streaming-markdown {
        position: relative;
      }

      .streaming-markdown > p:last-child::after,
      .streaming-markdown > h1:last-child::after,
      .streaming-markdown > h2:last-child::after,
      .streaming-markdown > h3:last-child::after,
      .streaming-markdown > h4:last-child::after,
      .streaming-markdown > blockquote:last-child > p:last-child::after,
      .streaming-markdown > :is(ul, ol):last-child > li:last-child:not(:has(ul, ol, p, pre, blockquote))::after,
      .streaming-markdown > :is(ul, ol):last-child > li:last-child > p:last-child::after,
      .streaming-markdown > :is(ul, ol):last-child > li:last-child > :is(ul, ol):last-child > li:last-child:not(:has(ul, ol, p))::after {
        content: '';
        position: absolute;
        width: 2px;
        height: 0.92em;
        margin-left: 3px;
        margin-top: 0.24em;
        border-radius: 1px;
        background-color: var(--teal, #14d4aa);
        pointer-events: none;
        animation: streamCaretBlink 0.9s steps(2, start) infinite;
      }

      .stopped-empty-note {
        margin: 0;
        font-size: 0.84rem;
        font-style: italic;
        color: var(--text-muted, #94a3b8);
      }

      .message-user-text {
        white-space: pre-wrap;
      }

      .message-image {
        display: block;
        max-width: 100%;
        max-height: 260px;
        border-radius: 10px;
        border: 1px solid var(--border, #262f40);
        margin: 4px 0 8px 0;
        object-fit: contain;
      }

      .message-audio {
        display: block;
        max-width: 100%;
        margin: 6px 0;
      }

      .message-media-missing {
        display: inline-block;
        padding: 2px 8px;
        margin: 0 2px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 8px;
        font-size: 0.8em;
        color: var(--text-muted, #94a3b8);
      }

      .message-content {
        white-space: normal;
      }

      .message-content p:first-child {
        margin-top: 0;
      }

      .message-content p:last-child {
        margin-bottom: 0;
      }

      .message-content h1,
      .message-content h2,
      .message-content h3,
      .message-content h4 {
        color: var(--text, #f1f5f9);
        line-height: 1.3;
        margin: 0.9em 0 0.45em 0;
        font-weight: 700;
      }

      .message-content h1 { font-size: 1.25rem; }
      .message-content h2 { font-size: 1.12rem; }
      .message-content h3 { font-size: 1.02rem; }

      .message-content a {
        color: var(--teal, #14d4aa);
        text-decoration: underline;
        text-underline-offset: 2px;
      }

      .message-content blockquote {
        margin: 10px 0;
        padding: 6px 14px;
        border-left: 3px solid var(--teal, #14d4aa);
        background-color: var(--bg-card, #171c26);
        border-radius: 0 8px 8px 0;
        color: var(--text-secondary, #cbd5e1);
      }

      .message-content hr {
        border: none;
        border-top: 1px solid var(--border, #262f40);
        margin: 14px 0;
      }

      .message-content table {
        display: block;
        width: max-content;
        max-width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border-collapse: separate;
        border-spacing: 0;
        margin: 12px 0;
        font-size: 0.85rem;
        border: 1px solid var(--border, #262f40);
        border-radius: 8px;
        word-wrap: normal;
        overflow-wrap: normal;
        word-break: normal;
      }

      .message-content th,
      .message-content td {
        padding: 8px 12px;
        border-right: 1px solid var(--border, #262f40);
        border-bottom: 1px solid var(--border, #262f40);
        text-align: left;
        word-wrap: normal;
        overflow-wrap: normal;
        word-break: normal;
      }

      .message-content th:last-child,
      .message-content td:last-child {
        border-right: none;
      }

      .message-content tbody tr:last-child td {
        border-bottom: none;
      }

      .message-content th {
        background-color: var(--bg-card, #171c26);
        color: var(--text, #f1f5f9);
        font-weight: 600;
        white-space: nowrap;
      }

      .message-content td:first-child {
        white-space: nowrap;
        font-weight: 500;
      }

      .message-content .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
        max-width: 100%;
        padding: 4px 0;
      }

      .message-content code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        padding: 1px 5px;
        border-radius: 5px;
        font-size: 0.85em;
        color: var(--text, #f1f5f9);
      }

      .message-content pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background-color: var(--code-bg, #080b10);
        border: 1px solid var(--border, #262f40);
        padding: 14px 16px;
        border-radius: 10px;
        overflow-x: auto;
        font-size: 0.84rem;
        margin: 8px 0;
      }

      .message-content pre code {
        background: transparent;
        border: none;
        padding: 0;
        font-size: inherit;
      }

      .code-content-pre {
        margin: 0 !important;
        padding: 14px 16px !important;
        overflow: visible !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.83rem;
        line-height: 20px;
        color: var(--hljs-fg, #e2e8f0);
        background-color: transparent !important;
        border-radius: 0 !important;
        border: none !important;
      }

      .message-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
        border-top: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding-top: 8px;
        justify-content: flex-end;
        align-items: center;
      }

      .btn-action {
        background: none;
        border: 1px solid transparent;
        color: var(--text-muted, #94a3b8);
        cursor: pointer;
        font-size: 0.74rem;
        font-weight: 500;
        font-family: inherit;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        min-height: 32px;
        padding: 0 10px;
        border-radius: 8px;
        transition: color 0.14s ease, background-color 0.14s ease, border-color 0.14s ease;
      }

      .btn-action svg {
        width: 13px;
        height: 13px;
        flex-shrink: 0;
      }

      .btn-action:hover:not(:disabled) {
        color: var(--text, #f1f5f9);
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border, #262f40);
      }

      .btn-action:disabled {
        opacity: 0.4;
        cursor: not-allowed;
        pointer-events: none;
      }

      .stopped-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 8px;
        border-radius: 999px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        color: var(--text-muted, #94a3b8);
        font-size: 0.68rem;
        font-weight: 600;
        font-family: inherit;
        letter-spacing: 0.01em;
        user-select: none;
      }

      .message-stats {
        font-size: 0.7rem;
        color: var(--text-muted, #94a3b8);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        display: inline-flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin-right: auto;
        user-select: none;
      }

      .message-stats > span {
        white-space: nowrap;
      }

      .stat-label-short,
      .stat-val-short {
        display: none;
      }

      .message-stats b {
        color: var(--teal, #14d4aa);
        font-weight: 600;
        white-space: nowrap;
      }

      .message-action-btns {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-left: auto;
        flex-shrink: 0;
      }

      @media (max-width: 768px), (pointer: coarse) {
        .btn-action {
          min-height: 36px;
          padding: 0 8px;
        }
      }

      @media (max-width: 768px) {
        .message-bubble.user {
          max-width: 88%;
          padding: 9px 14px;
        }

        .message-bubble.assistant {
          padding: 2px 0;
        }

        .message-actions {
          flex-wrap: wrap;
          gap: 6px;
        }

        .message-stats {
          font-size: 0.64rem;
          gap: 6px;
          min-width: 0;
        }

        .btn-action {
          font-size: 0.72rem;
          flex-shrink: 0;
        }
      }

      @media (max-width: 560px) {
        .stat-label,
        .stat-val-full {
          display: none;
        }

        .stat-label-short,
        .stat-val-short {
          display: inline;
        }

        .message-actions:not(:has(.stopped-chip)) {
          flex-wrap: nowrap;
          gap: 4px;
        }

        .message-stats {
          font-size: 0.61rem;
          gap: 3px;
          flex-wrap: nowrap;
          white-space: nowrap;
        }

        .message-stats > .stat-item + .stat-item::before {
          content: '· ';
          color: var(--text-muted, #94a3b8);
          opacity: 0.75;
        }

        .message-action-btns {
          gap: 2px;
        }

        .btn-action {
          min-height: 36px;
          padding: 0 6px;
          font-size: 0.68rem;
          gap: 3px;
        }

        .btn-action svg {
          width: 12px;
          height: 12px;
        }
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

  private renderMarkdownSegments(
      text: string, scopeKey: string, isStreamingBlock: boolean) {
    const htmlResult = this.renderMarkdown(text);
    if (!htmlResult) return '';
    if (!htmlResult.includes('<litert-code-block ')) {
      return renderHtml(htmlResult);
    }

    type Segment =
        | {kind: 'html'; key: string; html: string}
        | {
            kind: 'code';
            key: string;
            base64Code: string;
            language: string;
            preHtml: string;
          };

    const segments: Segment[] = [];
    const codeBlockRe =
        /<litert-code-block base-64-code="([^"]*)" language="([^"]*)">([\s\S]*?)<\/litert-code-block>/g;
    const wrapperTagRe =
        /<(\/?)(ul|ol|li|blockquote|table|thead|tbody|tr|td|th|details)\b[^>]*>/gi;

    let wrapperDepth = 0;
    let lastScanPos = 0;
    let segmentStart = 0;
    let codeIdx = 0;
    let htmlIdx = 0;

    const advanceDepth = (slice: string) => {
      let m: RegExpExecArray|null;
      while ((m = wrapperTagRe.exec(slice)) !== null) {
        wrapperDepth = Math.max(0, wrapperDepth + (m[1] === '/' ? -1 : 1));
      }
    };

    let match: RegExpExecArray|null;
    while ((match = codeBlockRe.exec(htmlResult)) !== null) {
      advanceDepth(htmlResult.slice(lastScanPos, match.index));
      lastScanPos = match.index + match[0].length;
      if (wrapperDepth !== 0) {
        continue;
      }
      const before = htmlResult.slice(segmentStart, match.index);
      if (before.trim().length > 0) {
        segments.push({
          kind: 'html',
          key: `${scopeKey}:html:${htmlIdx++}`,
          html: before,
        });
      }
      const language =
          (match[2] ?? '').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
      segments.push({
        kind: 'code',
        key: `${scopeKey}:code:${codeIdx++}`,
        base64Code: match[1] ?? '',
        language,
        preHtml: match[3] ?? '',
      });
      segmentStart = lastScanPos;
    }

    if (segments.length === 0) {
      return renderHtml(htmlResult);
    }

    const tail = htmlResult.slice(segmentStart);
    if (tail.trim().length > 0) {
      segments.push({
        kind: 'html',
        key: `${scopeKey}:html:${htmlIdx++}`,
        html: tail,
      });
    }

    return repeat(
        segments,
        seg => seg.key,
        (seg, idx) => seg.kind === 'code' ? html`<litert-code-block
          .base64Code=${seg.base64Code}
          .language=${seg.language}
          .streaming=${isStreamingBlock && idx === segments.length - 1}
        >${renderHtml(seg.preHtml)}</litert-code-block>` :
                                            renderHtml(seg.html),
    );
  }

  override connectedCallback() {
    super.connectedCallback();
    if (this.state?.addHost) {
      this.state.addHost(this);
      this.registeredState = this.state;
    }
  }

  override disconnectedCallback() {
    window.clearTimeout(this.copyResetTimer);
    this.copyResetTimer = undefined;
    if (this.registeredState?.removeHost) {
      this.registeredState.removeHost(this);
      this.registeredState = null;
    }
    super.disconnectedCallback();
  }

  protected override willUpdate(changedProperties: PropertyValues<this>) {
    if (changedProperties.has('state') &&
        this.state !== this.registeredState && this.isConnected) {
      this.registeredState?.removeHost?.(this);
      this.state?.addHost?.(this);
      this.registeredState = this.state ?? null;
    }
    const convId = this.state?.chatSession?.activeSavedConvId ?? 'draft';
    const role = this.message?.role ?? '';
    const messageKey = `${convId}:${this.index}:${role}`;
    if (messageKey !== this.lastMessageRoleAndIndex) {
      this.lastMessageRoleAndIndex = messageKey;
      this.userToggledThought = null;
      this.isCopied = false;
      clearCodeBlockScrollState(this.shadowRoot);
    }
  }

  private handleCopyMessage() {
    copyToClipboard(this.message.text)
        .then(() => {
          this.isCopied = true;
          window.clearTimeout(this.copyResetTimer);
          this.copyResetTimer = window.setTimeout(() => {
            this.isCopied = false;
          }, 2000);
        })
        .catch(err => console.error('[LiteRT-LM] Failed to copy message:', err));
  }

  /** Number of messages after this one, which redo and rewind discard. */
  private get laterMessageCount(): number {
    const messages = this.state?.chatSession?.messages;
    return messages ? Math.max(0, messages.length - 1 - this.index) : 0;
  }

  private async handleRetryResponse() {
    const count = this.laterMessageCount;
    if (count > 0 &&
        !await this.state.requestConfirm({
          title: 'Regenerate from earlier turn?',
          message: `Regenerating this response will discard ${count} later ${
              count === 1 ? 'message' : 'messages'} below it.`,
          confirmLabel: 'Regenerate',
          danger: true,
        })) {
      return;
    }
    await this.state.chatSession.redoResponse(this.index);
  }

  private async handleRewindEdit() {
    const count = this.laterMessageCount;
    if (count > 0 &&
        !await this.state.requestConfirm({
          title: 'Rewind and edit message?',
          message: `Moving this prompt back into the composer will discard ${
              count} ${count === 1 ? 'response' : 'subsequent messages'}.`,
          confirmLabel: 'Rewind & Edit',
          danger: true,
        })) {
      return;
    }
    const originalParts =
        await this.state.chatSession.rewindAndEdit(this.index);
    if (originalParts) {
      this.dispatchEvent(new CustomEvent('edit-prompt', {
        detail: {parts: originalParts},
        bubbles: true,
        composed: true,
      }));
    }
  }

  private handleThoughtSummaryClick(e: Event) {
    e.preventDefault();
    const currentlyOpen =
        this.userToggledThought ?? Boolean(this.streaming && !this.message?.text);
    this.userToggledThought = !currentlyOpen;
    this.requestUpdate();
  }

  /** Renders a user message's text with its images and audio inline. */
  private renderUserContent(msg: StoredMessage) {
    if (!msg.parts) {
      return html`<div class="message-user-text">${msg.text}</div>`;
    }
    const parts = msg.parts;
    const isBlockMediaPart = (idx: number): boolean => {
      const p = parts[idx];
      if (!p || p.type === 'text') return false;
      return Boolean(this.state?.chatSession?.getMedia(p.id));
    };

    return html`<div class="message-user-text">${parts.map((part, i) => {
      if (part.type === 'text') {
        let text = part.text;
        if (i > 0 && isBlockMediaPart(i - 1)) {
          text = text.replace(/^[ \t]+/, '');
        }
        if (i < parts.length - 1 && isBlockMediaPart(i + 1)) {
          text = text.replace(/[ \t]+$/, '');
        }
        return text;
      }
      const media = this.state.chatSession.getMedia(part.id);
      if (!media) {
        const icon = part.type === 'image' ? '🖼' : '🎵';
        return html`<span class="message-media-missing" title="Images and audio are not saved across page reloads">${
            icon} ${part.name}</span>`;
      }
      return part.type === 'image' ?
          html`<img class="message-image" src=${media.url} alt=${
              part.name} title=${part.name} />` :
          html`<audio class="message-audio" controls src=${media.url} title=${
              part.name}></audio>`;
    })}</div>`;
  }

  override render() {
    const msg = this.message;
    const isUser = msg.role === 'user';
    const convId = this.state?.chatSession?.activeSavedConvId ?? 'draft';
    const messageKey = `${convId}:${this.index}:${msg.role}`;

    // Clean size-free model names for bubble headers
    let senderName = msg.senderName;
    if (!isUser) {
      senderName = senderName
        .replace(/,\s*\d+(\.\d+)?\s*GB\)$/, ')')
        .replace(/\s*\(\d+(\.\d+)?\s*GB\)$/, '');
    }

    // While streaming, don't render a trailing line of bare markdown
    // punctuation; it would flash as a heading or divider (see
    // AMBIGUOUS_TAIL_RE).
    const visible = (text = '') =>
        this.streaming ? trimAmbiguousTail(text) : text;
    const visibleText = visible(msg.text);

    const activeToolName =
        this.streaming ? this.state?.chatSession?.activeToolName : null;
    const completedToolCalls = (msg.toolCalls ?? [])
        .filter(tc => !this.streaming || tc.status !== 'started');
    const showPreTokenPulse = !isUser && this.streaming && !msg.text &&
        !msg.thoughtText && !activeToolName && completedToolCalls.length === 0;
    const isAnyGenerating =
        this.disabled || Boolean(this.state?.chatSession?.isGenerating);
    const isThoughtOpen =
        this.userToggledThought ?? Boolean(this.streaming && !msg.text);
    const hasSpeedStats = Boolean(msg.prefillSpeed || msg.decodeSpeed);
    const hasPartialTokensOnly = !hasSpeedStats &&
        Boolean(msg.tokensCount && msg.tokensCount !== '-');
    const showStatsBlock =
        hasSpeedStats || hasPartialTokensOnly || Boolean(msg.stopped);
    const isStreamingBody =
        !isUser && this.streaming && Boolean(visibleText);

    return html`
      <div class="message-bubble ${msg.role}" data-raw-text="${msg.text}">
        ${isUser ? html`
          <span class="message-sender user sr-only">User</span>
        ` : html`
          <div class="bubble-header">
            <span class="message-sender assistant">${senderName}</span>
          </div>
        `}

        <div class="message-content">
          <!-- Thought block rendering (open while actively thinking, collapsed once answer arrives) -->
          ${msg.thoughtText ? html`
            <details class="thought-details" ?open=${isThoughtOpen}>
              <summary class="thought-summary" @click=${this.handleThoughtSummaryClick}><svg class="thought-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"></polyline></svg><span class="thought-summary-label">Thought Process</span></summary>
              <div class="thought-content">${this.renderMarkdownSegments(
                visible(msg.thoughtText), `${messageKey}:thought`,
                Boolean(this.streaming && !msg.text))}</div>
            </details>
          ` : ''}

          <!-- Persistent disclosure for completed/failed tool calls -->
          ${completedToolCalls.map(tc => html`
            <details class="tool-details ${tc.status === 'error' ? 'tool-error' : ''}">
              <summary class="tool-summary">
                <svg class="thought-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <svg class="tool-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
                <span>${tc.status === 'error' ? `Tool failed: ${tc.name}` : `Ran ${tc.name}`}</span>
              </summary>
              <div class="tool-details-body">
                ${tc.argsText ? html`
                  <div class="tool-io-section">
                    <span class="tool-io-label">Code</span>
                    <pre class="tool-io-pre"><code>${tc.argsText}</code></pre>
                  </div>
                ` : ''}
                ${tc.outputText ? html`
                  <div class="tool-io-section">
                    <span class="tool-io-label">${tc.status === 'error' ? 'Error' : 'Result'}</span>
                    <pre class="tool-io-pre"><code>${tc.outputText}</code></pre>
                  </div>
                ` : ''}
              </div>
            </details>
          `)}

          ${activeToolName ? html`
            <div class="tool-status-pill" role="status">
              <span>⚙ Running ${activeToolName}...</span>
            </div>
          ` : ''}

          <!-- Main message bubble content -->
          ${isUser ?
              this.renderUserContent(msg) :
              showPreTokenPulse ?
              html`<div class="streaming-placeholder" aria-live="polite"><span class="pulse-dots" aria-hidden="true"><span></span><span></span><span></span></span><span>Thinking...</span></div>` :
              (!msg.text && msg.stopped && !msg.thoughtText && completedToolCalls.length === 0) ?
              html`<p class="stopped-empty-note">Response stopped before any tokens were generated.</p>` :
              html`<div class=${isStreamingBody ? 'streaming-markdown' : ''}>${
                  this.renderMarkdownSegments(
                      visibleText, `${messageKey}:body`,
                      isStreamingBody)}</div>`}
        </div>

        <!-- Assistant message actions bar (hidden while streaming) -->
        ${!isUser && !this.streaming ? html`
          <div class="message-actions">
            ${showStatsBlock ? html`
              <div
                class="message-stats"
                title=${hasSpeedStats ?
                    `Prefill: ${msg.prefillSpeed || '-'} · Decode: ${msg.decodeSpeed || '-'} · Tokens: ${msg.tokensCount || '-'}` :
                    (msg.tokensCount ? `Tokens: ${msg.tokensCount}` : '')}
              >
                ${msg.stopped ? html`
                  <span class="stopped-chip" title="Generation stopped by user">
                    <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3"></rect></svg>
                    <span>Stopped</span>
                  </span>
                ` : ''}
                ${hasSpeedStats ? html`
                  <span class="stat-item"><span class="stat-label">Prefill: </span><span class="stat-label-short" aria-hidden="true">P: </span><b class="stat-val-full">${msg.prefillSpeed || '-'}</b><b class="stat-val-short" aria-hidden="true">${(msg.prefillSpeed || '-').replace(/\s*tk\/s$/i, '')}</b></span>
                  <span class="stat-item"><span class="stat-label">Decode: </span><span class="stat-label-short" aria-hidden="true">D: </span><b>${msg.decodeSpeed || '-'}</b></span>
                  <span class="stat-item"><span class="stat-label">Tokens: </span><b>${msg.tokensCount || '-'}</b><span class="stat-label-short" aria-hidden="true"> tok</span></span>
                ` : hasPartialTokensOnly ? html`
                  <span class="stat-item"><span class="stat-label">Tokens: </span><b>${msg.tokensCount}</b><span class="stat-label-short" aria-hidden="true"> tok</span></span>
                ` : ''}
              </div>
            ` : ''}

            <div class="message-action-btns">
              <button
                type="button"
                class="btn-action"
                ?disabled=${!msg.text}
                @click=${this.handleCopyMessage}
              >${this.isCopied ? html`
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>
                <span>Copied!</span>
              ` : html`
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <span>Copy</span>
              `}</button>
              <button
                type="button"
                class="btn-action"
                ?disabled=${isAnyGenerating}
                @click=${this.handleRetryResponse}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                <span>Retry</span>
              </button>
            </div>
          </div>
        ` : ''}
      </div>

      ${isUser ? html`
        <div class="user-meta-row">
          ${msg.tokensCount && msg.tokensCount !== '-' ? html`
            <span class="user-token-badge">${msg.tokensCount} tokens</span>
          ` : ''}
          <button
            type="button"
            class="btn-action"
            ?disabled=${isAnyGenerating}
            @click=${this.handleRewindEdit}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
            <span>Edit</span>
          </button>
        </div>
      ` : ''}
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-chat-bubble': LitertChatBubble;
  }
}
