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

import {css} from 'lit';

/** Shared CSS styles for the chat components. */
export const sharedStyles = css`
  * {
    box-sizing: border-box;
  }

  /* Custom subtle scrollbars across Shadow DOM */
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }

  ::-webkit-scrollbar-track {
    background: transparent;
  }

  ::-webkit-scrollbar-thumb {
    background: var(--border, #262f40);
    border-radius: 999px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: var(--border-strong, #364156);
  }

  label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted, #94a3b8);
    letter-spacing: 0.02em;
  }

  select, input[type="number"], input[type="text"], textarea {
    background-color: var(--bg-input, #10141c);
    border: 1px solid var(--border, #262f40);
    border-radius: 8px;
    padding: 8px 10px;
    color: var(--text, #f1f5f9);
    font-size: 0.8125rem;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease, opacity 0.15s ease;
    font-family: inherit;
  }

  select {
    appearance: none;
    -webkit-appearance: none;
    padding-right: 28px;
    cursor: pointer;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    background-size: 13px 13px;
  }

  select:disabled,
  input[type="number"]:disabled,
  input[type="text"]:disabled,
  textarea:disabled {
    opacity: 0.48;
    cursor: not-allowed;
    background-color: var(--bg-sidebar, #131720);
    color: var(--text-muted, #94a3b8);
  }

  select:hover:not(:disabled),
  input[type="number"]:hover:not(:disabled),
  input[type="text"]:hover:not(:disabled),
  textarea:hover:not(:disabled) {
    border-color: var(--border-strong, #364156);
  }

  select:focus, input:focus, textarea:focus {
    border-color: var(--teal, #14d4aa);
    box-shadow: 0 0 0 2px var(--teal-subtle, rgba(20, 212, 170, 0.15));
  }

  button:focus-visible,
  a:focus-visible,
  summary:focus-visible,
  [tabindex]:focus-visible {
    outline: 2px solid var(--teal, #14d4aa);
    outline-offset: 2px;
  }

  /* Base buttons */
  .btn {
    font-weight: 600;
    padding: 9px 16px;
    border-radius: 9px;
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 0.8125rem;
    line-height: 1.2;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease, opacity 0.15s ease, transform 0.1s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-family: inherit;
    white-space: nowrap;
  }

  .btn:active:not(:disabled) {
    transform: translateY(0.5px);
  }

  .btn-primary {
    background-color: var(--teal-btn-bg, #14d4aa);
    color: var(--teal-btn-text, #041f18);
  }

  .btn-primary:hover:not(:disabled) {
    background-color: var(--teal-btn-hover, #2ee0b8);
  }

  .btn-stop {
    background-color: var(--danger-btn-bg, #dc2626) !important;
    color: #ffffff !important;
  }

  .btn-stop:hover:not(:disabled) {
    background-color: var(--danger-btn-hover, #b91c1c) !important;
  }

  .btn-secondary {
    background-color: var(--bg-elevated, #1e2432);
    border: 1px solid var(--border, #262f40);
    color: var(--text, #f1f5f9);
  }

  .btn-secondary:hover:not(:disabled) {
    background-color: var(--border, #262f40);
    border-color: var(--border-strong, #364156);
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* Shadow DOM Highlight.js Theme Tokens (Dark + Light WCAG AA) */
  .hljs {
    color: var(--hljs-fg, #e2e8f0);
  }
  .hljs-comment,
  .hljs-quote {
    color: var(--hljs-comment, #94a3b8);
    font-style: italic;
  }
  .hljs-keyword,
  .hljs-selector-tag,
  .hljs-literal,
  .hljs-type,
  .hljs-built_in {
    color: var(--hljs-keyword, #7dd3fc);
  }
  .hljs-string,
  .hljs-regexp,
  .hljs-addition,
  .hljs-template-variable {
    color: var(--hljs-string, #6ee7b7);
  }
  .hljs-number,
  .hljs-symbol,
  .hljs-bullet {
    color: light-dark(#92400e, #fcd34d);
  }
  .hljs-title,
  .hljs-section,
  .hljs-function .hljs-title {
    color: var(--hljs-title, #f9a8d4);
  }
  .hljs-attr,
  .hljs-attribute,
  .hljs-variable,
  .hljs-params {
    color: var(--hljs-attr, #93c5fd);
  }
  .hljs-name,
  .hljs-tag {
    color: var(--hljs-tag, #5eead4);
  }
`;
