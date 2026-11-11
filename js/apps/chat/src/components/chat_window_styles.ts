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

/** Styles for `<litert-chat-window>`. */
export const chatWindowStyles = css`
  :host {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    width: 100%;
    overflow: hidden;
    position: relative;
    container-type: inline-size;
    background-color: var(--bg-chat, #0d1017);
  }

  /* Top status/error banners */
  .banner-stack {
    width: 100%;
    max-width: 820px;
    margin: 0 auto;
    padding: 12px 20px 0 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
  }

  .banner-stack:empty {
    display: none;
  }

  .alert-banner {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 12px;
    background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
    border: 1px solid var(--danger, #f87171);
    color: var(--text, #f1f5f9);
    font-size: 0.8125rem;
    line-height: 1.45;
  }

  .alert-banner-left {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    min-width: 0;
    flex: 1;
  }

  .alert-banner-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
    flex: 1;
  }

  .alert-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }

  .btn-alert-action {
    height: 28px;
    padding: 0 11px;
    border-radius: 8px;
    border: 1px solid var(--border-strong, #364156);
    background-color: var(--bg-card, #171c26);
    color: var(--text, #f1f5f9);
    font-family: inherit;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: background-color 0.14s ease, border-color 0.14s ease;
  }

  .btn-alert-action:hover:not(:disabled) {
    background-color: var(--bg-elevated, #1e2432);
    border-color: var(--teal, #14d4aa);
  }

  .btn-alert-action.primary {
    border-color: var(--danger, #f87171);
  }

  .btn-alert-action:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .alert-icon {
    color: var(--danger, #f87171);
    flex-shrink: 0;
    margin-top: 2px;
  }

  .btn-dismiss-alert {
    background: none;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 0.85rem;
    line-height: 1;
    flex-shrink: 0;
  }

  .btn-dismiss-alert:hover {
    color: var(--text, #f1f5f9);
    background-color: var(--bg-elevated, #1e2432);
  }

  .spinner {
    width: 15px;
    height: 15px;
    border-radius: 50%;
    border: 2px solid var(--border-strong, #364156);
    border-top-color: var(--teal, #14d4aa);
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-progress-bar-wrap {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background-color: var(--bg-input, #10141c);
    overflow: hidden;
  }

  .loading-progress-bar-fill {
    height: 100%;
    background-color: var(--teal, #14d4aa);
    border-radius: 999px;
    transition: width 0.15s ease;
  }

  /* Scrollable Message Stream + Top Fade Scrim */
  .messages-viewport {
    position: relative;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .messages-viewport.has-scroll-top::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 14px;
    background: linear-gradient(
      to bottom,
      color-mix(in srgb, var(--bg-chat, #0d1017) 55%, transparent) 0%,
      color-mix(in srgb, var(--bg-chat, #0d1017) 18%, transparent) 55%,
      transparent 100%
    );
    pointer-events: none;
    z-index: 10;
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0 20px 28px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    background-color: var(--bg-chat, #0d1017);
  }

  .chat-messages > :first-child {
    margin-top: 20px;
  }

  .chat-messages > .empty-hero:first-child {
    margin-top: auto;
  }

  litert-chat-bubble {
    width: 100%;
    max-width: 780px;
    margin: 0 auto;
  }

  /* Empty State Hero */
  .empty-hero {
    margin: auto;
    width: 100%;
    max-width: 680px;
    padding: 20px 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 16px;
  }

  .hero-mark {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background-color: var(--bg-card, #171c26);
    border: 1px solid var(--border, #262f40);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.3));
  }

  .hero-mark img {
    width: 26px;
    height: 26px;
    object-fit: contain;
    display: block;
  }

  .hero-heading {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin: 0;
    color: var(--text, #f1f5f9);
  }

  .hero-subtitle {
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--text-muted, #94a3b8);
    max-width: 540px;
    margin: 0;
  }

  .capability-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    row-gap: 4px;
    column-gap: 0;
    margin: 2px 0 0 0;
    padding: 0;
    list-style: none;
  }

  .capability-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0;
    border: none;
    border-radius: 0;
    background: transparent;
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-muted, #94a3b8);
    white-space: nowrap;
    cursor: default;
  }

  .capability-tag:not(:last-child)::after {
    content: '·';
    margin-left: 9px;
    margin-right: 9px;
    color: var(--text-muted, #94a3b8);
  }

  .capability-tag svg {
    color: var(--text-muted, #94a3b8);
    flex-shrink: 0;
  }

  .starters-container {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    width: 100%;
    margin-top: 8px;
  }

  @container (max-width: 620px) {
    .starters-container {
      grid-template-columns: 1fr;
      gap: 8px;
    }
  }

  .btn-starter {
    background-color: var(--bg-card, #171c26);
    border: 1px solid var(--border, #262f40);
    color: var(--text, #f1f5f9);
    padding: 12px 14px;
    border-radius: 12px;
    font-family: inherit;
    cursor: pointer;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    text-align: left;
    transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.12s ease, opacity 0.15s ease;
  }

  .btn-starter svg {
    color: var(--teal, #14d4aa);
    flex-shrink: 0;
    margin-top: 2px;
  }

  .starter-copy {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
  }

  .starter-title {
    font-size: 0.8125rem;
    font-weight: 600;
    line-height: 1.25;
    color: var(--text, #f1f5f9);
  }

  .starter-subtitle {
    font-size: 0.72rem;
    font-weight: 400;
    line-height: 1.35;
    color: var(--text-muted, #94a3b8);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .btn-starter:hover:not(:disabled) {
    background-color: var(--bg-elevated, #1e2432);
    border-color: var(--border-strong, #364156);
    transform: translateY(-1px);
  }

  .btn-starter:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* Floating Scroll-to-bottom button anchored dynamically above composer */
  .btn-scroll-bottom {
    position: absolute;
    left: auto;
    right: max(24px, calc(50% - 390px));
    bottom: calc(100% + 4px);
    z-index: 15;
    height: 28px;
    padding: 0 11px;
    border-radius: 999px;
    background-color: color-mix(in srgb, var(--bg-elevated, #1e2432) 94%, var(--bg-chat, #0d1017));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-strong, #364156);
    color: var(--text, #f1f5f9);
    font-size: 0.72rem;
    font-weight: 600;
    font-family: inherit;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
    box-shadow: var(--shadow-md, 0 8px 24px rgba(0, 0, 0, 0.45));
    transition: background-color 0.15s ease, border-color 0.15s ease;
    white-space: nowrap;
  }

  .btn-scroll-bottom:hover {
    background-color: var(--bg-elevated, #1e2432);
  }

  /* Modern 2-Row Rounded Composer Card */
  .chat-input-container {
    position: relative;
    z-index: 30;
    padding: 8px 20px 18px 20px;
    background-color: var(--bg-chat, #0d1017);
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
  }

  .chat-input-container.has-scroll-above::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: 100%;
    height: 24px;
    opacity: 1;
    background: linear-gradient(
      to bottom,
      transparent 0%,
      color-mix(in srgb, var(--bg-chat, #0d1017) 55%, transparent) 55%,
      var(--bg-chat, #0d1017) 100%
    );
    pointer-events: none;
    z-index: 10;
  }

  /* Inline Model Preparation & Download Progress Strip inside Composer */
  .composer-loading-bar {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 4px 4px 8px 4px;
    border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
  }

  .composer-loading-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 0.76rem;
    color: var(--text, #f1f5f9);
  }

  .composer-loading-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
    min-width: 0;
    overflow: hidden;
  }

  .composer-loading-title > span:last-child {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .composer-loading-right {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .composer-loading-stats {
    font-size: 0.69rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-muted, #94a3b8);
    white-space: nowrap;
  }

  .composer-loading-sub {
    display: none;
    font-size: 0.68rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-muted, #94a3b8);
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .btn-composer-cancel {
    height: 24px;
    padding: 0 9px;
    border-radius: 999px;
    border: 1px solid var(--border, #262f40);
    background-color: var(--bg-card, #171c26);
    color: var(--danger, #f87171);
    font-size: 0.7rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    transition: background-color 0.14s ease, border-color 0.14s ease;
  }

  .btn-composer-cancel:hover {
    background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
    border-color: var(--danger, #f87171);
  }

  .input-wrapper {
    position: relative;
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 780px;
    background-color: var(--bg-composer, #151a24);
    border: 1px solid var(--border, #262f40);
    border-radius: 20px;
    padding: 10px 12px 10px 14px;
    gap: 6px;
    box-shadow: var(--shadow-composer, 0 4px 20px rgba(0, 0, 0, 0.28));
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
  }

  .input-wrapper:focus-within {
    border-color: var(--border-strong, #364156);
    box-shadow: 0 0 0 2px var(--teal-subtle, rgba(20, 212, 170, 0.14)), var(--shadow-composer, 0 4px 20px rgba(0, 0, 0, 0.28));
  }

  .input-wrapper.drag-over {
    border-color: var(--teal, #14d4aa);
    background-color: var(--teal-subtle, rgba(20, 212, 170, 0.10));
  }

  /*
   * The prompt editor is a contenteditable so that images can be placed
   * inline between words. Its children are managed imperatively (see
   * setEditorParts) because re-rendering them with Lit on every keystroke
   * would reset the caret.
   */
  .editor {
    position: relative;
    width: 100%;
    min-height: 44px;
    max-height: 180px;
    overflow-y: auto;
    background: transparent;
    border: none;
    color: var(--text, #f1f5f9);
    padding: 6px 4px;
    font-size: 0.925rem;
    font-family: inherit;
    line-height: 1.55;
    outline: none;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  .editor[contenteditable='false'] {
    opacity: 0.65;
    cursor: not-allowed;
  }

  /*
   * Positioned out of flow: as in-flow generated content it would push
   * the caret after the placeholder text. Offsets match the padding.
   */
  .editor.is-empty::before {
    content: attr(data-placeholder);
    position: absolute;
    left: 4px;
    top: 6px;
    right: 4px;
    color: var(--text-muted, #94a3b8);
    pointer-events: none;
    user-select: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .composer-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .composer-tools-left {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    flex: 0 1 auto;
  }

  .composer-tools-right {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    flex-shrink: 0;
  }

  .composer-media-tools {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    margin-right: 4px;
  }

  .composer-model-anchor {
    display: inline-flex;
    align-items: center;
    min-width: 0;
    max-width: 100%;
  }

  .btn-composer-model {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 34px;
    max-width: 300px;
    min-width: 0;
    overflow: hidden;
    padding: 0 11px;
    border-radius: 999px;
    border: 1px solid var(--border, #262f40);
    background-color: var(--bg-card, #171c26);
    color: var(--text, #f1f5f9);
    font-family: inherit;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  }

  .btn-composer-model:hover {
    background-color: var(--bg-elevated, #1e2432);
    border-color: var(--border-strong, #364156);
  }

  .btn-composer-model.open {
    background-color: var(--bg-elevated, #1e2432);
    border-color: var(--teal, #14d4aa);
    box-shadow: 0 0 0 2px var(--teal-subtle, rgba(20, 212, 170, 0.14));
  }

  .model-pill-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    background-color: var(--text-muted, #94a3b8);
  }

  .model-pill-dot.ready {
    background-color: var(--teal, #14d4aa);
  }

  .model-pill-dot.generating {
    background-color: var(--teal, #14d4aa);
    animation: pillPulse 1.1s ease-in-out infinite;
  }

  .model-pill-dot.loading {
    background-color: var(--warning, #fbbf24);
    animation: pillPulse 1.4s ease-in-out infinite;
  }

  .model-pill-dot.error {
    background-color: var(--danger, #f87171);
  }

  @keyframes pillPulse {
    0%, 100% { opacity: 0.35; }
    50% { opacity: 1; }
  }

  .model-pill-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .model-pill-state {
    font-size: 0.68rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-muted, #94a3b8);
    flex-shrink: 0;
  }

  .model-pill-state.state-compact {
    display: none;
  }

  .model-pill-chevron {
    width: 12px;
    height: 12px;
    color: var(--text-muted, #94a3b8);
    flex-shrink: 0;
    transition: transform 0.18s ease, color 0.15s ease;
  }

  .btn-composer-model.open .model-pill-chevron {
    transform: rotate(180deg);
    color: var(--teal, #14d4aa);
  }

  .composer-add-model-popover {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    width: min(460px, 100%);
    max-width: calc(100vw - 24px);
    padding: 10px;
    border-radius: 12px;
    border: 1px solid var(--border-strong, #364156);
    background-color: var(--bg-card, #171c26);
    box-shadow: var(--shadow-md, 0 10px 28px rgba(0, 0, 0, 0.48));
    z-index: 100000;
    box-sizing: border-box;
  }

  #composer-add-model-input:focus,
  #composer-add-model-input:focus-visible {
    border-color: var(--teal, #14d4aa);
    outline: 2px solid var(--teal, #14d4aa);
    outline-offset: 1px;
    box-shadow: 0 0 0 3px var(--teal-subtle, rgba(20, 212, 170, 0.22));
  }

  .composer-add-model-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted, #94a3b8);
    margin-bottom: 6px;
    padding: 0 2px;
  }

  .composer-add-model-popover .add-model-form {
    margin-top: 0;
  }

  .media-chip {
    display: inline-flex;
    align-items: center;
    vertical-align: middle;
    gap: 6px;
    background: var(--bg-card, #171c26);
    border: 1px solid var(--border, #262f40);
    border-radius: 8px;
    padding: 2px 7px 2px 3px;
    margin: 2px 3px;
    font-size: 0.76rem;
    max-width: 230px;
  }

  .media-chip::selection,
  .media-chip *::selection {
    background: var(--teal-subtle, rgba(20, 212, 170, 0.25));
  }

  .chip-thumb {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    object-fit: cover;
    flex-shrink: 0;
  }

  .chip-play-btn {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: 1px solid var(--border, #262f40);
    background: var(--bg-input, #10141c);
    color: var(--teal, #14d4aa);
    cursor: pointer;
    font-size: 0.82rem;
    flex-shrink: 0;
    user-select: none;
  }

  .chip-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    user-select: none;
    color: var(--text, #f1f5f9);
  }

  .chip-del-btn {
    background: transparent;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    padding: 2px 3px;
    font-size: 0.75rem;
    line-height: 1;
    user-select: none;
  }

  .chip-del-btn:hover {
    color: var(--danger, #f87171);
  }

  .btn-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted, #94a3b8);
    font-size: 0.95rem;
    font-family: inherit;
    cursor: pointer;
    transition: background-color 0.14s ease, color 0.14s ease, border-color 0.14s ease;
    flex-shrink: 0;
    padding: 0;
  }

  .btn-icon:hover:not(:disabled) {
    background: var(--bg-elevated, #1e2432);
    color: var(--text, #f1f5f9);
  }

  .btn-icon:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .btn-icon.recording {
    width: auto;
    padding: 0 10px;
    gap: 6px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--danger, #f87171);
    background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
    border-color: var(--danger, #f87171);
  }

  .btn-cancel-record {
    height: 30px;
    padding: 0 10px;
    border-radius: 999px;
    border: 1px solid var(--border, #262f40);
    background-color: var(--bg-card, #171c26);
    color: var(--text-muted, #94a3b8);
    font-size: 0.72rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
  }

  .btn-cancel-record:hover {
    color: var(--text, #f1f5f9);
    background-color: var(--bg-elevated, #1e2432);
  }

  .btn-input {
    height: 36px;
    padding: 0 16px;
    font-size: 0.8125rem;
    font-weight: 600;
    border-radius: 999px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .btn-input:disabled {
    background-color: var(--bg-elevated, #1e2432);
    border-color: var(--border, #262f40);
    color: var(--text-muted, #94a3b8);
    opacity: 0.72;
    cursor: not-allowed;
  }

  .btn-spinner {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    border: 2px solid currentColor;
    border-top-color: transparent;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  #btn-send .send-icon-download {
    display: none;
  }

  @media (max-width: 768px) {
    .banner-stack {
      padding: 10px 12px 0 12px;
    }

    .chat-messages {
      padding: 0 12px 24px 12px;
      gap: 14px;
    }

    .chat-messages > :first-child {
      margin-top: 14px;
    }

    .chat-messages > .empty-hero:first-child {
      margin-top: auto;
    }

    .hero-heading {
      font-size: 1.22rem;
    }

    .hero-subtitle {
      font-size: 0.82rem;
    }

    .starters-container {
      grid-template-columns: 1fr;
      gap: 8px;
    }

    .btn-starter {
      padding: 10px 12px;
    }

    .chat-input-container {
      padding: 6px 10px 12px 10px;
    }

    .btn-scroll-bottom {
      right: 12px;
      bottom: calc(100% + 3px);
      height: 26px;
      padding: 0 10px;
      font-size: 0.69rem;
    }

    .input-wrapper {
      border-radius: 18px;
      padding: 8px 10px 8px 12px;
    }

    .btn-composer-model {
      height: 32px;
      padding: 0 9px;
      gap: 5px;
      font-size: 0.74rem;
      max-width: 220px;
    }

    .model-pill-state:not(.active-state) {
      display: none;
    }

    .composer-media-tools {
      gap: 1px;
      margin-right: 2px;
    }

    .composer-media-tools .btn-icon:not(.recording) {
      width: 30px;
      height: 30px;
    }

    .btn-input {
      height: 32px;
      padding: 0 11px;
      font-size: 0.75rem;
      gap: 5px;
    }
  }

  @media (max-width: 600px) {
    .capability-row {
      column-gap: 14px;
      row-gap: 6px;
    }

    .capability-tag:not(:last-child)::after {
      display: none;
    }

    .composer-loading-speed-inline {
      display: none;
    }

    .composer-loading-sub {
      display: block;
    }

    .composer-toolbar {
      flex-wrap: nowrap;
      gap: 6px;
    }

    .composer-tools-left {
      flex: 1 1 0;
      min-width: 0;
    }

    .btn-composer-model {
      max-width: 100%;
      min-width: 0;
    }

    .model-pill-state.state-full {
      display: none;
    }

    .model-pill-state.state-compact {
      display: inline;
    }

    .composer-media-tools.is-recording #btn-add-media,
    .composer-media-tools.is-recording #btn-camera {
      display: none;
    }

    #btn-send {
      width: 34px;
      min-width: 34px;
      height: 34px;
      padding: 0;
      border-radius: 999px;
      justify-content: center;
      flex-shrink: 0;
    }

    #btn-send .send-btn-label {
      display: none;
    }

    #btn-send.needs-download .send-icon-up {
      display: none;
    }

    #btn-send.needs-download .send-icon-download {
      display: block;
    }
  }

  @media (max-width: 380px) {
    .chat-input-container {
      padding: 6px 8px 10px 8px;
    }

    .input-wrapper {
      padding: 8px 8px 8px 10px;
    }

    .composer-toolbar {
      gap: 4px;
    }

    .composer-tools-right {
      gap: 2px;
    }

    .composer-media-tools {
      gap: 0;
      margin-right: 0;
    }

    .composer-media-tools .btn-icon:not(.recording) {
      width: 27px;
      height: 28px;
    }

    .btn-composer-model {
      padding: 0 7px;
      gap: 4px;
      font-size: 0.7rem;
    }

    .model-pill-state {
      font-size: 0.64rem;
    }

    #btn-send {
      width: 32px;
      min-width: 32px;
      height: 32px;
    }
  }
`;
