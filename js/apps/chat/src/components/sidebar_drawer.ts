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

import {css, html, LitElement, nothing, svg, SVGTemplateResult} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

import {LlmChatStateController} from '../state_controller.js';
import {getModelKey} from '../stores/model_loader_service.js';
import {getCpuTextMemoryWarning, getKnownModelSizeBytes, getModalityConfigs, getNonDefaultBackendSummary, isGpuOnlyTextModel, Modality, MODALITIES, PartialSettingsSchema, Settings, SettingsGroup} from '../stores/settings_store.js';
import {CHAT_TOOLS} from '../stores/tools.js';
import {sharedStyles} from '../styles/shared_styles.js';

import {getDeepActiveElement, getFocusableElementsDeep} from './focus_util.js';

/* tslint:disable:no-new-decorators */

/** Collapsible settings panels available in the sidebar. */
type SettingsPage = 'model-config'|'system-prompt'|'experimental';

const RESET_CONFIRMATIONS: Record<SettingsGroup, {title: string; message: string}> = {
  modelConfig: {
    title: 'Reset model configuration?',
    message:
        'Restore Context Length, Max Output, Sampler, Temperature, Top-P, Top-K, Thinking Mode, and Speculative Decoding to their default values?',
  },
  systemPrompt: {
    title: 'Reset system prompt & tools?',
    message:
        'Clear your custom System Prompt and re-enable all built-in tools?',
  },
  experimental: {
    title: 'Reset experimental settings?',
    message:
        'Restore Text, Vision, and Audio backends and activation precision to their default values?',
  },
};

/** Component representing the sidebar drawer. */
@customElement('litert-sidebar')
export class LitertSidebar extends LitElement {
  @property({type: Object})
  state!: LlmChatStateController;

  /** Currently expanded settings disclosure, or null when collapsed. */
  @state() private openSettingsPage: SettingsPage|null = null;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        height: 100%;
        width: 100%;
        padding: 0;
        box-sizing: border-box;
        gap: 0;
        overflow: hidden;
        color: var(--text, #f1f5f9);
      }

      .section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted, #94a3b8);
        margin: 0;
      }

      .mobile-sidebar-header {
        display: none;
      }

      /* Middle Zone: Conversations + Upward-Expanding Settings Tray */
      .sidebar-middle {
        flex: 1 1 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .conversations-section {
        display: flex;
        flex-direction: column;
        flex: 1 1 118px;
        min-height: 118px;
        padding: 12px 14px 8px 14px;
        overflow: hidden;
      }

      .conversations-list {
        display: flex;
        flex-direction: column;
        gap: 3px;
        flex: 1 1 0;
        min-height: 0;
        overflow: hidden;
        padding: 0;
        margin: 0;
      }

      .recent-scroll-list {
        display: flex;
        flex-direction: column;
        gap: 3px;
        flex: 1 1 0;
        min-height: 0;
        overflow-y: auto;
        overscroll-behavior: contain;
        padding: 0;
        margin: 0;
      }

      .recent-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted, #94a3b8);
        padding: 8px 6px 4px 6px;
      }

      .empty-convs-hint {
        font-size: 0.75rem;
        color: var(--text-muted, #94a3b8);
        padding: 10px 8px;
        line-height: 1.45;
      }

      .conv-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 4px;
        padding: 3px 6px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.8rem;
        color: var(--text-secondary, #cbd5e1);
        transition: background-color 0.14s ease, color 0.14s ease, border-color 0.14s ease;
        user-select: none;
        border: 1px solid transparent;
      }

      .conv-item:not(.active):hover {
        background-color: var(--bg-card, #171c26);
        color: var(--text, #f1f5f9);
      }

      .conv-item.active {
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border, #262f40);
        color: var(--text, #f1f5f9);
        font-weight: 600;
      }

      .new-chat-item {
        padding: 8px 10px;
        margin-bottom: 4px;
        border-radius: 9px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        color: var(--text, #f1f5f9);
        font-weight: 600;
      }

      .new-chat-item:hover {
        background-color: var(--bg-elevated, #1e2432) !important;
        border-color: var(--border-strong, #364156);
      }

      .new-chat-item.is-empty-chat {
        color: var(--text-secondary, #cbd5e1);
      }

      .conv-select-btn {
        flex: 1;
        min-width: 0;
        background: none;
        border: none;
        color: inherit;
        font: inherit;
        text-align: left;
        padding: 5px 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .new-chat-item .conv-select-btn {
        padding: 0;
      }

      .conv-title {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .btn-delete-conv {
        background: none;
        border: none;
        color: var(--text-muted, #94a3b8);
        cursor: pointer;
        font-size: 0.75rem;
        padding: 4px 6px;
        border-radius: 6px;
        transition: color 0.14s ease, background-color 0.14s ease;
        flex-shrink: 0;
      }

      .btn-delete-conv:hover {
        color: var(--danger, #f87171) !important;
        background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
      }

      /* Upward-Expanding Settings Tray + Anchored Bottom Dock */
      @keyframes trayExpandUp {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .settings-tray {
        flex: 0 1 auto;
        max-height: calc(100% - 118px);
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        background-color: var(--bg-input, #10141c);
        border-top: 1px solid var(--border-strong, #364156);
        box-shadow: 0 -8px 20px rgba(0, 0, 0, 0.18);
        animation: trayExpandUp 0.16s cubic-bezier(0.2, 0, 0, 1);
      }

      .settings-tray-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 7px 14px 6px 14px;
        border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        flex-shrink: 0;
      }

      .settings-tray-title {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted, #94a3b8);
        white-space: nowrap;
        min-width: 0;
      }

      .exp-badge {
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 1px 5px;
        border-radius: 999px;
        color: var(--warning-text, var(--warning, #fbbf24));
        background-color: color-mix(
            in srgb, var(--warning, #fbbf24) 11%, transparent);
        border: 1px solid color-mix(
            in srgb, var(--warning, #fbbf24) 32%, transparent);
      }

      .btn-close-tray {
        background: none;
        border: none;
        color: var(--text-muted, #94a3b8);
        font-size: 0.74rem;
        line-height: 1;
        padding: 3px 5px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.14s ease, color 0.14s ease;
      }

      .btn-close-tray:hover {
        background-color: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
      }

      .sidebar-bottom-dock {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        background-color: var(--bg-sidebar, #131720);
      }

      .settings-accordions {
        border-top: 1px solid var(--border, #262f40);
        border-bottom: 1px solid var(--border, #262f40);
        background-color: color-mix(in srgb, var(--bg-input, #10141c) 65%, var(--bg-sidebar, #131720));
        display: flex;
        flex-direction: column;
        gap: 0;
        flex-shrink: 0;
      }

      .settings-disclosure {
        display: flex;
        flex-direction: column;
      }

      .settings-disclosure + .settings-disclosure {
        border-top: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
      }

      .settings-disclosure-btn {
        width: 100%;
        height: 34px;
        min-height: 34px;
        max-height: 34px;
        box-sizing: border-box;
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        padding: 0 14px 0 11px;
        cursor: pointer;
        user-select: none;
        font-family: inherit;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary, #cbd5e1);
        transition: background-color 0.14s ease, color 0.14s ease, border-color 0.14s ease;
        white-space: nowrap;
        text-align: left;
        overflow: hidden;
      }

      .settings-disclosure-btn:focus-visible {
        outline: 2px solid var(--teal, #14d4aa);
        outline-offset: -2px;
      }

      .settings-disclosure-btn:hover {
        background-color: var(--bg-card, #171c26);
        color: var(--text, #f1f5f9);
      }

      .settings-disclosure-btn.active {
        background-color: var(--teal-subtle, rgba(20, 212, 170, 0.12));
        border-left-color: var(--teal, #14d4aa);
        color: var(--text, #f1f5f9);
      }

      .tuning-summary-left {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        flex: 1;
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
      }

      .tuning-summary-left svg {
        flex-shrink: 0;
        color: var(--text-muted, #94a3b8);
        transition: color 0.14s ease;
      }

      .settings-disclosure-btn.active .tuning-summary-left svg {
        color: var(--teal, #14d4aa);
      }

      .tuning-summary-left > span {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .tuning-summary-right {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        flex-shrink: 0;
      }

      .tuning-pill {
        font-size: 0.67rem;
        font-weight: 500;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: var(--text-muted, #94a3b8);
        background: none;
        border: none;
        padding: 0;
        white-space: nowrap;
        flex-shrink: 0;
      }

      .settings-disclosure-btn.active .tuning-pill {
        color: var(--teal, #14d4aa);
      }

      .tuning-chevron {
        width: 13px;
        height: 13px;
        color: var(--text-muted, #94a3b8);
        transition: transform 0.18s ease, color 0.14s ease;
        flex-shrink: 0;
      }

      .settings-disclosure-btn.active .tuning-chevron {
        transform: rotate(180deg);
        color: var(--teal, #14d4aa);
      }

      .tuning-body {
        flex: 1 1 auto;
        min-height: 0;
        padding: 10px 14px 12px 14px;
        overflow-y: auto;
        overscroll-behavior: contain;
        scrollbar-width: thin;
        scrollbar-color: var(--border-strong, #364156) transparent;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .tuning-body::-webkit-scrollbar {
        width: 5px;
      }

      .tuning-body::-webkit-scrollbar-track {
        background: transparent;
      }

      .tuning-body::-webkit-scrollbar-thumb {
        background-color: var(--border-strong, #364156);
        border-radius: 999px;
      }

      .tuning-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }

      .field-col {
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
      }

      .field-col.dimmed input {
        background-color: var(--bg-sidebar, #131720);
        color: var(--text-muted, #94a3b8);
      }

      .field-col label,
      .field-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted, #94a3b8);
        margin: 0;
      }

      .field-col input,
      .field-col select {
        width: 100%;
        padding: 6px 8px;
        font-size: 0.76rem;
      }

      #system-prompt {
        width: 100%;
        height: 76px;
        min-height: 76px;
        max-height: 180px;
        resize: vertical;
        font-size: 0.75rem;
        line-height: 1.45;
        padding: 8px 10px;
        box-sizing: border-box;
      }

      .setting-row {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .model-toggles,
      .tools-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .toggle-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin: 0;
        padding: 7px 9px;
        border-radius: 8px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        cursor: pointer;
        user-select: none;
        font-size: inherit;
        font-weight: 400;
        letter-spacing: normal;
        color: var(--text, #f1f5f9);
        transition: background-color 0.14s ease, border-color 0.14s ease, box-shadow 0.14s ease;
      }

      .toggle-row:hover {
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border-strong, #364156);
      }

      .toggle-row:has(input[type="checkbox"]:focus-visible) {
        border-color: var(--teal, #14d4aa);
        box-shadow: 0 0 0 2px var(--teal-subtle, rgba(20, 212, 170, 0.2));
        outline: 2px solid var(--teal, #14d4aa);
        outline-offset: 1px;
      }

      .toggle-title {
        margin: 0;
        cursor: pointer;
        text-transform: none;
        letter-spacing: normal;
        font-size: 0.76rem;
        font-weight: 600;
        line-height: normal;
        color: var(--text, #f1f5f9);
      }

      .toggle-row input[type="checkbox"] {
        flex-shrink: 0;
        width: 15px;
        height: 15px;
        margin: 0;
        accent-color: var(--teal, #14d4aa);
        cursor: pointer;
      }

      .toggle-row input[type="checkbox"]:focus,
      .toggle-row input[type="checkbox"]:focus-visible {
        outline: none;
        box-shadow: none;
      }

      .tool-text {
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 0;
      }

      .setting-hint {
        font-size: 0.67rem;
        color: var(--text-muted, #94a3b8);
        line-height: 1.38;
      }

      .tuning-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding-top: 2px;
      }

      .btn-reset-sm {
        font-size: 0.72rem;
        padding: 5px 10px;
        flex-shrink: 0;
      }

      /* Experimental Backend & Precision Matrix */
      .exp-matrix {
        display: flex;
        flex-direction: column;
        gap: 5px;
      }

      .exp-col-headers,
      .exp-row {
        display: grid;
        grid-template-columns: 56px minmax(0, 0.78fr) minmax(0, 1.22fr);
        gap: 7px;
        align-items: center;
      }

      .exp-col-headers span {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted, #94a3b8);
        letter-spacing: 0.02em;
      }

      .exp-modality-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text, #f1f5f9);
      }

      .exp-row select {
        width: 100%;
        min-width: 0;
        height: 30px;
        padding: 4px 24px 4px 7px;
        font-size: 0.72rem;
        background-position: right 6px center;
        background-size: 12px 12px;
      }

      .experimental-note {
        min-height: 36px;
        box-sizing: border-box;
        padding: 5px 8px;
        border-radius: 7px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        font-size: 0.66rem;
        line-height: 1.32;
        color: var(--text-muted, #94a3b8);
        display: flex;
        align-items: center;
      }

      .experimental-note.warning {
        background-color: color-mix(in srgb, var(--warning, #fbbf24) 10%, var(--bg-card, #171c26));
        border-color: color-mix(in srgb, var(--warning, #fbbf24) 45%, var(--border, #262f40));
        color: var(--warning-text, #fbbf24);
      }

      .experimental-note .nowrap {
        white-space: nowrap;
      }

      /* Compact Runtime Status Footer */
      .sidebar-status-footer {
        flex-shrink: 0;
        padding: 10px 14px calc(12px + env(safe-area-inset-bottom, 0px)) 14px;
      }

      .status-card {
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 9px;
        padding: 8px 11px;
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .status-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        height: 18px;
        min-height: 18px;
        max-height: 18px;
        line-height: 18px;
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text, #f1f5f9);
      }

      .status-header-left {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        min-width: 0;
      }

      .runtime-backend-badge {
        display: inline-flex;
        align-items: center;
        height: 16px;
        padding: 0 6px;
        border-radius: 999px;
        background-color: var(--bg-elevated, #1e2432);
        border: 1px solid var(--border, #262f40);
        color: var(--text-secondary, #cbd5e1);
        font-family: inherit;
        font-size: 0.62rem;
        font-weight: 600;
        line-height: 1;
        white-space: nowrap;
        flex-shrink: 0;
      }

      .status-header:has(.runtime-backend-badge) .load-time-prefix {
        display: none;
      }

      .status-indicator {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: var(--text-muted, #94a3b8);
        flex-shrink: 0;
      }

      .status-indicator.loading {
        background-color: var(--warning, #fbbf24);
        animation: pulse 1.4s infinite;
      }

      .status-indicator.ready {
        background-color: var(--teal, #14d4aa);
      }

      .status-indicator.error {
        background-color: var(--danger, #f87171);
      }

      @keyframes pulse {
        0%, 100% { opacity: 0.35; }
        50% { opacity: 1; }
      }

      .load-time-pill {
        font-size: 0.68rem;
        line-height: 1;
        white-space: nowrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: var(--text-muted, #94a3b8);
      }

      .load-time-pill b {
        color: var(--teal, #14d4aa);
        font-weight: 600;
      }

      .status-text {
        height: 18px;
        min-height: 18px;
        max-height: 18px;
        line-height: 18px;
        font-size: 0.73rem;
        color: var(--text-muted, #94a3b8);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      @media (max-height: 740px) {
        .conversations-section {
          flex-basis: 98px;
          min-height: 98px;
          padding: 8px 14px 4px 14px;
        }

        .new-chat-item {
          padding: 6px 10px;
          margin-bottom: 2px;
        }

        .recent-scroll-list {
          padding-bottom: 2px;
        }

        .recent-label {
          padding: 2px 6px 2px 6px;
        }

        .settings-tray {
          max-height: calc(100% - 98px);
        }

        .settings-disclosure-btn {
          height: 30px;
          min-height: 30px;
          max-height: 30px;
          font-size: 0.73rem;
        }

        .tuning-body {
          padding: 7px 12px 8px 12px;
          gap: 5px;
        }

        .tuning-grid {
          gap: 6px;
        }

        .field-col input,
        .field-col select {
          padding: 5px 8px;
        }

        #system-prompt {
          height: 48px;
          min-height: 48px;
          padding: 6px 9px;
        }

        .exp-matrix {
          gap: 4px;
        }

        .exp-row select {
          height: 28px;
          padding: 3px 22px 3px 7px;
        }

        .experimental-note {
          height: 30px;
          min-height: 30px;
          max-height: 30px;
          padding: 3px 8px;
          font-size: 0.64rem;
          line-height: 1.25;
        }

        .toggle-row {
          padding: 5px 9px;
        }

        .sidebar-status-footer {
          padding: 8px 14px 10px 14px;
        }
      }

      @media (max-height: 700px) {
        .conversations-section {
          padding: 6px 14px 4px 14px;
        }

        .conversations-list,
        .recent-scroll-list {
          gap: 2px;
        }

        .new-chat-item {
          padding: 5px 10px;
          margin-bottom: 1px;
        }

        .recent-label {
          font-size: 0.65rem;
          padding: 1px 6px 1px 6px;
        }

        .conv-item:not(.new-chat-item) {
          padding: 2px 6px;
        }

        .conv-item:not(.new-chat-item) .conv-select-btn {
          padding: 3px 4px;
        }

        .settings-tray-header {
          padding: 5px 12px 4px 12px;
        }

        .tuning-body {
          padding: 5px 12px 6px 12px;
          gap: 4px;
        }

        .tuning-grid {
          gap: 5px;
        }

        .field-col {
          gap: 2px;
        }

        .field-col label,
        .field-label {
          font-size: 0.64rem;
          line-height: 1.15;
        }

        .field-col input,
        .field-col select {
          padding: 4px 7px;
          font-size: 0.74rem;
        }

        .model-toggles,
        .tools-list {
          gap: 4px;
        }

        .toggle-row {
          padding: 4px 8px;
          gap: 8px;
        }

        .toggle-title {
          font-size: 0.73rem;
        }

        .tool-text {
          gap: 1px;
        }

        .setting-hint {
          font-size: 0.64rem;
          line-height: 1.24;
        }

        .tuning-footer {
          padding-top: 1px;
        }

        .btn-reset-sm {
          padding: 3px 9px;
          font-size: 0.7rem;
        }

        .sidebar-status-footer {
          padding: 6px 14px 8px 14px;
        }

        .status-card {
          padding: 7px 11px;
          gap: 3px;
        }
      }

      @media (min-width: 769px) and (max-width: 960px) {
        .conversations-section {
          padding-left: 11px;
          padding-right: 11px;
        }

        .settings-disclosure-btn {
          padding: 0 11px 0 9px;
          font-size: 0.72rem;
        }

        .tuning-pill {
          font-size: 0.63rem;
        }

        .tuning-body {
          padding-left: 11px;
          padding-right: 11px;
        }

        .exp-col-headers,
        .exp-row {
          grid-template-columns: 50px minmax(0, 0.78fr) minmax(0, 1.22fr);
          gap: 5px;
        }

        .exp-row select {
          font-size: 0.68rem;
          padding-right: 18px;
          background-position: right 6px center;
        }

        .experimental-note {
          height: 42px;
          min-height: 42px;
          max-height: 42px;
          font-size: 0.61rem;
          line-height: 1.2;
        }

        .sidebar-status-footer {
          padding-left: 11px;
          padding-right: 11px;
        }

        .status-card {
          padding-left: 9px;
          padding-right: 9px;
        }

        .status-header,
        .status-header-left {
          gap: 5px;
        }

        .runtime-backend-badge {
          padding: 0 5px;
          font-size: 0.59rem;
        }

        .load-time-prefix {
          display: none;
        }

        .status-text {
          font-size: 0.68rem;
        }
      }

      @media (max-height: 620px) {
        .conversations-section {
          flex-basis: 86px;
          min-height: 86px;
        }

        .settings-tray {
          max-height: calc(100% - 86px);
        }
      }

      .btn-dismiss-sidebar {
        display: none;
      }

      @media (max-width: 768px) {
        .mobile-sidebar-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 14px 10px 14px;
          border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
          flex-shrink: 0;
        }

        .conversations-section {
          padding-top: 10px;
        }

        .btn-dismiss-sidebar {
          display: inline-flex;
          align-items: center;
          background-color: var(--bg-card, #171c26);
          border: 1px solid var(--border, #262f40);
          color: var(--text, #f1f5f9);
          border-radius: 6px;
          font-size: 0.72rem;
          font-weight: 600;
          font-family: inherit;
          padding: 4px 10px;
          cursor: pointer;
          transition: background-color 0.15s ease;
        }

        .btn-dismiss-sidebar:hover {
          background-color: var(--bg-elevated, #1e2432);
        }
      }
    `
  ];

  override connectedCallback() {
    super.connectedCallback();

    // Receive state updates from the centralized state controller.
    this.state.addHost(this);
    this.renderRoot.addEventListener(
        'keydown', this.handleHostKeyDown as EventListener);
  }

  override disconnectedCallback() {
    this.renderRoot.removeEventListener(
        'keydown', this.handleHostKeyDown as EventListener);
    this.state.removeHost(this);
    super.disconnectedCallback();
  }

  /**
   * Focuses the logical last bottom-dock control when Shift+Tabbing back into
   * the sidebar while the bottommost settings disclosure (Experimental) is open.
   */
  focusLastBottomElement(): boolean {
    if (this.openSettingsPage !== 'experimental') return false;
    if (this.renderRoot.querySelector('#btn-cancel-download')) return false;
    const tray = this.renderRoot.querySelector<HTMLElement>('.settings-tray');
    const trayFocusables = tray ? getFocusableElementsDeep(tray) : [];
    const last = trayFocusables[trayFocusables.length - 1];
    if (!last) return false;
    last.focus();
    return true;
  }

  private closeSettingsTray(restoreFocus = true): boolean {
    const closedPage = this.openSettingsPage;
    if (!closedPage) return false;
    this.openSettingsPage = null;
    if (restoreFocus) {
      const triggerId = `#tab-${closedPage}`;
      const focusTrigger = () => {
        this.renderRoot.querySelector<HTMLButtonElement>(triggerId)?.focus();
      };
      focusTrigger();
      void this.updateComplete.then(focusTrigger);
    }
    return true;
  }

  private readonly handleHostKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && !e.defaultPrevented) {
      if (this.dismissTopTransientMenu()) {
        e.preventDefault();
      }
      return;
    }
    if (e.key !== 'Tab' || e.defaultPrevented || this.openSettingsPage === null) {
      return;
    }

    const active = getDeepActiveElement();
    if (!active) return;

    const tray = this.renderRoot.querySelector<HTMLElement>('.settings-tray');
    const trayFocusables = tray ? getFocusableElementsDeep(tray) : [];
    const convSection =
        this.renderRoot.querySelector<HTMLElement>('.conversations-section');
    const convFocusables =
        convSection ? getFocusableElementsDeep(convSection) : [];
    const tabModel =
        this.renderRoot.querySelector<HTMLElement>('#tab-model-config');
    const tabSys =
        this.renderRoot.querySelector<HTMLElement>('#tab-system-prompt');
    const tabExp =
        this.renderRoot.querySelector<HTMLElement>('#tab-experimental');
    const orderedPages: SettingsPage[] = [
      'model-config',
      'system-prompt',
      'experimental',
    ];
    const orderedTabs = [tabModel, tabSys, tabExp];
    const openIdx = orderedPages.indexOf(this.openSettingsPage);
    const ownerTab = openIdx >= 0 ? orderedTabs[openIdx] : null;
    const nextTab = openIdx >= 0 ? orderedTabs[openIdx + 1] : null;

    const cancelDownloadBtn =
        this.renderRoot.querySelector<HTMLElement>('#btn-cancel-download');
    const dismissSidebarBtn =
        this.renderRoot.querySelector<HTMLElement>('#btn-dismiss-sidebar');
    const isDismissSidebarVisible = Boolean(
        dismissSidebarBtn &&
        (dismissSidebarBtn.offsetParent !== null ||
         dismissSidebarBtn.getClientRects().length > 0));

    const firstTray = trayFocusables[0];
    const lastTray = trayFocusables[trayFocusables.length - 1];
    const lastConv = convFocusables[convFocusables.length - 1];

    if (!e.shiftKey) {
      if (lastConv && active === lastConv && tabModel) {
        e.preventDefault();
        tabModel.focus();
        return;
      }
      if (ownerTab && active === ownerTab && firstTray) {
        e.preventDefault();
        firstTray.focus();
        return;
      }
      if (lastTray && active === lastTray) {
        if (nextTab) {
          e.preventDefault();
          nextTab.focus();
          return;
        }
        if (cancelDownloadBtn) {
          e.preventDefault();
          cancelDownloadBtn.focus();
          return;
        }
        if (isDismissSidebarVisible && dismissSidebarBtn) {
          e.preventDefault();
          dismissSidebarBtn.focus();
          return;
        }
        // Desktop: move focus synchronously to #tab-experimental without
        // preventDefault so native Tab advances past the sidebar into <main>.
        ownerTab?.focus();
        return;
      }
    } else {
      if (active === tabModel && lastConv) {
        e.preventDefault();
        lastConv.focus();
        return;
      }
      if (firstTray && active === firstTray && ownerTab) {
        e.preventDefault();
        ownerTab.focus();
        return;
      }
      if (nextTab && active === nextTab && lastTray) {
        e.preventDefault();
        lastTray.focus();
        return;
      }
      if (!nextTab && lastTray) {
        if (cancelDownloadBtn && active === cancelDownloadBtn) {
          e.preventDefault();
          lastTray.focus();
          return;
        }
        if (!cancelDownloadBtn && isDismissSidebarVisible &&
            active === dismissSidebarBtn) {
          e.preventDefault();
          lastTray.focus();
          return;
        }
      }
    }
  };

  /**
   * Dismisses the open settings tray inside the sidebar on Escape,
   * returning true when an inner control handled the Escape key.
   */
  dismissTopTransientMenu(): boolean {
    if (this.openSettingsPage !== null) {
      return this.closeSettingsTray(true);
    }
    return false;
  }

  private async handleResetSettings(group: SettingsGroup = 'modelConfig') {
    const confirmed = await this.state.requestConfirm({
      ...RESET_CONFIRMATIONS[group],
      confirmLabel: 'Reset Defaults',
      danger: false,
    });
    if (confirmed) this.state.settings.resetDefaults(group);
  }

  private handleSettingInput(e: Event, prop: keyof Settings) {
    const target =
        e.target as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
    let val: unknown;

    if (target instanceof HTMLInputElement) {
      if (target.type === 'checkbox') {
        val = target.checked;
      } else {
        val = target.type === 'number' ? Number(target.value) : target.value;
      }
    } else {
      val = target.value;
    }

    this.updateSettings({[prop]: val});
  }

  /** Validates and saves changes to the settings. */
  private updateSettings(changes: Record<string, unknown>) {
    const parseResult = PartialSettingsSchema.safeParse(changes);
    if (parseResult.success) {
      Object.assign(this.state.settings, parseResult.data);
      this.state.settings.saveSettings();
    } else {
      console.warn('[LiteRT-LM] Invalid settings input:', parseResult.error);
    }
  }

  private handleToolToggle(e: Event, toolName: string) {
    const enabled = (e.target as HTMLInputElement).checked;
    const disabledTools =
        (this.state.settings.disabledTools ?? []).filter(n => n !== toolName);
    if (!enabled) disabledTools.push(toolName);
    this.updateSettings({disabledTools});
  }

  private dismissSidebar() {
    this.dispatchEvent(new CustomEvent('close', {
      bubbles: true,
      composed: true,
    }));
  }

  private dismissSidebarIfMobile() {
    if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      this.dismissSidebar();
    }
  }

  private renderModelConfigPanel(sampler: string) {
    const greedyReason =
        'Not used with Greedy sampling — select Top-K or Top-P to adjust Temperature, Top-P, and Top-K.';
    const topPReason = 'Top-P is only used when Top-P sampling is selected.';
    const tempDisabledReason = sampler === 'greedy' ? greedyReason : null;
    const topPDisabledReason = sampler === 'greedy' ?
        greedyReason :
        (sampler === 'top_k' ? topPReason : null);
    const topPDescribedBy = sampler === 'greedy' ?
        'greedy-disabled-desc' :
        (sampler === 'top_k' ? 'top-p-disabled-desc' : nothing);

    return html`
      <div
        id="panel-model-config"
        class="tuning-body"
        role="region"
        aria-labelledby="tab-model-config"
      >
        <div class="tuning-grid">
          <div class="field-col">
            <label for="context-length">Context Length</label>
            <input
              type="number"
              id="context-length"
              .value=${String(this.state.settings.contextLength)}
              min="256"
              step="256"
              @input=${(e: Event) => this.handleSettingInput(e, 'contextLength')}
            />
          </div>
          <div class="field-col">
            <label for="max-output-tokens">Max Output</label>
            <input
              type="number"
              id="max-output-tokens"
              .value=${String(this.state.settings.maxOutputTokens)}
              min="128"
              step="64"
              @input=${(e: Event) => this.handleSettingInput(e, 'maxOutputTokens')}
            />
          </div>
        </div>

        <div class="tuning-grid">
          <div class="field-col">
            <label for="sampler-type">Sampler</label>
            <select
              id="sampler-type"
              .value=${this.state.settings.samplerType}
              @change=${(e: Event) => this.handleSettingInput(e, 'samplerType')}
            >
              <option value="greedy">Greedy</option>
              <option value="top_k">Top-K</option>
              <option value="top_p">Top-P</option>
            </select>
          </div>
          <div
            class="field-col ${sampler === 'greedy' ? 'dimmed' : ''}"
            title=${tempDisabledReason ?? nothing}
          >
            <label for="temperature">Temperature</label>
            <input
              type="number"
              id="temperature"
              .value=${String(this.state.settings.temperature)}
              min="0.0"
              max="2.0"
              step="0.1"
              ?disabled=${sampler === 'greedy'}
              title=${tempDisabledReason ?? nothing}
              aria-describedby=${sampler === 'greedy' ? 'greedy-disabled-desc' : nothing}
              @input=${(e: Event) => this.handleSettingInput(e, 'temperature')}
            />
          </div>
        </div>

        <div class="tuning-grid">
          <div
            class="field-col ${sampler !== 'top_p' ? 'dimmed' : ''}"
            title=${topPDisabledReason ?? nothing}
          >
            <label for="top-p">Top-P</label>
            <input
              type="number"
              id="top-p"
              .value=${String(this.state.settings.topP)}
              min="0.0"
              max="1.0"
              step="0.05"
              ?disabled=${sampler !== 'top_p'}
              title=${topPDisabledReason ?? nothing}
              aria-describedby=${topPDescribedBy}
              @input=${(e: Event) => this.handleSettingInput(e, 'topP')}
            />
          </div>
          <div
            class="field-col ${sampler === 'greedy' ? 'dimmed' : ''}"
            title=${tempDisabledReason ?? nothing}
          >
            <label for="top-k">Top-K</label>
            <input
              type="number"
              id="top-k"
              .value=${String(this.state.settings.topK)}
              min="1"
              max="128"
              step="1"
              ?disabled=${sampler === 'greedy'}
              title=${tempDisabledReason ?? nothing}
              aria-describedby=${sampler === 'greedy' ? 'greedy-disabled-desc' : nothing}
              @input=${(e: Event) => this.handleSettingInput(e, 'topK')}
            />
          </div>
        </div>

        ${sampler === 'greedy' ? html`
          <span id="greedy-disabled-desc" class="sr-only">${greedyReason}</span>
        ` : sampler === 'top_k' ? html`
          <span id="top-p-disabled-desc" class="sr-only">${topPReason}</span>
        ` : nothing}

        <div class="model-toggles">
          <label class="toggle-row" for="enable-thinking">
            <span class="tool-text">
              <span id="enable-thinking-label" class="toggle-title">Thinking mode</span>
              <span id="enable-thinking-desc" class="setting-hint">Include step-by-step reasoning before answering.</span>
            </span>
            <input
              type="checkbox"
              id="enable-thinking"
              aria-labelledby="enable-thinking-label"
              aria-describedby="enable-thinking-desc"
              .checked=${this.state.settings.enableThinking}
              @change=${(e: Event) => this.handleSettingInput(e, 'enableThinking')}
            />
          </label>

          <label class="toggle-row" for="enable-speculative-decoding">
            <span class="tool-text">
              <span id="enable-speculative-decoding-label" class="toggle-title">Speculative decoding</span>
              <span id="enable-speculative-decoding-desc" class="setting-hint">Adaptive drafting on models with a drafter (e.g. MTP).</span>
            </span>
            <input
              type="checkbox"
              id="enable-speculative-decoding"
              aria-labelledby="enable-speculative-decoding-label"
              aria-describedby="enable-speculative-decoding-desc"
              .checked=${Boolean(this.state.settings.enableSpeculativeDecoding)}
              @change=${(e: Event) => this.handleSettingInput(e, 'enableSpeculativeDecoding')}
            />
          </label>
        </div>

        <div class="tuning-footer">
          <span class="setting-hint">Applies on next message</span>
          <button
            id="btn-reset-settings"
            type="button"
            class="btn btn-secondary btn-reset-sm"
            @click=${() => this.handleResetSettings('modelConfig')}
          >Reset</button>
        </div>
      </div>
    `;
  }

  private renderSystemPromptPanel() {
    const settings = this.state.settings;
    const disabledTools = settings.disabledTools ?? [];
    return html`
      <div
        id="panel-system-prompt"
        class="tuning-body"
        role="region"
        aria-labelledby="tab-system-prompt"
      >
        <div class="field-col">
          <label for="system-prompt">System Instructions</label>
          <textarea
            id="system-prompt"
            aria-label="System prompt"
            placeholder="Optional instructions for tone, persona, or constraints…"
            .value=${settings.systemPrompt ?? ''}
            @input=${(e: Event) => this.handleSettingInput(e, 'systemPrompt')}
            @change=${(e: Event) => this.handleSettingInput(e, 'systemPrompt')}
          ></textarea>
        </div>

        <div class="setting-row">
          <span class="field-label">Tools</span>
          <div class="tools-list">
            ${CHAT_TOOLS.map(tool => html`
              <label class="toggle-row" for="tool-${tool.name}">
                <span class="tool-text">
                  <span id="tool-${tool.name}-label" class="toggle-title">${tool.label}</span>
                  <span id="tool-${tool.name}-desc" class="setting-hint">${tool.summary}</span>
                </span>
                <input
                  type="checkbox"
                  id="tool-${tool.name}"
                  aria-labelledby="tool-${tool.name}-label"
                  aria-describedby="tool-${tool.name}-desc"
                  .checked=${!disabledTools.includes(tool.name)}
                  @change=${(e: Event) => this.handleToolToggle(e, tool.name)}
                />
              </label>
            `)}
          </div>
        </div>

        <div class="tuning-footer">
          <span class="setting-hint">Applies on next message</span>
          <button
            id="btn-reset-system-prompt"
            type="button"
            class="btn btn-secondary btn-reset-sm"
            @click=${() => this.handleResetSettings('systemPrompt')}
          >Reset</button>
        </div>
      </div>
    `;
  }

  private getExperimentalBadge(): string {
    const configs = getModalityConfigs(this.state.settings);
    const overrides = MODALITIES.flatMap(modality => {
      const {backend, precision} = configs[modality];
      const prefix = modality[0].toUpperCase();
      if (backend === 'off') return [`${prefix}:Off`];
      if (backend === 'cpu') return [`${prefix}:CPU`];
      if (precision !== 'default') {
        return [`${prefix}:${precision.toUpperCase()}`];
      }
      return [];
    });
    if (overrides.length === 0) {
      return 'GPU · Default';
    }
    if (overrides.length <= 2) {
      return overrides.join(' · ');
    }
    return `${overrides.slice(0, 2).join(' · ')} +${overrides.length - 2}`;
  }

  private renderExperimentalPanel() {
    const settings = this.state.settings;
    const gpuOnlyText =
        isGpuOnlyTextModel(settings.selectedModelPath, settings);
    const configs = getModalityConfigs(settings);
    const cachedSizeBytes = this.state.modelLoader.cachedModels?.get(
        getModelKey(settings.selectedModelPath));
    const knownSizeBytes = getKnownModelSizeBytes(
        settings.selectedModelPath, settings, cachedSizeBytes);
    const cpuMemoryWarning = getCpuTextMemoryWarning(
        settings.selectedModelPath, settings, cachedSizeBytes);
    const showCpuMemoryWarning =
        !gpuOnlyText && configs.text.backend === 'cpu' &&
        Boolean(cpuMemoryWarning);

    const textBackendReason =
        'This model uses a WebGPU-only text decoder (-gpu.litertlm). Choose a standard .litertlm model to try CPU.';
    const cpuPrecisionReason =
        'CPU backend always executes in FP32 / native graph precision.';
    const offPrecisionReason =
        'Turn this modality on (GPU or CPU) to configure activation precision.';

    const anyModalityOff =
        configs.vision.backend === 'off' || configs.audio.backend === 'off';
    const noteContent = gpuOnlyText ? html`
          <span>This model's text is <span class="nowrap">WebGPU-only</span> (<span class="nowrap">-gpu.litertlm</span>). Use a standard <span class="nowrap">.litertlm</span> for CPU.</span>
        ` : showCpuMemoryWarning ? html`
          <span>${cpuMemoryWarning}</span>
        ` : configs.text.backend === 'cpu' && knownSizeBytes === null ? html`
          <span>Text CPU runs in WASM (FP32). May not fit in WASM memory (<span class="nowrap">~1.5 GB</span> practical limit); loading may fail.</span>
        ` : configs.text.backend === 'cpu' ? html`
          <span>Text CPU runs in WASM (FP32; <span class="nowrap">~1.5 GB</span> practical WASM memory limit).</span>
        ` : anyModalityOff ? html`
          <span>Off modalities skip weight loading to save memory. GPU defaults: Text FP16, Vision/Audio FP32.</span>
        ` : (configs.vision.backend === 'cpu' || configs.audio.backend === 'cpu') ? html`
          <span>CPU modalities run in WASM (FP32). GPU defaults: Text FP16, Vision/Audio FP32.</span>
        ` : html`
          <span>Experimental overrides. Set Vision or Audio to Off to skip loading weights.</span>
        `;

    const renderRow = (modality: Modality, label: string,
                       defaultPrecision: string) => {
      const {backend, precision} = configs[modality];
      const backendLocked = modality === 'text' && gpuOnlyText;
      const canTurnOff = modality !== 'text';
      const isOff = backend === 'off';
      const onCpu = backend === 'cpu';
      const backendTitle = backendLocked ?
          textBackendReason :
          (modality === 'text' && cpuMemoryWarning ? cpuMemoryWarning :
                                                     nothing);
      return html`
          <div class="exp-row">
            <span class="exp-modality-label">${label}</span>
            <select
              id="${modality}-backend"
              aria-label="${label} backend"
              .value=${backend}
              ?disabled=${backendLocked}
              title=${backendTitle}
              aria-describedby=${
          modality === 'text' ? 'experimental-note' : nothing}
              @change=${(e: Event) =>
          this.handleSettingInput(e, `${modality}Backend`)}
            >
              <option value="gpu">GPU</option>
              <option value="cpu">CPU</option>
              ${canTurnOff ? html`<option value="off">Off</option>` : nothing}
            </select>
            <select
              id="${modality}-precision"
              aria-label="${label} precision"
              .value=${isOff ? 'off' : (onCpu ? 'fp32' : precision)}
              ?disabled=${isOff || onCpu}
              title=${
          isOff ? offPrecisionReason :
                  (onCpu ? cpuPrecisionReason : nothing)}
              aria-describedby=${
          modality === 'text' || onCpu || isOff ? 'experimental-note' : nothing}
              @change=${(e: Event) =>
          this.handleSettingInput(e, `${modality}Precision`)}
            >
              ${isOff ? html`
                <option value="off">Off</option>
              ` : onCpu ? html`
                <option value="fp32">FP32 (CPU)</option>
              ` : html`
                <option value="default">Default (${defaultPrecision})</option>
                <option value="fp16">FP16</option>
                <option value="fp32">FP32</option>
              `}
            </select>
          </div>
      `;
    };

    return html`
      <div
        id="panel-experimental"
        class="tuning-body"
        role="region"
        aria-labelledby="tab-experimental"
      >
        <div class="exp-matrix">
          <div class="exp-col-headers" aria-hidden="true">
            <span>Modality</span>
            <span>Backend</span>
            <span>Precision</span>
          </div>

          ${renderRow('text', 'Text', 'FP16')}
          ${renderRow('vision', 'Vision', 'FP32')}
          ${renderRow('audio', 'Audio', 'FP32')}
        </div>

        <div
          id="experimental-note"
          class="experimental-note ${showCpuMemoryWarning ? 'warning' : ''}"
        >
          ${noteContent}
        </div>

        <div class="tuning-footer">
          <span class="setting-hint">Reloads model on next message</span>
          <button
            id="btn-reset-experimental"
            type="button"
            class="btn btn-secondary btn-reset-sm"
            @click=${() => this.handleResetSettings('experimental')}
          >Reset</button>
        </div>
      </div>
    `;
  }

  override render() {
    const activeModelKey = getModelKey(this.state.settings.selectedModelPath);
    const sampler = this.state.settings.samplerType;
    const samplerLabel =
        sampler === 'top_k' ? 'Top-K' : sampler === 'top_p' ? 'Top-P' : 'Greedy';
    const ctxLabel = this.state.settings.contextLength >= 1024 ?
        `${(this.state.settings.contextLength / 1024).toFixed(
               this.state.settings.contextLength % 1024 === 0 ? 0 : 1)}k` :
        String(this.state.settings.contextLength);
    const modelConfigBadge = `${samplerLabel} · ${ctxLabel}${
        this.state.settings.enableSpeculativeDecoding ? '+' : ''}`;
    const disabledTools = this.state.settings.disabledTools ?? [];
    const enabledToolsCount =
        CHAT_TOOLS.filter(t => !disabledTools.includes(t.name)).length;
    const hasCustomSystemPrompt =
        Boolean(this.state.settings.systemPrompt?.trim());
    const systemPromptBadge = `${
        hasCustomSystemPrompt ? 'Custom · ' : 'Tools '}${enabledToolsCount}/${
        CHAT_TOOLS.length}`;

    const isModelConfigOpen = this.openSettingsPage === 'model-config';
    const isSystemPromptOpen = this.openSettingsPage === 'system-prompt';
    const isExperimentalOpen = this.openSettingsPage === 'experimental';
    const disclosures: Array<
        {page: SettingsPage, label: string, badge: string, icon: SVGTemplateResult}> = [
      {
        page: 'model-config',
        label: 'Model Config',
        badge: modelConfigBadge,
        icon: svg`
          <line x1="4" y1="21" x2="4" y2="14"></line>
          <line x1="4" y1="10" x2="4" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12" y2="3"></line>
          <line x1="20" y1="21" x2="20" y2="16"></line>
          <line x1="20" y1="12" x2="20" y2="3"></line>
          <line x1="1" y1="14" x2="7" y2="14"></line>
          <line x1="9" y1="8" x2="15" y2="8"></line>
          <line x1="17" y1="16" x2="23" y2="16"></line>`,
      },
      {
        page: 'system-prompt',
        label: 'System Prompt',
        badge: systemPromptBadge,
        icon: svg`
          <polyline points="4 17 10 11 4 5"></polyline>
          <line x1="12" y1="19" x2="20" y2="19"></line>`,
      },
      {
        page: 'experimental',
        label: 'Experimental',
        badge: this.getExperimentalBadge(),
        icon: svg`
          <path d="M9 3h6"></path>
          <path d="M10 3v6.5L4.8 18.2A2 2 0 0 0 6.5 21h11a2 2 0 0 0 1.7-2.8L14 9.5V3"></path>
          <path d="M7.5 15h9"></path>`,
      },
    ];

    const isErrorStatus = Boolean(this.state.errorMessage);
    const needsReload = Boolean(
        this.state.modelLoader.engine &&
        this.state.modelLoader.needsModelReload?.(
            this.state.settings.modelSettings));
    const isShowingProgress = this.state.modelLoader.isModelLoading &&
        this.state.modelLoader.downloadProgresses.has(activeModelKey);

    return html`
      <!-- 1. Mobile Drawer Header -->
      <div class="mobile-sidebar-header">
        <h2 class="section-title">Menu</h2>
        <button
          id="btn-dismiss-sidebar"
          type="button"
          class="btn-dismiss-sidebar"
          aria-label="Close menu"
          @click=${this.dismissSidebar}
        >Close</button>
      </div>

      <!-- 2. Middle Zone: Conversations + Upward-Expanding Settings Tray -->
      <div class="sidebar-middle">
        <div class="conversations-section">
          <div id="conversations-list" class="conversations-list">
            <div
              class="conv-item new-chat-item ${
                  this.state.chatSession.activeSavedConvId === null &&
                          (this.state.chatSession.messages?.length ?? 0) === 0 ?
                      'is-empty-chat' :
                      ''}"
              @click=${() => {
                void this.state.chatSession.startNewConversation();
                this.dismissSidebarIfMobile();
              }}
            >
              <button type="button" class="conv-select-btn">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <line x1="12" y1="5" x2="12" y2="19"></line>
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                <span class="conv-title">New Chat</span>
              </button>
            </div>

            ${this.state.chatSession.conversationsList.length > 0 ? html`
              <div class="recent-scroll-list">
                <div class="recent-label">Recent</div>
                ${this.state.chatSession.conversationsList.map(conv => html`
                  <div
                    class="conv-item ${
                        this.state.chatSession.activeSavedConvId === conv.id ?
                            'active' :
                            ''}"
                    @click=${() => {
                      void this.state.chatSession.selectConversation(conv.id);
                      this.dismissSidebarIfMobile();
                    }}
                  >
                    <button
                      type="button"
                      class="conv-select-btn"
                      title=${conv.title}
                      aria-current=${
                          this.state.chatSession.activeSavedConvId === conv.id ?
                              'true' :
                              nothing}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex-shrink: 0; opacity: 0.7;">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                      </svg>
                      <span class="conv-title">${conv.title}</span>
                    </button>
                    <button
                      type="button"
                      class="btn-delete-conv"
                      aria-label="Delete Conversation"
                      title="Delete conversation"
                      @click=${(e: Event) => {
                        e.stopPropagation();
                        this.state.chatSession.deleteConversation(conv.id);
                        this.dispatchEvent(new CustomEvent(
                            'conversation-deleted',
                            {bubbles: true, composed: true}));
                      }}
                    >✕</button>
                  </div>
                `)}
              </div>
            ` : html`
              <div class="empty-convs-hint">
                Conversations are saved locally in your browser as you chat.
              </div>
            `}
          </div>
        </div>

        ${isModelConfigOpen || isSystemPromptOpen || isExperimentalOpen ? html`
          <div class="settings-tray">
            <div class="settings-tray-header">
              <span class="settings-tray-title">${
                isModelConfigOpen ? 'Model Configuration' :
                isSystemPromptOpen ? 'System Prompt & Tools' : html`
                  <span>Experimental</span>
                  <span class="exp-badge">Preview</span>
                `}</span>
              <button
                type="button"
                class="btn-close-tray"
                aria-label="Close settings panel"
                title="Close"
                @click=${() => {
                  this.closeSettingsTray(true);
                }}
              >✕</button>
            </div>
            ${isModelConfigOpen  ? this.renderModelConfigPanel(sampler) :
              isSystemPromptOpen ? this.renderSystemPromptPanel() :
                                   this.renderExperimentalPanel()}
          </div>
        ` : nothing}
      </div>

      <!-- 3. Anchored Bottom Dock: Settings Disclosures + Runtime Status -->
      <div class="sidebar-bottom-dock">
        <div class="settings-accordions">
          ${disclosures.map(({page, label, badge, icon}) => {
            const isOpen = this.openSettingsPage === page;
            return html`
              <div class="settings-disclosure">
                <button
                  id="tab-${page}"
                  type="button"
                  class="settings-disclosure-btn ${isOpen ? 'active' : ''}"
                  aria-expanded=${isOpen ? 'true' : 'false'}
                  aria-controls=${isOpen ? `panel-${page}` : nothing}
                  @click=${() => {
                    this.openSettingsPage = isOpen ? null : page;
                  }}
                >
                  <span class="tuning-summary-left">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icon}</svg>
                    <span>${label}</span>
                  </span>
                  <span class="tuning-summary-right">
                    <span class="tuning-pill">${badge}</span>
                    <svg class="tuning-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <polyline points="18 15 12 9 6 15"></polyline>
                    </svg>
                  </span>
                </button>
              </div>
            `;
          })}
        </div>

        <!-- 4. Runtime Status Card -->
        <div class="sidebar-status-footer">
          <div class="status-card">
            <div class="status-header">
              <span class="status-header-left">
                <span
                  id="status-dot"
                  class="status-indicator ${
                      this.state.modelLoader.isModelLoading ? 'loading' :
                      isErrorStatus                         ? 'error' :
                      needsReload                           ? 'loading' :
                      this.state.modelLoader.engine         ? 'ready' :
                                                              ''}"
                ></span>
                <span>Runtime</span>
                ${(() => {
                  const backendOverride =
                      getNonDefaultBackendSummary(this.state.settings);
                  return backendOverride ? html`
                    <span
                      id="runtime-backend-badge"
                      class="runtime-backend-badge"
                      title=${backendOverride.detail}
                    >${backendOverride.label}</span>
                  ` : nothing;
                })()}
              </span>
              ${this.state.modelLoader.metricLoadTime !== '-' ? html`
                <span
                  class="load-time-pill"
                  title="Loaded in ${this.state.modelLoader.metricLoadTime}"
                ><span class="load-time-prefix">Loaded in </span><b>${this.state.modelLoader.metricLoadTime}</b></span>
              ` : nothing}
            </div>
            ${(() => {
              const progressPct = isShowingProgress ?
                  this.state.modelLoader.downloadProgresses.get(activeModelKey) :
                  undefined;
              const statusLineText = progressPct !== undefined ?
                  `${
                      this.state.statusText.includes('Downloading') ?
                          'Downloading weights' :
                          'Preparing model'} · ${progressPct}%` :
                  isErrorStatus ?
                  'Error · see message in chat' :
                  (needsReload && !this.state.modelLoader.isModelLoading) ?
                  'Reloads on next message' :
                  this.state.statusText;
              return html`
                <div
                  id="status-text"
                  class="status-text"
                  title=${statusLineText}
                >${statusLineText}</div>
              `;
            })()}
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-sidebar': LitertSidebar;
  }
}
