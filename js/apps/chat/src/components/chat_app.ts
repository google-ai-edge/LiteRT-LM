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

import './sidebar_drawer';
import './chat_window';
import './learn_more_drawer';

import {css, html, LitElement, nothing} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';
import {FOCUSABLE_SELECTOR, focusNowAndNextFrame, getDeepActiveElement, getFocusableElementsDeep, trapTabKey} from './focus_util.js';
import {registerAppServiceWorker, setSandboxIframeHtml} from './util.js';

import {ConfirmDialogOptions, LlmChatStateController} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';
import type {LitertChatWindow} from './chat_window.js';
import type {CustomDropdown} from './custom_dropdown.js';
import type {LitertSidebar} from './sidebar_drawer.js';

/* tslint:disable:no-new-decorators */

/** The `localStorage` key of the theme override (`light` or `dark`). */
export const THEME_STORAGE_KEY = 'litertlm-theme-mode';

/** Supported theme preference modes. */
export type ThemeMode = 'system'|'light'|'dark';

interface ActiveConfirmModal extends ConfirmDialogOptions {
  resolve: (confirmed: boolean) => void;
  returnFocusEl: HTMLElement|null;
}

/** Main application container for the LiteRT-LM chat interface. */
@customElement('litert-lm-chat-app')
export class LitertLmChatApp extends LitElement {
  // Central source of truth state controller
  private state = new LlmChatStateController(this);

  // Unified sidebar state: true = visible/open, false = hidden/collapsed
  @property({type: Boolean})
  isSidebarOpen = typeof window !== 'undefined' ? window.innerWidth > 768 : true;

  @property({type: Boolean}) isLearnMoreOpen = false;

  @property({type: Boolean}) isPreviewOpen = false;

  @property({type: String}) theme: 'dark'|'light' = 'dark';
  @state() private themeMode: ThemeMode = 'system';

  @state() private activeConfirm: ActiveConfirmModal|null = null;
  @state() private isCameraOpen = false;
  @state()
  private isMobileViewport =
      typeof window !== 'undefined' ? window.innerWidth <= 768 : false;
  private previewReturnFocusEl: HTMLElement|null = null;
  private learnMoreReturnFocusEl: HTMLElement|null = null;
  private mobileSidebarReturnFocusEl: HTMLElement|null = null;
  private lastClickedFocusable: HTMLElement|null = null;
  private lastClickTime = 0;
  private wasUndoToastRendered = false;
  private hadFocusInUndoToast = false;
  private handlingUndoToastClick = false;

  private readonly mobileMediaQuery = typeof window !== 'undefined' ?
      window.matchMedia('(max-width: 768px)') :
      null;
  private readonly colorSchemeMediaQuery = typeof window !== 'undefined' ?
      window.matchMedia('(prefers-color-scheme: light)') :
      null;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        width: 100%;
        height: 100%;
        overflow: hidden;
        color: var(--text, #f1f5f9);
        background-color: var(--bg-dark, #0d1017);
      }

      .skip-to-composer {
        position: fixed;
        top: 11px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10060;
        padding: 7px 14px;
        border-radius: 999px;
        border: 1px solid var(--teal, #14d4aa);
        background-color: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
        font-family: inherit;
        font-size: 0.78rem;
        font-weight: 600;
        box-shadow: var(--shadow-md, 0 8px 24px rgba(0, 0, 0, 0.45));
        cursor: pointer;
      }

      .skip-to-composer:not(:focus):not(:focus-visible) {
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        clip-path: inset(50%);
        white-space: nowrap;
        border: 0;
        box-shadow: none;
      }

      header {
        position: relative;
        height: 56px;
        background-color: var(--bg-sidebar, #131720);
        border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding: 0 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        flex-shrink: 0;
        z-index: 40;
      }

      .header-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        min-width: 0;
      }

      .brand-logo {
        width: 26px;
        height: 26px;
        object-fit: contain;
        display: block;
        pointer-events: none;
        flex-shrink: 0;
      }

      .brand-titles {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }

      h1 {
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
        color: var(--text, #f1f5f9);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
      }

      .btn-icon-header {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-muted, #94a3b8);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        flex-shrink: 0;
        padding: 0;
      }

      .btn-icon-header:hover {
        background-color: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
      }

      .btn-toggle-sidebar {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-muted, #94a3b8);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.15s ease, color 0.15s ease;
        flex-shrink: 0;
        padding: 0;
      }

      .btn-toggle-sidebar:hover {
        background-color: var(--bg-elevated, #1e2432);
        color: var(--text, #f1f5f9);
      }

      .btn-toggle-learn-more {
        height: 34px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        color: var(--text, #f1f5f9);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0 12px;
        border-radius: 8px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: border-color 0.15s ease, background-color 0.15s ease;
        font-family: inherit;
        white-space: nowrap;
      }

      .btn-toggle-learn-more:hover {
        border-color: var(--border-strong, #364156);
        background-color: var(--bg-elevated, #1e2432);
      }

      /* Layout wrapper */
      .app-container {
        display: flex;
        flex: 1;
        min-height: 0;
        overflow: hidden;
        position: relative;
      }

      litert-sidebar {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        gap: 0;
      }

      litert-chat-window {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        overflow: hidden;
      }

      litert-learn-more {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        overflow: hidden;
      }

      /* Desktop Sidebar: expanded when .open, collapsed when :not(.open) */
      .sidebar {
        width: 288px;
        background-color: var(--bg-sidebar, #131720);
        border-right: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 12px;
        flex-shrink: 0;
        overflow: hidden;
        box-sizing: border-box;
        visibility: visible;
        transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.18s ease, visibility 0s linear 0s;
      }

      @media (min-width: 769px) {
        .sidebar:not(.open) {
          width: 0;
          padding-left: 0;
          padding-right: 0;
          border-right-width: 0;
          opacity: 0;
          visibility: hidden;
          pointer-events: none;
          transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.18s ease, visibility 0s linear 0.22s;
        }

        .sidebar-overlay {
          display: none;
        }
      }

      /* Chat Panel */
      .chat-panel {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        background-color: var(--bg-chat, #0d1017);
        overflow: hidden;
        position: relative;
      }

      /* HTML Preview Overlay */
      .preview-overlay {
        position: fixed;
        inset: 0;
        background-color: var(--bg-dark, #0d1017);
        z-index: 10050;
        display: flex;
        flex-direction: column;
      }

      .preview-toolbar {
        height: 56px;
        padding: 0 16px;
        background-color: var(--bg-sidebar, #131720);
        border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-shrink: 0;
      }

      .preview-toolbar-title {
        display: flex;
        align-items: center;
        gap: 8px;
        flex: 1;
        min-width: 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text, #f1f5f9);
      }

      .preview-toolbar-title > svg {
        flex-shrink: 0;
      }

      .preview-toolbar-label {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .preview-sandbox-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        color: var(--text-muted, #94a3b8);
        opacity: 0.78;
        cursor: help;
        transition: opacity 0.15s ease, color 0.15s ease;
      }

      .preview-sandbox-icon:hover {
        opacity: 1;
        color: var(--teal, #14d4aa);
      }

      .preview-overlay iframe {
        width: 100%;
        flex: 1;
        border: none;
        background-color: #ffffff;
      }

      .btn-close-preview {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        white-space: nowrap;
        flex-shrink: 0;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 8px;
        padding: 7px 14px;
        color: var(--text, #f1f5f9);
        font-weight: 600;
        font-size: 0.8rem;
        line-height: 1.1;
        font-family: inherit;
        cursor: pointer;
        transition: background-color 0.15s ease, border-color 0.15s ease;
      }

      .btn-close-preview:hover {
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border-strong, #364156);
      }

      /* Collapsible Learn More Drawer */
      .sidebar-right {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        max-width: 440px;
        height: 100vh;
        height: 100dvh;
        z-index: 10000;
        background-color: var(--bg-sidebar, #131720);
        border-left: 1px solid var(--border, #262f40);
        padding: 20px;
        box-sizing: border-box;
        transform: translateX(100%);
        visibility: hidden;
        pointer-events: none;
        transition: transform 0.26s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.26s ease, visibility 0s linear 0.26s;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: none;
      }

      .sidebar-right.open {
        transform: translateX(0);
        visibility: visible;
        pointer-events: auto;
        transition: transform 0.26s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.26s ease, visibility 0s linear 0s;
        box-shadow: var(--shadow-md, -8px 0 28px rgba(0, 0, 0, 0.5));
      }

      @keyframes undoToastSlideDown {
        from {
          opacity: 0;
          transform: translate(-50%, -8px);
        }
        to {
          opacity: 1;
          transform: translate(-50%, 0);
        }
      }

      /* Floating Undo Toast (top-anchored below header, zero layout shift) */
      .undo-toast {
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translate(-50%, 0);
        z-index: 10040;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 9px 12px 9px 16px;
        border-radius: 999px;
        background-color: var(--bg-elevated, #1e2432);
        border: 1px solid var(--border-strong, #364156);
        box-shadow: var(--shadow-md, 0 10px 28px rgba(0, 0, 0, 0.5));
        color: var(--text, #f1f5f9);
        font-size: 0.8125rem;
        font-weight: 500;
        max-width: calc(100vw - 24px);
        white-space: nowrap;
        animation: undoToastSlideDown 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      }

      @media (prefers-reduced-motion: reduce) {
        .undo-toast {
          animation: none;
        }
      }

      @media (min-width: 769px) {
        .undo-toast.sidebar-open {
          left: calc(50% + 144px);
        }
      }

      @media (max-width: 768px) {
        .undo-toast {
          top: 10px;
        }

        .undo-toast.sidebar-open {
          position: fixed;
          top: 102px;
          left: calc(min(86vw, 310px) / 2);
          width: calc(min(86vw, 310px) - 24px);
          max-width: calc(min(86vw, 310px) - 24px);
          box-sizing: border-box;
          padding: 8px 10px 8px 14px;
          gap: 8px;
          font-size: 0.78rem;
        }

        .undo-toast.sidebar-open > span {
          flex: 1;
        }
      }

      .btn-undo-action {
        height: 28px;
        padding: 0 12px;
        border-radius: 999px;
        border: 1px solid var(--teal, #14d4aa);
        background-color: var(--bg-input, #10141c);
        color: var(--teal, #14d4aa);
        font-family: inherit;
        font-size: 0.76rem;
        font-weight: 700;
        cursor: pointer;
        transition: background-color 0.14s ease, color 0.14s ease;
      }

      .btn-undo-action:hover {
        background-color: var(--teal-btn-bg, #14d4aa);
        color: var(--teal-btn-text, #041f18);
      }

      .btn-undo-dismiss {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        border: none;
        background: transparent;
        color: var(--text-muted, #94a3b8);
        font-size: 0.78rem;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0;
      }

      .btn-undo-dismiss:hover {
        background-color: var(--bg-card, #171c26);
        color: var(--text, #f1f5f9);
      }

      /* Non-blocking Themed Confirmation Modal */
      .confirm-backdrop {
        position: fixed;
        inset: 0;
        z-index: 10100;
        background-color: var(--overlay-backdrop, rgba(4, 7, 13, 0.68));
        backdrop-filter: blur(4px);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
      }

      .confirm-dialog {
        width: 100%;
        max-width: 400px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        border-radius: 16px;
        padding: 20px;
        box-shadow: var(--shadow-md, 0 12px 32px rgba(0, 0, 0, 0.5));
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .confirm-dialog-title {
        margin: 0;
        font-size: 0.96rem;
        font-weight: 700;
        color: var(--text, #f1f5f9);
      }

      .confirm-dialog-message {
        margin: 0;
        font-size: 0.825rem;
        line-height: 1.55;
        color: var(--text-secondary, #cbd5e1);
      }

      .confirm-dialog-actions {
        display: flex;
        justify-content: flex-end;
        gap: 8px;
        margin-top: 4px;
      }

      .sidebar-right-overlay {
        position: fixed;
        inset: 0;
        background-color: var(--overlay-backdrop, rgba(4, 7, 13, 0.68));
        z-index: 9999;
        backdrop-filter: blur(3px);
        overscroll-behavior: none;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.24s ease;
      }

      .sidebar-right-overlay.open {
        opacity: 1;
        pointer-events: auto;
      }

      /* Mobile Screen Configurations */
      @media (max-width: 768px) {
        header {
          padding: 0 10px;
          gap: 6px;
        }

        .header-brand {
          gap: 8px;
        }

        .brand-logo {
          width: 24px;
          height: 24px;
        }

        h1 {
          font-size: 0.9rem;
        }

        .header-actions {
          gap: 4px;
        }

        .btn-icon-header,
        .btn-toggle-sidebar {
          width: 32px;
          height: 32px;
        }

        .learn-more-label {
          display: none;
        }

        .btn-toggle-learn-more {
          height: 32px;
          padding: 0 8px;
        }

        .preview-toolbar {
          padding: 0 12px;
          gap: 8px;
        }

        .preview-toolbar-title {
          gap: 6px;
          font-size: 0.8rem;
        }

        .btn-close-preview {
          padding: 6px 10px;
          font-size: 0.76rem;
        }

        .sidebar {
          position: fixed;
          top: 0;
          left: 0;
          bottom: 0;
          width: 86vw;
          max-width: 310px;
          height: 100vh;
          height: 100dvh;
          z-index: 10000;
          background-color: var(--bg-sidebar, #131720);
          box-shadow: none;
          transform: translateX(-100%);
          visibility: hidden;
          pointer-events: none;
          transition: transform 0.26s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.26s ease, visibility 0s linear 0.26s;
          border-right: 1px solid var(--border, #262f40);
          box-sizing: border-box;
          padding: 0 !important;
          overflow: hidden;
        }

        .sidebar.open {
          transform: translateX(0);
          visibility: visible;
          pointer-events: auto;
          transition: transform 0.26s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.26s ease, visibility 0s linear 0s;
          box-shadow: var(--shadow-md, 8px 0 28px rgba(0, 0, 0, 0.55));
        }

        .sidebar-overlay {
          position: fixed;
          inset: 0;
          background-color: var(--overlay-backdrop, rgba(4, 7, 13, 0.68));
          z-index: 9999;
          backdrop-filter: blur(3px);
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.24s ease;
        }

        .sidebar-overlay.open {
          opacity: 1;
          pointer-events: auto;
        }
      }

      @media (max-width: 480px) {
        .brand-suffix {
          display: none;
        }
      }

      @media (max-width: 420px) {
        .sidebar-right {
          max-width: 100% !important;
          border-left: none !important;
        }
      }
    `
  ];

  private readonly handleMediaChange = (e: MediaQueryListEvent) => {
    this.isMobileViewport = e.matches;
    this.isSidebarOpen = !e.matches;
    if (!this.isSidebarOpen) {
      this.mobileSidebarReturnFocusEl = null;
    }
  };

  private readonly captureTriggerClick = (e: Event) => {
    for (const node of e.composedPath()) {
      if (node instanceof HTMLElement && node.matches(FOCUSABLE_SELECTOR)) {
        this.lastClickedFocusable = node;
        this.lastClickTime = Date.now();
        break;
      }
    }
  };

  private readonly handleWindowKeyDown = (e: KeyboardEvent) => {
    if (e.defaultPrevented) return;
    if (e.key === 'Tab') {
      if (this.activeConfirm) {
        const dialog = this.renderRoot.querySelector('.confirm-backdrop') as
            HTMLElement |
            null;
        if (dialog) trapTabKey(e, dialog);
      } else if (this.isPreviewOpen) {
        const preview = this.renderRoot.querySelector('#preview-overlay') as
            HTMLElement |
            null;
        if (preview) trapTabKey(e, preview);
      } else if (this.isLearnMoreOpen) {
        const learnMore = this.renderRoot.querySelector('.sidebar-right') as
            HTMLElement |
            null;
        if (learnMore) trapTabKey(e, learnMore);
      } else if (this.isSidebarOpen && this.isMobileViewport) {
        const sidebarEl = this.renderRoot.querySelector('.sidebar') as
            HTMLElement |
            null;
        const toastEl = this.renderRoot.querySelector('.undo-toast') as
            HTMLElement |
            null;
        if (sidebarEl) trapTabKey(e, sidebarEl, toastEl);
      } else if (e.shiftKey && !e.defaultPrevented && this.isSidebarOpen) {
        const mainEl =
            this.renderRoot.querySelector('main.chat-panel') as HTMLElement |
            null;
        const active = getDeepActiveElement();
        if (mainEl && active &&
            getFocusableElementsDeep(mainEl)[0] === active &&
            this.getSidebar()?.focusLastBottomElement()) {
          e.preventDefault();
        }
      }
      return;
    }
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const isModalOrDrawerTrapped = Boolean(
          this.activeConfirm || this.isPreviewOpen || this.isCameraOpen ||
          this.isLearnMoreOpen ||
          (this.isSidebarOpen && this.isMobileViewport));
      const isEditable = (node: unknown): boolean => {
        if (!(node instanceof HTMLElement)) return false;
        const tag = node.tagName;
        return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
            node.isContentEditable ||
            node.getAttribute('contenteditable') === 'true';
      };
      const path = typeof e.composedPath === 'function' ? e.composedPath() : [];
      if (!isModalOrDrawerTrapped && !path.some(isEditable) &&
          !isEditable(getDeepActiveElement())) {
        e.preventDefault();
        this.focusComposer();
        return;
      }
    }
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    if (this.activeConfirm) {
      e.preventDefault();
      this.resolveConfirmDialog(false);
    } else if (this.isPreviewOpen) {
      e.preventDefault();
      this.closePreview();
    } else if (this.getChatWindow()?.dismissCameraModal()) {
      e.preventDefault();
    } else if (this.isLearnMoreOpen) {
      e.preventDefault();
      this.closeLearnMore();
    } else if (this.getChatWindow()?.dismissTransientMenus()) {
      e.preventDefault();
    } else if (
        this.isSidebarOpen && this.getSidebar()?.dismissTopTransientMenu()) {
      e.preventDefault();
    } else if (this.isSidebarOpen && this.isMobileViewport) {
      e.preventDefault();
      this.closeSidebar();
    }
  };

  private readonly handleColorSchemeChange = () => {
    if (this.themeMode === 'system') {
      this.syncResolvedTheme();
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    this.state.addHost?.(this);
    this.initTheme();
    this.isMobileViewport =
        typeof window !== 'undefined' ? window.innerWidth <= 768 : false;
    this.state.confirmHandler = (options: ConfirmDialogOptions) =>
        this.openConfirmDialog(options);
    this.mobileMediaQuery?.addEventListener('change', this.handleMediaChange);
    this.colorSchemeMediaQuery?.addEventListener(
        'change', this.handleColorSchemeChange);
    this.addEventListener('click', this.captureTriggerClick, true);
    window.addEventListener('keydown', this.handleWindowKeyDown);
  }

  override disconnectedCallback() {
    if (this.state.confirmHandler) {
      this.state.confirmHandler = null;
    }
    // Don't leave a caller waiting on a dialog that can no longer be answered.
    this.activeConfirm?.resolve(false);
    this.activeConfirm = null;
    this.state.removeHost?.(this);
    this.mobileMediaQuery?.removeEventListener('change', this.handleMediaChange);
    this.colorSchemeMediaQuery?.removeEventListener(
        'change', this.handleColorSchemeChange);
    this.removeEventListener('click', this.captureTriggerClick, true);
    window.removeEventListener('keydown', this.handleWindowKeyDown);
    super.disconnectedCallback();
  }

  private getSidebar(): LitertSidebar|null {
    return this.renderRoot.querySelector('litert-sidebar') as LitertSidebar |
        null;
  }

  private getChatWindow(): LitertChatWindow|null {
    return this.renderRoot.querySelector('litert-chat-window') as
        LitertChatWindow |
        null;
  }

  private getModalReturnFocusElement(): HTMLElement|null {
    if (this.lastClickedFocusable?.isConnected &&
        Date.now() - this.lastClickTime < 400) {
      return this.lastClickedFocusable;
    }
    return getDeepActiveElement();
  }

  private restoreFocusElement(returnEl: HTMLElement|null) {
    if (!returnEl) return;
    void this.updateComplete.then(() => {
      const focusTarget = (): HTMLElement | null => {
        if (returnEl.isConnected &&
            (returnEl.offsetParent !== null ||
             returnEl.getClientRects().length > 0)) {
          returnEl.focus?.();
          return returnEl;
        }
        const parentDropdown =
            returnEl.closest('custom-dropdown') as CustomDropdown | null;
        if (parentDropdown?.isConnected) {
          parentDropdown.focusButton();
          return null;
        }
        returnEl.focus?.();
        return returnEl;
      };
      const target = focusTarget();
      requestAnimationFrame(() => {
        if (target && getDeepActiveElement() !== target &&
            target.isConnected) {
          focusTarget();
        }
      });
    });
  }

  private openConfirmDialog(options: ConfirmDialogOptions): Promise<boolean> {
    if (this.activeConfirm) {
      this.activeConfirm.resolve(false);
    }
    const activeEl = this.getModalReturnFocusElement();
    return new Promise<boolean>((resolve) => {
      this.activeConfirm = {
        ...options,
        resolve,
        returnFocusEl: activeEl,
      };
      void this.updateComplete.then(() => {
        const cancelBtn = this.renderRoot.querySelector<HTMLButtonElement>(
            '#btn-confirm-cancel');
        focusNowAndNextFrame(() => {
          if (this.activeConfirm) cancelBtn?.focus();
        });
      });
    });
  }

  private resolveConfirmDialog(confirmed: boolean) {
    const pending = this.activeConfirm;
    if (!pending) return;
    const parentDropdown =
        pending.returnFocusEl?.closest('custom-dropdown') as CustomDropdown |
        null;
    this.activeConfirm = null;
    void this.updateComplete.then(() => {
      if (pending.returnFocusEl?.isConnected &&
          (pending.returnFocusEl.offsetParent !== null ||
           pending.returnFocusEl.getClientRects().length > 0)) {
        pending.returnFocusEl.focus?.();
      }
      pending.resolve(confirmed);
      if (!confirmed) {
        this.restoreFocusElement(pending.returnFocusEl);
      } else {
        requestAnimationFrame(() => {
          const deep = getDeepActiveElement();
          if (!deep || deep === document.body) {
            if (parentDropdown?.isConnected) {
              parentDropdown.focusButton();
              return;
            }
            this.getChatWindow()?.focusInput();
          }
        });
      }
    });
  }

  private getSystemTheme(): 'dark'|'light' {
    return this.colorSchemeMediaQuery?.matches ? 'light' : 'dark';
  }

  private initTheme() {
    const stored = this.state.storage.getItem(THEME_STORAGE_KEY);
    const mode: ThemeMode =
        stored === 'light' || stored === 'dark' || stored === 'system' ?
        stored :
        'system';
    this.setThemeMode(mode, false);
  }

  private syncResolvedTheme() {
    const resolved: 'dark'|'light' =
        this.themeMode === 'system' ? this.getSystemTheme() : this.themeMode;
    this.theme = resolved;
    document.documentElement.setAttribute('data-theme', resolved);
    this.setAttribute('data-theme', resolved);
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute(
          'content', resolved === 'light' ? '#f8fafc' : '#0d1017');
    }
  }

  private setThemeMode(mode: ThemeMode, persist = true) {
    this.themeMode = mode;
    this.syncResolvedTheme();
    if (persist) {
      if (mode === 'system') {
        this.state.storage.removeItem(THEME_STORAGE_KEY);
      } else {
        this.state.storage.setItem(THEME_STORAGE_KEY, mode);
      }
    }
  }

  private getNextThemeMode(): ThemeMode {
    const sys = this.getSystemTheme();
    const opposite: 'dark'|'light' = sys === 'light' ? 'dark' : 'light';
    if (this.themeMode === 'system') {
      return opposite;
    }
    if (this.themeMode === opposite) {
      return sys;
    }
    return 'system';
  }

  private toggleTheme() {
    this.setThemeMode(this.getNextThemeMode());
  }

  private handlePreviewHtml(e: CustomEvent<{base64Code: string}>) {
    const base64 = e.detail.base64Code;
    if (!base64) return;
    try {
      const code = decodeURIComponent(escape(atob(base64)));
      const iframe =
          this.renderRoot.querySelector('#preview-iframe') as HTMLIFrameElement;

      if (iframe) {
        this.previewReturnFocusEl = this.getModalReturnFocusElement();
        setSandboxIframeHtml(iframe, code);
        this.isPreviewOpen = true;
        void this.updateComplete.then(() => {
          const closeBtn = this.renderRoot.querySelector(
                               '#btn-close-preview') as HTMLButtonElement |
              null;
          closeBtn?.focus();
        });
      }
    } catch (err) {
      console.error('[LiteRT-LM] Failed to decode HTML code content:', err);
    }
  }

  override firstUpdated() {
    // Register PWA Service Worker only on secure origins (HTTPS / localhost).
    if (window.isSecureContext && 'serviceWorker' in navigator &&
        navigator.serviceWorker) {
      registerAppServiceWorker(navigator.serviceWorker)
          .then(
              reg => console.log(
                  '[PWA] Service Worker registered successfully on scope:',
                  reg.scope))
          .catch(
              err => console.warn(
                  '[PWA] Service Worker registration skipped:', err));
    }
  }

  private toggleSidebar() {
    this.getChatWindow()?.closeTransientMenus();
    if (this.isSidebarOpen) {
      this.closeSidebar();
      return;
    }
    const onMobile = this.isMobileViewport;
    if (onMobile) {
      this.mobileSidebarReturnFocusEl = this.getModalReturnFocusElement() ??
          (this.renderRoot.querySelector('#btn-toggle-sidebar') as
               HTMLElement |
           null);
    }
    this.isSidebarOpen = true;
    if (onMobile) {
      void this.updateComplete.then(async () => {
        const sidebar = this.getSidebar();
        await sidebar?.updateComplete;
        const focusFirstInDrawer = () => {
          if (!this.isSidebarOpen) return;
          const dismissBtn = sidebar?.shadowRoot?.querySelector(
                                 '#btn-dismiss-sidebar') as HTMLElement |
              null;
          if (dismissBtn &&
              (dismissBtn.offsetParent !== null ||
               dismissBtn.getClientRects().length > 0)) {
            dismissBtn.focus();
            return;
          }
          const sidebarRoot = this.renderRoot.querySelector('.sidebar');
          if (sidebarRoot) {
            getFocusableElementsDeep(sidebarRoot)[0]?.focus();
          }
        };
        focusNowAndNextFrame(focusFirstInDrawer);
      });
    }
  }

  private closeSidebar() {
    const onMobile = this.isMobileViewport;
    this.isSidebarOpen = false;
    if (onMobile) {
      const returnEl = this.mobileSidebarReturnFocusEl ??
          (this.renderRoot.querySelector('#btn-toggle-sidebar') as
               HTMLElement |
           null);
      this.mobileSidebarReturnFocusEl = null;
      this.restoreFocusElement(returnEl);
    }
  }

  private focusComposer() {
    this.getChatWindow()?.closeTransientMenus();
    this.getChatWindow()?.focusInput();
  }

  private openComposerModelSelector(e?: Event) {
    e?.preventDefault();
    if (this.isSidebarOpen && this.isMobileViewport) {
      this.closeSidebar();
    }
    this.getChatWindow()?.openModelPicker();
  }

  private toggleLearnMore() {
    this.getChatWindow()?.closeTransientMenus();
    if (this.isLearnMoreOpen) {
      this.closeLearnMore();
      return;
    }
    this.learnMoreReturnFocusEl = this.getModalReturnFocusElement();
    this.isLearnMoreOpen = true;
    void this.updateComplete.then(async () => {
      const drawer =
          this.renderRoot.querySelector('litert-learn-more') as LitElement |
          null;
      await drawer?.updateComplete;
      const focusCloseBtn = () => {
        if (!this.isLearnMoreOpen) return;
        const closeBtn = drawer?.shadowRoot?.querySelector(
                             '.btn-dismiss-right-drawer') as HTMLButtonElement |
            null;
        closeBtn?.focus();
      };
      focusNowAndNextFrame(focusCloseBtn);
    });
  }

  private focusSidebarFallback(preferActiveConv = false) {
    const sidebar = this.getSidebar();
    if (this.isSidebarOpen && sidebar?.shadowRoot) {
      const target =
          (preferActiveConv ? sidebar.shadowRoot.querySelector<HTMLElement>(
                                  '.conv-item.active .conv-select-btn') :
                              null) ??
          sidebar.shadowRoot.querySelector<HTMLElement>(
              '.new-chat-item .conv-select-btn');
      if (target) {
        target.focus();
        return;
      }
    }
    this.getChatWindow()?.focusInput();
  }

  override willUpdate() {
    const toastEl = this.renderRoot.querySelector('.undo-toast');
    const active = getDeepActiveElement();
    this.hadFocusInUndoToast =
        Boolean(toastEl && active && toastEl.contains(active));
  }

  override updated() {
    const hasToast = Boolean(this.state.chatSession.deletedConversation);
    if (this.wasUndoToastRendered && !hasToast &&
        this.hadFocusInUndoToast && !this.handlingUndoToastClick) {
      this.hadFocusInUndoToast = false;
      this.focusSidebarFallback(false);
    }
    this.wasUndoToastRendered = hasToast;
  }

  private handleConversationDeleted() {
    void this.updateComplete.then(() => {
      const focusUndo = () => {
        const undoBtn = this.renderRoot.querySelector(
                            '#btn-undo-delete') as HTMLButtonElement |
            null;
        if (undoBtn) {
          undoBtn.focus();
        } else {
          this.focusSidebarFallback(false);
        }
      };
      focusNowAndNextFrame(focusUndo);
    });
  }

  private handleUndoDeleteConversation() {
    this.handlingUndoToastClick = true;
    this.state.chatSession.undoDeleteConversation?.();
    void this.updateComplete.then(async () => {
      await this.getSidebar()?.updateComplete;
      this.focusSidebarFallback(true);
      this.handlingUndoToastClick = false;
    });
  }

  private handleDismissDeletedConversation() {
    this.handlingUndoToastClick = true;
    this.state.chatSession.dismissDeletedConversation?.();
    void this.updateComplete.then(() => {
      this.focusSidebarFallback(false);
      this.handlingUndoToastClick = false;
    });
  }

  private closeLearnMore() {
    const returnEl = this.learnMoreReturnFocusEl;
    this.learnMoreReturnFocusEl = null;
    this.isLearnMoreOpen = false;
    this.restoreFocusElement(returnEl);
  }

  private closePreview() {
    const returnEl = this.previewReturnFocusEl;
    this.previewReturnFocusEl = null;
    this.isPreviewOpen = false;
    const iframe =
        this.renderRoot.querySelector('#preview-iframe') as HTMLIFrameElement;
    if (iframe) {
      setSandboxIframeHtml(iframe, '');
    }
    this.restoreFocusElement(returnEl);
  }

  override render() {
    const isAppModalOpen = Boolean(this.activeConfirm || this.isPreviewOpen);
    const isMobileSidebarModal = this.isMobileViewport && this.isSidebarOpen;
    const deletedConv = this.state.chatSession.deletedConversation;
    const nextThemeMode = this.getNextThemeMode();
    const nextThemeLabel = nextThemeMode === 'system' ?
        'System' :
        (nextThemeMode === 'light' ? 'Light' : 'Dark');
    const currentThemeLabel = this.themeMode === 'system' ?
        `System (${this.theme === 'light' ? 'Light' : 'Dark'})` :
        (this.themeMode === 'light' ? 'Light' : 'Dark');
    const themeTooltip =
        `Theme: ${currentThemeLabel} — click for ${nextThemeLabel}`;

    return html`
      <!-- Mobile sidebar backdrop overlay -->
      <div
        class="sidebar-overlay ${this.isSidebarOpen ? 'open' : ''}"
        ?inert=${isAppModalOpen || this.isCameraOpen || this.isLearnMoreOpen}
        @click=${this.closeSidebar}
      ></div>

      <!-- Keyboard Skip Link to Composer -->
      <button
        id="skip-to-composer"
        type="button"
        class="skip-to-composer"
        ?inert=${isAppModalOpen || this.isCameraOpen || this.isLearnMoreOpen || isMobileSidebarModal}
        @click=${this.focusComposer}
      >Skip to composer</button>

      <!-- Top Header Bar -->
      <header ?inert=${isAppModalOpen || this.isCameraOpen || this.isLearnMoreOpen || isMobileSidebarModal}>
        <div class="header-brand">
          <button
            id="btn-toggle-sidebar"
            class="btn-toggle-sidebar"
            title="Toggle sidebar"
            aria-label="Toggle sidebar"
            aria-expanded=${this.isSidebarOpen ? 'true' : 'false'}
            @click=${this.toggleSidebar}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2"></rect>
              <line x1="9" y1="3" x2="9" y2="21"></line>
            </svg>
          </button>
          <img
            src=${this.theme === 'light' ? './assets/LiteRT_Logo_Symbol-only_RGB_Color_Blue.svg' : './assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg'}
            alt=""
            aria-hidden="true"
            class="brand-logo"
          />
          <div class="brand-titles">
            <h1>LiteRT-LM<span class="brand-suffix">.js Chat</span></h1>
          </div>
        </div>

        <div class="header-actions">
          <button
            id="btn-toggle-theme"
            class="btn-icon-header"
            title=${themeTooltip}
            aria-label=${themeTooltip}
            @click=${this.toggleTheme}
          >
            ${this.themeMode === 'system' ? html`
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                <line x1="8" y1="21" x2="16" y2="21"></line>
                <line x1="12" y1="17" x2="12" y2="21"></line>
              </svg>
            ` : this.themeMode === 'light' ? html`
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="4"></circle>
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path>
              </svg>
            ` : html`
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            `}
          </button>

          <button
            id="btn-toggle-learn-more"
            class="btn-toggle-learn-more"
            aria-label="Toggle Learn More"
            aria-expanded=${this.isLearnMoreOpen ? 'true' : 'false'}
            @click=${this.toggleLearnMore}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span class="learn-more-label">Learn More</span>
          </button>
        </div>
      </header>

      <div class="app-container" ?inert=${isAppModalOpen || this.isLearnMoreOpen}>
        <!-- Collapsible left navigation & configuration sidebar -->
        <div
          class="sidebar ${this.isSidebarOpen ? 'open' : ''}"
          role=${isMobileSidebarModal ? 'dialog' : 'complementary'}
          aria-modal=${isMobileSidebarModal ? 'true' : nothing}
          ?inert=${!this.isSidebarOpen || isAppModalOpen || this.isCameraOpen}
          aria-hidden=${!this.isSidebarOpen ? 'true' : 'false'}
          aria-label="Conversations and model settings"
        >
          <litert-sidebar
            .state=${this.state}
            @close=${this.closeSidebar}
            @conversation-deleted=${this.handleConversationDeleted}
          ></litert-sidebar>
        </div>

        <!-- Chat window messages & inputs area -->
        <main class="chat-panel" ?inert=${isMobileSidebarModal}>
          <litert-chat-window
            .state=${this.state}
            .theme=${this.theme}
            @preview-html=${this.handlePreviewHtml}
            @choose-model=${this.openComposerModelSelector}
            @camera-open-change=${(e: CustomEvent<{open: boolean}>) => {
              this.isCameraOpen = e.detail.open;
            }}
          ></litert-chat-window>
        </main>

        <!-- Floating Undo Toast on Conversation Delete -->
        ${deletedConv ? html`
          <div class="undo-toast ${this.isSidebarOpen ? 'sidebar-open' : ''}" role="status" aria-live="polite">
            <span>Conversation deleted</span>
            <button
              id="btn-undo-delete"
              type="button"
              class="btn-undo-action"
              @click=${this.handleUndoDeleteConversation}
            >Undo</button>
            <button
              type="button"
              class="btn-undo-dismiss"
              aria-label="Dismiss notification"
              title="Dismiss"
              @click=${this.handleDismissDeletedConversation}
            >✕</button>
          </div>
        ` : nothing}
      </div>

      <!-- Right-side Learn More Sidebar drawer backdrop overlay -->
      <div class="sidebar-right-overlay ${
        this.isLearnMoreOpen ? 'open' :
                               ''}" @click=${this.closeLearnMore}></div>
      <div
        class="sidebar-right ${this.isLearnMoreOpen ? 'open' : ''}"
        role="dialog"
        aria-modal="true"
        ?inert=${!this.isLearnMoreOpen || isAppModalOpen || this.isCameraOpen}
        aria-hidden=${!this.isLearnMoreOpen ? 'true' : 'false'}
        aria-label="About LiteRT-LM"
      >
        <litert-learn-more @close=${this.closeLearnMore}></litert-learn-more>
      </div>

      <!-- Fullscreen HTML Preview Overlay -->
      <div
        id="preview-overlay"
        class="preview-overlay"
        role="dialog"
        aria-modal="true"
        aria-label="HTML Sandbox Preview"
        ?inert=${!this.isPreviewOpen || Boolean(this.activeConfirm)}
        style="display: ${this.isPreviewOpen ? 'flex' : 'none'};"
      >
        <div class="preview-toolbar">
          <div class="preview-toolbar-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="16 18 22 12 16 6"></polyline>
              <polyline points="8 6 2 12 8 18"></polyline>
            </svg>
            <span class="preview-toolbar-label">HTML Sandbox Preview</span>
            <span
              class="preview-sandbox-icon"
              role="img"
              aria-label="Sandboxed iframe (isolated origin)"
              title="Runs in a sandboxed iframe (isolated origin)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              </svg>
            </span>
          </div>
          <button id="btn-close-preview" class="btn-close-preview" @click=${
        this.closePreview}>Exit Preview ✕</button>
        </div>
        <iframe id="preview-iframe" title="HTML Sandbox Preview" sandbox="allow-scripts"></iframe>
      </div>

      <!-- Themed Non-Blocking Confirmation Modal -->
      ${this.activeConfirm ? html`
        <div
          class="confirm-backdrop"
          @click=${(e: Event) => {
            if (e.target === e.currentTarget) {
              this.resolveConfirmDialog(false);
            }
          }}
        >
          <div
            class="confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-dialog-title"
            aria-describedby="confirm-dialog-desc"
          >
            <h2 id="confirm-dialog-title" class="confirm-dialog-title">
              ${this.activeConfirm.title}
            </h2>
            <p id="confirm-dialog-desc" class="confirm-dialog-message">
              ${this.activeConfirm.message}
            </p>
            <div class="confirm-dialog-actions">
              <button
                id="btn-confirm-cancel"
                type="button"
                class="btn btn-secondary"
                @click=${() => this.resolveConfirmDialog(false)}
              >Cancel</button>
              <button
                id="btn-confirm-accept"
                type="button"
                class="btn ${this.activeConfirm.danger ? 'btn-stop' : 'btn-primary'}"
                @click=${() => this.resolveConfirmDialog(true)}
              >${this.activeConfirm.confirmLabel || 'Confirm'}</button>
            </div>
          </div>
        </div>
      ` : nothing}
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-lm-chat-app': LitertLmChatApp;
  }
}
