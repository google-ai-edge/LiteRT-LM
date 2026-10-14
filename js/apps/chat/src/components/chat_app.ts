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

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {registerAppServiceWorker, setSandboxIframeHtml} from './util.js';

import {LlmChatStateController} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

/** Main application container for the LiteRT-LM chat interface. */
@customElement('litert-lm-chat-app')
export class LitertLmChatApp extends LitElement {
  // Central source of truth state controller
  private state = new LlmChatStateController(this);

  // Native Component UI Modals State
  @property({type: Boolean}) isSidebarOpen = false;

  @property({type: Boolean}) isLearnMoreOpen = false;

  @property({type: Boolean}) isPreviewOpen = false;

  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        width: 100%;
        height: 100%;
        overflow: hidden;
      }

      header {
        background-color: var(--bg-card);
        border-bottom: 1px solid var(--border);
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
      }

      h1 {
        font-size: 1.25rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
      }

      .header-subtitle {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: ui-monospace, monospace;
      }

      /* Layout wrapper */
      .app-container {
        display: flex;
        flex: 1;
        overflow: hidden;
      }

      litert-sidebar {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0; 
        width: 100%;
        overflow: hidden;
        gap: 12px;
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

      /* Sidebar */
      .sidebar {
        width: 280px;
        background-color: var(--bg-card);
        border-right: 1px solid var(--border);
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        flex-shrink: 0;
        overflow-y: hidden;
        box-sizing: border-box;
      }

      /* Chat Panel */
      .chat-panel {
        flex: 1;
        display: flex;
        flex-direction: column;
        background-color: var(--bg-chat);
        overflow: hidden;
      }

      /* HTML Preview Overlay */
      .preview-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: var(--bg-dark);
        z-index: 9999;
        display: flex;
        flex-direction: column;
      }

      .preview-overlay iframe {
        width: 100%;
        height: 100%;
        border: none;
        background-color: #ffffff;
      }

      .btn-close-preview {
        position: absolute;
        top: 16px;
        right: 24px;
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 16px;
        color: var(--text);
        font-weight: 700;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.15s;
        backdrop-filter: blur(4px);
        z-index: 10000;
      }

      .btn-close-preview:hover {
        background-color: rgba(30, 41, 59, 0.9);
        border-color: var(--teal);
      }

      /* Collapsible Learn More Drawer */
      .sidebar-right {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        max-width: 480px;
        height: 100vh;
        z-index: 10000;
        background-color: var(--bg-card);
        border-left: 1px solid var(--border);
        padding: 24px 20px;
        box-sizing: border-box;
        transform: translateX(100%);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        display: flex;
        flex-direction: column;
        overflow-y: hidden;
        box-shadow: none;
      }

      .sidebar-right.open {
        transform: translateX(0);
        box-shadow: -8px 0 24px rgba(0, 0, 0, 0.65);
      }

      .sidebar-right-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.65);
        z-index: 9999;
        backdrop-filter: blur(4px);
        overscroll-behavior: none;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
      }
      
      .sidebar-right-overlay.open {
        opacity: 1;
        pointer-events: auto;
      }

      /* Header toggle buttons */
      .btn-toggle-learn-more {
        background: rgba(0, 201, 158, 0.04);
        border: 1px solid var(--teal);
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: border-color 0.15s, color 0.15s, background-color 0.15s, box-shadow 0.15s;
        outline: none;
        font-family: inherit;
      }

      .btn-toggle-learn-more:hover {
        border-color: var(--teal);
        color: #ffffff;
        background-color: rgba(0, 201, 158, 0.1);
        box-shadow: 0 0 8px rgba(0, 201, 158, 0.25);
      }

      .btn-toggle-sidebar {
        display: none;
        background: none;
        border: none;
        color: var(--text);
        font-size: 1.5rem;
        cursor: pointer;
        padding: 2px 8px;
        border-radius: 4px;
        transition: background-color 0.15s;
        line-height: 1;
      }

      .btn-toggle-sidebar:hover {
        background-color: var(--border);
      }

      /* Mobile Screen Configurations */
      @media (max-width: 768px) {
        .btn-toggle-sidebar {
          display: block; 
        }

        .sidebar {
          position: fixed;
          top: 0;
          left: 0;
          bottom: 0;
          width: 100%; 
          max-width: 280px; 
          height: 100vh;
          z-index: 10000;
          background-color: var(--bg-card);
          box-shadow: none; 
          transform: translateX(-100%); 
          transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
          border-right: 1px solid var(--border);
          box-sizing: border-box;
          padding: 16px !important; 
          overflow-y: auto; 
        }

        .sidebar.open {
          transform: translateX(0); 
          box-shadow: 8px 0 24px rgba(0, 0, 0, 0.65); 
        }

        .sidebar-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.65);
          z-index: 9999;
          backdrop-filter: blur(4px);
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.3s ease;
        }

        .sidebar-overlay.open {
          opacity: 1;
          pointer-events: auto;
        }
      }

      @media (max-width: 480px) {
        .sidebar, .sidebar-right {
          max-width: 100% !important;
          border: none !important;
          border-radius: 0 !important;
        }
      }
    `
  ];

  private handlePreviewHtml(e: CustomEvent<{base64Code: string}>) {
    const base64 = e.detail.base64Code;
    if (!base64) return;
    try {
      const code = decodeURIComponent(escape(atob(base64)));
      const iframe =
          this.renderRoot.querySelector('#preview-iframe') as HTMLIFrameElement;

      if (iframe) {
        setSandboxIframeHtml(iframe, code);
        this.isPreviewOpen = true;
      }
    } catch (err) {
      console.error('[LiteRT-LM] Failed to decode HTML code content:', err);
    }
  }

  override firstUpdated() {
    // Register PWA Service Worker for offline shell caching
    if ('serviceWorker' in navigator) {
      registerAppServiceWorker(navigator.serviceWorker)
          .then(
              reg => console.log(
                  '[PWA] Service Worker registered successfully on scope:',
                  reg.scope))
          .catch(
              err => console.error(
                  '[PWA] Service Worker registration failed:', err));
    }
  }

  private toggleSidebar() {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  private toggleLearnMore() {
    this.isLearnMoreOpen = !this.isLearnMoreOpen;
  }

  private closePreview() {
    this.isPreviewOpen = false;
    const iframe =
        this.renderRoot.querySelector('#preview-iframe') as HTMLIFrameElement;
    if (iframe) {
      setSandboxIframeHtml(iframe, '');
    }
  }

  override render() {
    return html`
      <!-- PWA configurations backdrop overlay -->
      <div class="sidebar-overlay ${this.isSidebarOpen ? 'open' : ''}" @click=${
        this.toggleSidebar}></div>

      <!-- Top Header Bar -->
      <header>
        <div style="display: flex; align-items: center; gap: 12px;">
          <button id="btn-toggle-sidebar" class="btn-toggle-sidebar" aria-label="Toggle Configurations" @click=${
        this.toggleSidebar}>☰</button>
          <img src="./assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg" alt="LiteRT Logo" style="height: 42px; width: auto; display: block; pointer-events: none;">
          <div style="display: flex; flex-direction: column;">
            <h1 style="color: #ffffff; margin: 0; font-size: 1.25rem; font-weight: 800; line-height: 1.2;">LiteRT-LM.js Chat</h1>
            <span class="header-subtitle">Experimental On-Device WebGPU LLM Runner</span>
          </div>
        </div>
        <div class="header-links" style="display: flex; gap: 16px; align-items: center;">
          <button id="btn-toggle-learn-more" class="btn-toggle-learn-more" aria-label="Toggle Learn More" @click=${
        this.toggleLearnMore}>Learn More</button>
        </div>
      </header>

      <div class="app-container">
        <!-- Collapsible drawer sidebar panel -->
        <aside class="sidebar ${this.isSidebarOpen ? 'open' : ''}">
          <litert-sidebar .state=${this.state} @close=${
        this.toggleSidebar}></litert-sidebar>
        </aside>

        <!-- Chat window messages & inputs area -->
        <main class="chat-panel" style="flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; background-color: var(--bg-chat);">
          <litert-chat-window
            .state=${this.state}
            @preview-html=${this.handlePreviewHtml}
          ></litert-chat-window>
        </main>
      </div>

      <!-- Right-side Learn More Sidebar drawer backdrop overlay -->
      <div class="sidebar-right-overlay ${
        this.isLearnMoreOpen ? 'open' :
                               ''}" @click=${this.toggleLearnMore}></div>
      <aside class="sidebar-right ${this.isLearnMoreOpen ? 'open' : ''}">
        <litert-learn-more @close=${this.toggleLearnMore}></litert-learn-more>
      </aside>

      <!-- Fullscreen HTML Preview Overlay -->
      <div id="preview-overlay" class="preview-overlay" style="display: ${
        this.isPreviewOpen ? 'flex' : 'none'};">
        <button id="btn-close-preview" class="btn-close-preview" @click=${
        this.closePreview}>Exit Preview ✕</button>
        <iframe id="preview-iframe" sandbox="allow-scripts"></iframe>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-lm-chat-app': LitertLmChatApp;
  }
}
