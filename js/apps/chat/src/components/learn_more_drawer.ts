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

import {css, html, LitElement} from 'lit';
import {customElement} from 'lit/decorators.js';

import {sharedStyles} from '../styles/shared_styles.js';
/* tslint:disable:no-new-decorators */

/** Component for the right sidebar drawer with "Learn More" information. */
@customElement('litert-learn-more')
export class LitertLearnMore extends LitElement {
  static override styles = [
    sharedStyles, css`
      :host {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        width: 100%;
        overflow: hidden;
        color: var(--text, #f1f5f9);
      }

      .drawer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding-bottom: 12px;
        flex-shrink: 0;
      }

      .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text, #f1f5f9);
        margin: 0;
      }

      .btn-dismiss-right-drawer {
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        color: var(--text, #f1f5f9);
        border-radius: 7px;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 5px 12px;
        cursor: pointer;
        transition: background-color 0.15s ease, border-color 0.15s ease;
        font-family: inherit;
      }

      .btn-dismiss-right-drawer:hover {
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border-strong, #364156);
      }

      .drawer-body {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 4px;
        padding-bottom: 16px;
        overscroll-behavior: contain;
        display: flex;
        flex-direction: column;
        gap: 20px;
      }

      .doc-section {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .doc-section + .doc-section {
        border-top: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding-top: 18px;
      }

      h3 {
        font-size: 0.85rem;
        color: var(--text, #f1f5f9);
        font-weight: 700;
        margin: 0;
      }

      p {
        font-size: 0.8125rem;
        line-height: 1.55;
        color: var(--text-secondary, #cbd5e1);
        margin: 0;
      }

      .feature-list {
        font-size: 0.8rem;
        line-height: 1.5;
        color: var(--text-secondary, #cbd5e1);
        padding-left: 18px;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 7px;
      }

      .feature-list b {
        color: var(--text, #f1f5f9);
      }

      .sdk-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 2px;
      }

      .sdk-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 999px;
        background-color: var(--bg-card, #171c26);
        border: 1px solid var(--border, #262f40);
        color: var(--teal, #14d4aa);
        font-size: 0.74rem;
        font-weight: 600;
        text-decoration: none;
        transition: background-color 0.14s ease, border-color 0.14s ease;
      }

      .sdk-chip:hover {
        background-color: var(--bg-elevated, #1e2432);
        border-color: var(--border-strong, #364156);
      }

      .resource-list {
        display: flex;
        flex-direction: column;
        margin: 0;
        padding: 0;
        list-style: none;
      }

      .resource-list li + li {
        border-top: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
      }

      .resource-link {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 8px;
        margin: 0;
        border-radius: 7px;
        color: var(--text, #f1f5f9);
        text-decoration: none;
        font-size: 0.8125rem;
        font-weight: 500;
        transition: background-color 0.14s ease, color 0.14s ease;
      }

      .resource-link:hover {
        background-color: var(--bg-card, #171c26);
        color: var(--teal, #14d4aa);
      }

      .resource-icon {
        width: 15px;
        height: 15px;
        color: var(--text-muted, #94a3b8);
        flex-shrink: 0;
        transition: color 0.14s ease;
      }

      .resource-link:hover .resource-icon {
        color: var(--teal, #14d4aa);
      }

      .resource-label {
        flex: 1;
        min-width: 0;
      }

      .resource-link .arrow {
        color: var(--text-muted, #94a3b8);
        font-size: 0.8rem;
        opacity: 0.7;
        flex-shrink: 0;
        transition: color 0.14s ease, opacity 0.14s ease, transform 0.14s ease;
      }

      .resource-link:hover .arrow {
        color: var(--teal, #14d4aa);
        opacity: 1;
        transform: translate(1px, -1px);
      }

      .footer-note {
        border-top: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.08));
        padding-top: 14px;
        margin-top: auto;
        font-size: 0.74rem;
        line-height: 1.5;
        color: var(--text-muted, #94a3b8);
      }

      .footer-note a {
        color: var(--teal, #14d4aa);
        font-weight: 600;
        text-decoration: underline;
        text-underline-offset: 2px;
      }
    `
  ];

  private dismissLearnMore() {
    this.dispatchEvent(new CustomEvent('close', {
      bubbles: true,
      composed: true,
    }));
  }

  override render() {
    return html`
      <div class="drawer-header">
        <h2 class="section-title">About LiteRT-LM</h2>
        <button type="button" class="btn-dismiss-right-drawer" @click=${this.dismissLearnMore}>Done</button>
      </div>

      <div class="drawer-body">
        <div class="doc-section">
          <h3>What is LiteRT-LM?</h3>
          <p>
            LiteRT-LM is a production-ready, open-source inference framework built for high-performance, cross-platform LLM deployments directly on edge devices and browsers.
          </p>
          <ul class="feature-list">
            <li><b>Cross-Platform:</b> Run on Android, iOS, Web, Desktop, and IoT devices.</li>
            <li><b>Hardware Acceleration:</b> Leverage WebGPU, GPU, and NPU accelerators for low-latency prefill and decode.</li>
            <li><b>Multimodal Ready:</b> Native vision and audio encoders alongside text generation.</li>
            <li><b>Agentic Tool Use:</b> Function calling and constrained decoding for reliable tool workflows.</li>
            <li><b>Broad Model Ecosystem:</b> Run Gemma, Llama, Phi-4, Qwen, and custom <code>.litertlm</code> weights.</li>
          </ul>

          <div>
            <p style="font-size: 0.74rem; color: var(--text-muted, #94a3b8); margin-bottom: 6px; font-weight: 600;">Platform SDKs</p>
            <div class="sdk-chips">
              <a class="sdk-chip" href="https://ai.google.dev/edge/litert-lm/android" target="_blank" rel="noopener">Kotlin</a>
              <a class="sdk-chip" href="https://ai.google.dev/edge/litert-lm/swift" target="_blank" rel="noopener">Swift</a>
              <a class="sdk-chip" href="https://ai.google.dev/edge/litert-lm/python" target="_blank" rel="noopener">Python</a>
              <a class="sdk-chip" href="https://ai.google.dev/edge/litert-lm/flutter" target="_blank" rel="noopener">Flutter</a>
              <a class="sdk-chip" href="https://ai.google.dev/edge/litert-lm/js" target="_blank" rel="noopener">JavaScript</a>
            </div>
          </div>
        </div>

        <div class="doc-section">
          <h3>Documentation &amp; Resources</h3>
          <ul class="resource-list">
            <li>
              <a class="resource-link" href="https://ai.google.dev/edge/litert-lm/overview" target="_blank" rel="noopener">
                <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                  <polyline points="2 17 12 22 22 17"></polyline>
                  <polyline points="2 12 12 17 22 12"></polyline>
                </svg>
                <span class="resource-label">LiteRT-LM Architecture &amp; Overview</span>
                <span class="arrow" aria-hidden="true">↗</span>
              </a>
            </li>
            <li>
              <a class="resource-link" href="https://www.npmjs.com/package/@litert-lm/core" target="_blank" rel="noopener">
                <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <span class="resource-label">@litert-lm/core on npm</span>
                <span class="arrow" aria-hidden="true">↗</span>
              </a>
            </li>
            <li>
              <a class="resource-link" href="https://ai.google.dev/edge/litert-lm/js" target="_blank" rel="noopener">
                <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                </svg>
                <span class="resource-label">JavaScript Developer Documentation</span>
                <span class="arrow" aria-hidden="true">↗</span>
              </a>
            </li>
            <li>
              <a class="resource-link" href="https://github.com/google-ai-edge/LiteRT-LM" target="_blank" rel="noopener">
                <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"></path>
                  <path d="M9 18c-4.51 2-5-2-7-2"></path>
                </svg>
                <span class="resource-label">google-ai-edge/LiteRT-LM on GitHub</span>
                <span class="arrow" aria-hidden="true">↗</span>
              </a>
            </li>
          </ul>
        </div>

        <div class="footer-note">
          Running standard TFLite / LiteRT models instead of LLMs? Check out
          <a href="https://www.npmjs.com/package/@litertjs/core?activeTab=readme" target="_blank" rel="noopener">@litertjs/core</a>.
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'litert-learn-more': LitertLearnMore;
  }
}
