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

import {css, html, svg} from 'lit';
import {customElement} from 'lit/decorators.js';

import {PROMPT_TEMPLATE_LABELS, PromptTemplateId} from '../app_config.js';
import {StatefulElement} from '../state_controller.js';
import {sharedStyles} from '../styles/shared_styles.js';

/* tslint:disable:no-new-decorators */

function liteRtSymbol(size = 34) {
  return svg`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="180 100 700 880"
         width=${size} height=${size} fill="var(--teal)" aria-hidden="true">
      <path d="M836.8,358.7c-1-1.1-20.2-20.2-21.2-21.2L197,962.3l580.3-328.2c105.3-59.6,143.1-187.9,59.5-275.3h0Z"/>
      <path d="M698.9,228.6c-.8-.8-15-15.1-15.8-15.8l-460.7,465.4,432.2-244.5c78.4-44.4,106.6-139.9,44.3-205.1h0Z"/>
      <path d="M598.8,128.5c-.5-.5-10-10-10.5-10.5l-306,309.1,287-162.4c52.1-29.5,70.8-92.9,29.5-136.2h0Z"/>
    </svg>
  `;
}

/** App header with branding, prompt template selector and model status. */
@customElement('embedding-search-header')
export class AppHeader extends StatefulElement {

  static override styles = [
    sharedStyles,
    css`
      :host {
        display: block;
        background-color: var(--bg-card);
        border-bottom: 1px solid var(--border);
        padding: 12px 24px;
        flex-shrink: 0;
      }

      .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1400px;
        margin: 0 auto;
        gap: 16px;
      }

      .logo-group {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .logo-icon {
        display: flex;
        filter: drop-shadow(0 0 10px rgba(0, 201, 158, 0.35));
      }

      .title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
      }

      .title .brand { color: var(--teal); }
      .subtitle { font-size: 0.75rem; color: var(--text-muted); }

      .actions {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .template-picker {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .select-wrap {
        position: relative;
        display: inline-flex;
        align-items: center;
      }

      .template-picker select {
        appearance: none;
        -webkit-appearance: none;
        padding: 6px 30px 6px 12px;
        font-size: 0.8rem;
        border-radius: 20px;
        background: var(--bg-dark);
        cursor: pointer;
      }

      .template-picker select:hover,
      .model-status:hover {
        border-color: var(--teal);
      }

      .select-caret {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
      }

      .model-status {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid var(--border);
        background: var(--bg-dark);
        color: var(--text);
        font: inherit;
        font-size: 0.8rem;
        transition: border-color 0.15s;
        max-width: 320px;
      }

      .model-status:focus-visible {
        outline: none;
        border-color: var(--teal);
        box-shadow: 0 0 0 2px rgba(0, 201, 158, 0.25);
      }

      .model-label {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--amber);
        flex-shrink: 0;
      }

      .status-dot.active {
        background-color: var(--teal);
        box-shadow: 0 0 8px rgba(0, 201, 158, 0.6);
      }

      .status-dot.loading {
        background-color: var(--cyan);
        animation: blink 1s ease-in-out infinite;
      }

      @keyframes blink { 50% { opacity: 0.3; } }

      .model-caret {
        font-size: 1.1rem;
        color: var(--text-muted);
      }
    `,
  ];

  private onTemplateChange(e: Event) {
    void this.state.setPromptTemplate(
        (e.target as HTMLSelectElement).value as PromptTemplateId);
  }

  override render() {
    const {models} = this.state;
    const isModelLoaded = !!models.engine;
    let modelLabel = models.loadedModelName || 'Load a model';
    if (models.isLoading) {
      modelLabel = models.loadingPercent !== null ?
          `Loading model… ${models.loadingPercent}%` :
          'Loading model…';
    }
    const dotClass =
        models.isLoading ? 'loading' : (isModelLoaded ? 'active' : '');

    return html`
      <div class="header-container">
        <div class="logo-group">
          <div class="logo-icon">${liteRtSymbol()}</div>
          <div>
            <div class="title"><span class="brand">LiteRT-LM</span> EmbeddingGemma Search</div>
            <div class="subtitle">On-device multimodal search · WebGPU</div>
          </div>
        </div>

        <div class="actions">
          <label
            class="template-picker"
            title="How text is wrapped before it is embedded. The AI Edge Gallery Smart Album embeds raw text; the EmbeddingGemma model card recommends task prompts for text retrieval."
          >
            Prompt
            <span class="select-wrap">
              <select @change=${this.onTemplateChange}>
                ${(Object.keys(PROMPT_TEMPLATE_LABELS) as PromptTemplateId[])
                    .map((id) => html`
                  <option value=${id} ?selected=${id === this.state.promptTemplate}>
                    ${PROMPT_TEMPLATE_LABELS[id]}
                  </option>
                `)}
              </select>
              <span class="select-caret model-caret icon" aria-hidden="true">expand_more</span>
            </span>
          </label>

          <button
            type="button"
            class="model-status"
            aria-haspopup="dialog"
            @click=${() => {
              this.state.isModelModalOpen = true;
              this.state.requestUpdate();
            }}
            title="Click to change model"
          >
            <div class="status-dot ${dotClass}"></div>
            <span class="model-label" style="color: ${
                isModelLoaded || models.isLoading ? '#ffffff' : 'var(--text-muted)'}">
              ${modelLabel}
            </span>
            <span class="model-caret icon" aria-hidden="true">expand_more</span>
          </button>
        </div>
      </div>
    `;
  }
}
