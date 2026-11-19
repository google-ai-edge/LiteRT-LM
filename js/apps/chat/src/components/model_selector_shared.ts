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

import {css, html, nothing, TemplateResult} from 'lit';

import {LlmChatStateController} from '../state_controller.js';
import {getModelKey} from '../stores/model_loader_service.js';
import {parseModelReference} from '../stores/model_reference.js';
import {coerceGpuOnlyTextBackend, CustomModel, findModel, getNonDefaultBackendSummary, MODELS, NonDefaultBackendSummary, selectCustomModel} from '../stores/settings_store.js';

const LARGE_MODEL_WARNING_KEY = 'litertlm-large-model-warning-dismissed';

/** Shared CSS styles for model dropdown rows, badges, and inline URL forms. */
export const modelSelectorStyles = css`
  /* Add remote model form */
  .add-model-form {
    display: flex;
    gap: 6px;
    align-items: center;
    width: 100%;
    margin-top: 4px;
  }

  .add-model-form input {
    flex: 1;
    min-width: 0;
    padding: 7px 10px;
    font-size: 0.76rem;
  }

  .btn-cancel-add {
    background: none;
    border: 1px solid var(--border, #262f40);
    color: var(--text-muted, #94a3b8);
    border-radius: 8px;
    width: 30px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
    padding: 0;
  }

  .btn-cancel-add:hover {
    background-color: var(--bg-elevated, #1e2432);
    color: var(--text, #f1f5f9);
  }

  /* Dropdown item rows and inner badges */
  .dropdown-row {
    display: flex;
    align-items: center;
    gap: 2px;
    border-radius: 7px;
  }

  .dropdown-row:hover,
  .dropdown-row:has(.dropdown-item[data-focused]) {
    background-color: var(--bg-elevated, #1e2432);
  }

  .dropdown-row:has(.dropdown-item[data-focused]) {
    box-shadow: inset 0 0 0 1.5px var(--teal, #14d4aa);
  }

  .dropdown-row:has(.dropdown-item[data-selected]) {
    background-color: var(--teal-subtle, rgba(20, 212, 170, 0.12));
  }

  .dropdown-row .dropdown-item {
    flex: 1;
    min-width: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    padding: 7px 9px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 0.775rem;
    color: var(--text-secondary, #cbd5e1);
    outline: none;
    transition: color 0.12s ease;
  }

  .dropdown-row .dropdown-item:focus,
  .dropdown-row .dropdown-item:focus-visible {
    outline: none;
  }

  .dropdown-row .dropdown-item:hover,
  .dropdown-row .dropdown-item[data-focused] {
    background-color: transparent;
    color: var(--text, #f1f5f9);
  }

  .dropdown-row .dropdown-item[data-selected] {
    color: var(--text, #f1f5f9);
    font-weight: 600;
  }

  .model-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }

  .item-meta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }

  .size-badge {
    font-size: 0.66rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background-color: var(--bg-elevated, #1e2432);
    border: 1px solid var(--border, #262f40);
    padding: 1px 6px;
    border-radius: 999px;
    color: var(--text-muted, #94a3b8);
  }

  .download-badge {
    font-size: 0.66rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-muted, #94a3b8);
    white-space: nowrap;
  }

  .dropdown-action-icon {
    width: 14px;
    height: 14px;
    color: var(--teal, #14d4aa);
    flex-shrink: 0;
  }

  .local-badge {
    font-size: 0.66rem;
    font-weight: 600;
    color: var(--teal, #14d4aa);
  }

  .downloading-info {
    font-size: 0.68rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--teal, #14d4aa);
    font-weight: 700;
  }

  .delete-cache-btn {
    background: none;
    border: none;
    color: var(--danger, #f87171);
    cursor: pointer;
    font-size: 0.76rem;
    line-height: 1;
    padding: 4px 6px;
    margin-right: 3px;
    border-radius: 5px;
    flex-shrink: 0;
    transition: background-color 0.12s ease;
  }

  .delete-cache-btn:hover {
    background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
  }

  .dropdown-action-divider {
    border-top: 1px solid var(--border, #262f40);
    margin-top: 4px;
    padding-top: 4px;
  }

  .dropdown-action-label {
    color: var(--teal, #14d4aa);
    font-weight: 600;
  }

  .dropdown-listbox-owner {
    height: 0;
    overflow: visible;
  }

  .dropdown-footer-section {
    border-top: 1px solid var(--border, #262f40);
    margin-top: 4px;
    padding: 6px 8px 3px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .dropdown-footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: var(--text-muted, #94a3b8);
  }

  .local-dir-info {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cache-bar-value {
    color: var(--text, #f1f5f9);
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .btn-rescan-dir {
    background: none;
    border: 1px solid var(--border, #262f40);
    color: var(--teal, #14d4aa);
    font-size: 0.69rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    padding: 2px 7px;
    border-radius: 5px;
    flex-shrink: 0;
    transition: background-color 0.14s ease, border-color 0.14s ease;
  }

  .btn-rescan-dir:hover:not(:disabled) {
    background-color: var(--teal-subtle, rgba(20, 212, 170, 0.12));
    border-color: var(--teal, #14d4aa);
  }

  .btn-clear-cache {
    background: none;
    border: none;
    color: var(--danger, #f87171);
    font-size: 0.72rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 5px;
    flex-shrink: 0;
    transition: background-color 0.15s ease, color 0.15s ease, opacity 0.15s ease;
  }

  .btn-clear-cache:hover:not(:disabled) {
    background-color: var(--danger-subtle, rgba(248, 113, 113, 0.12));
  }

  .btn-clear-cache:disabled {
    color: var(--text-muted, #94a3b8);
    opacity: 0.45;
    cursor: not-allowed;
    pointer-events: none;
  }
`;

/** Returns the compact size/action label for a model that isn't cached. */
export function getModelDownloadLabel(model: CustomModel): string {
  if (model.path.startsWith('https://local-model/')) return 'Local';
  return model.size ? `↓ ${model.size}` : '↓ Download';
}

/** Returns all built-in, custom, and mounted local-directory models. */
export function getAllAvailableModels(
    state: LlmChatStateController): CustomModel[] {
  return [
    ...MODELS,
    ...(state.settings.customModels ?? []),
    ...(state.settings.localDirModels ?? []),
  ];
}

/** Formatted total cached weights size across all cached models. */
export function getTotalCachedWeightsLabel(
    state: LlmChatStateController): string {
  let totalBytes = 0;
  for (const [_, size] of state.modelLoader.cachedModels ?? []) {
    totalBytes += size;
  }
  return `${(totalBytes / 1e9).toFixed(2)} GB`;
}

/** Compact summary of the currently selected model and its runtime status. */
export interface ActiveModelSummary {
  name: string;
  stateLabel: string;
  compactStateLabel: string;
  dotClass: string;
  isActiveState: boolean;
  backendOverride: NonDefaultBackendSummary|null;
}

/** Computes the active model's display name, status dot, and state label. */
export function getActiveModelSummary(
    state: LlmChatStateController): ActiveModelSummary {
  const selectedPath = state.settings.selectedModelPath || '';
  const match = findModel(state.settings, selectedPath);
  const rawName = match?.name || selectedPath.split('/').pop() || 'Select model';
  const cleanName = rawName
                        .replace(/,\s*\d+(\.\d+)?\s*GB\)$/, ')')
                        .replace(/\s*\(\d+(\.\d+)?\s*GB\)$/, '');
  const backendOverride = getNonDefaultBackendSummary(state.settings);
  const summary =
      (stateLabel: string, dotClass = '', isActiveState = false,
       compactStateLabel = ''): ActiveModelSummary => ({
        name: cleanName,
        stateLabel,
        compactStateLabel,
        dotClass,
        isActiveState,
        backendOverride,
      });

  const key = getModelKey(selectedPath);
  const progress = state.modelLoader.downloadProgresses?.get(key);
  const needsReload = Boolean(
      state.modelLoader.engine &&
      state.modelLoader.needsModelReload?.(state.settings.modelSettings));
  if (state.modelLoader.isModelLoading) {
    const label = progress !== undefined ? `${progress}%` : 'Loading';
    return summary(label, 'loading', true, label);
  }
  if (state.errorMessage) return summary('Error', 'error', true, 'Error');
  if (state.chatSession.isGenerating) {
    return summary('Generating', 'generating', true);
  }
  if (needsReload) return summary('Reload on send', 'loading', true, 'Reload');
  if (state.modelLoader.engine) return summary('Ready', 'ready');
  if (selectedPath.startsWith('local-dir://')) return summary('Local Dir');
  const isCached = Boolean(state.modelLoader.cachedModels?.has(key));
  return summary(isCached ? 'Cached' : 'Not loaded');
}

/** Returns true if `modelPath` belongs to a user-added custom model. */
export function isCustomModelPath(
    state: LlmChatStateController, modelPath: string): boolean {
  return state.settings.customModels.some(m => m.path === modelPath);
}

/** Removes a custom model from settings and falls back to default if active. */
export function removeCustomModelFromSettings(
    state: LlmChatStateController, modelPath: string) {
  state.settings.customModels =
      state.settings.customModels.filter(m => m.path !== modelPath);
  if (state.settings.selectedModelPath === modelPath) {
    state.settings.selectedModelPath = MODELS[0]!.path;
    coerceGpuOnlyTextBackend(state.settings);
  }
}

/** Confirms and removes an uncached custom model from the model list. */
export async function confirmAndRemoveCustomModel(
    state: LlmChatStateController, modelPath: string) {
  const confirmed = await state.requestConfirm({
    title: 'Remove custom model?',
    message: 'Remove this custom model from your model list?',
    confirmLabel: 'Remove',
    danger: true,
  });
  if (!confirmed) return;
  removeCustomModelFromSettings(state, modelPath);
  state.settings.saveSettings();
}

/** Deletes cached weights for `modelPath` and resets selection if needed. */
export async function deleteCachedModelAndSyncSettings(
    state: LlmChatStateController, modelPath: string) {
  const deleted = await state.modelLoader.deleteModelFromCache(modelPath);
  if (!deleted) return;

  let settingsChanged = false;
  if (isCustomModelPath(state, modelPath)) {
    removeCustomModelFromSettings(state, modelPath);
    settingsChanged = true;
  }
  if (state.settings.selectedModelPath === modelPath) {
    state.settings.selectedModelPath = MODELS[0]!.path;
    coerceGpuOnlyTextBackend(state.settings);
    settingsChanged = true;
  }
  if (settingsChanged) {
    state.settings.saveSettings();
  }
}

function isLargeModel(path: string): boolean {
  return /gemma-4-(26|31)b/i.test(path);
}

/**
 * Selects `path` (prompting once for high-VRAM models) and starts loading its
 * weights if the selection changed. Returns false if the user cancelled.
 */
export async function selectModelWithLargeModelCheck(
    state: LlmChatStateController, path: string): Promise<boolean> {
  const dismissed =
      state.storage.getItem(LARGE_MODEL_WARNING_KEY) === 'true';
  if (isLargeModel(path) && !dismissed) {
    const matched = findModel(state.settings, path);
    const modelName =
        matched?.name || path.split('/').pop() || 'Selected model';
    const modelWithSize =
        matched?.size ? `${modelName} (${matched.size})` : modelName;
    const warningMsg =
        `${modelWithSize} needs significant video memory (VRAM). Running it may cause the app or browser tab to crash or your system to become unresponsive if you do not have enough VRAM.`;
    const proceed = await state.requestConfirm({
      title: `Load ${modelWithSize}?`,
      message: `${warningMsg} Do you want to proceed?`,
      confirmLabel: 'Load Anyway',
      danger: false,
    });
    if (!proceed) {
      return false;
    }
    state.storage.setItem(LARGE_MODEL_WARNING_KEY, 'true');
  }

  if (state.settings.selectedModelPath !== path) {
    state.settings.selectedModelPath = path;
    coerceGpuOnlyTextBackend(state.settings);
    state.settings.saveSettings();

    void state.modelLoader.loadModelWeights(
        state.settings.modelSettings, async () => {
          await state.chatSession.createConversationSession?.();
        });
  }
  return true;
}

/** Opens a native file picker to import and load a local `.litertlm` file. */
export function triggerLitertModelUpload(
    state: LlmChatStateController, onSelected?: () => void) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.litertlm';
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const customModel = await state.modelLoader.importCustomModel(file);
      selectCustomModel(state.settings, customModel);
      onSelected?.();

      await state.modelLoader.loadModelWeights(
          state.settings.modelSettings, async () => {
            await state.chatSession.createConversationSession?.();
          });
    } catch (e) {
      console.error('[LiteRT-LM] Failed to import/load custom model:', e);
    }
  };
  input.click();
}

/**
 * Resolves a remote URL or Hugging Face repo, adds it to custom models, selects
 * it, and kicks off loading. Returns true if resolved and selected.
 */
export async function addRemoteModelFromInput(
    state: LlmChatStateController,
    text: string,
    onSelected?: () => void): Promise<boolean> {
  const reference = parseModelReference(text);
  const builtIn =
      reference ? MODELS.find(m => m.path === reference) : undefined;
  let customModel: CustomModel;
  if (builtIn) {
    customModel = builtIn;
  } else {
    try {
      customModel = await state.modelLoader.addRemoteModel(
          text, state.settings.textBackend);
    } catch {
      return false;
    }
  }
  selectCustomModel(state.settings, customModel);
  onSelected?.();

  await state.modelLoader.loadModelWeights(
      state.settings.modelSettings, async () => {
        await state.chatSession.createConversationSession?.();
      });
  return true;
}

/** Renders the shared model items and action rows inside a `custom-dropdown`. */
export function renderModelDropdownItems(
    state: LlmChatStateController,
    idPrefix = 'model-dropdown'): TemplateResult {
  const allModels = getAllAvailableModels(state);
  const localDirSupported = Boolean(state.localDirService?.isSupported);
  const localDirAuthorized = Boolean(state.localDirService?.isAuthorized);
  const localDirName = state.localDirService?.directoryName || 'Mounted folder';
  const hasCachedWeights = (state.modelLoader.cachedModels?.size ?? 0) > 0;
  const actionIds = [
    ...(localDirSupported ? [`${idPrefix}-opt-select-dir`] : []),
    `${idPrefix}-opt-upload`,
    `${idPrefix}-opt-add-url`,
  ];
  const optionIds = [
    ...allModels.map((_, idx) => `${idPrefix}-opt-${idx}`),
    ...actionIds,
  ];
  return html`
    <div
      class="dropdown-listbox-owner"
      role="listbox"
      aria-label="Available models"
      aria-owns=${optionIds.join(' ')}
    ></div>
    ${allModels.map((model, idx) => {
      const modelKey = getModelKey(model.path);
      const cachedSizeBytes = state.modelLoader.cachedModels?.get(modelKey);
      const downloadProgress =
          state.modelLoader.downloadProgresses?.get(modelKey);
      const isLocalDir = model.path.startsWith('local-dir://');

      return html`
        <div class="dropdown-row" role="presentation">
          <div
            id="${idPrefix}-opt-${idx}"
            class="dropdown-item"
            data-value="${model.path}"
          >
            <span class="model-name">${model.name}</span>

            ${isLocalDir ? html`
              <span class="item-meta">
                ${model.size ? html`<span class="download-badge">${model.size}</span>` : nothing}
                <span class="local-badge">Local Dir</span>
              </span>
            ` : cachedSizeBytes ? html`
              <span class="item-meta">
                <span class="size-badge">${(cachedSizeBytes / 1e9).toFixed(1)} GB</span>
              </span>
            ` : downloadProgress !== undefined ? html`
              <span class="downloading-info">${downloadProgress}%</span>
            ` : html`
              <span class="item-meta">
                <span class="download-badge">${getModelDownloadLabel(model)}</span>
              </span>
            `}
          </div>
          ${!isLocalDir && cachedSizeBytes ? html`
            <button
              type="button"
              class="delete-cache-btn"
              title="Remove from cache"
              aria-label="Remove ${model.name} from cache"
              @click=${(e: Event) => {
                e.stopPropagation();
                void deleteCachedModelAndSyncSettings(state, model.path);
              }}
            >✕</button>
          ` : !isLocalDir && !cachedSizeBytes &&
                  downloadProgress === undefined &&
                  isCustomModelPath(state, model.path) ? html`
            <button
              type="button"
              class="delete-cache-btn"
              title="Remove from list"
              aria-label="Remove ${model.name} from list"
              @click=${(e: Event) => {
                e.stopPropagation();
                void confirmAndRemoveCustomModel(state, model.path);
              }}
            >✕</button>
          ` : nothing}
        </div>
      `;
    })}
    ${localDirSupported ? html`
      <div class="dropdown-row dropdown-action-divider" role="presentation">
        <div
          id="${idPrefix}-opt-select-dir"
          class="dropdown-item"
          data-value="select-dir"
        >
          <span class="model-name dropdown-action-label">${
            localDirAuthorized ?
                'Change local directory…' :
                'Select local directory…'}</span>
          <svg class="dropdown-action-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
          </svg>
        </div>
      </div>
    ` : nothing}
    <div
      class="dropdown-row ${
        localDirSupported ? '' : 'dropdown-action-divider'}"
      role="presentation"
    >
      <div
        id="${idPrefix}-opt-upload"
        class="dropdown-item"
        data-value="upload"
      >
        <span class="model-name dropdown-action-label">Upload .litertlm file…</span>
        <svg class="dropdown-action-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
      </div>
    </div>
    <div class="dropdown-row" role="presentation">
      <div
        id="${idPrefix}-opt-add-url"
        class="dropdown-item"
        data-value="add-url"
      >
        <span class="model-name dropdown-action-label">Add URL / Hugging Face…</span>
        <svg class="dropdown-action-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
        </svg>
      </div>
    </div>
    <div class="dropdown-footer-section">
      ${(() => {
        const backendOverride = getNonDefaultBackendSummary(state.settings);
        const overrideHeading =
            backendOverride?.detail.startsWith('Modalities:') ?
            'Modalities:' :
            'Backend override:';
        return backendOverride ? html`
          <div
            id="${idPrefix}-backend-note"
            class="dropdown-footer-row backend-override-bar"
            title=${backendOverride.detail}
          >
            <span>${overrideHeading}</span>
            <span class="cache-bar-value">${backendOverride.label}</span>
          </div>
        ` : nothing;
      })()}
      ${localDirAuthorized ? html`
        <div class="dropdown-footer-row local-dir-bar">
          <span class="local-dir-info" title=${localDirName}>
            Local dir: <span class="cache-bar-value">${localDirName}</span>
          </span>
          <button
            type="button"
            class="btn-rescan-dir"
            aria-label="Rescan local directory"
            @click=${(e: Event) => {
              e.stopPropagation();
              void state.localDirService.scanDirectory();
            }}
          >Rescan</button>
        </div>
      ` : nothing}
      <div class="dropdown-footer-row cache-footer-row">
        <span>Cached weights: <span class="cache-bar-value">${getTotalCachedWeightsLabel(state)}</span></span>
        <button
          id="${idPrefix}-clear-cache"
          type="button"
          class="btn-clear-cache"
          ?disabled=${!hasCachedWeights || state.modelLoader.isModelLoading}
          @click=${(e: Event) => {
            e.stopPropagation();
            void state.modelLoader.clearAllCache?.();
          }}
        >Clear All</button>
      </div>
    </div>
  `;
}
