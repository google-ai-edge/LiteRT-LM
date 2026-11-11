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

import {findModel, migrateModelPath, MODELS, SettingsStore} from './settings_store.js';

describe('SettingsStore', () => {
  let mockCallbackCount = 0;
  function mockUpdateCallback() {
    mockCallbackCount++;
  }

  beforeEach(() => {
    mockCallbackCount = 0;
    // Clear localStorage before testing
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('initializes with default values', () => {
    const store = new SettingsStore(mockUpdateCallback);
    expect(store.selectedModelPath).toBe(MODELS[0]!.path);
    expect(store.contextLength).toBe(4096);
    expect(store.maxOutputTokens).toBe(2048);
    expect(store.samplerType).toBe('greedy');
    expect(store.temperature).toBe(1.0);
    expect(store.topP).toBe(0.95);
    expect(store.topK).toBe(64);
    expect(store.enableThinking).toBe(true);
    expect(store.enableSpeculativeDecoding).toBe(false);
    expect(store.systemPrompt).toBe('');
    expect(store.disabledTools).toEqual([]);
  });

  it('saves and loads the system prompt and model config', () => {
    const store = new SettingsStore(mockUpdateCallback);
    store.systemPrompt = 'Be concise.';
    store.disabledTools = ['run_javascript'];
    store.enableSpeculativeDecoding = true;
    store.saveSettings();

    const loaded = new SettingsStore(mockUpdateCallback);
    expect(loaded.systemPrompt).toBe('Be concise.');
    expect(loaded.disabledTools).toEqual(['run_javascript']);
    expect(loaded.enableSpeculativeDecoding).toBeTrue();
    expect(loaded.modelSettings).toEqual(jasmine.objectContaining({
      systemPrompt: 'Be concise.',
      disabledTools: ['run_javascript'],
      enableSpeculativeDecoding: true,
    }));
  });

  it('resets only the system prompt and tools', () => {
    const store = new SettingsStore(mockUpdateCallback);
    store.systemPrompt = 'Be concise.';
    store.disabledTools = ['run_javascript'];
    store.temperature = 0.1;

    store.resetDefaults('systemPrompt');

    expect(store.systemPrompt).toBe('');
    expect(store.disabledTools).toEqual([]);
    expect(store.temperature).toBe(0.1);
  });

  it('saves settings to localStorage', () => {
    const store = new SettingsStore(mockUpdateCallback);
    store.temperature = 0.5;
    store.saveSettings();

    const storedData = window.localStorage.getItem('litertlm-chat-settings');
    expect(storedData).toBeDefined();

    if (storedData) {
      const parsed = JSON.parse(storedData) as Record<string, unknown>;
      expect(parsed['temperature']).toBe(0.5);
      expect(parsed['samplerType']).toBe('greedy');
    }

    expect(mockCallbackCount).toBe(1);
  });

  it('loads valid settings from localStorage', () => {
    const customPayload = {
      contextLength: 1024,
      maxOutputTokens: 512,
      samplerType: 'top_k',
      topK: 40,
    };
    window.localStorage.setItem(
        'litertlm-chat-settings', JSON.stringify(customPayload));

    const store = new SettingsStore(mockUpdateCallback);

    // Initial loaded items should match stored payload
    expect(store.contextLength).toBe(1024);
    expect(store.maxOutputTokens).toBe(512);
    expect(store.samplerType).toBe('top_k');
    expect(store.topK).toBe(40);

    // Everything else remains default
    expect(store.temperature).toBe(1.0);
    expect(store.selectedModelPath).toBe(MODELS[0]!.path);
  });

  it('ignores invalid types from localStorage and maintains defaults', () => {
    const corruptedPayload = {
      contextLength: 'a string instead of a number',
      topK: -100  // invalid negative value
    };
    window.localStorage.setItem(
        'litertlm-chat-settings', JSON.stringify(corruptedPayload));

    // When the schema parsing fails, it safely falls back to defaults.
    const store = new SettingsStore(mockUpdateCallback);
    expect(store.contextLength).toBe(4096);
    expect(store.topK).toBe(64);
  });

  it('resets configurations to defaults', () => {
    const store = new SettingsStore(mockUpdateCallback);
    store.temperature = 0.1;
    store.contextLength = 1024;
    store.enableSpeculativeDecoding = true;
    store.systemPrompt = 'Be concise.';

    store.resetDefaults();

    expect(store.temperature).toBe(1.0);
    expect(store.contextLength).toBe(4096);
    expect(store.enableSpeculativeDecoding).toBeFalse();
    // The system prompt is reset separately.
    expect(store.systemPrompt).toBe('Be concise.');
    expect(mockCallbackCount).toBe(1);  // the reset triggers a save callback
  });

  it('loads custom models from localStorage', () => {
    const customPayload = {
      customModels: [
        {
          name: 'Custom Model',
          filename: 'custom.litertlm',
          path: 'https://local-model/custom.litertlm',
          size: '1.0 GB',
        }
      ]
    };
    window.localStorage.setItem(
        'litertlm-chat-settings', JSON.stringify(customPayload));

    const store = new SettingsStore(mockUpdateCallback);
    expect(store.customModels.length).toBe(1);
    expect(store.customModels[0]!.name).toBe('Custom Model');
  });

  it('replaces the default models\' old paths', () => {
    const oldPath =
        'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm';
    expect(migrateModelPath(oldPath))
        .toBe('hf://litert-community/gemma-4-E4B-it-litert-lm');
    expect(migrateModelPath('https://example.com/model.litertlm'))
        .toBe('https://example.com/model.litertlm');

    window.localStorage.setItem(
        'litertlm-chat-settings', JSON.stringify({selectedModelPath: oldPath}));
    expect(new SettingsStore(mockUpdateCallback).selectedModelPath)
        .toBe('hf://litert-community/gemma-4-E4B-it-litert-lm');
  });

  it('finds built-in, custom, and local-directory models by path', () => {
    const custom =
        {name: 'Mine', filename: 'mine.litertlm', path: 'p/mine', size: '1 GB'};
    const local =
        {name: 'Dir', filename: 'dir.litertlm', path: 'p/local', size: '2 GB'};
    const models = {customModels: [custom], localDirModels: [local]};

    expect(findModel(models, MODELS[1]!.path)).toBe(MODELS[1]);
    expect(findModel(models, 'p/mine')).toBe(custom);
    expect(findModel(models, 'p/local')).toBe(local);
    expect(findModel(models, 'p/unknown')).toBeUndefined();
  });
});
