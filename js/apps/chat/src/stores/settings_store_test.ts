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

import {MODELS, SettingsStore} from './settings_store.js';

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

    // Auto-mock window.confirm to bypass browser blocking prompt
    spyOn(window, 'confirm').and.returnValue(true);

    const wasReset = store.resetDefaults();

    expect(wasReset).toBeTrue();
    expect(store.temperature).toBe(1.0);
    expect(store.contextLength).toBe(4096);
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
});
