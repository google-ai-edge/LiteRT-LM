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

import {CPU_TEXT_WASM_WARNING_BYTES, findModel, getCpuTextMemoryWarning, getModalityConfigs, getNonDefaultBackendSummary, isGpuOnlyTextModel, migrateModelPath, MODELS, parseModelSizeBytes, selectCustomModel, SettingsStore} from './settings_store.js';

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
    expect(store.textBackend).toBe('gpu');
    expect(store.visionBackend).toBe('gpu');
    expect(store.audioBackend).toBe('gpu');
    expect(store.textPrecision).toBe('default');
    expect(store.visionPrecision).toBe('default');
    expect(store.audioPrecision).toBe('default');
  });

  it('persists, resets, and coerces experimental backend and precision settings', () => {
    const store = new SettingsStore(mockUpdateCallback);
    store.selectedModelPath = 'https://example.com/custom-cpu.litertlm';
    store.textBackend = 'cpu';
    store.visionBackend = 'cpu';
    store.audioBackend = 'gpu';
    store.textPrecision = 'fp32';
    store.visionPrecision = 'fp16';
    store.audioPrecision = 'fp16';
    store.temperature = 0.4;
    store.saveSettings();

    const loaded = new SettingsStore(mockUpdateCallback);
    expect(loaded.textBackend).toBe('cpu');
    expect(loaded.visionBackend).toBe('cpu');
    expect(loaded.audioPrecision).toBe('fp16');
    expect(isGpuOnlyTextModel(MODELS[0]!.path)).toBeTrue();
    expect(isGpuOnlyTextModel('https://example.com/custom-cpu.litertlm'))
        .toBeFalse();

    // If a GPU-only -gpu.litertlm model is saved with textBackend='cpu',
    // loadSettings coerces textBackend back to 'gpu'.
    window.localStorage.setItem(
        'litertlm-chat-settings',
        JSON.stringify({
          selectedModelPath: MODELS[0]!.path,
          textBackend: 'cpu',
          visionBackend: 'cpu',
        }));
    const coerced = new SettingsStore(mockUpdateCallback);
    expect(coerced.textBackend).toBe('gpu');
    expect(coerced.visionBackend).toBe('cpu');

    // Resetting 'experimental' leaves Model Config untouched.
    loaded.resetDefaults('experimental');
    expect(loaded.textBackend).toBe('gpu');
    expect(loaded.visionBackend).toBe('gpu');
    expect(loaded.audioBackend).toBe('gpu');
    expect(loaded.textPrecision).toBe('default');
    expect(loaded.visionPrecision).toBe('default');
    expect(loaded.audioPrecision).toBe('default');
    expect(loaded.temperature).toBe(0.4);
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

  it('summarizes non-default modality backend overrides and returns null for all-GPU',
     () => {
       const store = new SettingsStore(mockUpdateCallback);
       expect(getNonDefaultBackendSummary(store)).toBeNull();

       store.visionBackend = 'cpu';
       expect(getNonDefaultBackendSummary(store)).toEqual({
         label: 'WebGPU · CPU',
         detail: 'Backend: WebGPU · CPU (Vision on CPU)',
       });

       store.selectedModelPath = 'https://example.com/gemma-3-1b-it.litertlm';
       store.textBackend = 'cpu';
       store.audioBackend = 'cpu';
       expect(getNonDefaultBackendSummary(store)).toEqual({
         label: 'CPU',
         detail: 'Backend: CPU (Text, Vision, Audio in WASM FP32)',
       });

       // -gpu.litertlm models coerce Text back to GPU -> mixed WebGPU · CPU
       store.selectedModelPath = MODELS[0]!.path;
       expect(getNonDefaultBackendSummary(store)).toEqual({
         label: 'WebGPU · CPU',
         detail: 'Backend: WebGPU · CPU (Vision, Audio on CPU)',
       });
     });

  it('selects built-in models in selectCustomModel without adding duplicates to customModels',
     () => {
       const store = new SettingsStore(mockUpdateCallback);
       store.textBackend = 'cpu';

       selectCustomModel(store, MODELS[1]!);
       expect(store.selectedModelPath).toBe(MODELS[1]!.path);
       expect(store.customModels).toEqual([]);
       expect(store.textBackend).toBe('gpu');

       const custom = {
         name: 'Mine',
         filename: 'mine.litertlm',
         path: 'hf://org/mine',
         size: '1 GB',
       };
       const otherCustom = {
         name: 'Other',
         filename: 'other.litertlm',
         path: 'hf://org/other',
         size: '2 GB',
       };
       selectCustomModel(store, custom);
       selectCustomModel(store, custom);
       selectCustomModel(store, otherCustom);
       expect(store.selectedModelPath).toBe(otherCustom.path);
       expect(store.customModels).toEqual([custom, otherCustom]);
     });

  it('never blocks CPU text unless the model is -gpu.litertlm, regardless of size or format',
     () => {
       const store = new SettingsStore(mockUpdateCallback);
       store.customModels = [
         {
           name: 'Huge Standard Model',
           filename: 'gemma-4-9B-it.litertlm',
           path: 'https://example.com/gemma-4-9B-it.litertlm',
           size: '5.40 GB',
         },
         {
           name: 'External Weights Repo',
           filename: 'model.litertlm',
           path: 'hf://org/external-weights-repo',
           size: '2.10 GB',
         },
         {
           name: 'Artisan GPU Package',
           filename: 'gemma-4-E2B-it-gpu.litertlm',
           path: 'hf://org/artisan-gpu-repo',
           size: '1.10 GB',
         },
       ];

       store.textBackend = 'cpu';

       store.selectedModelPath = 'https://example.com/gemma-4-9B-it.litertlm';
       expect(isGpuOnlyTextModel(store.selectedModelPath, store)).toBeFalse();
       expect(getModalityConfigs(store).text.backend).toBe('cpu');

       store.selectedModelPath = 'hf://org/external-weights-repo';
       expect(isGpuOnlyTextModel(store.selectedModelPath, store)).toBeFalse();
       expect(getModalityConfigs(store).text.backend).toBe('cpu');

       store.selectedModelPath = 'local-dir://big-unlisted-4gb.litertlm';
       expect(isGpuOnlyTextModel(store.selectedModelPath, store)).toBeFalse();
       expect(getModalityConfigs(store).text.backend).toBe('cpu');

       // Only -gpu.litertlm is coerced / hard-blocked from CPU text.
       store.selectedModelPath = 'hf://org/artisan-gpu-repo';
       expect(isGpuOnlyTextModel(store.selectedModelPath, store)).toBeTrue();
       expect(getModalityConfigs(store).text.backend).toBe('gpu');
     });

  it('computes non-blocking CPU WASM memory warnings around the ~1.5 GB threshold',
     () => {
       expect(parseModelSizeBytes('900 MB')).toBe(900_000_000);
       expect(parseModelSizeBytes('1.49 GB')).toBe(1_490_000_000);
       expect(parseModelSizeBytes('1.50 GB')).toBe(CPU_TEXT_WASM_WARNING_BYTES);
       expect(parseModelSizeBytes('')).toBeNull();
       expect(parseModelSizeBytes('unknown')).toBeNull();

       const models = {
         customModels: [
           {
             name: 'Small',
             filename: 'small.litertlm',
             path: 'https://example.com/small.litertlm',
             size: '1.20 GB',
           },
           {
             name: 'AtThreshold',
             filename: 'threshold.litertlm',
             path: 'https://example.com/threshold.litertlm',
             size: '1.50 GB',
           },
           {
             name: 'Large',
             filename: 'large.litertlm',
             path: 'https://example.com/large.litertlm',
             size: '2.40 GB',
           },
           {
             name: 'UnknownSize',
             filename: 'unknown.litertlm',
             path: 'https://example.com/unknown.litertlm',
             size: '',
           },
         ],
       };

       // Below 1.5 GB -> no warning
       expect(getCpuTextMemoryWarning(
                  'https://example.com/small.litertlm', models))
           .toBeNull();

       // At or above 1.5 GB -> non-blocking warning citing model size
       const atThreshold = getCpuTextMemoryWarning(
           'https://example.com/threshold.litertlm', models);
       expect(atThreshold).toContain('1.50 GB');
       expect(atThreshold).toContain('~1.5 GB practical limit');
       expect(atThreshold).toContain('loading may fail');

       const largeWarn = getCpuTextMemoryWarning(
           'https://example.com/large.litertlm', models);
       expect(largeWarn).toContain('2.40 GB may not fit in WASM memory');

       // Unknown metadata size falls back to cached byte count when available
       expect(getCpuTextMemoryWarning(
                  'https://example.com/unknown.litertlm', models))
           .toBeNull();
       expect(getCpuTextMemoryWarning(
                  'https://example.com/unknown.litertlm', models, 1_850_000_000))
           .toContain('1.85 GB may not fit in WASM memory');

       // -gpu.litertlm uses its own WebGPU-only block rather than the size warning
       expect(getCpuTextMemoryWarning(MODELS[0]!.path, models, 2_300_000_000))
           .toBeNull();
     });
});
