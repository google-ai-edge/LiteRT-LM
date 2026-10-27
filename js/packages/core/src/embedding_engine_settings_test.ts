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

import {type EmbeddingEngineSettings, fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings, LiteRtLm, loadLiteRtLm, unloadLiteRtLm, type Wasm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

import {Cleanup} from './cleanup.js';
import {createStreamingModelAssets} from './stream_utils.js';

describe('EmbeddingEngineSettings', () => {
  let liteRtLm: LiteRtLm;
  let cleanup: Cleanup;
  let modelAssets: Wasm.ModelAssets;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm('/wasm');
    cleanup = new Cleanup();
    ({modelAssets} = await createStreamingModelAssets(
         new Blob([]), liteRtLm.liteRtLmWasm, cleanup));
  });

  afterAll(() => {
    cleanup.run();
  });

  it('creates EmbeddingEngineSettings', () => {
    const settings =
        liteRtLm.liteRtLmWasm.EmbeddingEngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
    expect(settings).toBeDefined();
    settings.delete();
  });

  describe('input length bounds', () => {
    function fill(settings: Partial<EmbeddingEngineSettings>):
        Wasm.EmbeddingEngineSettings {
      const wasmSettings =
          liteRtLm.liteRtLmWasm.EmbeddingEngineSettings.createDefault(
              modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
      fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings(
          wasmSettings, {...settings, model: '/path/to/model'});
      return wasmSettings;
    }

    it('prepares a single signature when neither bound is set', () => {
      const wasmSettings = fill({});
      expect(wasmSettings.getMaxInputLength()).toBe(1024);
      expect(wasmSettings.getMinInputLength()).toBe(1024);
      wasmSettings.delete();
    });

    it('makes minInputLength follow an explicit maxInputLength', () => {
      const wasmSettings = fill({maxInputLength: 256});
      expect(wasmSettings.getMaxInputLength()).toBe(256);
      expect(wasmSettings.getMinInputLength()).toBe(256);
      wasmSettings.delete();
    });

    it('keeps an explicit minInputLength', () => {
      const wasmSettings = fill({minInputLength: 128, maxInputLength: 512});
      expect(wasmSettings.getMaxInputLength()).toBe(512);
      expect(wasmSettings.getMinInputLength()).toBe(128);
      wasmSettings.delete();
    });
  });

  describe('vision token bounds', () => {
    function fill(settings: Partial<EmbeddingEngineSettings>):
        Wasm.EmbeddingEngineSettings {
      const wasmSettings =
          liteRtLm.liteRtLmWasm.EmbeddingEngineSettings.createDefault(
              modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
      fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings(
          wasmSettings, {...settings, model: '/path/to/model'});
      return wasmSettings;
    }

    it('leaves both bounds unset by default', () => {
      // The engine resolves them from the model's metadata at creation.
      const wasmSettings = fill({});
      expect(wasmSettings.getVisionTokensPerImage()).toBeUndefined();
      expect(wasmSettings.getMinVisionTokensPerImage()).toBeUndefined();
      wasmSettings.delete();
    });

    it('passes explicit vision token bounds through', () => {
      const wasmSettings =
          fill({visionTokensPerImage: 280, minVisionTokensPerImage: 140});
      expect(wasmSettings.getVisionTokensPerImage()).toBe(280);
      expect(wasmSettings.getMinVisionTokensPerImage()).toBe(140);
      wasmSettings.delete();
    });
  });
});
