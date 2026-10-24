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

import {Backend, fillWasmEngineSettingsFromEngineSettings, LiteRtLm, loadLiteRtLm, unloadLiteRtLm, type Wasm, wasmEngineSettingsToEngineSettings} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

import {Cleanup} from './cleanup.js';
import {createStreamingModelAssets} from './stream_utils.js';

describe('EngineSettings', () => {
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

  it('creates EngineSettings', async () => {
    const engineSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
    expect(engineSettings).toBeDefined();
    engineSettings.delete();
  });

  it('fills partial CPU backendConfig with defaults', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.CPU,
          mainExecutorSettings: {
            backendConfig: {number_of_threads: 4},
          },
        },
        Backend.CPU, liteRtLm.liteRtLmWasm);
    const cpuConfig =
        wasmSettings.getMutableMainExecutorSettings().getBackendConfigCpu();
    expect(cpuConfig).toEqual({
      kv_increment_size: jasmine.any(Number),
      prefill_chunk_size: jasmine.any(Number),
      number_of_threads: 4,
    });
    wasmSettings.delete();
  });

  it('fills partial GPU backendConfig with defaults', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          mainExecutorSettings: {
            backendConfig: {max_top_k: 3},
          },
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    const gpuConfig =
        wasmSettings.getMutableMainExecutorSettings().getBackendConfigGpu();
    expect(gpuConfig).toEqual({
      max_top_k: 3,
      external_tensor_mode: jasmine.any(Boolean),
    });
    wasmSettings.delete();
  });

  it('fills partial GPU_ARTISAN backendConfig with defaults', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU_ARTISAN,
          mainExecutorSettings: {
            backendConfig: {max_top_k: 5},
          },
        },
        Backend.GPU_ARTISAN, liteRtLm.liteRtLmWasm);
    const gpuArtisanConfig = wasmEngineSettingsToEngineSettings(wasmSettings)
                                 .mainExecutorSettings.backendConfig;
    expect(gpuArtisanConfig).toEqual({
      num_output_candidates: jasmine.any(Number),
      wait_for_weight_uploads: jasmine.any(Boolean),
      num_decode_steps_per_sync: jasmine.any(Number),
      sequence_batch_size: jasmine.any(Number),
      supported_lora_ranks: jasmine.any(Array),
      max_top_k: 5,
      enable_decode_logits: jasmine.any(Boolean),
      enable_external_embeddings: jasmine.any(Boolean),
      use_submodel: jasmine.any(Boolean),
      use_autosized_ringbuffers: jasmine.any(Boolean),
    });
    wasmSettings.delete();
  });

  it('round-trips maxNumImages', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU_ARTISAN,
          mainExecutorSettings: {maxNumImages: 3},
        },
        Backend.GPU_ARTISAN, liteRtLm.liteRtLmWasm);
    expect(wasmEngineSettingsToEngineSettings(wasmSettings)
               .mainExecutorSettings.maxNumImages)
        .toBe(3);
    wasmSettings.delete();
  });

  it('round-trips multimodal vision settings', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU,
            liteRtLm.liteRtLmWasm.Backend.CPU, undefined);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          visionBackend: Backend.CPU,
          maxVisionTokensPerImage: 280,
          minVisionTokensPerImage: 140,
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
    expect(resolved.visionBackend).toBe(Backend.CPU);
    expect(resolved.maxVisionTokensPerImage).toBe(280);
    expect(resolved.minVisionTokensPerImage).toBe(140);
    wasmSettings.delete();
  });

  it('round-trips multimodal audio settings', async () => {
    const wasmSettings =
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU, undefined,
            liteRtLm.liteRtLmWasm.Backend.CPU);
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          audioBackend: Backend.CPU,
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
    expect(resolved.audioBackend).toBe(Backend.CPU);
    expect(resolved.visionBackend).toBeUndefined();
    wasmSettings.delete();
  });

  it('leaves vision and audio backends unset for text-only settings',
     async () => {
       const wasmSettings =
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
       const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
       expect(resolved.visionBackend).toBeUndefined();
       expect(resolved.audioBackend).toBeUndefined();
       wasmSettings.delete();
     });
});

