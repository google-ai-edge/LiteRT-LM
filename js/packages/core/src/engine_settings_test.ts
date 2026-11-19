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

import {ActivationDataType, Backend, fillWasmEngineSettingsFromEngineSettings, LiteRtLm, loadLiteRtLm, SpeculativeDecodingMode, unloadLiteRtLm, type Wasm, wasmEngineSettingsToEngineSettings} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

import {Cleanup} from './cleanup.js';
import {createStreamingModelAssets} from './stream_utils.js';
import {unwrapWasmResult} from './wasm_binding_types.js';

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
    const engineSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU));
    expect(engineSettings).toBeDefined();
    engineSettings.delete();
  });

  async function resolveWith(
      wasmBackend: Wasm.EmscriptenEnumElement<Backend>, backend: Backend,
      mainExecutorSettings?:
          Parameters<typeof fillWasmEngineSettingsFromEngineSettings>[
            1
          ]['mainExecutorSettings']) {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, wasmBackend));
    try {
      fillWasmEngineSettingsFromEngineSettings(
          wasmSettings, {
            model: '/path/to/model',
            backend,
            ...(mainExecutorSettings !== undefined ? {mainExecutorSettings} :
                                                     {}),
          },
          backend, liteRtLm.liteRtLmWasm);
      return wasmEngineSettingsToEngineSettings(
          wasmSettings, liteRtLm.liteRtLmWasm);
    } finally {
      wasmSettings.delete();
    }
  }

  it('produces identical CPU backendConfig defaults when omitted, empty, or partial',
     async () => {
       const wasmCpu = liteRtLm.liteRtLmWasm.Backend.CPU;
       const omittedMain = await resolveWith(wasmCpu, Backend.CPU);
       const omittedBackendConfig = await resolveWith(wasmCpu, Backend.CPU, {});
       const emptyBackendConfig =
           await resolveWith(wasmCpu, Backend.CPU, {backendConfig: {}});
       const undefinedFieldConfig = await resolveWith(wasmCpu, Backend.CPU, {
         backendConfig: {number_of_threads: undefined},
       });

       expect(omittedMain.mainExecutorSettings.backendConfig).toEqual({
         kv_increment_size: 16,
         prefill_chunk_size: -1,
         number_of_threads: 4,
       });
       expect(omittedBackendConfig).toEqual(omittedMain);
       expect(emptyBackendConfig).toEqual(omittedMain);
       expect(undefinedFieldConfig).toEqual(omittedMain);

       const partial = await resolveWith(wasmCpu, Backend.CPU, {
         backendConfig: {number_of_threads: 2},
       });
       expect(partial.mainExecutorSettings.backendConfig).toEqual({
         ...omittedMain.mainExecutorSettings.backendConfig,
         number_of_threads: 2,
       });
     });

  it('produces identical GPU backendConfig defaults when omitted, empty, or partial',
     async () => {
       const wasmGpu = liteRtLm.liteRtLmWasm.Backend.GPU;
       const omittedMain = await resolveWith(wasmGpu, Backend.GPU);
       const omittedBackendConfig = await resolveWith(wasmGpu, Backend.GPU, {});
       const emptyBackendConfig =
           await resolveWith(wasmGpu, Backend.GPU, {backendConfig: {}});
       const undefinedFieldConfig = await resolveWith(wasmGpu, Backend.GPU, {
         backendConfig: {max_top_k: undefined},
       });

       expect(omittedMain.mainExecutorSettings.backendConfig).toEqual({
         max_top_k: 64,
         external_tensor_mode: false,
       });
       expect(omittedBackendConfig).toEqual(omittedMain);
       expect(emptyBackendConfig).toEqual(omittedMain);
       expect(undefinedFieldConfig).toEqual(omittedMain);

       const partial = await resolveWith(wasmGpu, Backend.GPU, {
         backendConfig: {max_top_k: 3},
       });
       expect(partial.mainExecutorSettings.backendConfig).toEqual({
         ...omittedMain.mainExecutorSettings.backendConfig,
         max_top_k: 3,
       });
     });

  it('produces identical GPU_ARTISAN backendConfig defaults when omitted, empty, or partial',
     async () => {
       const wasmGpuArtisan = liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN;
       const omittedMain =
           await resolveWith(wasmGpuArtisan, Backend.GPU_ARTISAN);
       const omittedBackendConfig =
           await resolveWith(wasmGpuArtisan, Backend.GPU_ARTISAN, {});
       const emptyBackendConfig = await resolveWith(
           wasmGpuArtisan, Backend.GPU_ARTISAN, {backendConfig: {}});
       const undefinedFieldConfig =
           await resolveWith(wasmGpuArtisan, Backend.GPU_ARTISAN, {
             backendConfig: {
               max_top_k: undefined,
               use_autosized_ringbuffers: undefined,
               supported_lora_ranks: undefined,
             },
           });

       expect(omittedMain.mainExecutorSettings.backendConfig).toEqual({
         num_output_candidates: 1,
         wait_for_weight_uploads: false,
         num_decode_steps_per_sync: 1,
         sequence_batch_size: 0,
         supported_lora_ranks: [],
         max_top_k: 64,
         enable_decode_logits: false,
         enable_external_embeddings: false,
         use_submodel: false,
         use_autosized_ringbuffers: false,
       });
       expect(omittedBackendConfig).toEqual(omittedMain);
       expect(emptyBackendConfig).toEqual(omittedMain);
       expect(undefinedFieldConfig).toEqual(omittedMain);

       const partialTopK =
           await resolveWith(wasmGpuArtisan, Backend.GPU_ARTISAN, {
             backendConfig: {max_top_k: 5},
           });
       expect(partialTopK.mainExecutorSettings.backendConfig).toEqual({
         ...omittedMain.mainExecutorSettings.backendConfig,
         max_top_k: 5,
       });

       const partialLoraAndRingbuffers =
           await resolveWith(wasmGpuArtisan, Backend.GPU_ARTISAN, {
             backendConfig: {
               supported_lora_ranks: [4, 8],
               use_autosized_ringbuffers: true,
             },
           });
       expect(partialLoraAndRingbuffers.mainExecutorSettings.backendConfig)
           .toEqual({
             ...omittedMain.mainExecutorSettings.backendConfig,
             supported_lora_ranks: [4, 8],
             use_autosized_ringbuffers: true,
           });
     });

  it('propagates VectorUint32 allocation errors without double-deleting base supported_lora_ranks',
     async () => {
       const wasmSettings = unwrapWasmResult(
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN));
       const allocError = new Error('WASM VectorUint32 out of memory');
       const throwingWasm: Wasm.LiteRtLmWasm = {
         ...liteRtLm.liteRtLmWasm,
         VectorUint32: class {
           constructor() {
             throw allocError;
           }
         } as unknown as Wasm.LiteRtLmWasm['VectorUint32'],
       };

       try {
         expect(
             () => fillWasmEngineSettingsFromEngineSettings(
                 wasmSettings, {
                   model: '/path/to/model',
                   backend: Backend.GPU_ARTISAN,
                   mainExecutorSettings: {
                     backendConfig: {supported_lora_ranks: [4, 8]},
                   },
                 },
                 Backend.GPU_ARTISAN, throwingWasm))
             .toThrow(allocError);
       } finally {
         wasmSettings.delete();
       }
     });

  it('round-trips maxNumImages', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN));
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
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU,
            liteRtLm.liteRtLmWasm.Backend.CPU, undefined));
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
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU, undefined,
            liteRtLm.liteRtLmWasm.Backend.CPU));
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
       const wasmSettings = unwrapWasmResult(
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU));
       const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
       expect(resolved.visionBackend).toBeUndefined();
       expect(resolved.audioBackend).toBeUndefined();
       wasmSettings.delete();
     });

  it('leaves activation data types unset by default', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU,
            liteRtLm.liteRtLmWasm.Backend.GPU,
            liteRtLm.liteRtLmWasm.Backend.GPU));
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          visionBackend: Backend.GPU,
          audioBackend: Backend.GPU,
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
    expect(resolved.activationDataType).toBeUndefined();
    expect(resolved.visionActivationDataType).toBeUndefined();
    expect(resolved.audioActivationDataType).toBeUndefined();
    wasmSettings.delete();
  });

  it('round-trips activationDataType on GPU and GPU_ARTISAN', async () => {
    for (const [wasmBackend, backend, activationDataType] of [
             [
               liteRtLm.liteRtLmWasm.Backend.GPU,
               Backend.GPU,
               ActivationDataType.FLOAT32,
             ],
             [
               liteRtLm.liteRtLmWasm.Backend.GPU_ARTISAN,
               Backend.GPU_ARTISAN,
               ActivationDataType.FLOAT16,
             ],
    ] as const) {
      const wasmSettings = unwrapWasmResult(
          await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
              modelAssets, wasmBackend));
      fillWasmEngineSettingsFromEngineSettings(
          wasmSettings, {
            model: '/path/to/model',
            backend,
            activationDataType,
          },
          backend, liteRtLm.liteRtLmWasm);
      expect(wasmEngineSettingsToEngineSettings(wasmSettings)
                 .activationDataType)
          .toBe(activationDataType);
      wasmSettings.delete();
    }
  });

  it('round-trips multimodal visionActivationDataType and audioActivationDataType',
     async () => {
       const wasmSettings = unwrapWasmResult(
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefaultMultimodal(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU,
               liteRtLm.liteRtLmWasm.Backend.GPU,
               liteRtLm.liteRtLmWasm.Backend.GPU));
       fillWasmEngineSettingsFromEngineSettings(
           wasmSettings, {
             model: '/path/to/model',
             backend: Backend.GPU,
             activationDataType: ActivationDataType.FLOAT32,
             visionBackend: Backend.GPU,
             visionActivationDataType: ActivationDataType.FLOAT16,
             audioBackend: Backend.GPU,
             audioActivationDataType: ActivationDataType.FLOAT32,
           },
           Backend.GPU, liteRtLm.liteRtLmWasm);
       const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
       expect(resolved.activationDataType).toBe(ActivationDataType.FLOAT32);
       expect(resolved.visionActivationDataType)
           .toBe(ActivationDataType.FLOAT16);
       expect(resolved.audioActivationDataType)
           .toBe(ActivationDataType.FLOAT32);
       wasmSettings.delete();
     });

  it('ignores visionActivationDataType and audioActivationDataType when disabled',
     async () => {
       const wasmSettings = unwrapWasmResult(
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
       fillWasmEngineSettingsFromEngineSettings(
           wasmSettings, {
             model: '/path/to/model',
             backend: Backend.GPU,
             visionActivationDataType: ActivationDataType.FLOAT16,
             audioActivationDataType: ActivationDataType.FLOAT32,
           },
           Backend.GPU, liteRtLm.liteRtLmWasm);
       const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
       expect(resolved.visionBackend).toBeUndefined();
       expect(resolved.visionActivationDataType).toBeUndefined();
       expect(resolved.audioBackend).toBeUndefined();
       expect(resolved.audioActivationDataType).toBeUndefined();
       wasmSettings.delete();
     });

  it('leaves the speculative decoding config unset by default', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {model: '/path/to/model', backend: Backend.GPU},
        Backend.GPU, liteRtLm.liteRtLmWasm);
    expect(wasmSettings.getSpeculativeDecodingConfig())
        .toEqual({mode: undefined, lenience: undefined});
    expect(wasmEngineSettingsToEngineSettings(wasmSettings)
               .speculativeDecodingConfig)
        .toEqual({});
    wasmSettings.delete();
  });

  it('resolves unset advanced settings to the C++ defaults', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
    expect(wasmSettings.getMutableMainExecutorSettings().getAdvancedSettings())
        .toBeUndefined();
    const defaultAdvanced =
        liteRtLm.liteRtLmWasm.createDefaultAdvancedSettings();
    expect(
        wasmEngineSettingsToEngineSettings(wasmSettings, liteRtLm.liteRtLmWasm)
            .mainExecutorSettings.advancedSettings)
        .toEqual(defaultAdvanced);

    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          mainExecutorSettings: {
            advancedSettings: {num_threads_to_upload: undefined},
          },
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    expect(
        wasmEngineSettingsToEngineSettings(wasmSettings, liteRtLm.liteRtLmWasm)
            .mainExecutorSettings.advancedSettings)
        .toEqual(defaultAdvanced);
    wasmSettings.delete();
  });

  it('round-trips the speculative decoding config', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
    fillWasmEngineSettingsFromEngineSettings(
        wasmSettings, {
          model: '/path/to/model',
          backend: Backend.GPU,
          speculativeDecodingConfig: {
            mode: SpeculativeDecodingMode.ADAPTIVE,
            lenience: 0.5,
          },
        },
        Backend.GPU, liteRtLm.liteRtLmWasm);
    expect(wasmSettings.getSpeculativeDecodingConfig().mode)
        .toBe(liteRtLm.liteRtLmWasm.SpeculativeDecodingMode.ADAPTIVE);
    expect(wasmEngineSettingsToEngineSettings(wasmSettings)
               .speculativeDecodingConfig)
        .toEqual({mode: SpeculativeDecodingMode.ADAPTIVE, lenience: 0.5});
    wasmSettings.delete();
  });

  it('keeps the speculative decoding config when advancedSettings are set',
     async () => {
       const wasmSettings = unwrapWasmResult(
           await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
               modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
       fillWasmEngineSettingsFromEngineSettings(
           wasmSettings, {
             model: '/path/to/model',
             backend: Backend.GPU,
             speculativeDecodingConfig: {mode: SpeculativeDecodingMode.ON},
             mainExecutorSettings: {
               advancedSettings: {num_threads_to_upload: 2},
             },
           },
           Backend.GPU, liteRtLm.liteRtLmWasm);
       const resolved = wasmEngineSettingsToEngineSettings(wasmSettings);
       expect(resolved.speculativeDecodingConfig)
           .toEqual({mode: SpeculativeDecodingMode.ON});
       expect(resolved.mainExecutorSettings.advancedSettings
                  .num_threads_to_upload)
           .toBe(2);
       wasmSettings.delete();
     });

  it('rejects an out-of-range lenience', async () => {
    const wasmSettings = unwrapWasmResult(
        await liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.GPU));
    expect(
        () => fillWasmEngineSettingsFromEngineSettings(
            wasmSettings, {
              model: '/path/to/model',
              backend: Backend.GPU,
              speculativeDecodingConfig: {lenience: 1.5},
            },
            Backend.GPU, liteRtLm.liteRtLmWasm))
        .toThrowError(/lenience must be in \[0, 1\]/);
    wasmSettings.delete();
  });
});
