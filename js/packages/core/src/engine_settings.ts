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

import {getGlobalLiteRtLm} from './global_litertlm.js';
import {RecursiveRequired} from './types.js';
import {type AdvancedSettings, Backend, type CpuConfig, EngineSettings as WasmEngineSettings, GpuArtisanConfig as WasmGpuArtisanConfig, type GpuConfig} from './wasm_binding_types.js';
import {consumeEmscriptenVectorToArray, fillEmscriptenVector} from './wasm_utils.js';

// Re-export these types since they're already compatible with JS.
export {AdvancedSettings, CpuConfig, GpuConfig};

/**
 * LiteRT-LM GpuArtisanConfig
 */
export interface GpuArtisanConfig extends
    Omit<WasmGpuArtisanConfig, 'supported_lora_ranks'> {
  // We replace the EmscriptenVector with a plain array for better JS interop.
  supported_lora_ranks: number[];
}

/**
 * LiteRT-LM LlmExecutorSettings
 */
export interface LlmExecutorSettings {
  maxNumTokens?: number;
  backendConfig?: CpuConfig|GpuConfig|GpuArtisanConfig;
  samplerBackend?: Backend;
  advancedSettings?: Partial<AdvancedSettings>;
}

/**
 * LiteRT-LM EngineSettings
 */
export interface EngineSettings {
  model: string|ReadableStream<Uint8Array>;
  backend?: Backend;
  mainExecutorSettings?: LlmExecutorSettings;
}

/**
 * Fills a WasmEngineSettings with the values from a EngineSettings.
 */
export function fillWasmEngineSettingsFromEngineSettings(
    wasmEngineSettings: WasmEngineSettings, engineSettings: EngineSettings,
    backend: Backend, wasm = getGlobalLiteRtLm().liteRtLmWasm): void {
  // This function only fills an existing WasmEngineSettings object because
  // constructing the WasmEngineSettings object requires loading the model file,
  // and that is best handled in the Engine class.

  const wasmExecutorSettings =
      wasmEngineSettings.getMutableMainExecutorSettings();
  wasmExecutorSettings.setCacheDir(':nocache');  // Not supported in JS.

  if (engineSettings.mainExecutorSettings) {
    const mainExecutorSettings = engineSettings.mainExecutorSettings;

    if (mainExecutorSettings.maxNumTokens !== undefined) {
      wasmExecutorSettings.setMaxNumTokens(mainExecutorSettings.maxNumTokens);
    }
    if (mainExecutorSettings.samplerBackend !== undefined) {
      wasmExecutorSettings.setSamplerBackend(
          {value: mainExecutorSettings.samplerBackend});
    }
    if (mainExecutorSettings.backendConfig !== undefined) {
      const backendConfig = mainExecutorSettings.backendConfig;
      if (backend === Backend.CPU) {
        wasmExecutorSettings.setBackendConfigCpu(backendConfig as CpuConfig);
      } else if (backend === Backend.GPU) {
        wasmExecutorSettings.setBackendConfigGpu(backendConfig as GpuConfig);
      } else if (backend === Backend.GPU_ARTISAN) {
        const gpuArtisanConfig = backendConfig as GpuArtisanConfig;
        const loraRanksVec = new wasm.VectorUint32();
        fillEmscriptenVector(
            gpuArtisanConfig.supported_lora_ranks, loraRanksVec);
        const wasmGpuArtisanConfig: WasmGpuArtisanConfig = {
          ...gpuArtisanConfig,
          supported_lora_ranks: loraRanksVec,
        };
        wasmExecutorSettings.setBackendConfigGpuArtisan(wasmGpuArtisanConfig);
        loraRanksVec.delete();
      }
    }
    if (mainExecutorSettings.advancedSettings !== undefined) {
      const defaultAdv = wasmExecutorSettings.getAdvancedSettings();
      const newAdv = {...defaultAdv, ...mainExecutorSettings.advancedSettings};
      wasmExecutorSettings.setAdvancedSettings(newAdv);
    }
  }
}

/**
 * Converts a WasmEngineSettings to a EngineSettings.
 *
 * Fields are populated with the WASM object's values, which may be default
 * values if they were not set.
 *
 * The returned value omits the `model` field since its TS representation is a
 * URL or stream, which does not match what the WASM object holds.
 */
export function wasmEngineSettingsToEngineSettings(
    wasmEngineSettings: WasmEngineSettings):
    RecursiveRequired<Omit<EngineSettings, 'model'>> {
  const wasmExecutorSettings =
      wasmEngineSettings.getMutableMainExecutorSettings();

  let backendConfig: CpuConfig|GpuConfig|GpuArtisanConfig;
  let backend: Backend;
  try {
    // This isn't ideal, but there's no way to get what backend config is set.
    backendConfig = wasmExecutorSettings.getBackendConfigCpu();
    backend = Backend.CPU;
  } catch (e) {
    try {
      backendConfig = wasmExecutorSettings.getBackendConfigGpu();
      backend = Backend.GPU;
    } catch (e) {
      try {
        const wasmGpuArtisanConfig =
            wasmExecutorSettings.getBackendConfigGpuArtisan();
        backendConfig = {
          ...wasmGpuArtisanConfig,
          supported_lora_ranks: consumeEmscriptenVectorToArray(
              wasmGpuArtisanConfig.supported_lora_ranks),
        };
        backend = Backend.GPU_ARTISAN;
      } catch (e) {
        throw new Error('Unsupported backend config');
      }
    }
  }

  return {
    backend,
    mainExecutorSettings: {
      maxNumTokens: wasmExecutorSettings.getMaxNumTokens(),
      samplerBackend: wasmExecutorSettings.getSamplerBackend().value,
      backendConfig,
      advancedSettings: wasmExecutorSettings.getAdvancedSettings(),
    },
  };
}
