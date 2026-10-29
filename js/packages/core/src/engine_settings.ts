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
import {ModelInput} from './model_input.js';
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
  /**
   * Maximum number of images in a conversation (until the context is reset).
   * Required for vision input with `Backend.GPU_ARTISAN` models, which reserve
   * space for this many images' embeddings up front.
   */
  maxNumImages?: number;
  backendConfig?: Partial<CpuConfig>|Partial<GpuConfig>|
      Partial<GpuArtisanConfig>;
  samplerBackend?: Backend;
  advancedSettings?: Partial<AdvancedSettings>;
}

/**
 * LiteRT-LM EngineSettings
 */
export interface EngineSettings {
  /** The model: a URL, a `hf://` URI, a `ModelSource`, or the model's data. */
  model: ModelInput;
  /**
   * Backend for the model. Defaults to `Backend.GPU`. For a model loaded from
   * a `hf://` URI, it also chooses the model file, and the model must use it
   * if it's set (see `ModelInput`).
   */
  backend?: Backend;
  /**
   * Backend for the vision encoder. Can be `Backend.CPU` or `Backend.GPU`.
   * If unset, vision input is disabled.
   */
  visionBackend?: Backend;
  /**
   * Maximum number of vision tokens per image. Only used when vision is
   * enabled. If unset, the model's default is used.
   */
  maxVisionTokensPerImage?: number;
  /**
   * Smallest vision signature, in tokens per image, to prepare. Defaults to
   * `maxVisionTokensPerImage`, which prepares exactly one vision signature.
   *
   * Lowering this prepares additional smaller signatures so that small images
   * run on a tighter graph instead of being padded, but each extra signature
   * costs a lot of memory and can exceed the 4GB wasm32 limit.
   */
  minVisionTokensPerImage?: number;
  /**
   * Backend for the audio encoder. Can be `Backend.CPU` or `Backend.GPU`.
   * If unset, audio input is disabled.
   */
  audioBackend?: Backend;
  mainExecutorSettings?: LlmExecutorSettings;
  benchmarkEnabled?: boolean;
}

type OptionalEngineSettingsKeys = 'visionBackend'|'maxVisionTokensPerImage'|
    'minVisionTokensPerImage'|'audioBackend';

/**
 * EngineSettings with all defaults resolved. Fields that have no default
 * (e.g. vision settings when vision is disabled) remain optional.
 */
export type ResolvedEngineSettings =
    RecursiveRequired<Omit<EngineSettings, OptionalEngineSettingsKeys|'model'>>&
    Pick<EngineSettings, OptionalEngineSettingsKeys|'model'>;

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

  if (engineSettings.benchmarkEnabled) {
    wasmEngineSettings.enableBenchmark();
  }

  if (engineSettings.maxVisionTokensPerImage !== undefined) {
    wasmEngineSettings.setMaxVisionTokensPerImage(
        engineSettings.maxVisionTokensPerImage);
  }
  if (engineSettings.minVisionTokensPerImage !== undefined) {
    wasmEngineSettings.setMinVisionTokensPerImage(
        engineSettings.minVisionTokensPerImage);
  }

  if (engineSettings.mainExecutorSettings) {
    const mainExecutorSettings = engineSettings.mainExecutorSettings;

    if (mainExecutorSettings.maxNumTokens !== undefined) {
      wasmExecutorSettings.setMaxNumTokens(mainExecutorSettings.maxNumTokens);
    }
    if (mainExecutorSettings.maxNumImages !== undefined) {
      wasmExecutorSettings.setMaxNumImages(mainExecutorSettings.maxNumImages);
    }
    if (mainExecutorSettings.samplerBackend !== undefined) {
      wasmExecutorSettings.setSamplerBackend(
          {value: mainExecutorSettings.samplerBackend});
    }
    if (mainExecutorSettings.backendConfig !== undefined) {
      const backendConfig = mainExecutorSettings.backendConfig;
      if (backend === Backend.CPU) {
        const partialCpuConfig = backendConfig as Partial<CpuConfig>;
        const cpuConfig: CpuConfig = {
          kv_increment_size: partialCpuConfig.kv_increment_size ?? 0,
          prefill_chunk_size: partialCpuConfig.prefill_chunk_size ?? 0,
          number_of_threads: partialCpuConfig.number_of_threads ?? 0,
        };
        wasmExecutorSettings.setBackendConfigCpu(cpuConfig);
      } else if (backend === Backend.GPU) {
        const partialGpuConfig = backendConfig as Partial<GpuConfig>;
        const gpuConfig: GpuConfig = {
          max_top_k: partialGpuConfig.max_top_k ?? 1,
          external_tensor_mode: partialGpuConfig.external_tensor_mode ?? false,
        };
        wasmExecutorSettings.setBackendConfigGpu(gpuConfig);
      } else if (backend === Backend.GPU_ARTISAN) {
        const partialGpuArtisanConfig =
            backendConfig as Partial<GpuArtisanConfig>;
        const loraRanksVec = new wasm.VectorUint32();
        fillEmscriptenVector(
            partialGpuArtisanConfig.supported_lora_ranks ?? [], loraRanksVec);
        const wasmGpuArtisanConfig: WasmGpuArtisanConfig = {
          num_output_candidates:
              partialGpuArtisanConfig.num_output_candidates ?? 1,
          wait_for_weight_uploads:
              partialGpuArtisanConfig.wait_for_weight_uploads ?? true,
          num_decode_steps_per_sync:
              partialGpuArtisanConfig.num_decode_steps_per_sync ?? 1,
          sequence_batch_size:
              partialGpuArtisanConfig.sequence_batch_size ?? 0,
          supported_lora_ranks: loraRanksVec,
          max_top_k: partialGpuArtisanConfig.max_top_k ?? 1,
          enable_decode_logits:
              partialGpuArtisanConfig.enable_decode_logits ?? false,
          enable_external_embeddings:
              partialGpuArtisanConfig.enable_external_embeddings ?? false,
          use_submodel: partialGpuArtisanConfig.use_submodel ?? true,
          use_autosized_ringbuffers:
              partialGpuArtisanConfig.use_autosized_ringbuffers ?? true,
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
    Omit<ResolvedEngineSettings, 'model'> {
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
    visionBackend: wasmEngineSettings.getVisionBackend()?.value,
    maxVisionTokensPerImage: wasmEngineSettings.getMaxVisionTokensPerImage(),
    minVisionTokensPerImage: wasmEngineSettings.getMinVisionTokensPerImage(),
    audioBackend: wasmEngineSettings.getAudioBackend()?.value,
    mainExecutorSettings: {
      maxNumTokens: wasmExecutorSettings.getMaxNumTokens(),
      maxNumImages: wasmExecutorSettings.getMaxNumImages(),
      samplerBackend: wasmExecutorSettings.getSamplerBackend().value,
      backendConfig,
      advancedSettings: wasmExecutorSettings.getAdvancedSettings(),
    },
    benchmarkEnabled: false,
  };
}
