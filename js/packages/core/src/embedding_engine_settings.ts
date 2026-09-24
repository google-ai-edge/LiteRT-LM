/**
 * Copyright 2026 Google LLC
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

import {Backend, EmbeddingEngineSettings as WasmEmbeddingEngineSettings} from './wasm_binding_types.js';

/**
 * LiteRT-LM EmbeddingExecutorSettings
 */
export interface EmbeddingExecutorSettings {
  numThreads?: number;
}

/**
 * LiteRT-LM EmbeddingEngineSettings
 */
export interface EmbeddingEngineSettings {
  model: string | Blob | ReadableStream<Uint8Array>;
  backend?: Backend;
  visionBackend?: Backend;
  audioBackend?: Backend;
  /**
   * Longest input, in tokens, that the engine must accept. Defaults to 1024.
   *
   * The engine prepares every text encoder signature between `minInputLength`
   * and this value, and rejects the model if no signature is long enough.
   */
  maxInputLength?: number;
  /**
   * Shortest input, in tokens, to prepare a signature for. Defaults to
   * `maxInputLength`, which prepares exactly one signature.
   *
   * Lowering this prepares additional shorter signatures, so that short inputs
   * run on a tighter graph instead of being padded up to `maxInputLength`.
   */
  minInputLength?: number;
  visionTokensPerImage?: number;
  mainExecutorSettings?: EmbeddingExecutorSettings;
}

/**
 * Longest input the engine accepts when the caller does not say.
 *
 * Together with defaulting `minInputLength` to `maxInputLength` this prepares
 * a single signature, which keeps a default-configured engine inside the 4GB
 * wasm32 address space.
 */
const DEFAULT_MAX_INPUT_LENGTH = 1024;

/**
 * Fills a WasmEmbeddingEngineSettings with the values from an EmbeddingEngineSettings.
 */
export function fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings(
    wasmSettings: WasmEmbeddingEngineSettings,
    settings: EmbeddingEngineSettings,
    backend: Backend,
): void {
  const wasmExecutorSettings = wasmSettings.getMutableMainExecutorSettings();
  wasmExecutorSettings.setCacheDir(':nocache');

  const maxInputLength = settings.maxInputLength ?? DEFAULT_MAX_INPUT_LENGTH;
  wasmSettings.setMaxInputLength(maxInputLength);
  wasmSettings.setMinInputLength(settings.minInputLength ?? maxInputLength);
  if (settings.visionTokensPerImage !== undefined) {
    wasmSettings.setVisionTokensPerImage(settings.visionTokensPerImage);
  }

  if (settings.mainExecutorSettings) {
    const mainExecutorSettings = settings.mainExecutorSettings;
    if (mainExecutorSettings.numThreads !== undefined) {
      wasmExecutorSettings.setNumThreads(mainExecutorSettings.numThreads);
    }
  }
}
