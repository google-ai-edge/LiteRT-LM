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
  maxInputLength?: number;
  /**
   * Upper bound on how many text encoder signatures to prepare. Defaults to 1.
   *
   * Models ship one signature per supported sequence length, and each is a
   * full private copy of the graph. Preparing all of them can exhaust the 4GB
   * wasm32 address space, so the web bindings only prepare one by default.
   *
   * The longest signature is always prepared, so this never lowers the input
   * length the engine accepts. Raising it lets shorter inputs run on a tighter
   * signature instead of being padded up to the longest one, trading memory
   * for speed. Pass `Infinity` to prepare every signature.
   */
  maxNumSignatures?: number;
  visionTokensPerImage?: number;
  mainExecutorSettings?: EmbeddingExecutorSettings;
}

/**
 * Number of text encoder signatures prepared when the caller does not say.
 */
const DEFAULT_MAX_NUM_SIGNATURES = 1;

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

  if (settings.maxInputLength !== undefined) {
    wasmSettings.setMaxInputLength(settings.maxInputLength);
  }
  const maxNumSignatures =
      settings.maxNumSignatures ?? DEFAULT_MAX_NUM_SIGNATURES;
  // `Infinity` means "no cap", which the C++ side spells as an unset value.
  if (Number.isFinite(maxNumSignatures)) {
    wasmSettings.setMaxNumSignatures(maxNumSignatures);
  }
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
