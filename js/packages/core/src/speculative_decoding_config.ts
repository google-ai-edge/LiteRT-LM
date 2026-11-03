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

import {SpeculativeDecodingConfig as WasmSpeculativeDecodingConfig, SpeculativeDecodingMode} from './wasm_binding_types.js';

// Re-export SpeculativeDecodingMode as it's compatible with the TS API.
export {SpeculativeDecodingMode};

/**
 * Speculative decoding (e.g. Multi-Token Prediction (MTP)) options.
 *
 * The engine holds the default configuration
 * (`EngineSettings.speculativeDecodingConfig`), and each session can override
 * it (`SessionConfig.speculativeDecodingConfig`). Every field is optional: a
 * field left unset in the session inherits the engine's value, and a field left
 * unset in the engine uses the runtime default.
 */
export interface SpeculativeDecodingConfig {
  /**
   * How speculative decoding is used. Runtime default:
   * `SpeculativeDecodingMode.OFF`.
   *
   * - `OFF`: Never speculate.
   * - `ON`: Speculate on every decode step.
   * - `ADAPTIVE`: Speculate only while it is measured to be faster than
   *   regular decoding. Speculative decoding speeds up predictable text (e.g.
   *   code), but can be slower than regular decoding on less predictable text
   *   (e.g. creative writing).
   *
   * Setting `ON` or `ADAPTIVE` on the engine loads the drafter when the engine
   * is created. Sessions can only speculate if the engine loaded the drafter.
   */
  mode?: SpeculativeDecodingMode;
  /**
   * Lenience factor c in [0, 1] of the speculative decoding acceptance test.
   *
   * - undefined or 0 (default): Exact. Emitted tokens follow the session's
   *   sampling distribution.
   * - c > 0: A drafted token is also accepted with probability c when the
   *   model disagrees with it. This trades output quality for speed; c = 1
   *   accepts every drafted token.
   */
  lenience?: number;
}

/**
 * Converts a SpeculativeDecodingConfig to the WASM representation.
 */
export function speculativeDecodingConfigToWasm(
    config: SpeculativeDecodingConfig): WasmSpeculativeDecodingConfig {
  if (config.lenience !== undefined &&
      !(config.lenience >= 0 && config.lenience <= 1)) {
    throw new Error(
        `lenience must be in [0, 1] or undefined, but got ${config.lenience}.`);
  }
  // Embind requires every field to be present, even if undefined.
  return {
    mode: config.mode === undefined ? undefined : {value: config.mode},
    lenience: config.lenience,
  };
}

/**
 * Converts a WASM SpeculativeDecodingConfig to a SpeculativeDecodingConfig.
 * Unset fields are omitted.
 */
export function wasmSpeculativeDecodingConfigToConfig(
    config: WasmSpeculativeDecodingConfig): SpeculativeDecodingConfig {
  const result: SpeculativeDecodingConfig = {};
  if (config.mode !== undefined) {
    result.mode = config.mode.value;
  }
  if (config.lenience !== undefined) {
    result.lenience = config.lenience;
  }
  return result;
}
