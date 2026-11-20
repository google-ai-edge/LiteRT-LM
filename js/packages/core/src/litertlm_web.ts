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

import {WasmModule} from '@litertjs/wasm-utils';
import {createDefaultWebGpuDevice} from '@litertjs/webgpu-utils';
// Placeholder for internal dependency on trusted resource url

import {getGlobalLiteRtLm} from './global_litertlm.js';
import {LiteRtLmWasm} from './wasm_binding_types.js';

/** Options for `setupDefaultWebGpuDevice`. */
export declare interface WebGpuDeviceOptions {
  /**
   * Whether to briefly create a WebGL context to identify the GPU, when the
   * browser's WebGPU adapter info is too generic to pick the right kernels
   * (for example on Apple Silicon Macs in Chrome). Defaults to true.
   *
   * The probe runs at most once per page and releases its context right away,
   * but it takes one of the page's limited WebGL context slots while it runs.
   */
  identifyGpuWithWebGl?: boolean;
}

/**
 * Set up the default WebGPU device for the LiteRT LM Wasm module.
 *
 * Engines call this automatically. Call it yourself before creating an engine
 * to pass `options`; later calls do nothing once the device exists.
 */
export async function setupDefaultWebGpuDevice(options?: WebGpuDeviceOptions) {
  await getGlobalLiteRtLm().setupDefaultWebGpuDevice(options);
}

/**
 * Run LiteRt LM models in the browser.
 */
export class LiteRtLm {
  readonly liteRtLmWasm: LiteRtLmWasm;
  static DEFAULT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.12.1/wasm';

  constructor(wasmModule: WasmModule) {
    this.liteRtLmWasm = wasmModule as LiteRtLmWasm;
    this.liteRtLmWasm.setupLogging();
  }

  async setupDefaultWebGpuDevice(options: WebGpuDeviceOptions = {}) {
    if (this.liteRtLmWasm.preinitializedWebGPUDevice) {
      return;
    }
    this.liteRtLmWasm.preinitializedWebGPUDevice =
        await createDefaultWebGpuDevice(options);
  }

  delete() {
    // no op for now.
  }
}
