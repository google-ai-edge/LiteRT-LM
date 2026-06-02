/**
 * Copyright 2025 Google LLC
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

/**
 * Checks for relaxed SIMD support.
 *
 * (module
 *  (func (result v128)
 *    i32.const 1
 *    i8x16.splat
 *    i32.const 2
 *    i8x16.splat
 *    i8x16.relaxed_swizzle
 *  )
 * )
 */
const WASM_RELAXED_SIMD_CHECK = new Uint8Array([
  0, 97, 115, 109, 1,  0, 0,  0, 1,   5,  1,  96, 0,   1,  123, 3,   2, 1,
  0, 10, 15,  1,   13, 0, 65, 1, 253, 15, 65, 2,  253, 15, 253, 128, 2, 11
]);

/**
 * Checks for WebAssembly Pthreads emulation.
 *
 * (module
 *  (memory 1 1 shared)
 *  (func
 *    i32.const 0
 *    i32.atomic.load
 *    drop
 *  )
 * )
 */
const WASM_THREADS_CHECK = new Uint8Array([
  0, 97, 115, 109, 1, 0,  0,  0, 1, 4, 1,  96, 0,   0,  3, 2, 1,  0, 5,
  4, 1,  3,   1,   1, 10, 11, 1, 9, 0, 65, 0,  254, 16, 2, 0, 26, 11
]);

interface SupportStatus {
  supported: boolean;
  error?: Error;
}

// This must use 'declare' to prevent property renaming since string keys are
// used to access these properties.
declare interface WasmFeatureValues {
  relaxedSimd: Promise<SupportStatus>|undefined;
  threads: Promise<SupportStatus>|undefined;
}

const WASM_FEATURE_VALUES: WasmFeatureValues = {
  'relaxedSimd': undefined,
  'threads': undefined,
};

async function tryWasm(wasm: Uint8Array): Promise<SupportStatus> {
  try {
    await WebAssembly.instantiate(wasm);
    return {supported: true};
  } catch (e) {
    return {supported: false, error: e as Error};
  }
}

const WASM_FEATURE_CHECKS:
    Record<keyof typeof WASM_FEATURE_VALUES, () => Promise<SupportStatus>> = {
      'relaxedSimd': () => {
        if (WASM_FEATURE_VALUES.relaxedSimd === undefined) {
          WASM_FEATURE_VALUES.relaxedSimd = tryWasm(WASM_RELAXED_SIMD_CHECK);
        }
        return WASM_FEATURE_VALUES.relaxedSimd!;
      },
      'threads': () => {
        if (WASM_FEATURE_VALUES.threads === undefined) {
          try {
            if (typeof MessageChannel !== 'undefined') {
              new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
            }
            WASM_FEATURE_VALUES.threads = tryWasm(WASM_THREADS_CHECK);
          } catch (e) {
            WASM_FEATURE_VALUES.threads =
                Promise.resolve({supported: false, error: e as Error});
          }
        }
        return WASM_FEATURE_VALUES.threads!;
      },
    };

/**
 * Check if a given WASM feature is supported.
 *
 * @param feature The feature to check.
 * @return A promise that resolves to true if the feature is supported,
 *     false otherwise.
 */
export async function supportsFeature(
    feature: keyof typeof WASM_FEATURE_CHECKS): Promise<boolean> {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    // Then we don't know how to check for this feature.
    throw new Error(`Unknown feature: ${feature}`);
  }
  return (await check).supported;
}

/**
 * Throw an error if a given WASM feature is not supported.
 *
 * @param feature The feature to check.
 * @throws An error if the feature is not supported.
 */
export async function throwIfFeatureNotSupported(
    feature: keyof typeof WASM_FEATURE_CHECKS): Promise<void> {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    // Then we don't know how to check for this feature.
    throw new Error(`Unknown feature: ${feature}`);
  }
  const result = await check;
  if (!result.supported) {
    throw result.error!;
  }
}
