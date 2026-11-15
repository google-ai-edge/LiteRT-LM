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

/** Options for `waitForWebGpuAdapter()`. */
export declare interface WaitForWebGpuAdapterOptions {
  /** How many times to request an adapter. Defaults to 10. */
  maxAttempts?: number;
  /**
   * How long to wait for each request to resolve before trying again, in
   * milliseconds. Defaults to 5000.
   */
  attemptTimeoutMs?: number;
  /** How long to wait between attempts, in milliseconds. Defaults to 2000. */
  delayMs?: number;
}

/**
 * Waits until the browser provides a hardware WebGPU adapter, and returns
 * whether it did. A software fallback adapter (SwiftShader) doesn't count.
 *
 * On Forge GPU machines, the browser's GPU process can take several seconds to
 * come up (or may restart). Until it does, `navigator.gpu.requestAdapter()`
 * either returns null or doesn't resolve. Call this in a `beforeAll` (with a
 * timeout longer than the total wait) before creating a WebGPU device or a GPU
 * engine.
 */
export async function waitForWebGpuAdapter(
    options: WaitForWebGpuAdapterOptions = {}): Promise<boolean> {
  const {maxAttempts = 10, attemptTimeoutMs = 5000, delayMs = 2000} = options;
  // `navigator.gpu` is undefined in browsers without WebGPU support.
  const gpu: GPU|undefined = navigator.gpu;
  if (!gpu) {
    console.log('WebGPU is not supported by this browser.');
    return false;
  }
  for (let attempt = 1; attempt <= maxAttempts; ++attempt) {
    const adapter = await requestAdapterWithTimeout(gpu, attemptTimeoutMs);
    if (adapter) {
      const description = describeAdapter(adapter);
      console.log(`Got a WebGPU adapter on attempt ${attempt}: ${
          JSON.stringify(description)}`);
      if (isFallbackAdapter(description)) {
        // Chrome falls back to SwiftShader for the rest of the session once its
        // GPU process fails to start (e.g. on H100 MIG slices, which don't
        // support the Vulkan extensions it needs), so don't retry.
        console.log(
            'This is a software (SwiftShader) fallback adapter, not a ' +
            'hardware GPU.');
        return false;
      }
      return true;
    }
    if (attempt < maxAttempts) {
      console.log(
          `WebGPU adapter not available yet (attempt ${attempt} of ` +
          `${maxAttempts}); retrying in ${delayMs}ms.`);
      await new Promise(resolve => void setTimeout(resolve, delayMs));
    }
  }
  console.log(
      `WebGPU adapter still unavailable after ${maxAttempts} attempts.`);
  return false;
}

/** The fields of `GPUAdapterInfo` that identify an adapter. */
declare interface AdapterDescription {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
}

/** Returns what's known about `adapter`, e.g. to tell if it's SwiftShader. */
function describeAdapter(adapter: GPUAdapter): AdapterDescription {
  // Not all versions of the WebGPU types have `info.isFallbackAdapter`.
  const info = (adapter as unknown as {info?: AdapterDescription}).info;
  return {
    vendor: info?.vendor,
    architecture: info?.architecture,
    device: info?.device,
    description: info?.description,
    isFallbackAdapter: info?.isFallbackAdapter,
  };
}

/** Returns whether `description` is of a software fallback adapter. */
function isFallbackAdapter(description: AdapterDescription): boolean {
  return !!description.isFallbackAdapter ||
      description.architecture === 'swiftshader';
}

/**
 * Requests an adapter like `createDefaultWebGpuDevice()` does, returning it if
 * it was provided within `timeoutMs`.
 */
async function requestAdapterWithTimeout(
    gpu: GPU, timeoutMs: number): Promise<GPUAdapter|null> {
  let timeoutId: ReturnType<typeof setTimeout>|undefined;
  const timeout = new Promise<null>(resolve => {
    timeoutId = setTimeout(() => {
      resolve(null);
    }, timeoutMs);
  });
  const request = gpu.requestAdapter({powerPreference: 'high-performance'})
                      .catch(() => null);
  try {
    return await Promise.race([request, timeout]);
  } finally {
    clearTimeout(timeoutId);
  }
}
