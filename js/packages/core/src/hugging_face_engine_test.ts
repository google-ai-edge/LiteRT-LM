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

// Placeholder for internal dependency on trusted resource url

import {Engine} from './engine.js';
import {loadLiteRtLm, unloadLiteRtLm} from './load_litertlm.js';
import {Backend} from './wasm_binding_types.js';

const CPU_MODEL_PATH = '/models/tiny_gemma.litertlm';
const GPU_ARTISAN_MODEL_PATH = '/models/gemma3-1b-hw.litertlm';
const COMMIT = 'c0ffee'.padEnd(40, '0');
const DOWNLOAD_URL_PREFIX =
    `https://huggingface.co/org/repo/resolve/${COMMIT}/`;

/** Returns whether `path` is served and starts with the `.litertlm` magic. */
async function servesLiteRtLmFile(path: string): Promise<boolean> {
  try {
    const response = await fetch(path);
    if (!response.ok || !response.body) {
      return false;
    }
    // Only read the first chunk, since the models are large.
    const reader = response.body.getReader();
    const {value} = await reader.read();
    await reader.cancel();
    return new TextDecoder().decode(value?.subarray(0, 8)) === 'LITERTLM';
  } catch {
    return false;
  }
}

/** Returns whether the browser provides a WebGPU adapter. */
async function hasWebGpuAdapter(): Promise<boolean> {
  const gpu =
      (navigator as Navigator & {gpu?: {requestAdapter(): Promise<unknown>}})
          .gpu;
  try {
    return !!(await gpu?.requestAdapter());
  } catch {
    return false;
  }
}

// Loads real engines from a fake Hugging Face Hub, whose files are served from
// local models.
describe('Engine with a hf:// model', () => {
  /** The local model that each file in the fake repo is served from. */
  let repoFiles: Map<string, string>;
  let downloads: string[];
  let downloadCancelled: boolean;
  // The local models and a WebGPU adapter are available when the tests run
  // internally on a GPU, but not everywhere: e.g. the open-source unit test
  // runner doesn't serve `/models`, and its browser may have no GPU.
  let modelsAvailable = false;
  let webGpuAvailable = false;

  beforeAll(async () => {
    unloadLiteRtLm();
    await loadLiteRtLm('/wasm');
    modelsAvailable = await servesLiteRtLmFile(CPU_MODEL_PATH) &&
        await servesLiteRtLmFile(GPU_ARTISAN_MODEL_PATH);
    webGpuAvailable = await hasWebGpuAdapter();
  });

  /**
   * Marks the current spec pending unless the local models, and WebGPU if
   * `needsWebGpu`, are available.
   */
  function skipUnlessAvailable(needsWebGpu: boolean) {
    if (!modelsAvailable) {
      pending('The local models are not served at /models.');
    }
    if (needsWebGpu && !webGpuAvailable) {
      pending('No WebGPU adapter is available.');
    }
  }

  beforeEach(() => {
    downloads = [];
    downloadCancelled = false;
    const realFetch = globalThis.fetch.bind(globalThis);
    spyOn(globalThis, 'fetch')
        .and.callFake(async (input: RequestInfo|URL, init?: RequestInit) => {
          const url = String(input);
          if (url === 'https://huggingface.co/api/models/org/repo/revision/' +
                  'main?blobs=true') {
            return new Response(JSON.stringify({
              sha: COMMIT,
              siblings: [...repoFiles.keys()].map(
                  rfilename => ({rfilename, size: 1})),
            }));
          }
          if (url.startsWith(DOWNLOAD_URL_PREFIX)) {
            const path = url.slice(DOWNLOAD_URL_PREFIX.length);
            downloads.push(path);
            const response = await realFetch(repoFiles.get(path)!);
            return new Response(trackCancel(response.body!));
          }
          return realFetch(input, init);
        });
  });

  /** Returns a copy of `stream` that records whether it was cancelled. */
  function trackCancel(stream: ReadableStream<Uint8Array>):
      ReadableStream<Uint8Array> {
    const reader = stream.getReader();
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        const {done, value} = await reader.read();
        if (done) {
          controller.close();
        } else {
          controller.enqueue(value);
        }
      },
      async cancel(reason) {
        downloadCancelled = true;
        await reader.cancel(reason);
      },
    });
  }

  it('loads the file for the backend', async () => {
    skipUnlessAvailable(/* needsWebGpu= */ false);
    repoFiles = new Map([
      ['model.litertlm', CPU_MODEL_PATH],
      ['model-gpu.litertlm', GPU_ARTISAN_MODEL_PATH],
    ]);

    const engine = await Engine.create({
      model: 'hf://org/repo',
      backend: Backend.CPU,
      mainExecutorSettings: {maxNumTokens: 128},
    });
    await engine.delete();

    expect(downloads).toEqual(['model.litertlm']);
    expect(engine.settings.model).toBe('hf://org/repo');
    expect(engine.settings.backend).toBe(Backend.CPU);
  });

  it('prefers the GPU_ARTISAN model if no backend is given', async () => {
    skipUnlessAvailable(/* needsWebGpu= */ true);
    repoFiles = new Map([
      ['model.litertlm', CPU_MODEL_PATH],
      ['model-gpu.litertlm', GPU_ARTISAN_MODEL_PATH],
    ]);

    const engine = await Engine.create({
      model: 'hf://org/repo',
      mainExecutorSettings: {maxNumTokens: 128},
    });
    await engine.delete();

    expect(downloads).toEqual(['model-gpu.litertlm']);
    // The engine switches to GPU_ARTISAN when it reads the model's header.
    expect(engine.settings.backend).toBe(Backend.GPU_ARTISAN);
  });

  it('rejects a model for a different backend than the one requested',
     async () => {
       // Requesting Backend.GPU sets up WebGPU before the model's header is
       // read.
       skipUnlessAvailable(/* needsWebGpu= */ true);
       repoFiles = new Map([['model.litertlm', GPU_ARTISAN_MODEL_PATH]]);

       await expectAsync(Engine.create({
         model: 'hf://org/repo',
         backend: Backend.GPU,
         mainExecutorSettings: {maxNumTokens: 128},
       }))
           .toBeRejectedWithError(
               'hf://org/repo@main/model.litertlm is a Backend.GPU_ARTISAN ' +
               'model, but Backend.GPU was requested. Choose another file, ' +
               'or leave `backend` unset to use the model\'s backend.');
       // The download is stopped once the model's header has been read.
       expect(downloadCancelled).toBeTrue();
     });

  it('fails before downloading if no file matches', async () => {
    repoFiles = new Map([['model-gpu.litertlm', GPU_ARTISAN_MODEL_PATH]]);

    await expectAsync(Engine.create({
      model: 'hf://org/repo',
      backend: Backend.CPU,
    })).toBeRejectedWithError(/has no suitable model file for Backend.CPU/);
    expect(downloads).toEqual([]);
  });
});
