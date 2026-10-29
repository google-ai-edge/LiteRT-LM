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

import {setCacheEnvironmentForTesting} from './model-cache/environment.js';
import {HuggingFaceFile, ResolvedHuggingFaceModel} from './model-cache/index.js';
import {checkModelBackend, resolveModelInput, selectModelFile} from './model_input.js';
import {Backend} from './wasm_binding_types.js';

function files(...paths: string[]): HuggingFaceFile[] {
  return paths.map(path => ({path, size: 1}));
}

describe('selectModelFile', () => {
  function select(
      paths: string[], backend: Backend|undefined,
      repo = 'org/repo'): string|undefined {
    return selectModelFile(files(...paths), repo, backend)?.file.path;
  }

  it('uses the only model file for any backend but GPU_ARTISAN', () => {
    const paths = ['README.md', 'model.task', 'model_int8.litertlm'];
    expect(select(paths, undefined)).toBe('model_int8.litertlm');
    expect(select(paths, Backend.GPU)).toBe('model_int8.litertlm');
    expect(select(paths, Backend.CPU)).toBe('model_int8.litertlm');
    expect(select(paths, Backend.GPU_ARTISAN)).toBeUndefined();
  });

  it('chooses between the GPU and GPU_ARTISAN files', () => {
    const paths = [
      'name.litertlm',
      'name-gpu.litertlm',
      'name-web.litertlm',
      'name_Google_Tensor_G5.litertlm',
      'name_intel_LNL.litertlm',
      'name_NPU_Uncompiled.litertlm',
      'name_Qualcomm_QCS8275.litertlm',
      'name.mediatek.mt6993.litertlm',
    ];
    expect(select(paths, undefined)).toBe('name-gpu.litertlm');
    expect(select(paths, Backend.GPU)).toBe('name.litertlm');
    expect(select(paths, Backend.GPU_ARTISAN)).toBe('name-gpu.litertlm');
  });

  it('never chooses hardware-specific or old -web files', () => {
    const paths = [
      'name-web.litertlm',
      'name-Intel_PTL_NPU.litertlm',
      'name_Google_Tensor_G5.litertlm',
    ];
    expect(select(paths, undefined)).toBeUndefined();
    expect(select(paths, Backend.GPU)).toBeUndefined();
    expect(select(paths, Backend.GPU_ARTISAN)).toBeUndefined();
  });

  it('only uses a GPU_ARTISAN file without a backend or with GPU_ARTISAN',
     () => {
       const paths = ['name-gpu.litertlm'];
       expect(select(paths, undefined)).toBe('name-gpu.litertlm');
       expect(select(paths, Backend.GPU)).toBeUndefined();
       expect(select(paths, Backend.CPU)).toBeUndefined();
     });

  it('matches suffixes case-insensitively', () => {
    const paths = ['Name.LiteRTLM', 'Name-GPU.litertlm'];
    expect(select(paths, Backend.GPU)).toBe('Name.LiteRTLM');
    expect(select(paths, Backend.GPU_ARTISAN)).toBe('Name-GPU.litertlm');
  });

  it('prefers the file that matches the GPU_ARTISAN file', () => {
    const paths = [
      'model.litertlm',
      'name_int4.litertlm',
      'name.litertlm',
      'name-gpu.litertlm',
    ];
    expect(select(paths, Backend.GPU)).toBe('name.litertlm');
  });

  it('prefers the file named after the repo', () => {
    expect(select(
               ['model.litertlm', 'name_q4.litertlm', 'name.litertlm'],
               Backend.GPU, 'org/name'))
        .toBe('name.litertlm');
    expect(select(
               ['name_q4.litertlm', 'Name.litertlm'], Backend.GPU,
               'org/name-litert-lm'))
        .toBe('Name.litertlm');
  });

  it('prefers model.litertlm', () => {
    const paths =
        ['model_block32.litertlm', 'name_int4.litertlm', 'model.litertlm'];
    expect(select(paths, Backend.GPU)).toBe('model.litertlm');
  });

  it('prefers int4 models', () => {
    expect(select(['name_int8.litertlm', 'name_int4.litertlm'], undefined))
        .toBe('name_int4.litertlm');
    expect(select(
               ['name_int8.litertlm', 'name_mixed_int4.litertlm'], Backend.GPU))
        .toBe('name_mixed_int4.litertlm');
  });

  it('prefers shorter names, then sorts by name', () => {
    expect(select(
               [
                 'name-vision_int4.litertlm',
                 'name_int4.litertlm',
                 'name-vision_int4_gpu.litertlm',
               ],
               Backend.GPU))
        .toBe('name_int4.litertlm');
    expect(select(['b-gpu.litertlm', 'a-gpu.litertlm'], Backend.GPU_ARTISAN))
        .toBe('a-gpu.litertlm');
  });

  it('returns the other matching files in order of preference', () => {
    const choice = selectModelFile(
        files(
            'name_int8.litertlm', 'name-gpu.litertlm', 'name_int4.litertlm',
            'model.litertlm'),
        'org/repo', Backend.GPU);

    expect(choice?.file.path).toBe('model.litertlm');
    expect(choice?.others.map(file => file.path)).toEqual([
      'name_int4.litertlm',
      'name_int8.litertlm',
    ]);
  });

  it('returns undefined if there are no model files', () => {
    expect(select(['README.md', 'model.task'], undefined)).toBeUndefined();
  });
});

describe('resolveModelInput', () => {
  const COMMIT = 'c0ffee'.padEnd(40, '0');
  const WEB_COMMIT = 'beef'.padEnd(40, '0');
  const SHA256 = 'ab'.repeat(32);
  const GPU_SHA256 = 'cd'.repeat(32);
  let routes: Map<string, () => Response>;
  let fetchSpy:
      jasmine.Spy<(url: string, init?: RequestInit) => Promise<Response>>;

  function json(body: unknown, init?: ResponseInit): Response {
    return new Response(JSON.stringify(body), init);
  }

  function revisionUrl(repo: string, revision: string): string {
    return `https://huggingface.co/api/models/${repo}/revision/${
        revision}?blobs=true`;
  }

  function lfsFile(path: string, size: number, sha256: string) {
    return {rfilename: path, size, lfs: {sha256, size}};
  }

  function revision(commit: string) {
    return () => json({
      sha: commit,
      siblings: [
        {rfilename: 'README.md', size: 10},
        lfsFile('model.litertlm', 1000, SHA256),
        lfsFile('model-gpu.litertlm', 2000, GPU_SHA256),
      ],
    });
  }

  beforeEach(() => {
    routes = new Map([
      [revisionUrl('org/repo', 'main'), revision(COMMIT)],
      [revisionUrl('litert-community/repo', 'main'), revision(COMMIT)],
    ]);
    fetchSpy = jasmine.createSpy('fetch').and.callFake(async (url: string) => {
      const route = routes.get(url);
      if (route) return route();
      if (url.includes('/revision/')) {
        return json({error: 'Invalid rev id'}, {
          status: 404,
          headers: {'X-Error-Code': 'RevisionNotFound'},
        });
      }
      return json({error: 'Not found'}, {status: 404});
    });
    setCacheEnvironmentForTesting({fetch: fetchSpy});
  });

  afterEach(() => {
    setCacheEnvironmentForTesting();
  });

  function requestedUrls(): string[] {
    return fetchSpy.calls.allArgs().map(([url]) => url);
  }

  it('returns other inputs unchanged', async () => {
    const blob = new Blob([]);
    const stream = new ReadableStream<Uint8Array>();
    const source = {url: 'https://example.com/model.litertlm', sha256: SHA256};
    for (const model of ['model.litertlm', blob, stream, source]) {
      const resolved = await resolveModelInput(model, undefined);
      expect(resolved.model).toBe(model);
      expect(resolved.huggingFace).toBeUndefined();
      expect(() => {
        resolved.checkBackend(Backend.GPU_ARTISAN);
      }).not.toThrow();
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('resolves a hf:// URI to a download at the current commit', async () => {
    const {model, huggingFace} =
        await resolveModelInput('hf://org/repo', Backend.GPU);

    expect(model).toEqual({
      url: `https://huggingface.co/org/repo/resolve/${COMMIT}/model.litertlm`,
      size: 1000,
      sha256: SHA256,
    });
    expect(huggingFace).toEqual(jasmine.objectContaining({
      repo: 'org/repo',
      revision: 'main',
      commit: COMMIT,
      path: 'model.litertlm',
    }));
    // Other orgs' models are never looked up on a web branch.
    expect(requestedUrls()).toEqual([revisionUrl('org/repo', 'main')]);
  });

  it('chooses the file by backend', async () => {
    for (const backend of [Backend.GPU_ARTISAN, undefined]) {
      const {model} = await resolveModelInput('hf://org/repo', backend);

      expect(model).toEqual(jasmine.objectContaining({
        url: `https://huggingface.co/org/repo/resolve/${
            COMMIT}/model-gpu.litertlm`,
        sha256: GPU_SHA256,
      }));
    }
  });

  it('uses the path in the URI', async () => {
    const {huggingFace} = await resolveModelInput(
        'hf://org/repo/model-gpu.litertlm', Backend.GPU);

    expect(huggingFace?.path).toBe('model-gpu.litertlm');
  });

  it('prefers the web branch for litert-community models', async () => {
    routes.set(
        revisionUrl('litert-community/repo', 'web'), revision(WEB_COMMIT));

    const {huggingFace} =
        await resolveModelInput('hf://litert-community/repo', undefined);

    expect(huggingFace?.revision).toBe('web');
    expect(huggingFace?.commit).toBe(WEB_COMMIT);
  });

  it('uses main for litert-community models without a web branch',
     async () => {
       const {huggingFace} =
           await resolveModelInput('hf://litert-community/repo', undefined);

       expect(huggingFace?.revision).toBe('main');
       expect(requestedUrls()).toEqual(jasmine.arrayWithExactContents([
         revisionUrl('litert-community/repo', 'web'),
         revisionUrl('litert-community/repo', 'main'),
       ]));
     });

  it('uses the revision in the URI', async () => {
    routes.set(
        revisionUrl('litert-community/repo', 'web'), revision(WEB_COMMIT));

    const {huggingFace} =
        await resolveModelInput('hf://litert-community/repo@main', undefined);

    expect(huggingFace?.revision).toBe('main');
    expect(requestedUrls()).toEqual(
        [revisionUrl('litert-community/repo', 'main')]);
  });

  it('warns about the other files it could have chosen', async () => {
    routes.set(revisionUrl('org/repo', 'v1'), () => json({
      sha: COMMIT,
      siblings: [
        lfsFile('a_int8.litertlm', 2, GPU_SHA256),
        lfsFile('a_int4.litertlm', 1, SHA256),
        lfsFile('a_block32.litertlm', 3, SHA256),
      ],
    }));
    const warn = spyOn(console, 'warn');

    const {huggingFace} =
        await resolveModelInput('hf://org/repo@v1', Backend.GPU);

    expect(huggingFace?.path).toBe('a_int4.litertlm');
    expect(warn).toHaveBeenCalledOnceWith(
        'Loading hf://org/repo@v1/a_int4.litertlm. Other model files in the ' +
        'repo, which can be loaded by URI: hf://org/repo@v1/a_int8.litertlm, ' +
        'hf://org/repo@v1/a_block32.litertlm');
  });

  it('names the revision that was used in the warning', async () => {
    routes.set(revisionUrl('litert-community/repo', 'web'), () => json({
      sha: WEB_COMMIT,
      siblings: [
        lfsFile('a_int4.litertlm', 1, SHA256),
        lfsFile('a_int8.litertlm', 2, GPU_SHA256),
      ],
    }));
    const warn = spyOn(console, 'warn');

    await resolveModelInput('hf://litert-community/repo', Backend.GPU);

    expect(warn).toHaveBeenCalledOnceWith(
        'Loading hf://litert-community/repo@web/a_int4.litertlm. Other model ' +
        'files in the repo, which can be loaded by URI: ' +
        'hf://litert-community/repo@web/a_int8.litertlm');
  });

  it('checks the backend of a hf:// model', async () => {
    const {checkBackend} =
        await resolveModelInput('hf://org/repo', Backend.GPU);

    expect(() => {
      checkBackend(Backend.GPU);
    }).not.toThrow();
    expect(() => {
      checkBackend(Backend.GPU_ARTISAN);
    }).toThrowError(/is a Backend.GPU_ARTISAN model, but Backend.GPU was/);
  });

  it('throws before downloading if no file matches', async () => {
    routes.set(revisionUrl('org/repo', 'main'), () => json({
      sha: COMMIT,
      siblings: [
        {rfilename: 'a-gpu.litertlm', size: 1},
        {rfilename: 'a_intel_LNL.litertlm', size: 1},
      ],
    }));

    await expectAsync(resolveModelInput('hf://org/repo', Backend.GPU))
        .toBeRejectedWithError(
            'hf://org/repo@main has no suitable model file for Backend.GPU. ' +
            'Add one of a-gpu.litertlm, a_intel_LNL.litertlm to the URI, ' +
            'e.g. hf://org/repo@main/a-gpu.litertlm');
    expect(requestedUrls()).toEqual([revisionUrl('org/repo', 'main')]);
  });

  it('rejects invalid hf:// URIs', async () => {
    await expectAsync(resolveModelInput('hf://datasets/org/data', undefined))
        .toBeRejectedWithError(/Invalid Hugging Face model URI/);
  });

  it('resolves a ModelSource with a hf:// URI', async () => {
    const {model} = await resolveModelInput(
        {
          url: 'hf://org/repo/model.litertlm',
          sha256: SHA256.toUpperCase(),
          size: 1000,
          cacheKey: 'my-model',
          accessToken: 'token',
        },
        undefined);

    expect(model).toEqual({
      url: `https://huggingface.co/org/repo/resolve/${COMMIT}/model.litertlm`,
      size: 1000,
      sha256: SHA256,
      cacheKey: 'my-model',
      accessToken: 'token',
    });
    const [, init] = fetchSpy.calls.mostRecent().args;
    expect(init?.headers).toEqual({'Authorization': 'Bearer token'});
  });

  it('rejects a ModelSource with the wrong SHA-256', async () => {
    const source = {url: 'hf://org/repo/model.litertlm', sha256: GPU_SHA256};
    await expectAsync(resolveModelInput(source, undefined))
        .toBeRejectedWithError(`hf://org/repo@main/model.litertlm has SHA-256 ${
            SHA256}, not ${GPU_SHA256}`);
  });

  it('rejects a ModelSource with the wrong size', async () => {
    const source = {url: 'hf://org/repo/model.litertlm', size: 5};
    await expectAsync(resolveModelInput(source, undefined))
        .toBeRejectedWithError(
            'hf://org/repo@main/model.litertlm is 1000 bytes, not 5 bytes');
  });
});

describe('checkModelBackend', () => {
  const huggingFace: ResolvedHuggingFaceModel = {
    source: {url: 'https://huggingface.co/org/repo/resolve/c0ffee/m.litertlm'},
    repo: 'org/repo',
    revision: 'main',
    commit: 'c0ffee',
    path: 'm.litertlm',
  };

  it('allows any backend if none was requested', () => {
    expect(() => {
      checkModelBackend(huggingFace, undefined, Backend.GPU_ARTISAN);
    }).not.toThrow();
  });

  it('allows the requested backend', () => {
    expect(() => {
      checkModelBackend(huggingFace, Backend.GPU, Backend.GPU);
    }).not.toThrow();
  });

  it('rejects a different backend', () => {
    expect(() => {
      checkModelBackend(huggingFace, Backend.GPU, Backend.GPU_ARTISAN);
    })
        .toThrowError(
            'hf://org/repo@main/m.litertlm is a Backend.GPU_ARTISAN model, ' +
            'but Backend.GPU was requested. Choose another file, or leave ' +
            '`backend` unset to use the model\'s backend.');
  });
});
