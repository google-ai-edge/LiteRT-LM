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

import {setCacheEnvironmentForTesting} from './environment.js';
import {HuggingFaceFile, HuggingFaceModelRef, parseHuggingFaceUri, resolveHuggingFaceModel, toHuggingFaceUri} from './hugging_face.js';

describe('parseHuggingFaceUri', () => {
  const cases: Array<[string, HuggingFaceModelRef]> = [
    ['hf://org/repo', {repo: 'org/repo'}],
    ['hf://org/repo@main', {repo: 'org/repo', revision: 'main'}],
    [
      'hf://org/repo/model.litertlm',
      {repo: 'org/repo', path: 'model.litertlm'},
    ],
    [
      'hf://org/repo@external-weights/dir/model.litertlm',
      {
        repo: 'org/repo',
        revision: 'external-weights',
        path: 'dir/model.litertlm',
      },
    ],
    ['hf://org/repo@refs%2Fpr%2F1', {repo: 'org/repo', revision: 'refs/pr/1'}],
    // Like HfFileSystem, PR and conversion refs don't need to be encoded.
    ['hf://org/repo@refs/pr/1', {repo: 'org/repo', revision: 'refs/pr/1'}],
    [
      'hf://org/repo@refs/pr/12/dir/model.litertlm',
      {repo: 'org/repo', revision: 'refs/pr/12', path: 'dir/model.litertlm'},
    ],
    [
      'hf://org/repo@refs/convert/parquet/model.litertlm',
      {
        repo: 'org/repo',
        revision: 'refs/convert/parquet',
        path: 'model.litertlm',
      },
    ],
    // Other revisions with a `/` must be encoded.
    [
      'hf://org/repo@refs/heads/x/model.litertlm',
      {repo: 'org/repo', revision: 'refs', path: 'heads/x/model.litertlm'},
    ],
    [
      'hf://org/repo@refs/pr/x/model.litertlm',
      {repo: 'org/repo', revision: 'refs', path: 'pr/x/model.litertlm'},
    ],
    ['hf://org/repo/', {repo: 'org/repo'}],
  ];
  for (const [uri, ref] of cases) {
    it(`parses ${uri}`, () => {
      expect(parseHuggingFaceUri(uri)).toEqual(ref);
    });
  }

  const invalid = [
    'org/repo',
    'https://huggingface.co/org/repo',
    'hf://',
    'hf://org',
    'hf://org/',
    'hf:///repo',
    'hf://org/@main',
    'hf://org/repo@',
    'hf://org/repo@%E0%A4%A',
    'hf://datasets/org/data',
    'hf://spaces/org/space',
  ];
  for (const uri of invalid) {
    it(`rejects ${uri}`, () => {
      expect(parseHuggingFaceUri(uri)).toBeUndefined();
    });
  }
});

describe('toHuggingFaceUri', () => {
  it('builds URIs', () => {
    expect(toHuggingFaceUri({repo: 'org/repo'})).toBe('hf://org/repo');
    expect(toHuggingFaceUri(
               {repo: 'org/repo', revision: 'v1', path: 'dir/model.litertlm'}))
        .toBe('hf://org/repo@v1/dir/model.litertlm');
  });

  it('percent-encodes revisions', () => {
    expect(toHuggingFaceUri({repo: 'org/repo', revision: 'refs/pr/1'}))
        .toBe('hf://org/repo@refs%2Fpr%2F1');
  });

  it('round-trips with parseHuggingFaceUri', () => {
    const refs: HuggingFaceModelRef[] = [
      {repo: 'org/repo'},
      {repo: 'org/repo', revision: 'refs/pr/1', path: 'a/b.litertlm'},
      {repo: 'org/repo', revision: 'feature@2 #3', path: 'model.litertlm'},
    ];
    for (const ref of refs) {
      expect(parseHuggingFaceUri(toHuggingFaceUri(ref))).toEqual(ref);
    }
  });
});

describe('resolveHuggingFaceModel', () => {
  const COMMIT = 'c0ffee'.padEnd(40, '0');
  const SHA256 = 'ab'.repeat(32);
  const API = 'https://huggingface.co/api/models/org/repo';
  let routes: Map<string, () => Response>;
  let fetchSpy:
      jasmine.Spy<(url: string, init?: RequestInit) => Promise<Response>>;

  function json(body: unknown, init?: ResponseInit): Response {
    return new Response(JSON.stringify(body), init);
  }

  /** The Hub's response for a revision that doesn't exist. */
  function revisionNotFound(): Response {
    return json({error: 'Invalid rev id'}, {
      status: 404,
      headers: {
        'X-Error-Code': 'RevisionNotFound',
        'X-Error-Message': 'Invalid rev id',
      },
    });
  }

  /** The URL of the Hub API's revision info, with the file listing. */
  function revisionUrl(revision: string): string {
    return `${API}/revision/${revision}?blobs=true`;
  }

  /** A file stored with Git LFS, as listed in the Hub API's revision info. */
  function lfsFile(path: string, size: number, sha256 = SHA256) {
    return {
      rfilename: path,
      blobId: 'git-sha1',
      size,
      lfs: {sha256, size, pointerSize: 135},
    };
  }

  beforeEach(() => {
    routes = new Map([
      [
        revisionUrl('main'),
        () => json({
          sha: COMMIT,
          siblings: [
            {rfilename: 'README.md', blobId: 'git-sha1', size: 10},
            lfsFile('model.litertlm', 1000),
            lfsFile('model-gpu.litertlm', 2000, 'cd'.repeat(32)),
            lfsFile('dir/sub/model.litertlm', 7, 'ee'.repeat(32)),
          ],
        }),
      ],
    ]);
    fetchSpy = jasmine.createSpy('fetch').and.callFake(async (url: string) => {
      const route = routes.get(url);
      return route ? route() : json({error: 'Not found'}, {status: 404});
    });
    setCacheEnvironmentForTesting({fetch: fetchSpy});
  });

  afterEach(() => {
    setCacheEnvironmentForTesting();
  });

  it('resolves a file to a download at the current commit', async () => {
    const resolved =
        await resolveHuggingFaceModel('hf://org/repo/model.litertlm');

    expect(resolved).toEqual({
      source: {
        url: `https://huggingface.co/org/repo/resolve/${COMMIT}/model.litertlm`,
        size: 1000,
        sha256: SHA256,
      },
      repo: 'org/repo',
      revision: 'main',
      commit: COMMIT,
      path: 'model.litertlm',
    });
  });

  it('accepts a HuggingFaceModelRef', async () => {
    const resolved = await resolveHuggingFaceModel(
        {repo: 'org/repo', path: 'model-gpu.litertlm'});

    expect(resolved.source.sha256).toBe('cd'.repeat(32));
  });

  it('pins a revision to its commit', async () => {
    routes.set(
        revisionUrl('refs%2Fpr%2F1'),
        () => json({
          sha: 'pr'.padEnd(40, '0'),
          siblings: [lfsFile('model.litertlm', 5)],
        }));

    const resolved = await resolveHuggingFaceModel(
        'hf://org/repo@refs%2Fpr%2F1/model.litertlm');

    expect(resolved.revision).toBe('refs/pr/1');
    expect(resolved.commit).toBe('pr'.padEnd(40, '0'));
    expect(resolved.source.url)
        .toBe(`https://huggingface.co/org/repo/resolve/${
            'pr'.padEnd(40, '0')}/model.litertlm`);
  });

  it('finds a file in a subdirectory', async () => {
    const resolved =
        await resolveHuggingFaceModel('hf://org/repo/dir/sub/model.litertlm');

    expect(resolved.path).toBe('dir/sub/model.litertlm');
    expect(resolved.source.sha256).toBe('ee'.repeat(32));
    expect(resolved.source.url)
        .toBe(`https://huggingface.co/org/repo/resolve/${
            COMMIT}/dir/sub/model.litertlm`);
  });

  it('chooses a top-level file with selectFile', async () => {
    let offered: HuggingFaceFile[] = [];
    let offeredRevision = '';

    const resolved = await resolveHuggingFaceModel('hf://org/repo', {
      selectFile: (files, revision) => {
        offered = files;
        offeredRevision = revision;
        return files.find(f => f.path.endsWith('-gpu.litertlm'));
      },
    });

    expect(offered).toEqual([
      {path: 'README.md', size: 10},
      {path: 'model.litertlm', size: 1000, sha256: SHA256},
      {path: 'model-gpu.litertlm', size: 2000, sha256: 'cd'.repeat(32)},
    ]);
    expect(offeredRevision).toBe('main');
    expect(resolved.path).toBe('model-gpu.litertlm');
  });

  it('looks up the commit and files with one request', async () => {
    await resolveHuggingFaceModel('hf://org/repo/model.litertlm');

    expect(fetchSpy).toHaveBeenCalledOnceWith(
        revisionUrl('main'), jasmine.anything());
  });

  it('fails if no file matches, before downloading anything', async () => {
    await expectAsync(resolveHuggingFaceModel('hf://org/repo', {
      selectFile: () => undefined,
    })).toBeRejectedWithError(new RegExp(
        'No suitable model file in hf://org/repo@main. Its files are: ' +
        'README.md, model.litertlm, model-gpu.litertlm'));

    for (const [url] of fetchSpy.calls.allArgs()) {
      expect(url).toContain('/api/');
    }
  });

  it('fails if the file does not exist', async () => {
    await expectAsync(resolveHuggingFaceModel('hf://org/repo/missing.litertlm'))
        .toBeRejectedWithError(/missing.litertlm doesn't exist/);
  });

  it('fails without a path or selectFile', async () => {
    await expectAsync(resolveHuggingFaceModel('hf://org/repo'))
        .toBeRejectedWithError(/doesn't give a file path/);
  });

  it('fails if the revision does not exist', async () => {
    routes.set(revisionUrl('nope'), revisionNotFound);

    await expectAsync(resolveHuggingFaceModel('hf://org/repo@nope/m.litertlm'))
        .toBeRejectedWithError(new RegExp(
            "Couldn't resolve hf://org/repo/m.litertlm: " +
            "no revision named 'nope' exists"));
  });

  it('fails if a revision lookup fails for another reason', async () => {
    routes.set(
        revisionUrl('main'),
        () => json({error: 'Server error'}, {status: 500}));

    await expectAsync(resolveHuggingFaceModel('hf://org/repo/model.litertlm'))
        .toBeRejectedWithError(/Couldn't resolve .*: 500/);
  });

  describe('with defaultRevisions', () => {
    const WEB_COMMIT = 'web'.padEnd(40, '0');

    beforeEach(() => {
      routes.set(
          revisionUrl('web'),
          () => json({
            sha: WEB_COMMIT,
            siblings: [lfsFile('model.litertlm', 1001, 'ef'.repeat(32))],
          }));
    });

    it('uses the first revision that exists', async () => {
      const resolved = await resolveHuggingFaceModel(
          'hf://org/repo/model.litertlm', {defaultRevisions: ['web', 'main']});

      expect(resolved.revision).toBe('web');
      expect(resolved.commit).toBe(WEB_COMMIT);
      expect(resolved.source.sha256).toBe('ef'.repeat(32));
    });

    it('falls back to a later revision', async () => {
      routes.set(revisionUrl('web'), revisionNotFound);

      const resolved = await resolveHuggingFaceModel(
          'hf://org/repo/model.litertlm', {defaultRevisions: ['web', 'main']});

      expect(resolved.revision).toBe('main');
      expect(resolved.commit).toBe(COMMIT);
    });

    it('looks up the revisions in parallel', async () => {
      let releaseWeb!: () => void;
      const webReleased = new Promise<void>(resolve => {
        releaseWeb = resolve;
      });
      const webRoute = routes.get(revisionUrl('web'))!;
      fetchSpy.and.callFake(async (url: string) => {
        if (url === revisionUrl('web')) {
          await webReleased;
          return webRoute();
        }
        const found = routes.get(url);
        return found ? found() : json({}, {status: 404});
      });

      const resolving = resolveHuggingFaceModel(
          'hf://org/repo/model.litertlm', {defaultRevisions: ['web', 'main']});
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(fetchSpy.calls.allArgs().map(([url]) => url)).toEqual([
        revisionUrl('web'),
        revisionUrl('main'),
      ]);

      releaseWeb();
      expect((await resolving).revision).toBe('web');
    });

    it('does not wait for later revisions once one exists', async () => {
      let rejectMain!: (reason: Error) => void;
      fetchSpy.and.callFake(async (url: string) => {
        if (url === revisionUrl('main')) {
          return new Promise<Response>((resolve, reject) => {
            rejectMain = reject;
          });
        }
        return routes.get(url)!();
      });

      const resolved = await resolveHuggingFaceModel(
          'hf://org/repo/model.litertlm', {defaultRevisions: ['web', 'main']});

      expect(resolved.revision).toBe('web');
      // A later lookup that fails afterwards is ignored.
      rejectMain(new Error('Network error'));
    });

    it('does not fall back if the first revision lacks the file', async () => {
      await expectAsync(resolveHuggingFaceModel(
                            'hf://org/repo/model-gpu.litertlm',
                            {defaultRevisions: ['web', 'main']}))
          .toBeRejectedWithError(
              'hf://org/repo@web/model-gpu.litertlm doesn\'t exist');
    });

    it('is ignored for a URI with a revision', async () => {
      const resolved = await resolveHuggingFaceModel(
          'hf://org/repo@main/model.litertlm',
          {defaultRevisions: ['web', 'main']});

      expect(resolved.revision).toBe('main');
      expect(fetchSpy).not.toHaveBeenCalledWith(
          revisionUrl('web'), jasmine.anything());
    });

    it('fails if an earlier revision lookup fails', async () => {
      routes.set(
          revisionUrl('web'),
          () => json({error: 'Unauthorized'}, {status: 401}));

      await expectAsync(resolveHuggingFaceModel(
                            'hf://org/repo/model.litertlm',
                            {defaultRevisions: ['web', 'main']}))
          .toBeRejectedWithError(/Couldn't access/);
    });

    it('fails if no revision exists', async () => {
      routes.set(revisionUrl('web'), revisionNotFound);
      routes.set(revisionUrl('main'), revisionNotFound);

      await expectAsync(resolveHuggingFaceModel(
                            'hf://org/repo/model.litertlm',
                            {defaultRevisions: ['web', 'main']}))
          .toBeRejectedWithError(/no revision named 'web' or 'main' exists/);
    });

    it('must not be empty', async () => {
      await expectAsync(resolveHuggingFaceModel(
                            'hf://org/repo/model.litertlm',
                            {defaultRevisions: []}))
          .toBeRejectedWithError(/defaultRevisions must not be empty/);
    });
  });

  it('explains when the repo is private or gated', async () => {
    routes.set(
        revisionUrl('main'),
        () => json({error: 'Unauthorized'}, {status: 401}));

    await expectAsync(resolveHuggingFaceModel('hf://org/repo/model.litertlm'))
        .toBeRejectedWithError(
            /private or gated, which needs the access token of a user/);
  });

  it('rejects URIs that are not hf://', async () => {
    await expectAsync(resolveHuggingFaceModel('https://example.com/m'))
        .toBeRejectedWithError(/Not a Hugging Face model URI/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('sends the access token and keeps it in the source', async () => {
    const resolved = await resolveHuggingFaceModel(
        'hf://org/repo/model.litertlm', {accessToken: 'token'});

    expect(resolved.source.accessToken).toBe('token');
    for (const [, init] of fetchSpy.calls.allArgs()) {
      expect(init!.headers).toEqual({'Authorization': 'Bearer token'});
    }
  });

  it('treats an empty access token as none', async () => {
    const resolved = await resolveHuggingFaceModel(
        'hf://org/repo/model.litertlm', {accessToken: ''});

    expect(resolved.source.accessToken).toBeUndefined();
    for (const [, init] of fetchSpy.calls.allArgs()) {
      expect(init!.headers).toBeUndefined();
    }
  });

  it('passes the signal to the requests', async () => {
    const controller = new AbortController();

    await resolveHuggingFaceModel(
        'hf://org/repo/model.litertlm', {signal: controller.signal});

    for (const [, init] of fetchSpy.calls.allArgs()) {
      expect(init!.signal).toBe(controller.signal);
    }
  });
});
