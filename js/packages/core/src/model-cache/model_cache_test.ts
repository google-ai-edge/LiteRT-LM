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

import {getCacheEnvironment, setCacheEnvironmentForTesting} from './environment.js';
import {createCacheEntryHeaders, getCachedModelInfo, getCacheKeyFromRequestUrl, getCacheRequestUrl, getCacheRequestUrlForSource} from './model_cache.js';
import {getCacheKey, ModelSource, toModelSource} from './model_source.js';

describe('model cache keys', () => {
  beforeEach(() => {
    setCacheEnvironmentForTesting({baseUrl: 'https://example.com/app/'});
  });

  afterEach(() => {
    setCacheEnvironmentForTesting();
  });

  it('prefers cacheKey, then sha256, then the URL', () => {
    expect(getCacheKey({url: 'm.litertlm', sha256: 'abc', cacheKey: 'key'}))
        .toBe('key');
    expect(getCacheKey({url: 'm.litertlm', sha256: 'abc'})).toBe('abc');
    expect(getCacheKey({url: 'm.litertlm'}))
        .toBe('https://example.com/app/m.litertlm');
  });

  it('maps keys to request URLs and back', () => {
    for (const key of ['abc', 'my key/with?odd#chars', 'https://x.com/m']) {
      expect(getCacheKeyFromRequestUrl(new Request(getCacheRequestUrl(key)).url))
          .toBe(key);
    }
    expect(getCacheRequestUrl('https://x.com/m')).toBe('https://x.com/m');
  });

  it('maps a source to the request URL for its key', () => {
    expect(getCacheRequestUrlForSource({url: 'm.litertlm', sha256: 'abc'}))
        .toBe(getCacheRequestUrl('abc'));
    expect(getCacheRequestUrlForSource({url: 'm.litertlm'}))
        .toBe('https://example.com/app/m.litertlm');
  });

  it('converts a URL to a ModelSource', () => {
    const source: ModelSource = {url: 'm.litertlm', sha256: 'abc'};

    expect(toModelSource('m.litertlm')).toEqual({url: 'm.litertlm'});
    expect(toModelSource(source)).toBe(source);
  });

  it('round-trips metadata through headers', () => {
    const headers = createCacheEntryHeaders(
        {url: 'm.litertlm', sha256: 'abc', size: 42, accessToken: 'secret'});

    expect(getCachedModelInfo('abc', headers)).toEqual({
      key: 'abc',
      url: 'https://example.com/app/m.litertlm',
      sha256: 'abc',
      size: 42,
      storedAt: jasmine.any(Date),
    });
  });

  it('rejects hf:// URIs that have not been resolved', () => {
    expect(() => getCacheKey({url: 'hf://org/repo'}))
        .toThrowError(/resolveHuggingFaceModel/);
  });

  it('ignores malformed metadata headers', () => {
    const headers = new Headers({
      'Content-Length': 'NaN',
      'X-LiteRTLM-Stored-At': 'not a date',
    });

    expect(getCachedModelInfo('abc', headers, 7))
        .toEqual({key: 'abc', size: 7});
    expect(getCachedModelInfo('abc', headers)).toEqual({key: 'abc', size: 0});
    for (const contentLength of ['-1', '1.5', '', '1e3']) {
      headers.set('Content-Length', contentLength);
      expect(getCachedModelInfo('abc', headers, 7).size).toBe(7);
    }
  });

  it('rejects an invalid size when creating headers', () => {
    for (const size of [NaN, -1, 1.5, Infinity]) {
      expect(() => createCacheEntryHeaders({url: 'm.litertlm', size}))
          .toThrowError(/non-negative integer/);
    }
    expect(createCacheEntryHeaders({url: 'm.litertlm', size: 0})
               .get('Content-Length'))
        .toBe('0');
  });
});

describe('getCacheEnvironment', () => {
  it('calls the global fetch with the global scope as `this`', async () => {
    // Storing `globalThis.fetch` itself and calling it as
    // `environment.fetch()` would throw "Illegal invocation".
    const response = new Response('model');
    const fetchSpy = spyOn(globalThis, 'fetch').and.resolveTo(response);
    const init: RequestInit = {credentials: 'same-origin'};

    expect(await getCacheEnvironment().fetch('https://x.com/m', init))
        .toBe(response);
    expect(fetchSpy).toHaveBeenCalledOnceWith('https://x.com/m', init);
    expect(fetchSpy.calls.mostRecent().object).toBe(globalThis);
  });
});
