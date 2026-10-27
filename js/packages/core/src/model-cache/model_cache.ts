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

import {getCacheKey, ModelSource, resolveModelUrl} from './model_source.js';

/** Options for `ModelCache.put()`. */
export declare interface ModelCachePutOptions {
  /** Aborts the write. Nothing is stored if it's aborted. */
  signal?: AbortSignal;
  /**
   * Whether to request persistent storage after the write succeeds.
   * Overrides the cache's own setting if set.
   */
  persistentStorage?: boolean;
}

/** Information about a model stored in a cache. */
export declare interface CachedModelInfo {
  /** The model's cache key (see `ModelSource.cacheKey`). */
  key: string;
  /** The URL the model was downloaded from, if known. */
  url?: string;
  sha256?: string;
  size: number;
  storedAt?: Date;
  hfRepo?: string;
  hfPath?: string;
  hfRevision?: string;
}

/**
 * Stores downloaded models.
 *
 * `get()` and `put()` don't take locks or wait for each other. Since `put()` is
 * atomic, a `get()` during a `put()` of the same model sees the previous entry
 * (or a miss), never a partial one.
 *
 * Callers coordinate access across tabs and workers with the Web Lock named
 * `getModelLockName(getCacheKey(source))`. The model loader does this:
 *
 * - It holds the lock shared while reading the `Blob` returned by `get()`,
 *   since deleting or replacing the entry may make that `Blob` unreadable.
 * - It holds the lock exclusively while downloading a model and storing it
 *   with `put()`, so that other tabs wait for the download instead of
 *   repeating it, and check the cache again afterwards.
 *
 * Implementations must not take that lock in `get()` or `put()`, since the
 * caller may already hold it and Web Locks aren't reentrant. `delete()` should
 * take it exclusively, so that it waits for reads of the entry to finish.
 */
export declare interface ModelCache {
  readonly name: string;
  /**
   * Returns the cached model, or undefined if it isn't cached. Never throws
   * for a miss. The `Blob` may become unreadable if the entry is deleted or
   * replaced, so read it while holding the model's lock shared.
   */
  get(source: ModelSource): Promise<Blob|undefined>;
  /**
   * Stores a model. The model only becomes visible once it has been written
   * completely, and nothing is left behind if the write fails or is aborted.
   * Callers that may replace an entry that's being read should hold the
   * model's lock exclusively.
   *
   * `data` is always either consumed or cancelled, even if this fails.
   */
  put(source: ModelSource, data: ReadableStream<Uint8Array>|Blob,
      options?: ModelCachePutOptions): Promise<void>;
  /** Lists the cached models. */
  list?(): Promise<CachedModelInfo[]>;
  /**
   * Deletes a cached model, given its source or cache key. Returns whether it
   * was cached. Waits for reads of the model to finish (see above).
   */
  delete?(source: ModelSource|string): Promise<boolean>;
}

/**
 * Names of the headers that store a model's metadata in a Cache API entry.
 */
export const CACHE_ENTRY_HEADERS = {
  contentLength: 'Content-Length',
  sha256: 'X-LiteRTLM-SHA256',
  sourceUrl: 'X-LiteRTLM-Source-URL',
  hfRepo: 'X-LiteRTLM-HF-Repo',
  hfPath: 'X-LiteRTLM-HF-Path',
  hfRevision: 'X-LiteRTLM-HF-Revision',
  backend: 'X-LiteRTLM-Backend',
  storedAt: 'X-LiteRTLM-Stored-At',
} as const;

/**
 * Returns the headers to store with a model in a Cache API entry. They hold
 * the model's metadata, which `ModelCache.list()` reports.
 *
 * Useful for apps that call `cache.put()` themselves, e.g. from a Background
 * Fetch handler, along with `getCacheRequestUrlForSource()` for the request.
 *
 * Throws if `source.size` is set but isn't a non-negative integer.
 */
export function createCacheEntryHeaders(source: ModelSource): Headers {
  const headers = new Headers({'Content-Type': 'application/octet-stream'});
  if (source.size !== undefined) {
    if (!Number.isSafeInteger(source.size) || source.size < 0) {
      throw new Error(
          `Model size must be a non-negative integer, got ${source.size}`);
    }
    headers.set(CACHE_ENTRY_HEADERS.contentLength, String(source.size));
  }
  if (source.sha256 !== undefined) {
    headers.set(CACHE_ENTRY_HEADERS.sha256, source.sha256);
  }
  headers.set(CACHE_ENTRY_HEADERS.sourceUrl, resolveModelUrl(source.url));
  headers.set(CACHE_ENTRY_HEADERS.storedAt, new Date().toISOString());
  return headers;
}

/**
 * Returns information about a cached model from its key and the headers it
 * was stored with. `size` is used if the headers don't include a valid size.
 * Malformed headers are ignored.
 */
export function getCachedModelInfo(
    key: string, headers: Headers, size?: number): CachedModelInfo {
  const contentLength =
      parseContentLength(headers.get(CACHE_ENTRY_HEADERS.contentLength));
  const info: CachedModelInfo = {key, size: contentLength ?? size ?? 0};
  const url = headers.get(CACHE_ENTRY_HEADERS.sourceUrl) ??
      (isHttpUrl(key) ? key : null);
  if (url !== null) info.url = url;
  const sha256 = headers.get(CACHE_ENTRY_HEADERS.sha256);
  if (sha256 !== null) info.sha256 = sha256;
  const storedAt = headers.get(CACHE_ENTRY_HEADERS.storedAt);
  if (storedAt !== null) {
    const date = new Date(storedAt);
    if (!Number.isNaN(date.getTime())) info.storedAt = date;
  }
  const hfRepo = headers.get(CACHE_ENTRY_HEADERS.hfRepo);
  if (hfRepo !== null) info.hfRepo = hfRepo;
  const hfPath = headers.get(CACHE_ENTRY_HEADERS.hfPath);
  if (hfPath !== null) info.hfPath = hfPath;
  const hfRevision = headers.get(CACHE_ENTRY_HEADERS.hfRevision);
  if (hfRevision !== null) info.hfRevision = hfRevision;
  return info;
}

/**
 * Parses a Content-Length header value, returning undefined if it's missing or
 * isn't a non-negative integer.
 */
function parseContentLength(value: string|null): number|undefined {
  if (value === null || !/^\d+$/.test(value.trim())) return undefined;
  const length = Number(value);
  return Number.isSafeInteger(length) ? length : undefined;
}

/**
 * Prefix of the request URLs that store models whose cache key isn't a URL.
 * The Cache API only accepts http(s) URLs, and the `.invalid` top-level domain
 * is reserved, so these never collide with real URLs.
 */
const KEY_URL_PREFIX = 'https://litertlm-model-cache.invalid/';

/** Returns the Cache API request URL that stores the model with `key`. */
export function getCacheRequestUrl(key: string): string {
  return isHttpUrl(key) ? key : KEY_URL_PREFIX + encodeURIComponent(key);
}

/** Inverse of `getCacheRequestUrl()`. */
export function getCacheKeyFromRequestUrl(url: string): string {
  return url.startsWith(KEY_URL_PREFIX) ?
      decodeURIComponent(url.slice(KEY_URL_PREFIX.length)) :
      url;
}

/**
 * Returns the Cache API request URL that stores a model: its cache key if
 * that's an http(s) URL, and a URL derived from the key otherwise.
 */
export function getCacheRequestUrlForSource(source: ModelSource): string {
  return getCacheRequestUrl(getCacheKey(source));
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}
