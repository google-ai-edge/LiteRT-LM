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

// The web platform APIs used by the model cache, looked up lazily so that
// tests can replace them. Everything here is available in service workers,
// except where noted.

/** The parts of `Cache` that the model cache uses. */
export declare interface CacheLike {
  match(request: string): Promise<Response|undefined>;
  put(request: string, response: Response): Promise<void>;
  delete(request: string): Promise<boolean>;
  keys(): Promise<readonly Request[]>;
}

/** The parts of `CacheStorage` that the model cache uses. */
export declare interface CacheStorageLike {
  open(cacheName: string): Promise<CacheLike>;
}

/** Options for `navigator.storageBuckets.open()`. */
export declare interface StorageBucketOpenOptions {
  persisted?: boolean;
  durability?: 'strict'|'relaxed';
}

/** A Storage Bucket (https://wicg.github.io/storage-buckets/). */
export declare interface StorageBucketLike {
  readonly caches: CacheStorageLike;
  persist?(): Promise<boolean>;
}

/** `navigator.storageBuckets`, which only Chromium supports. */
export declare interface StorageBucketManagerLike {
  open(name: string, options?: StorageBucketOpenOptions):
      Promise<StorageBucketLike>;
}

/** The parts of `StorageManager` that the model cache uses. */
export declare interface StorageManagerLike {
  estimate?(): Promise<StorageEstimate>;
  /** Only available in windows, not workers. */
  persist?(): Promise<boolean>;
}

/** The parts of `LockManager` that the model cache uses. */
export declare interface LockManagerLike {
  request(
      name: string, options: LockOptions,
      callback: (lock: Lock|null) => Promise<void>): Promise<void>;
}

/** Web platform APIs used by the model cache. */
export declare interface ModelCacheEnvironment {
  caches?: CacheStorageLike;
  storageBuckets?: StorageBucketManagerLike;
  storage?: StorageManagerLike;
  locks?: LockManagerLike;
  fetch: (input: string, init?: RequestInit) => Promise<Response>;
  /** Base URL for resolving relative model URLs. */
  baseUrl?: string;
}

/** `navigator` in browsers that support Storage Buckets. */
export declare interface NavigatorWithStorageBuckets {
  storageBuckets?: StorageBucketManagerLike;
}

/** The global scope of a window. */
export declare interface GlobalWithDocument {
  document?: {baseURI?: string};
}

let environmentOverride: Partial<ModelCacheEnvironment>|undefined;

/** Returns the web platform APIs to use. */
export function getCacheEnvironment(): ModelCacheEnvironment {
  const nav: (Navigator&NavigatorWithStorageBuckets)|undefined =
      globalThis.navigator;
  const defaults: ModelCacheEnvironment = {
    caches: globalThis.caches,
    storageBuckets: nav?.storageBuckets,
    storage: nav?.storage,
    locks: nav?.locks,
    fetch: (input, init) => globalThis.fetch(input, init),
    baseUrl: (globalThis as GlobalWithDocument).document?.baseURI ??
        globalThis.location?.href,
  };
  return {...defaults, ...environmentOverride};
}

/**
 * Replaces some of the web platform APIs, for tests. Call with no argument to
 * restore the real ones.
 */
export function setCacheEnvironmentForTesting(
    environment?: Partial<ModelCacheEnvironment>): void {
  environmentOverride = environment;
}
