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

import {getCacheEnvironment} from './environment.js';

/**
 * Returns the name of the Web Lock that guards the cache entries for `key`.
 *
 * A shared lock is held while reading a cached model, since deleting an entry
 * invalidates Blobs read from it. An exclusive lock is held while downloading
 * and writing a model, so that other tabs wait for it instead of downloading
 * it again, and while deleting one.
 */
export function getModelLockName(key: string): string {
  return `litertlm-model:${key}`;
}

/**
 * Runs `fn` while holding a Web Lock, and returns its result. The lock is
 * released when `fn` settles, including when it throws. Rejects with the
 * signal's reason if `signal` is aborted before the lock is acquired.
 *
 * Just runs `fn` if Web Locks aren't available.
 */
export async function withLock<T>(
    name: string, mode: LockMode, fn: () => Promise<T>| T,
    signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted();
  const locks = getCacheEnvironment().locks;
  if (!locks) {
    return fn();
  }
  const options: LockOptions = {mode};
  if (signal) options.signal = signal;
  let result!: T;
  await locks.request(name, options, async () => {
    result = await fn();
  });
  return result;
}
