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

import {Mutex} from './mutex.js';

const engineCreationMutexes = new WeakMap<object, Mutex>();

/**
 * Runs `fn` while holding the engine creation mutex of `wasmModule`, so that
 * at most one `Engine` or `EmbeddingEngine` is created at a time per Wasm
 * module.
 *
 * Creation suspends Wasm (JSPI / Asyncify) while it waits for weights to
 * download, and the JS event loop keeps running in the meantime. Without this
 * mutex, a second creation could start while the first is suspended and race
 * with it on state that is global to the Wasm module, e.g. to set up the
 * default WebGPU device, or deadlock on a C++ mutex held by the suspended
 * stack. (The streamed weights themselves are not shared: each model's weights
 * are uploaded from its own stream.)
 */
export function runExclusiveEngineCreation<T>(
    wasmModule: object, fn: () => Promise<T>| T): Promise<T> {
  let mutex = engineCreationMutexes.get(wasmModule);
  if (!mutex) {
    mutex = new Mutex();
    engineCreationMutexes.set(wasmModule, mutex);
  }
  return mutex.acquireAndRun(fn);
}
