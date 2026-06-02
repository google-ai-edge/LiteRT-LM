/**
 * Copyright 2026 The ODML Authors.
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
 * A mutual exclusion lock for managing asynchronous operations.
 *
 * In an Emscripten WebAssembly environment using Asyncify or JSPI, C++
 * execution can be paused during asynchronous operations (such as waiting for
 * WebGPU execution). While Wasm is paused, the JS event loop continues running.
 * If JS invokes another Wasm binding function that attempts to acquire the same
 * C++ mutex held by the paused stack frame, single-threaded C++ cannot yield
 * and triggers an unrecoverable deadlock.
 *
 * This class prevents re-entrancy deadlocks by serializing all entry points
 * from JavaScript into Wasm that interact with underlying C++ mutexes. Even
 * seemingly synchronous tasks (like creating or deleting a session) must
 * asynchronously wait for this mutex to ensure any paused C++ tasks have fully
 * completed and released their locks.
 */
export class Mutex {
  private mutexPromise = Promise.resolve();
  private queueCount = 0;

  /**
   * Whether the mutex is currently locked.
   */
  get isLocked(): boolean {
    return this.queueCount > 0;
  }

  /**
   * Executes the provided function once the mutex is acquired, and releases
   * the mutex when the function completes.
   */
  async acquireAndRun<T>(fn: () => Promise<T>| T): Promise<T> {
    this.queueCount++;
    const previousPromise = this.mutexPromise;
    let resolveMutex!: () => void;
    this.mutexPromise = new Promise<void>((resolve) => {
      resolveMutex = resolve;
    });
    try {
      await previousPromise;
      return await fn();
    } finally {
      this.queueCount--;
      resolveMutex();
    }
  }
}
