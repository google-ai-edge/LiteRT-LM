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

import {runExclusiveEngineCreation} from './engine_creation.js';

/** Returns a promise and the function that resolves it. */
function deferred(): {promise: Promise<void>, resolve: () => void} {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return {promise, resolve};
}

describe('runExclusiveEngineCreation', () => {
  it('does not interleave creations for the same Wasm module', async () => {
    const wasmModule = {};
    const events: string[] = [];
    const firstCanFinish = deferred();

    const first = runExclusiveEngineCreation(wasmModule, async () => {
      events.push('first start');
      await firstCanFinish.promise;
      events.push('first end');
      return 1;
    });
    const second = runExclusiveEngineCreation(wasmModule, async () => {
      events.push('second start');
      events.push('second end');
      return 2;
    });

    // Let the event loop run so that the second creation would start if it
    // were not serialized.
    await new Promise((r) => setTimeout(r, 0));
    expect(events).toEqual(['first start']);

    firstCanFinish.resolve();
    expect(await first).toBe(1);
    expect(await second).toBe(2);
    expect(events).toEqual(
        ['first start', 'first end', 'second start', 'second end']);
  });

  it('runs the next creation after a failed creation', async () => {
    const wasmModule = {};
    const failed = runExclusiveEngineCreation(wasmModule, async () => {
      throw new Error('creation failed');
    });
    const next = runExclusiveEngineCreation(wasmModule, async () => 'ok');

    await expectAsync(failed).toBeRejectedWithError('creation failed');
    expect(await next).toBe('ok');
  });

  it('does not serialize creations for different Wasm modules', async () => {
    const firstCanFinish = deferred();
    let secondRan = false;

    const first = runExclusiveEngineCreation({}, async () => {
      await firstCanFinish.promise;
    });
    const second = runExclusiveEngineCreation({}, async () => {
      secondRan = true;
    });

    await second;
    expect(secondRan).toBeTrue();
    firstCanFinish.resolve();
    await first;
  });
});
