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
import {getModelLockName, withLock} from './locks.js';

/** A promise that the test resolves. */
class Deferred {
  resolve!: () => void;
  readonly promise = new Promise<void>(resolve => {
    this.resolve = resolve;
  });
}

// Uses the browser's Web Locks, with a unique lock name per test.
describe('locks', () => {
  let name: string;

  beforeEach(() => {
    name = `locks-test-${Math.random()}`;
  });

  afterEach(() => {
    setCacheEnvironmentForTesting();
  });

  /** Returns the modes `name` is held in. */
  async function heldModes(): Promise<LockMode[]> {
    const {held} = await navigator.locks.query();
    return (held ?? []).filter(lock => lock.name === name)
        .map(lock => lock.mode!);
  }

  /** Returns whether `promise` settles within a few event loop turns. */
  async function settlesSoon(promise: Promise<unknown>): Promise<boolean> {
    const timeout = new Promise(resolve => {
      setTimeout(resolve, 20, 'timeout');
    });
    return (await Promise.race([promise.then(() => 'settled'), timeout])) ===
        'settled';
  }

  it('names locks after the cache key', () => {
    expect(getModelLockName('abc')).toBe('litertlm-model:abc');
  });

  it('holds the lock while the function runs', async () => {
    let modesWhileRunning: LockMode[] = [];

    const result = await withLock(name, 'exclusive', async () => {
      modesWhileRunning = await heldModes();
      return 42;
    });

    expect(result).toBe(42);
    expect(modesWhileRunning).toEqual(['exclusive']);
    expect(await heldModes()).toEqual([]);
  });

  it('releases the lock when the function throws', async () => {
    await expectAsync(withLock(name, 'exclusive', () => {
      throw new Error('failed');
    })).toBeRejectedWithError('failed');

    expect(await heldModes()).toEqual([]);
  });

  it('lets shared locks be held together', async () => {
    const done = new Deferred();
    const first = withLock(name, 'shared', () => done.promise);
    let modesInSecond: LockMode[] = [];

    await withLock(name, 'shared', async () => {
      modesInSecond = await heldModes();
    });

    expect(modesInSecond).toEqual(['shared', 'shared']);
    done.resolve();
    await first;
  });

  it('waits for an exclusive lock to be released', async () => {
    const done = new Deferred();
    const exclusive = withLock(name, 'exclusive', () => done.promise);
    const shared = withLock(name, 'shared', () => {});
    expect(await settlesSoon(shared)).toBeFalse();

    done.resolve();
    await exclusive;
    await shared;
  });

  it('stops waiting when the signal is aborted', async () => {
    const done = new Deferred();
    const holder = withLock(name, 'exclusive', () => done.promise);
    const controller = new AbortController();
    const fn = jasmine.createSpy('fn');
    const waiting = withLock(name, 'exclusive', fn, controller.signal);

    controller.abort(new Error('stop'));

    await expectAsync(waiting).toBeRejectedWithError('stop');
    expect(fn).not.toHaveBeenCalled();
    done.resolve();
    await holder;
  });

  it('rejects if the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort(new Error('stop'));
    const fn = jasmine.createSpy('fn');

    await expectAsync(withLock(name, 'shared', fn, controller.signal))
        .toBeRejectedWithError('stop');
    expect(fn).not.toHaveBeenCalled();
  });

  describe('without Web Locks', () => {
    beforeEach(() => {
      setCacheEnvironmentForTesting({locks: undefined});
    });

    it('just runs the function', async () => {
      let modesWhileRunning: LockMode[] = [];

      const result = await withLock(name, 'exclusive', async () => {
        modesWhileRunning = await heldModes();
        return 42;
      });

      expect(result).toBe(42);
      expect(modesWhileRunning).toEqual([]);
    });

    it('rejects if the signal is already aborted', async () => {
      const controller = new AbortController();
      controller.abort(new Error('stop'));

      await expectAsync(withLock(name, 'shared', () => {}, controller.signal))
          .toBeRejectedWithError('stop');
    });
  });
});
