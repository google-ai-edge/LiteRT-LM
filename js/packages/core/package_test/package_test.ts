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
 *
 * @fileoverview Smoke tests for the `@litert-lm/core` package.
 *
 * These tests only use the public API of `@litert-lm/core`, and they load the
 * Wasm files that ship with it. In open source, they run against the packed
 * npm tarball before it is published to make sure its JavaScript and Wasm
 * files are compatible (see
 * https://github.com/google-ai-edge/LiteRT-LM/issues/4051).
 *
 * They only use the CPU backend so that they can run without a GPU.
 */

import {Backend, Engine, LiteRtLm, loadLiteRtLm, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

const TEST_MODEL_PATH = '/testdata/test_lm.litertlm';

describe('@litert-lm/core package', () => {
  let liteRtLm: LiteRtLm;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm('/wasm');
  }, 300_000);

  afterAll(() => {
    unloadLiteRtLm();
  });

  it('loads the Wasm module', () => {
    expect(liteRtLm).toBeDefined();
    expect(liteRtLm.liteRtLmWasm).toBeDefined();
  });

  it('reaches the model parser when the model is invalid', async () => {
    // The JavaScript and Wasm must agree on the shape of the Wasm API to get
    // this far. A mismatch fails earlier with an unrelated error, such as a
    // TypeError.
    const invalidModel = new Blob([new Uint8Array(1024)]);
    let error: unknown;
    try {
      const engine =
          await Engine.create({model: invalidModel, backend: Backend.CPU});
      await engine.delete();
    } catch (e) {
      error = e;
    }
    expect(error).toBeDefined();
    expect(String(error)).toMatch(/magic number/i);
  });

  describe('with a small model on CPU', () => {
    let engine: Engine;

    beforeAll(async () => {
      engine = await Engine.create({
        model: TEST_MODEL_PATH,
        backend: Backend.CPU,
        mainExecutorSettings: {
          maxNumTokens: 128,
        },
      });
    }, 300_000);

    afterAll(async () => {
      await engine?.delete();
    });

    it('creates an Engine with the requested settings', () => {
      expect(engine).toBeDefined();
      expect(engine.settings.backend).toBe(Backend.CPU);
      expect(engine.settings.mainExecutorSettings.maxNumTokens).toBe(128);
    });

    it('runs prefill and decode in a Session', async () => {
      const session = await engine.createSession();
      try {
        await session.runPrefill(['Hello']);
        const responses = await session.runDecode();
        try {
          const texts = responses.getTexts();
          expect(texts.length).toBeGreaterThan(0);
          expect(texts[0].length).toBeGreaterThan(0);
        } finally {
          responses.delete();
        }
      } finally {
        await session.delete();
      }
    });
  });
});
