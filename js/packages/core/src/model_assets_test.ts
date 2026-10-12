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

import {LiteRtLm, loadLiteRtLm, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

import {Cleanup} from './cleanup.js';
import {createStreamingModelAssets} from './stream_utils.js';

describe('ModelAssets tests', () => {
  let liteRtLm: LiteRtLm;
  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
  });

  describe('ModelAssets', () => {
    it('creates streaming ModelAssets', async () => {
      const cleanup = new Cleanup();
      const {modelAssets} = await createStreamingModelAssets(
          new Blob([]), liteRtLm.liteRtLmWasm, cleanup);
      expect(modelAssets).toBeDefined();
      cleanup.run();
    });
  });
});
