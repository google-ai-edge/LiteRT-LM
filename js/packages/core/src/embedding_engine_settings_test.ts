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

import {LiteRtLm, loadLiteRtLm, unloadLiteRtLm, type Wasm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

describe('EmbeddingEngineSettings', () => {
  let liteRtLm: LiteRtLm;
  let modelAssets: Wasm.ModelAssets;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
    modelAssets = liteRtLm.liteRtLmWasm.ModelAssets.create('/path/to/model');
  });

  afterAll(() => {
    modelAssets.delete();
  });

  it('creates EmbeddingEngineSettings', () => {
    const settings =
        liteRtLm.liteRtLmWasm.EmbeddingEngineSettings.createDefault(
            modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
    expect(settings).toBeDefined();
    settings.delete();
  });
});
