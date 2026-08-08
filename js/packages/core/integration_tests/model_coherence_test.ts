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

import {Backend, Engine, LiteRtLm, loadLiteRtLm, Message, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

const TEST_TIMEOUT_MS = 3600_000;  // 60 minutes
jasmine.DEFAULT_TIMEOUT_INTERVAL = TEST_TIMEOUT_MS;

interface ModelConfig {
  name: string;
  path: string;
  backend: Backend;
}

const MODELS: ModelConfig[] = [
  {
    name: 'Gemma 4 E2B',
    path: '/models/gemma4-e2b-hw.litertlm',
    backend: Backend.GPU_ARTISAN,
  },
  {
    name: 'Gemma 4 E4B',
    path: '/models/gemma4-e4b-hw.litertlm',
    backend: Backend.GPU_ARTISAN,
  },
];

describe('Model Coherence Tests', () => {
  let liteRtLm: LiteRtLm;

  async function resetLiteRtLm() {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
    await liteRtLm.setupDefaultWebGpuDevice();
  }

  beforeAll(async () => {
    await resetLiteRtLm();
  }, TEST_TIMEOUT_MS);

  for (const model of MODELS) {
    describe(`${model.name} coherence tests`, () => {
      let engine: Engine;

      beforeAll(async () => {
        engine = await Engine.create({
          model: model.path,
          backend: model.backend,
        });
      }, TEST_TIMEOUT_MS);

      afterAll(async () => {
        if (engine) {
          await engine.delete();
        }
      });

      it('answers "What is the capital of France?" correctly', async () => {
        const conversation = await engine.createConversation({
          sessionConfig: {
            maxOutputTokens: 100,
          },
        });
        const response: Message =
            await conversation.sendMessage('What is the capital of France?');

        expect(response).toBeDefined();
        expect(response.content).toBeDefined();

        let textContent = '';
        if (typeof response.content === 'string') {
          textContent = response.content;
        } else if (Array.isArray(response.content)) {
          textContent = response.content.filter((part) => part.type === 'text')
                            .map((part: any) => part.text)
                            .join(' ');
        }

        console.log(`[${model.name}] Response:`, textContent);
        expect(textContent.toLowerCase()).toContain('paris');

        await conversation.delete();
      }, TEST_TIMEOUT_MS);
    });
  }
});
