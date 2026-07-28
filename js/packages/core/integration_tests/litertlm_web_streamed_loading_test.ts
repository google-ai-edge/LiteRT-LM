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

import {Backend, Conversation, Engine, LiteRtLm, loadLiteRtLm, SamplerType, SessionConfig, unloadLiteRtLm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

jasmine.DEFAULT_TIMEOUT_INTERVAL = 3_000_000;  // 3000 seconds (50 minutes)

describe('LiteRtLm Streamed Loading Tests', () => {
  let liteRtLm: LiteRtLm;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
    await liteRtLm.setupDefaultWebGpuDevice();
  }, 300_000);

  describe('Gemma 4 E2B on GPU', () => {
    let engine: Engine | undefined;
    let initialized = false;

    beforeAll(async () => {
      try {
        const externalModelPath =
            '/models/gemma-4-E2B-it-external-perlayer-prefilldecode.litertlm';
        engine = await Engine.create({
          model: externalModelPath,
          mainExecutorSettings: {
            maxNumTokens: 128,
            backendConfig: {
              max_top_k: 3,
              external_tensor_mode: false,
            },
          },
        });
        initialized = true;
      } catch (e) {
        console.warn(
            'Failed to initialize Gemma 4 E2B on GPU, skipping tests. Error:',
            e);
      }
    });

    afterAll(async () => {
      if (engine) {
        await engine.delete();
      }
    });

    it('runs a conversation with Gemma 4 E2B on GPU', async () => {
      if (!initialized) {
        pending('Gemma 4 E2B model not available or GPU not supported');
        return;
      }
      expect(engine).toBeDefined();

      const conversation = await engine!.createConversation({
        sessionConfig: {
          maxOutputTokens: 32,
          samplerParams: {
            type: SamplerType.TOP_P,
            k: 1,
            p: 0.9,
          },
        },
      });

      const response = await conversation.sendMessage({
        role: 'user',
        content: 'What is the capital of France?',
      });

      expect(response).toBeDefined();
      expect(response.content).toBeDefined();

      let text = '';
      if (typeof response.content === 'string') {
        text = response.content;
      } else if (Array.isArray(response.content)) {
        text = response.content
            .filter(item => item.type === 'text')
            .map(item => item.text)
            .join(' ');
      }
      expect(text.toLowerCase()).toContain('paris');

      await conversation.delete();
    });
  });

  xdescribe('Gemma 4 E4B on GPU', () => {
    let engine: Engine | undefined;
    let initialized = false;

    beforeAll(async () => {
      try {
        const externalModelPath =
            '/models/gemma-4-E4B-it-external-perlayer-prefilldecode.litertlm';
        engine = await Engine.create({
          model: externalModelPath,
          backend: Backend.GPU,
          mainExecutorSettings: {
            maxNumTokens: 128,
            backendConfig: {
              max_top_k: 3,
              external_tensor_mode: false,
            },
          },
        });
        initialized = true;
      } catch (e) {
        console.warn(
            'Failed to initialize Gemma 4 E4B on GPU, skipping tests. Error:',
            e);
      }
    });

    afterAll(async () => {
      if (engine) {
        await engine.delete();
      }
    });

    it('runs a conversation with Gemma 4 E4B on GPU', async () => {
      if (!initialized) {
        pending('Gemma 4 E4B model not available or GPU not supported');
        return;
      }
      expect(engine).toBeDefined();

      const conversation = await engine!.createConversation({
        sessionConfig: {
          maxOutputTokens: 32,
          samplerParams: {
            type: SamplerType.TOP_P,
            k: 1,
            p: 0.9,
          },
        },
      });

      const response = await conversation.sendMessage({
        role: 'user',
        content: 'What is the capital of France?',
      });

      expect(response).toBeDefined();
      expect(response.content).toBeDefined();

      let text = '';
      if (typeof response.content === 'string') {
        text = response.content;
      } else if (Array.isArray(response.content)) {
        text = response.content
            .filter(item => item.type === 'text')
            .map(item => item.text)
            .join(' ');
      }
      expect(text.toLowerCase()).toContain('paris');

      await conversation.delete();
    });
  });

  xdescribe('Gemma 4 12B on GPU', () => {
    let engine: Engine | undefined;
    let initialized = false;

    beforeAll(async () => {
      try {
        const externalModelPath =
            '/models/gemma-4-12B-it-external-perlayer-prefilldecode.litertlm';
        engine = await Engine.create({
          model: externalModelPath,
          backend: Backend.GPU,
          mainExecutorSettings: {
            maxNumTokens: 128,
            backendConfig: {
              max_top_k: 3,
              external_tensor_mode: false,
            },
          },
        });
        initialized = true;
      } catch (e) {
        console.warn(
            'Failed to initialize Gemma 4 12B on GPU, skipping tests. Error:',
            e);
      }
    });

    afterAll(async () => {
      if (engine) {
        await engine.delete();
      }
    });

    xit('runs a conversation with Gemma 4 12B on GPU', async () => {
      if (!initialized) {
        pending('Gemma 4 12B model not available or GPU not supported');
        return;
      }
      expect(engine).toBeDefined();

      const conversation = await engine!.createConversation({
        sessionConfig: {
          maxOutputTokens: 32,
          samplerParams: {
            type: SamplerType.TOP_P,
            k: 1,
            p: 0.9,
          },
        },
      });

      const response = await conversation.sendMessage({
        role: 'user',
        content: 'What is the capital of France?',
      });

      expect(response).toBeDefined();
      expect(response.content).toBeDefined();

      let text = '';
      if (typeof response.content === 'string') {
        text = response.content;
      } else if (Array.isArray(response.content)) {
        text = response.content
            .filter(item => item.type === 'text')
            .map(item => item.text)
            .join(' ');
      }
      expect(text.toLowerCase()).toContain('paris');

      await conversation.delete();
    });
  });
});
