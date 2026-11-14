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

import {MemoryStorage} from '../dev/memory_storage.js';

import {buildModelCodeSnippet} from './code_snippet.js';
import {MODELS, SettingsStore} from './settings_store.js';

describe('buildModelCodeSnippet', () => {
  function makeSettings(): SettingsStore {
    return new SettingsStore(() => {}, new MemoryStorage());
  }

  it('generates minimal default multimodal GPU snippet for built-in hf:// models',
     () => {
       const settings = makeSettings();
       const code = buildModelCodeSnippet(settings);

       expect(code).toContain(
           'import {Backend, Engine} from \'@litert-lm/core\';');
       expect(code).toContain(
           `model: '${MODELS[0]!.path}',`);
       expect(code).toContain('// Reserves space for up to 8 images\n    maxNumImages: 8,');
       expect(code).toContain('// Enables image input\n  visionBackend: Backend.GPU,');
       expect(code).toContain('// Enables audio input\n  audioBackend: Backend.GPU,');
       expect(code).toContain('// Enables thinking mode\n    extra_context: {enable_thinking: true},');
       // Defaults are omitted:
       expect(code).not.toContain('backend: Backend.');
       expect(code).not.toContain('maxNumTokens:');
       expect(code).not.toContain('maxOutputTokens:');
       expect(code).not.toContain('samplerParams:');
       expect(code).not.toContain('ActivationDataType');
       expect(code).not.toContain('SpeculativeDecodingMode');
       expect(code).toContain(
           'for await (const chunk of conversation.sendMessageStreaming(\'Hello!\'))');
       expect(code).toContain('for (const part of chunk.content ?? [])');
     });

  it('omits visionBackend, audioBackend, maxNumImages, and createConversation args when text-only and thinking is off',
     () => {
       const settings = makeSettings();
       settings.visionBackend = 'off';
       settings.audioBackend = 'off';
       settings.enableThinking = false;

       const code = buildModelCodeSnippet(settings);

       expect(code).toContain('import {Engine} from \'@litert-lm/core\';');
       expect(code).not.toContain('Backend');
       expect(code).not.toContain('visionBackend');
       expect(code).not.toContain('audioBackend');
       expect(code).not.toContain('mainExecutorSettings');
       expect(code).not.toContain('maxNumImages');
       expect(code).toContain(
           'const conversation = await engine.createConversation();');
     });

  it('includes non-default contextLength, maxOutputTokens, Top-P sampler, max_top_k, systemPrompt, precisions, and speculative decoding',
     () => {
       const settings = makeSettings();
       settings.contextLength = 8192;
       settings.maxOutputTokens = 512;
       settings.samplerType = 'top_p';
       settings.topP = 0.9;
       settings.topK = 80;
       settings.temperature = 0.6;
       settings.systemPrompt = 'Reply concisely.\nDon\'t guess.';
       settings.enableThinking = false;
       settings.enableSpeculativeDecoding = true;
       settings.textPrecision = 'fp32';
       settings.visionPrecision = 'fp16';
       settings.audioBackend = 'cpu';
       settings.audioPrecision = 'fp16';  // Ignored on CPU.

       const code = buildModelCodeSnippet(settings);

       expect(code).toContain([
         'import {',
         '  ActivationDataType,',
         '  Backend,',
         '  Engine,',
         '  SamplerType,',
         '  SpeculativeDecodingMode,',
         '} from \'@litert-lm/core\';',
       ].join('\n'));
       expect(code).toContain(
           'activationDataType: ActivationDataType.FLOAT32,');
       expect(code).toContain('maxNumTokens: 8192,');
       expect(code).toContain('maxNumImages: 8,');
       expect(code).toContain('backendConfig: {max_top_k: 80},');
       expect(code).toContain(
           'visionActivationDataType: ActivationDataType.FLOAT16,');
       expect(code).toContain('audioBackend: Backend.CPU,');
       expect(code).not.toContain('audioActivationDataType');
       expect(code).toContain(
           'speculativeDecodingConfig: {mode: SpeculativeDecodingMode.ADAPTIVE},');
       expect(code).toContain('maxOutputTokens: 512,');
       expect(code).toContain('type: SamplerType.TOP_P,');
       expect(code).toContain('p: 0.9,');
       expect(code).toContain('k: 80,');
       expect(code).toContain('temperature: 0.6,');
       expect(code).toContain(
           '{role: \'system\', content: \'Reply concisely.\\nDon\\\'t guess.\'},');
       expect(code).not.toContain('enable_thinking');
     });

  it('emits Backend.CPU without maxNumImages for standard CPU text models and omits p for Top-K sampling',
     () => {
       const settings = makeSettings();
       settings.selectedModelPath =
           'https://example.com/models/gemma-3-1b-it.litertlm';
       settings.textBackend = 'cpu';
       settings.samplerType = 'top_k';
       settings.topK = 32;
       settings.temperature = 0.8;

       const code = buildModelCodeSnippet(settings);

       expect(code).toContain(
           'model: \'https://example.com/models/gemma-3-1b-it.litertlm\',');
       expect(code).toContain('backend: Backend.CPU,');
       expect(code).not.toContain('maxNumImages');
       expect(code).toContain('type: SamplerType.TOP_K,');
       expect(code).toContain('k: 32,');
       expect(code).not.toContain('p: ');
     });

  it('uses clear valid file placeholders for uploaded local files and local-dir:// models',
     () => {
       const settings = makeSettings();
       settings.customModels = [{
         name: 'my-local',
         filename: 'my-local.litertlm',
         path: 'https://local-model/my-local.litertlm',
         size: '1.20 GB',
       }];
       settings.selectedModelPath = 'https://local-model/my-local.litertlm';

       const localUploadCode = buildModelCodeSnippet(settings);
       expect(localUploadCode).toContain('// Local file:');
       expect(localUploadCode).toContain('model: \'./my-local.litertlm\',');
       expect(localUploadCode).not.toContain('https://local-model/');

       settings.localDirModels = [{
         name: 'mounted-gemma',
         filename: 'mounted-gemma.litertlm',
         path: 'local-dir://mounted-gemma.litertlm',
         size: '2.10 GB',
       }];
       settings.selectedModelPath = 'local-dir://mounted-gemma.litertlm';

       const localDirCode = buildModelCodeSnippet(settings);
       expect(localDirCode).toContain('// Local directory file:');
       expect(localDirCode).toContain('model: \'./mounted-gemma.litertlm\',');
       expect(localDirCode).not.toContain('local-dir://');
     });
});
