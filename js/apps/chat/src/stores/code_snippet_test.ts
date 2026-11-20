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

import {ActivationDataType, Backend, ConversationConfig, EngineSettings, GpuArtisanConfig, LiteRtLm, SamplerType, SpeculativeDecodingMode} from '@litert-lm/core';
import {EngineFake} from '@litert-lm/core/testing';

import {MemoryStorage} from '../dev/memory_storage.js';
import {parseHtmlDocument} from '../testing/parse_html.js';
import {getUserConversationConfig} from './chat_session_store.js';
import {buildModelCodeSnippet, getBuiltCoreVersion, getNpmInstallCommand, isUnreleasedCoreBuild, LITERT_LM_CORE_CDN_URL} from './code_snippet.js';
import {CORE_GPU_ARTISAN_DEFAULTS, ModelLoaderService} from './model_loader_service.js';
import {MODELS, SettingsStore} from './settings_store.js';

const ENUM_SUBSTITUTIONS: Readonly<Record<string, number>> = {
  'Backend.CPU': Backend.CPU,
  'Backend.GPU': Backend.GPU,
  'Backend.GPU_ARTISAN': Backend.GPU_ARTISAN,
  'ActivationDataType.FLOAT16': ActivationDataType.FLOAT16,
  'ActivationDataType.FLOAT32': ActivationDataType.FLOAT32,
  'SamplerType.GREEDY': SamplerType.GREEDY,
  'SamplerType.TOP_K': SamplerType.TOP_K,
  'SamplerType.TOP_P': SamplerType.TOP_P,
  'SpeculativeDecodingMode.OFF': SpeculativeDecodingMode.OFF,
  'SpeculativeDecodingMode.ON': SpeculativeDecodingMode.ON,
  'SpeculativeDecodingMode.ADAPTIVE': SpeculativeDecodingMode.ADAPTIVE,
};

/**
 * Extracts the balanced `{...}` object literal passed to `callPrefix` (e.g.
 * `'Engine.create('`) and parses it into a plain JS object without `eval`.
 */
function extractSnippetCallArg(code: string, callPrefix: string): unknown {
  const startCall = code.indexOf(callPrefix);
  if (startCall === -1) return undefined;
  const afterParen = code.slice(startCall + callPrefix.length).trimStart();
  if (afterParen.startsWith(')')) return undefined;

  // Walk balanced braces while respecting single-quoted strings and // comments.
  let depth = 0;
  let inString = false;
  let endIdx = -1;
  for (let i = 0; i < afterParen.length; i++) {
    const ch = afterParen[i]!;
    if (inString) {
      if (ch === '\\') {
        i++;
      } else if (ch === '\'') {
        inString = false;
      }
      continue;
    }
    if (ch === '/' && afterParen[i + 1] === '/') {
      const nl = afterParen.indexOf('\n', i);
      if (nl === -1) break;
      i = nl;
      continue;
    }
    if (ch === '\'') {
      inString = true;
    } else if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        endIdx = i + 1;
        break;
      }
    }
  }
  if (endIdx === -1) {
    throw new Error(`Unclosed object literal after ${callPrefix}`);
  }
  const objSrc = afterParen.slice(0, endIdx);

  // Convert JS object literal (single-quoted strings, // comments, enum refs,
  // unquoted keys, trailing commas) into strict JSON.
  let jsonLike = '';
  for (let i = 0; i < objSrc.length; i++) {
    const ch = objSrc[i]!;
    if (ch === '/' && objSrc[i + 1] === '/') {
      const nl = objSrc.indexOf('\n', i);
      if (nl === -1) break;
      i = nl - 1;
      continue;
    }
    if (ch === '\'') {
      let strOut = '"';
      i++;
      while (i < objSrc.length && objSrc[i] !== '\'') {
        const sc = objSrc[i]!;
        if (sc === '\\') {
          const next = objSrc[i + 1] ?? '';
          if (next === '\'') {
            strOut += '\'';
          } else {
            strOut += '\\' + next;
          }
          i += 2;
          continue;
        }
        if (sc === '"') {
          strOut += '\\"';
        } else {
          strOut += sc;
        }
        i++;
      }
      strOut += '"';
      jsonLike += strOut;
      continue;
    }
    jsonLike += ch;
  }

  for (const [expr, num] of Object.entries(ENUM_SUBSTITUTIONS)) {
    jsonLike = jsonLike.split(expr).join(String(num));
  }
  // Quote unquoted object property keys (only outside double-quoted strings).
  let quoted = '';
  let inDouble = false;
  for (let i = 0; i < jsonLike.length; i++) {
    const ch = jsonLike[i]!;
    if (inDouble) {
      quoted += ch;
      if (ch === '\\') {
        quoted += jsonLike[++i] ?? '';
      } else if (ch === '"') {
        inDouble = false;
      }
      continue;
    }
    if (ch === '"') {
      inDouble = true;
      quoted += ch;
      continue;
    }
    const keyMatch = /^([A-Za-z_$][A-Za-z0-9_$]*)\s*:/.exec(jsonLike.slice(i));
    if (keyMatch) {
      const prev = quoted.trimEnd().slice(-1);
      if (prev === '{' || prev === ',') {
        quoted += `"${keyMatch[1]}":`;
        i += keyMatch[0].length - 1;
        continue;
      }
    }
    quoted += ch;
  }
  const withoutTrailingCommas = quoted.replace(/,(\s*[}\]])/g, '$1');
  return JSON.parse(withoutTrailingCommas);
}

/**
 * `@litert-lm/core`'s C++ default for `LlmExecutorSettings.maxNumImages`
 * (`kDefaultMaxNumImages` in `llm_executor_settings.h`), which applies even
 * when `visionBackend` is unset.
 */
const CORE_DEFAULT_MAX_NUM_IMAGES = 4;

/**
 * Resolves `@litert-lm/core`'s effective `EngineSettings` (applying C++
 * `CORE_DEFAULT_MAX_NUM_IMAGES` and `CORE_GPU_ARTISAN_DEFAULTS` on GPU text and
 * stripping `model` / `benchmarkEnabled`) so app options and snippet options
 * can be compared.
 */
function normalizeEffectiveEngineOptions(raw: EngineSettings) {
  const isCpu = raw.backend === Backend.CPU;
  const exec = raw.mainExecutorSettings ?? {};
  const backendConfig: GpuArtisanConfig|undefined = isCpu ?
      undefined :
      {
        ...CORE_GPU_ARTISAN_DEFAULTS,
        supported_lora_ranks: [...CORE_GPU_ARTISAN_DEFAULTS.supported_lora_ranks],
        ...(exec.backendConfig as Partial<GpuArtisanConfig> | undefined),
      };
  return {
    backend: raw.backend,
    activationDataType: raw.activationDataType,
    mainExecutorSettings: {
      maxNumTokens: exec.maxNumTokens,
      maxNumImages:
          isCpu ? exec.maxNumImages :
                  (exec.maxNumImages ?? CORE_DEFAULT_MAX_NUM_IMAGES),
      backendConfig,
    },
    visionBackend: raw.visionBackend,
    visionActivationDataType: raw.visionActivationDataType,
    audioBackend: raw.audioBackend,
    audioActivationDataType: raw.audioActivationDataType,
    speculativeDecodingConfig: raw.speculativeDecodingConfig,
  };
}

/**
 * Resolves `@litert-lm/core`'s effective `ConversationConfig` (`GREEDY` ignores
 * `k`/`p`/`temperature`; `TOP_K` ignores `p`; `enable_thinking` defaults to
 * `false` in the model chat template when omitted).
 */
function normalizeEffectiveConversationConfig(raw: ConversationConfig|undefined) {
  const session = raw?.sessionConfig ?? {};
  const sp = session.samplerParams;
  let normalizedSampler: Record<string, unknown>|undefined;
  if (sp?.type === SamplerType.GREEDY) {
    normalizedSampler = {type: SamplerType.GREEDY};
  } else if (sp?.type === SamplerType.TOP_K) {
    normalizedSampler = {
      type: SamplerType.TOP_K,
      k: sp.k,
      temperature: sp.temperature,
    };
  } else if (sp?.type === SamplerType.TOP_P) {
    normalizedSampler = {
      type: SamplerType.TOP_P,
      p: sp.p,
      k: sp.k,
      temperature: sp.temperature,
    };
  }
  const messages = raw?.preface?.messages;
  return {
    sessionConfig: {
      maxOutputTokens: session.maxOutputTokens,
      samplerParams: normalizedSampler,
    },
    preface: {
      messages: messages && messages.length > 0 ? messages : undefined,
      enableThinking: Boolean(raw?.preface?.extra_context?.['enable_thinking']),
    },
    filterChannelContentFromKvCache: raw?.filterChannelContentFromKvCache,
  };
}

describe('buildModelCodeSnippet', () => {
  function makeSettings(): SettingsStore {
    return new SettingsStore(() => {}, new MemoryStorage());
  }

  it('generates default multimodal GPU snippet matching the app config', () => {
    const settings = makeSettings();
    const code = buildModelCodeSnippet(settings);

    expect(code).toContain(
        'import {Backend, Engine, SamplerType} from \'@litert-lm/core\';');
    expect(code).toContain(`model: '${MODELS[0]!.path}',`);
    expect(code).toContain('maxNumTokens: 4096,');
    expect(code).toContain(
        '// Reserves space for up to 8 images\n    maxNumImages: 8,');
    expect(code).toContain('num_decode_steps_per_sync: 3,');
    expect(code).toContain(
        '// Saves GPU memory on long contexts\n      use_autosized_ringbuffers: true,');
    // backendConfig fields equal to CORE_GPU_ARTISAN_DEFAULTS are omitted:
    expect(code).not.toContain('wait_for_weight_uploads:');
    expect(code).not.toContain('max_top_k:');
    expect(code).not.toContain('use_submodel:');
    expect(code).not.toContain('num_output_candidates:');
    expect(code).toContain(
        '// Enables image input\n  visionBackend: Backend.GPU,');
    expect(code).toContain(
        '// Enables audio input\n  audioBackend: Backend.GPU,');
    expect(code).toContain('maxOutputTokens: 2048,');
    expect(code).toContain('samplerParams: {type: SamplerType.GREEDY},');
    expect(code).toContain(
        '// Enables thinking mode\n    extra_context: {enable_thinking: true},');
    expect(code).not.toContain('backend: Backend.');
    expect(code).not.toContain('ActivationDataType');
    expect(code).not.toContain('SpeculativeDecodingMode');
    expect(code).toContain(
        'for await (const chunk of conversation.sendMessageStreaming(\'Hello!\'))');
    expect(code).toContain('for (const part of chunk.content ?? [])');
  });

  it('sets maxNumImages to 0 and omits visionBackend, audioBackend, use_autosized_ringbuffers, and preface when disabled',
     () => {
       const settings = makeSettings();
       settings.visionBackend = 'off';
       settings.audioBackend = 'off';
       settings.useAutosizedRingbuffers = false;
       settings.enableThinking = false;

       const code = buildModelCodeSnippet(settings);

       expect(code).toContain(
           'import {Engine, SamplerType} from \'@litert-lm/core\';');
       expect(code).not.toContain('Backend');
       expect(code).not.toContain('visionBackend');
       expect(code).not.toContain('audioBackend');
       expect(code).toContain(
           '// Disables image cache allocation (defaults to 4)\n    maxNumImages: 0,');
       expect(code).not.toContain('use_autosized_ringbuffers');
       expect(code).not.toContain('wait_for_weight_uploads');
       expect(code).toContain('num_decode_steps_per_sync: 3,');
       expect(code).not.toContain('preface:');
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
       expect(code).toContain('max_top_k: 80,');
       expect(code).not.toContain('wait_for_weight_uploads');
       expect(code).toContain('num_decode_steps_per_sync: 3,');
       expect(code).toContain('use_autosized_ringbuffers: true,');
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

  it('emits Backend.CPU without maxNumImages or backendConfig for CPU text models and omits p for Top-K sampling',
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
       expect(code).not.toContain('backendConfig');
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

  it('keeps effective Engine.create and createConversation options identical between ModelLoaderService/ChatSessionStore and the generated snippet',
     async () => {
       const fakeCache = {
         match: jasmine.createSpy().and.callFake(
             async () => ({
               headers: new Headers({'content-length': '3'}),
               body: new Blob([new Uint8Array([1, 2, 3])]).stream(),
             })),
       } as unknown as Cache;
       if (!(window as unknown as Record<string, unknown>)['caches']) {
         (window as unknown as Record<string, unknown>)['caches'] = {
           open: async () => fakeCache,
           delete: async () => true,
         };
       }
       spyOn(window.caches, 'open').and.resolveTo(fakeCache);

       const scenarios: Array<{name: string; configure: (s: SettingsStore) => void}> =
           [
             {
               name: 'default Gemma 4 E2B GPU config',
               configure: (s) => {
                 s.selectedModelPath = 'https://example.com/gemma-4-E2B-it-gpu.litertlm';
               },
             },
             {
               name: 'GPU text with Ring buffers disabled and Vision/Audio off',
               configure: (s) => {
                 s.selectedModelPath = 'https://example.com/gemma-4-E2B-it-gpu.litertlm';
                 s.useAutosizedRingbuffers = false;
                 s.visionBackend = 'off';
                 s.audioBackend = 'off';
                 s.enableThinking = false;
               },
             },
             {
               name: 'Top-P sampler with topK=40 (< 64), custom system prompt, and speculative decoding',
               configure: (s) => {
                 s.selectedModelPath = 'https://example.com/gemma-4-E2B-it-gpu.litertlm';
                 s.contextLength = 8192;
                 s.maxOutputTokens = 256;
                 s.samplerType = 'top_p';
                 s.topP = 0.9;
                 s.topK = 40;
                 s.temperature = 0.6;
                 s.systemPrompt = 'Always begin with PONG.';
                 s.enableSpeculativeDecoding = true;
                 s.textPrecision = 'fp32';
                 s.visionPrecision = 'fp16';
                 s.audioBackend = 'cpu';
               },
             },
             {
               name: 'Top-K sampler with topK=96 (> 64)',
               configure: (s) => {
                 s.selectedModelPath = 'https://example.com/gemma-4-E2B-it-gpu.litertlm';
                 s.samplerType = 'top_k';
                 s.topK = 96;
                 s.temperature = 0.75;
               },
             },
             {
               name: 'standard .litertlm on CPU text',
               configure: (s) => {
                 s.selectedModelPath = 'https://example.com/gemma-3-1b-it.litertlm';
                 s.textBackend = 'cpu';
                 s.visionBackend = 'off';
                 s.audioBackend = 'cpu';
                 s.useAutosizedRingbuffers = false;
               },
             },
           ];

       for (const scenario of scenarios) {
         const settings = makeSettings();
         scenario.configure(settings);

         const createEngineSpy =
             jasmine.createSpy('createEngine')
                 .and.callFake(
                     async (opts, hint) => await EngineFake.create(opts, hint));
         const loader = new ModelLoaderService(
             () => {}, () => {}, undefined, createEngineSpy,
             async () => ({} as LiteRtLm));
         await loader.loadModelWeights(settings.modelSettings, async () => {});

         const appEngineCall = createEngineSpy.calls.mostRecent().args[0];
         const appConvConfig = getUserConversationConfig(settings.modelSettings);

         for (const variant of ['npm', 'html'] as const) {
           const code = buildModelCodeSnippet(settings, variant);
           const snippetEngineCall =
               extractSnippetCallArg(code, 'Engine.create(') as EngineSettings |
               undefined;
           const snippetConvCall =
               extractSnippetCallArg(code, 'engine.createConversation(') as
               ConversationConfig |
               undefined;

           expect(snippetEngineCall)
               .withContext(`${scenario.name} [${variant}]: Engine.create parsed`)
               .toBeDefined();
           expect(snippetEngineCall!.model)
               .withContext(`${scenario.name} [${variant}]: model path`)
               .toBe(settings.selectedModelPath);
           expect(normalizeEffectiveEngineOptions(snippetEngineCall!))
               .withContext(`${scenario.name} [${variant}]: engine options parity`)
               .toEqual(normalizeEffectiveEngineOptions(appEngineCall));
           expect(normalizeEffectiveConversationConfig(snippetConvCall))
               .withContext(
                   `${scenario.name} [${variant}]: conversation options parity`)
               .toEqual(normalizeEffectiveConversationConfig(appConvConfig));
         }
       }
     });

  it('generates a complete self-contained HTML + importmap sample and tracks stamped vs unreleased versions',
     () => {
       const settings = makeSettings();
       const htmlCode = buildModelCodeSnippet(settings, 'html');

       expect(htmlCode).toContain('<!DOCTYPE html>');
       expect(htmlCode).toContain('<script type="importmap">');
       expect(htmlCode).toContain(
           `"@litert-lm/core": "${LITERT_LM_CORE_CDN_URL}"`);
       expect(htmlCode).not.toContain('@latest');
       expect(htmlCode).toContain('<form id="prompt-form">');
       expect(htmlCode).toContain('<input id="prompt"');
       expect(htmlCode).toContain('<button id="send"');
       expect(htmlCode).toContain('<pre id="output"></pre>');
       expect(htmlCode).toContain('<script type="module">');
       expect(htmlCode).toContain('output.textContent += part.text;');

       // Unstamped template build vs stamped release version:
       expect(isUnreleasedCoreBuild()).toBeTrue();
       expect(getNpmInstallCommand()).toBe('npm i @litert-lm/core');

       const stampedUrl =
           'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.20.0/+esm';
       expect(getBuiltCoreVersion(stampedUrl)).toBe('0.20.0');
       expect(isUnreleasedCoreBuild(stampedUrl)).toBeFalse();
       expect(getNpmInstallCommand(stampedUrl))
           .toBe('npm i @litert-lm/core@0.20.0');
       expect(buildModelCodeSnippet(settings, 'html', stampedUrl))
           .toContain(`"@litert-lm/core": "${stampedUrl}"`);
     });

  it('escapes < in systemPrompt, model filenames, and importmap URLs so generated HTML cannot break out of <script> tags',
     () => {
       const settings = makeSettings();
       const maliciousFilename =
           'model</script><img src=x onerror=alert(1)><!--.litertlm';
       const maliciousPrompt =
           'Ignore </script><img src=x onerror=alert(2)> and <!-- </SCRIPT>';
       const maliciousCdnUrl =
           'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.20.0/+esm#</script><img onerror=alert(3)><!--';

       settings.customModels = [{
         name: 'malicious',
         filename: maliciousFilename,
         path: `https://local-model/${encodeURIComponent(maliciousFilename)}`,
         size: '1.00 GB',
       }];
       settings.selectedModelPath = settings.customModels[0]!.path;
       settings.systemPrompt = maliciousPrompt;

       const htmlCode =
           buildModelCodeSnippet(settings, 'html', maliciousCdnUrl);
       const doc = parseHtmlDocument(htmlCode);
       const scripts = Array.from(doc.querySelectorAll('script'));
       expect(scripts.length).toBe(2);
       expect(scripts[0]!.getAttribute('type')).toBe('importmap');
       expect(scripts[1]!.getAttribute('type')).toBe('module');
       expect(doc.querySelectorAll('img').length).toBe(0);

       // Importmap JSON round-trips the exact CDN URL without breaking script.
       const importMap = JSON.parse(scripts[0]!.textContent!) as {
         imports: Record<string, string>;
       };
       expect(importMap.imports['@litert-lm/core']).toBe(maliciousCdnUrl);
       expect(scripts[0]!.textContent).not.toMatch(/<\/script|<!--/i);

       // Module script preserves exact string values without raw < sequences.
       const moduleScript = scripts[1]!.textContent!;
       expect(moduleScript).not.toMatch(/<\/script|<!--/i);
       const htmlEngineCall =
           extractSnippetCallArg(moduleScript, 'Engine.create(') as
           EngineSettings;
       const htmlConvCall =
           extractSnippetCallArg(moduleScript, 'engine.createConversation(') as
           ConversationConfig;
       expect(htmlEngineCall.model).toBe(`./${maliciousFilename}`);
       expect(htmlConvCall.preface?.messages?.[0]?.content)
           .toBe(maliciousPrompt);

       // npm variant uses the same valid JS \u003c escape and preserves values.
       const npmCode = buildModelCodeSnippet(settings, 'npm', maliciousCdnUrl);
       expect(npmCode).not.toMatch(/<\/script|<!--/i);
       expect(npmCode).toContain('\\u003c/script>');
       expect(npmCode).toContain('\\u003c!--');
       const npmEngineCall =
           extractSnippetCallArg(npmCode, 'Engine.create(') as EngineSettings;
       const npmConvCall =
           extractSnippetCallArg(npmCode, 'engine.createConversation(') as
           ConversationConfig;
       expect(npmEngineCall.model).toBe(`./${maliciousFilename}`);
       expect(npmConvCall.preface?.messages?.[0]?.content).toBe(maliciousPrompt);
     });
});
