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

import {ActivationDataType, Backend, type GpuArtisanConfig, SamplerType, SpeculativeDecodingMode} from '@litert-lm/core';

import {getUserConversationConfig} from './chat_session_store.js';
import {CORE_GPU_ARTISAN_DEFAULTS, getUserEngineOptions} from './model_loader_service.js';
import {CustomModel, DEFAULT_MODEL_CONFIG, findModel, ModelSettings} from './settings_store.js';

/** Output format for the "Get code" popover. */
export type CodeSnippetVariant = 'npm'|'html';

/**
 * Unstamped template version present in local development builds (matching
 * `packages/core/package.json`) before a release build stamps a version.
 */
export const UNRELEASED_CORE_TEMPLATE_VERSION = '0.12.1';

/**
 * Pinned jsDelivr ESM URL for `@litert-lm/core`. Stamped to the release version
 * during release packaging alongside `LiteRtLm.DEFAULT_WASM_PATH`.
 */
export const LITERT_LM_CORE_CDN_URL =
    'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.12.1/+esm';

/** Extracts the `@litert-lm/core` version embedded in `cdnUrl`. */
export function getBuiltCoreVersion(
    cdnUrl: string = LITERT_LM_CORE_CDN_URL): string {
  const match = /@litert-lm\/core@([^/]+)\//.exec(cdnUrl);
  return match?.[1] ?? UNRELEASED_CORE_TEMPLATE_VERSION;
}

/**
 * Returns true when the app was built without a release version tag (e.g.
 * local development or untagged builds).
 */
export function isUnreleasedCoreBuild(
    cdnUrl: string = LITERT_LM_CORE_CDN_URL): boolean {
  return getBuiltCoreVersion(cdnUrl) === UNRELEASED_CORE_TEMPLATE_VERSION;
}

/** Returns the `npm install` command matching the built `@litert-lm/core`. */
export function getNpmInstallCommand(
    cdnUrl: string = LITERT_LM_CORE_CDN_URL): string {
  if (isUnreleasedCoreBuild(cdnUrl)) {
    return 'npm i @litert-lm/core';
  }
  return `npm i @litert-lm/core@${getBuiltCoreVersion(cdnUrl)}`;
}

/** Settings input accepted by `buildModelCodeSnippet`. */
export type CodeSnippetSettings = ModelSettings&Partial<{
  customModels: CustomModel[];
  localDirModels: CustomModel[];
}>;

/**
 * Formats a string as a valid single-quoted JS/TS string literal that is safe
 * to embed inside an HTML `<script>` element.
 */
function formatJsString(value: string): string {
  const escaped = value
                      .replace(/\\/g, '\\\\')
                      .replace(/'/g, '\\\'')
                      .replace(/</g, '\\u003c')
                      .replace(/\r/g, '\\r')
                      .replace(/\n/g, '\\n')
                      .replace(/\t/g, '\\t')
                      .replace(/\u2028/g, '\\u2028')
                      .replace(/\u2029/g, '\\u2029');
  return `'${escaped}'`;
}

function backendExpr(backend: Backend): string {
  return backend === Backend.CPU ? 'Backend.CPU' : 'Backend.GPU';
}

function activationExpr(dtype: ActivationDataType): string {
  return dtype === ActivationDataType.FLOAT32 ? 'ActivationDataType.FLOAT32' :
                                                'ActivationDataType.FLOAT16';
}

function samplerTypeExpr(type: SamplerType): string {
  if (type === SamplerType.TOP_K) return 'SamplerType.TOP_K';
  if (type === SamplerType.TOP_P) return 'SamplerType.TOP_P';
  return 'SamplerType.GREEDY';
}

function areEqualBackendValues(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((val, idx) => val === b[idx]);
  }
  return a === b;
}

function formatBackendConfigLines(config: GpuArtisanConfig): string[] {
  const fieldLines: string[] = [];
  const keys =
      Object.keys(CORE_GPU_ARTISAN_DEFAULTS) as Array<keyof GpuArtisanConfig>;
  for (const key of keys) {
    const val = config[key];
    if (areEqualBackendValues(val, CORE_GPU_ARTISAN_DEFAULTS[key])) {
      continue;
    }
    if (key === 'use_autosized_ringbuffers' && val === true) {
      fieldLines.push('      // Saves GPU memory on long contexts');
    }
    const valExpr = Array.isArray(val) ? `[${val.join(', ')}]` : String(val);
    fieldLines.push(`      ${key}: ${valExpr},`);
  }
  if (fieldLines.length === 0) {
    return [];
  }
  return ['    backendConfig: {', ...fieldLines, '    },'];
}

function resolveModelSourceSnippet(settings: CodeSnippetSettings): {
  comment?: string;
  valueExpr: string;
} {
  const path = settings.selectedModelPath || '';
  const matched = findModel(settings, path);
  if (path.startsWith('https://local-model/')) {
    const rawName = path.slice('https://local-model/'.length);
    let decoded = rawName;
    try {
      decoded = decodeURIComponent(rawName);
    } catch {
      // Keep rawName when URI decoding fails.
    }
    const filename = matched?.filename || decoded || 'model.litertlm';
    return {
      comment:
          '// Local file: pass a File, Blob, ReadableStream, URL, or hf:// URI',
      valueExpr: formatJsString(`./${filename}`),
    };
  }
  if (path.startsWith('local-dir://')) {
    const filename =
        matched?.filename || path.slice('local-dir://'.length) || 'model.litertlm';
    return {
      comment:
          '// Local directory file: pass a File, Blob, ReadableStream, URL, or hf:// URI',
      valueExpr: formatJsString(`./${filename}`),
    };
  }
  return {valueExpr: formatJsString(path)};
}

interface CoreSnippetParts {
  importLines: string[];
  createEngineLines: string[];
  createConversationLines: string[];
  enableThinking: boolean;
}

/**
 * Builds the shared `@litert-lm/core` imports, `Engine.create(...)`, and
 * `engine.createConversation(...)` code blocks used by both the `npm` and
 * `html` sample variants so they cannot drift.
 */
function buildCoreSnippetParts(settings: CodeSnippetSettings): CoreSnippetParts {
  const imports = new Set<string>(['Engine']);
  const engineOpts = getUserEngineOptions(settings);
  const convConfig = getUserConversationConfig(settings);
  const modelSource = resolveModelSourceSnippet(settings);

  const engineLines: string[] = [];
  if (modelSource.comment) {
    engineLines.push(`  ${modelSource.comment}`);
  }
  engineLines.push(`  model: ${modelSource.valueExpr},`);

  if (engineOpts.backend !== undefined) {
    imports.add('Backend');
    engineLines.push(`  backend: ${backendExpr(engineOpts.backend)},`);
  }
  if (engineOpts.activationDataType !== undefined) {
    imports.add('ActivationDataType');
    engineLines.push(
        `  activationDataType: ${activationExpr(engineOpts.activationDataType)},`);
  }

  const executorLines: string[] = [
    `    maxNumTokens: ${engineOpts.maxNumTokens},`,
  ];
  if (engineOpts.maxNumImages !== undefined) {
    const comment = engineOpts.maxNumImages > 0 ?
        `    // Reserves space for up to ${engineOpts.maxNumImages} images` :
        '    // Disables image cache allocation (defaults to 4)';
    executorLines.push(comment);
    executorLines.push(`    maxNumImages: ${engineOpts.maxNumImages},`);
  }
  if (engineOpts.backendConfig !== undefined) {
    executorLines.push(...formatBackendConfigLines(engineOpts.backendConfig));
  }
  engineLines.push('  mainExecutorSettings: {', ...executorLines, '  },');

  if (engineOpts.visionBackend !== undefined) {
    imports.add('Backend');
    engineLines.push('  // Enables image input');
    engineLines.push(
        `  visionBackend: ${backendExpr(engineOpts.visionBackend)},`);
  }
  if (engineOpts.visionActivationDataType !== undefined) {
    imports.add('ActivationDataType');
    engineLines.push(`  visionActivationDataType: ${
        activationExpr(engineOpts.visionActivationDataType)},`);
  }
  if (engineOpts.audioBackend !== undefined) {
    imports.add('Backend');
    engineLines.push('  // Enables audio input');
    engineLines.push(
        `  audioBackend: ${backendExpr(engineOpts.audioBackend)},`);
  }
  if (engineOpts.audioActivationDataType !== undefined) {
    imports.add('ActivationDataType');
    engineLines.push(`  audioActivationDataType: ${
        activationExpr(engineOpts.audioActivationDataType)},`);
  }
  if (engineOpts.speculativeDecodingConfig?.mode ===
      SpeculativeDecodingMode.ADAPTIVE) {
    imports.add('SpeculativeDecodingMode');
    engineLines.push(
        '  speculativeDecodingConfig: {mode: SpeculativeDecodingMode.ADAPTIVE},');
  }

  const sessionLines: string[] = [];
  const maxOutputTokens = convConfig.sessionConfig?.maxOutputTokens ??
      DEFAULT_MODEL_CONFIG.maxOutputTokens;
  sessionLines.push(`    maxOutputTokens: ${maxOutputTokens},`);

  const samplerParams = convConfig.sessionConfig?.samplerParams;
  if (samplerParams?.type !== undefined) {
    imports.add('SamplerType');
    if (samplerParams.type === SamplerType.GREEDY) {
      sessionLines.push('    samplerParams: {type: SamplerType.GREEDY},');
    } else {
      sessionLines.push('    samplerParams: {');
      sessionLines.push(`      type: ${samplerTypeExpr(samplerParams.type)},`);
      if (samplerParams.type === SamplerType.TOP_P &&
          samplerParams.p !== undefined) {
        sessionLines.push(`      p: ${samplerParams.p},`);
      }
      if (samplerParams.k !== undefined) {
        sessionLines.push(`      k: ${samplerParams.k},`);
      }
      if (samplerParams.temperature !== undefined) {
        sessionLines.push(`      temperature: ${samplerParams.temperature},`);
      }
      sessionLines.push('    },');
    }
  }

  const prefaceLines: string[] = [];
  const systemMsg = convConfig.preface?.messages?.[0];
  if (systemMsg && typeof systemMsg.content === 'string') {
    prefaceLines.push('    messages: [');
    prefaceLines.push(
        `      {role: 'system', content: ${formatJsString(systemMsg.content)}},`);
    prefaceLines.push('    ],');
  }
  const enableThinking =
      Boolean(convConfig.preface?.extra_context?.['enable_thinking']);
  if (enableThinking) {
    prefaceLines.push('    // Enables thinking mode');
    prefaceLines.push('    extra_context: {enable_thinking: true},');
  }

  const convLines: string[] = [
    '  sessionConfig: {',
    ...sessionLines,
    '  },',
  ];
  if (prefaceLines.length > 0) {
    convLines.push('  preface: {', ...prefaceLines, '  },');
  }

  const sortedImports = Array.from(imports).sort();
  const singleLineImport =
      `import {${sortedImports.join(', ')}} from '@litert-lm/core';`;
  const importLines = singleLineImport.length <= 68 ?
      [singleLineImport] :
      [
        'import {',
        ...sortedImports.map(sym => `  ${sym},`),
        '} from \'@litert-lm/core\';',
      ];
  return {
    importLines,
    createEngineLines: [
      'const engine = await Engine.create({',
      ...engineLines,
      '});',
    ],
    createConversationLines: [
      'const conversation = await engine.createConversation({',
      ...convLines,
      '});',
    ],
    enableThinking,
  };
}

function indentLines(lines: readonly string[], prefix: string): string[] {
  return lines.map(line => (line ? `${prefix}${line}` : ''));
}

/**
 * Generates a runnable `@litert-lm/core` code sample (either an `npm` module
 * snippet or a self-contained `.html` file with a jsDelivr `<script
 * type="importmap">`) that reproduces the exact configuration the app runs.
 */
export function buildModelCodeSnippet(
    settings: CodeSnippetSettings,
    variant: CodeSnippetVariant = 'npm',
    cdnUrl: string = LITERT_LM_CORE_CDN_URL): string {
  const {importLines, createEngineLines, createConversationLines, enableThinking} =
      buildCoreSnippetParts(settings);

  if (variant === 'npm') {
    const streamBodyLines: string[] = [];
    if (enableThinking) {
      streamBodyLines.push(
          '  // Reasoning chunks stream on chunk.channels?.[\'thought\'].');
    }
    streamBodyLines.push(
        '  for (const part of chunk.content ?? []) {',
        '    if (part.type === \'text\') {',
        '      console.log(part.text);',
        '    }',
        '  }');

    return [
      ...importLines,
      '',
      ...createEngineLines,
      '',
      ...createConversationLines,
      '',
      'for await (const chunk of conversation.sendMessageStreaming(\'Hello!\')) {',
      ...streamBodyLines,
      '}',
      '',
      'await conversation.delete();',
      'await engine.delete();',
    ].join('\n');
  }

  const htmlStreamLines: string[] = [];
  if (enableThinking) {
    htmlStreamLines.push(
        '          // Reasoning chunks stream on chunk.channels?.[\'thought\'].');
  }
  htmlStreamLines.push(
      '          for (const part of chunk.content ?? []) {',
      '            if (part.type === \'text\') {',
      '              output.textContent += part.text;',
      '            }',
      '          }');

  return [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head>',
    '  <meta charset="UTF-8" />',
    '  <title>LiteRT-LM Web Demo</title>',
    '  <style>pre { white-space: pre-wrap; }</style>',
    '  <script type="importmap">',
    '    {',
    '      "imports": {',
    `        "@litert-lm/core": ${
        JSON.stringify(cdnUrl).replace(/</g, '\\u003c')}`,
    '      }',
    '    }',
    '  </script>',
    '</head>',
    '<body>',
    '  <form id="prompt-form">',
    '    <input id="prompt" value="Hello!" placeholder="Ask something..." required />',
    '    <button id="send" type="submit" disabled>Loading model...</button>',
    '  </form>',
    '  <pre id="output"></pre>',
    '',
    '  <script type="module">',
    ...indentLines(importLines, '    '),
    '',
    '    const form = document.getElementById(\'prompt-form\');',
    '    const input = document.getElementById(\'prompt\');',
    '    const sendBtn = document.getElementById(\'send\');',
    '    const output = document.getElementById(\'output\');',
    '',
    ...indentLines(createEngineLines, '    '),
    '',
    ...indentLines(createConversationLines, '    '),
    '',
    '    sendBtn.disabled = false;',
    '    sendBtn.textContent = \'Send\';',
    '',
    '    form.addEventListener(\'submit\', async (e) => {',
    '      e.preventDefault();',
    '      const prompt = input.value.trim();',
    '      if (!prompt) return;',
    '      sendBtn.disabled = true;',
    '      output.textContent = \'\';',
    '      try {',
    '        for await (const chunk of conversation.sendMessageStreaming(prompt)) {',
    ...htmlStreamLines,
    '        }',
    '      } finally {',
    '        sendBtn.disabled = false;',
    '      }',
    '    });',
    '  </script>',
    '</body>',
    '</html>',
  ].join('\n');
}
