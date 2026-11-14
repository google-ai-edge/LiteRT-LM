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

import {ActivationDataType, Backend, SamplerType, SpeculativeDecodingMode} from '@litert-lm/core';

import {getUserConversationConfig} from './chat_session_store.js';
import {getUserEngineOptions} from './model_loader_service.js';
import {CustomModel, DEFAULT_MODEL_CONFIG, findModel, ModelSettings} from './settings_store.js';

/** Settings input accepted by `buildModelCodeSnippet`. */
export type CodeSnippetSettings = ModelSettings&Partial<{
  customModels: CustomModel[];
  localDirModels: CustomModel[];
}>;

/** Formats a string as a valid single-quoted JS/TS string literal. */
function formatJsString(value: string): string {
  const escaped = value
                      .replace(/\\/g, '\\\\')
                      .replace(/'/g, '\\\'')
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

/**
 * Generates a runnable `@litert-lm/core` snippet matching the active model and
 * user-configured settings, omitting options left at their defaults.
 */
export function buildModelCodeSnippet(settings: CodeSnippetSettings): string {
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

  const executorLines: string[] = [];
  if (engineOpts.maxNumTokens !== DEFAULT_MODEL_CONFIG.contextLength) {
    executorLines.push(`    maxNumTokens: ${engineOpts.maxNumTokens},`);
  }
  if (engineOpts.maxNumImages !== undefined) {
    executorLines.push('    // Reserves space for up to 8 images');
    executorLines.push(`    maxNumImages: ${engineOpts.maxNumImages},`);
  }
  if (engineOpts.backend !== Backend.CPU &&
      (settings.samplerType ?? DEFAULT_MODEL_CONFIG.samplerType) !==
          DEFAULT_MODEL_CONFIG.samplerType &&
      engineOpts.maxTopK > DEFAULT_MODEL_CONFIG.topK) {
    executorLines.push(
        `    backendConfig: {max_top_k: ${engineOpts.maxTopK}},`);
  }
  if (executorLines.length > 0) {
    engineLines.push('  mainExecutorSettings: {', ...executorLines, '  },');
  }

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
  const maxOutputTokens = convConfig.sessionConfig?.maxOutputTokens;
  if (maxOutputTokens !== undefined &&
      maxOutputTokens !== DEFAULT_MODEL_CONFIG.maxOutputTokens) {
    sessionLines.push(`    maxOutputTokens: ${maxOutputTokens},`);
  }
  const samplerParams = convConfig.sessionConfig?.samplerParams;
  const activeSamplerType =
      settings.samplerType ?? DEFAULT_MODEL_CONFIG.samplerType;
  if (samplerParams?.type !== undefined &&
      activeSamplerType !== DEFAULT_MODEL_CONFIG.samplerType) {
    imports.add('SamplerType');
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

  const convLines: string[] = [];
  if (sessionLines.length > 0) {
    convLines.push('  sessionConfig: {', ...sessionLines, '  },');
  }
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
  const createConvBlock = convLines.length > 0 ?
      [
        'const conversation = await engine.createConversation({',
        ...convLines,
        '});',
      ] :
      ['const conversation = await engine.createConversation();'];

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
    'const engine = await Engine.create({',
    ...engineLines,
    '});',
    '',
    ...createConvBlock,
    '',
    'for await (const chunk of conversation.sendMessageStreaming(\'Hello!\')) {',
    ...streamBodyLines,
    '}',
    '',
    'await conversation.delete();',
    'await engine.delete();',
  ].join('\n');
}
