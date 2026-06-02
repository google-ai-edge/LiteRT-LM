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

import {Cleanup} from './cleanup.js';
import {Conversation} from './conversation.js';
import {ConversationConfig} from './conversation_config.js';
import {EngineSettings, fillWasmEngineSettingsFromEngineSettings, wasmEngineSettingsToEngineSettings} from './engine_settings.js';
import {getGlobalLiteRtLm} from './global_litertlm.js';
import {getOrLoadGlobalLiteRtLm} from './load_litertlm.js';
import {Mutex} from './mutex.js';
import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';
import {Session} from './session.js';
import {sessionConfigToWasmSessionConfig} from './session_config.js';
import {RecursiveRequired} from './types.js';
import {Backend, ConversationConfig as WasmConversationConfig, Deletable, Engine as WasmEngine, LiteRtLmWasm, ModelAssets, SessionConfig as WasmSessionConfig} from './wasm_binding_types.js';

/**
 * Global index to avoid writing to the sane VFS path.
 */
// TODO: b/477709280 - Remove this when streaming loading works.
let modelPathIndex = 0;

/**
 * LiteRT-LM Engine
 */
export class Engine implements Deletable {
  readonly settings: RecursiveRequired<EngineSettings>;
  private mutexes = {
    executor: new Mutex(),
  };

  private constructor(
      private readonly wasm: LiteRtLmWasm,
      private readonly engine: WasmEngine,
      modelSource: EngineSettings['model'],
      private readonly deleteCallback: () => void,
  ) {
    const wasmEngineSettings = engine.getEngineSettings();
    const settingsWithoutModel =
        wasmEngineSettingsToEngineSettings(wasmEngineSettings);
    this.settings = {
      ...settingsWithoutModel,
      model: modelSource,
    };
    wasmEngineSettings.delete();
  }

  static async create(engineSettings: EngineSettings, inputPromptAsHint = ''):
      Promise<Engine> {
    const litertlm = await getOrLoadGlobalLiteRtLm();
    const wasm = litertlm.liteRtLmWasm;
    // TODO: b/477709280 - Support other formats like .task.
    const dstPath = `/model_${modelPathIndex++}.litertlm`;

    // Default to GPU_ARTISAN if not specified.
    const backend = engineSettings.backend ?? Backend.GPU_ARTISAN;
    engineSettings = {...engineSettings, backend};

    const samplerBackend = engineSettings.mainExecutorSettings?.samplerBackend;
    if (backend === Backend.GPU || backend === Backend.GPU_ARTISAN ||
        samplerBackend === Backend.GPU ||
        samplerBackend === Backend.GPU_ARTISAN) {
      await litertlm.setupDefaultWebGpuDevice();
    }

    const cleanup = new Cleanup();

    const modelStream = await modelToStream(engineSettings.model);
    let engine: WasmEngine;
    try {
      let modelAssets: ModelAssets;
      const isStreaming = backend === Backend.GPU_ARTISAN;

      if (isStreaming) {
        // GPU Artisan supports streamed loading.
        const streamWrapper =
            new ReadableStreamDataStreamWrapper(modelStream, () => wasm.HEAPU8);
        const dataStream = wasm.ReadableStreamDataStream.create(streamWrapper);
        cleanup.add(() => {
          dataStream.delete();
        });
        modelAssets = wasm.ModelAssets.createStreaming(dataStream);
      } else {
        // Other backends must be fully loaded into Wasm memory first.
        await loadModelToVfs(wasm, modelStream, dstPath);
        cleanup.add(() => {
          try {
            wasm.FS.unlink(dstPath);
          } catch (e) {
            console.error(`Error removing file from VFS:`, e);
          }
        });

        modelAssets = wasm.ModelAssets.create(dstPath);
      }

      const cleanupModelAssets = cleanup.add(() => {
        modelAssets.delete();
      });

      const wasmEngineSettings =
          wasm.EngineSettings.createDefault(modelAssets, {value: backend});
      // Delete our copy of the ModelAssets object (not the underlying file).
      cleanupModelAssets();

      const cleanupWasmEngineSettings = cleanup.add(() => {
        wasmEngineSettings.delete();
      });

      fillWasmEngineSettingsFromEngineSettings(
          wasmEngineSettings, engineSettings, backend, wasm);
      wasmEngineSettings.setParallelFileSectionLoading(false);
      wasmEngineSettings.setSingleThreadedExecution(true);

      if (isStreaming) {
        engine = await wasm.Engine.createStreaming(
            wasmEngineSettings, inputPromptAsHint);
      } else {
        engine = await wasm.Engine.createEngine(
            wasmEngineSettings, inputPromptAsHint);
      }
      cleanupWasmEngineSettings();

      cleanup.add(() => {
        engine.delete();
      });
      return new Engine(
          wasm, engine, engineSettings.model, () => cleanup.run());
    } catch (e) {
      cleanup.run();
      throw e;
    }
  }

  async createSession(sessionConfig = {}): Promise<Session> {
    return this.mutexes.executor.acquireAndRun(() => {
      const wasmSessionConfig =
          sessionConfigToWasmSessionConfig(sessionConfig, this.wasm);

      const wasmSession = this.engine.createSession(wasmSessionConfig);
      wasmSessionConfig.delete();
      return new Session(wasmSession, this.mutexes);
    });
  }

  async createConversation(config?: ConversationConfig): Promise<Conversation> {
    return this.mutexes.executor.acquireAndRun(() => {
      let wasmConfig: WasmConversationConfig;
      let wasmSessionConfig: WasmSessionConfig|undefined;

      if (config) {
        wasmSessionConfig = sessionConfigToWasmSessionConfig(
            config.sessionConfig || {}, this.wasm);
        const prefaceJson =
            config.preface ? JSON.stringify(config.preface) : '';
        wasmConfig = this.wasm.ConversationConfig.createCustom(
            this.engine, wasmSessionConfig, !!config.enableConstrainedDecoding,
            !!config.prefillPrefaceOnInit,
            !!config.filterChannelContentFromKvCache, prefaceJson);
      } else {
        wasmConfig = this.wasm.ConversationConfig.createDefault(this.engine);
      }

      const wasmConversation =
          this.wasm.Conversation.create(this.engine, wasmConfig);

      wasmConfig.delete();
      if (wasmSessionConfig) {
        wasmSessionConfig.delete();
      }

      return new Conversation(wasmConversation, this.engine, this.mutexes);
    });
  }

  async delete(): Promise<void> {
    await this.mutexes.executor.acquireAndRun(() => {
      this.deleteCallback();
    });
  }
}

async function modelToStream(model: EngineSettings['model']):
    Promise<ReadableStream<Uint8Array>> {
  if (model instanceof ReadableStream) {
    return model;
  }

  const modelUrl = model;
  const response = await fetch(modelUrl, {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch model file from ${modelUrl}`);
  }
  return response.body!;
}

// TODO: b/477709280 - Remove this when streaming loading works.
async function loadModelToVfs(
    module: LiteRtLmWasm,
    modelStream: ReadableStream<Uint8Array>,
    dstPath: string,
) {
  let fileContent: Uint8Array;

  const reader = modelStream.getReader();
  const chunks: Uint8Array[] = [];
  let totalLength = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      totalLength += value.length;
    }
  }
  fileContent = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    fileContent.set(chunk, offset);
    offset += chunk.length;
  }

  try {
    module.FS.writeFile(dstPath, fileContent);
  } catch (e) {
    console.error(`Error writing file to VFS:`, e);
    throw e;
  }
}
