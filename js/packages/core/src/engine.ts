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
import {Backend, ConversationConfig as WasmConversationConfig, Deletable, Engine as WasmEngine, LiteRtLmWasm, SessionConfig as WasmSessionConfig} from './wasm_binding_types.js';



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
    // Default to GPU if not specified.
    const backend = engineSettings.backend ?? Backend.GPU;
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
      const streamWrapper =
          new ReadableStreamDataStreamWrapper(modelStream, () => wasm.HEAPU8);
      const dataStream = wasm.ReadableStreamDataStream.create(streamWrapper);
      cleanup.add(() => {
        dataStream.delete();
      });
      const modelAssets = wasm.ModelAssets.createStreaming(dataStream);

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

      const resolvedBackend =
          wasmEngineSettings.getMutableMainExecutorSettings().getBackend().value;

      fillWasmEngineSettingsFromEngineSettings(
          wasmEngineSettings, engineSettings, resolvedBackend, wasm);
      wasmEngineSettings.setParallelFileSectionLoading(false);
      wasmEngineSettings.setSingleThreadedExecution(true);

      if (resolvedBackend === Backend.GPU) {
        const gpuDevice = wasm.preinitializedWebGPUDevice;
        if (!gpuDevice) {
          throw new Error('WebGPU device not initialized');
        }

        wasm.registerStreamWeightsCallback(async (
            tflIds: Int32Array,
            wgpuBufferIds: Uint32Array,
            offsets: Float64Array,
            lengths: Float64Array,
        ) => {
          const requests = [];
          if (tflIds.length !== wgpuBufferIds.length) {
            throw new Error(
                `Stream weights callback received arrays of different lengths: ` +
                `tflIds=${tflIds.length}, wgpuBufferIds=${wgpuBufferIds.length}`);
          }
          for (let i = 0; i < tflIds.length; i++) {
            requests.push({
              id: tflIds[i],
              wgpuBufferId: wgpuBufferIds[i],
              offset: offsets[i],
              length: lengths[i],
            });
          }
          requests.sort((a, b) => a.offset - b.offset);

          const CHUNK_SIZE = 4 * 1024 * 1024;
          const tempPtr = wasm._malloc(CHUNK_SIZE);
          try {
            for (const req of requests) {
              const gpuBuffer = wasm.WebGPU.getJsObject(req.wgpuBufferId) as GPUBuffer;
              if (!gpuBuffer) {
                throw new Error(`Failed to find GPUBuffer for ID: ${req.wgpuBufferId}`);
              }
              const modelType = wasm.getCurrentlyCompilingModel();
              let bytesUploaded = 0;
              while (bytesUploaded < req.length) {
                const chunkSize = Math.min(CHUNK_SIZE, req.length - bytesUploaded);
                await wasm.readStoredWeights(
                    modelType, req.offset + bytesUploaded, chunkSize, tempPtr);

                let chunkData = new Uint8Array(wasm.HEAPU8.buffer, tempPtr, chunkSize);
                if (chunkData.byteLength % 4 !== 0 &&
                    bytesUploaded + chunkSize === req.length) {
                  const paddedSize = (chunkData.byteLength + 3) & ~3;
                  const paddedData = new Uint8Array(paddedSize);
                  paddedData.set(chunkData);
                  chunkData = paddedData;
                }

                gpuDevice.queue.writeBuffer(
                    gpuBuffer, bytesUploaded, chunkData as GPUAllowSharedBufferSource);
                bytesUploaded += chunkSize;
              }
            }
          } finally {
            wasm._free(tempPtr);
          }
        });
      }

      try {
        if (resolvedBackend === Backend.GPU_ARTISAN) {
          engine = await wasm.Engine.createStreaming(
              wasmEngineSettings, inputPromptAsHint);
        } else {
          engine = await wasm.Engine.createEngine(
              wasmEngineSettings, inputPromptAsHint);
        }
      } finally {
        if (resolvedBackend === Backend.GPU) {
          try {
            wasm.registerStreamWeightsCallback(undefined);
            await wasm.clearStoredWeightsStreams();
          } catch (cleanupError) {
            console.error('Error during cleanup:', cleanupError);
          }
        }
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
    return this.mutexes.executor.acquireAndRun(async () => {
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
          await this.wasm.Conversation.create(this.engine, wasmConfig);

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
  if (model instanceof Blob) {
    return model.stream();
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
