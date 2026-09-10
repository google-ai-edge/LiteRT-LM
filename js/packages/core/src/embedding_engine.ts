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

import {Cleanup} from './cleanup.js';
import {ContentPart, SUPPORTED_AUDIO_MIME_TYPES, SUPPORTED_IMAGE_MIME_TYPES} from './conversation_config.js';
import {EmbeddingEngineSettings, fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings} from './embedding_engine_settings.js';
import {getOrLoadGlobalLiteRtLm} from './load_litertlm.js';
import {Mutex} from './mutex.js';
import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';
import {modelToStream, setupStreamWeightsCallback} from './stream_utils.js';
import {Backend, Deletable, EmbeddingEngine as WasmEmbeddingEngine, EmbeddingEngineSettings as WasmEmbeddingEngineSettings, EmbeddingOptions, InputOverflowStrategy} from './wasm_binding_types.js';
import {consumeEmscriptenVectorToArray} from './wasm_utils.js';

export {
  type EmbeddingOptions,
  InputOverflowStrategy,
  SUPPORTED_AUDIO_MIME_TYPES,
  SUPPORTED_IMAGE_MIME_TYPES,
};

/**
 * A single input part to the embedding engine.
 */
export type SingleEmbeddingInput = string | ContentPart | Blob;

/**
 * An input or sequence of inputs to compute an embedding for.
 */
export type EmbeddingInput = SingleEmbeddingInput | SingleEmbeddingInput[];

/**
 * Result of an embedding computation.
 */
export interface EmbeddingResponse {
  embedding: number[];
  inputLength: number;
  truncatedLength?: number;
  numChunks: number;
}

function toWasmEmbeddingOptions(options?: EmbeddingOptions): EmbeddingOptions {
  return {
    normalize: options?.normalize ?? true,
    insertSpecialTokens: options?.insertSpecialTokens ?? true,
    inputOverflowStrategy:
        options?.inputOverflowStrategy ?? InputOverflowStrategy.ERROR,
    visionTokensPerImage: options?.visionTokensPerImage,
    outputSize: options?.outputSize,
  };
}

async function normalizeInputPart(part: SingleEmbeddingInput):
    Promise<ContentPart> {
  if (typeof part === 'string') {
    return {type: 'text', text: part};
  }
  if (part instanceof Blob) {
    const mimeType = part.type.toLowerCase().split(';')[0].trim();
    if (SUPPORTED_IMAGE_MIME_TYPES.has(mimeType)) {
      const arrayBuffer = await part.arrayBuffer();
      return {type: 'image', data: arrayBuffer};
    }
    if (SUPPORTED_AUDIO_MIME_TYPES.has(mimeType)) {
      const arrayBuffer = await part.arrayBuffer();
      return {type: 'audio', data: arrayBuffer};
    }
    throw new Error(
        `Unsupported Blob mime type: "${part.type}". Supported image types: ` +
        `${[...SUPPORTED_IMAGE_MIME_TYPES].join(', ')}. Supported audio types: ` +
        `${[...SUPPORTED_AUDIO_MIME_TYPES].join(', ')}.`);
  }
  if (typeof part === 'object' && part !== null) {
    if (part.type === 'image') {
      if (part.data instanceof Blob) {
        const arrayBuffer = await part.data.arrayBuffer();
        return {type: 'image', data: arrayBuffer};
      }
      return part;
    }
    if (part.type === 'audio') {
      if (part.data instanceof Blob) {
        const arrayBuffer = await part.data.arrayBuffer();
        return {type: 'audio', data: arrayBuffer};
      }
      return part;
    }
    return part;
  }
  throw new Error(`Unsupported input part: ${JSON.stringify(part)}`);
}

async function normalizeInput(input: EmbeddingInput):
    Promise<ContentPart | ContentPart[]> {
  if (Array.isArray(input)) {
    return Promise.all(input.map((p) => normalizeInputPart(p)));
  }
  return normalizeInputPart(input);
}

/**
 * LiteRT-LM Embedding Engine
 */
export class EmbeddingEngine implements Deletable {
  readonly settings: EmbeddingEngineSettings;
  private readonly mutex = new Mutex();

  private constructor(
      private readonly engine: WasmEmbeddingEngine,
      settings: EmbeddingEngineSettings,
      private readonly deleteCallback: () => void,
  ) {
    this.settings = settings;
  }

  static async create(settings: EmbeddingEngineSettings):
      Promise<EmbeddingEngine> {
    const litertlm = await getOrLoadGlobalLiteRtLm();
    const wasm = litertlm.liteRtLmWasm;
    const backend = settings.backend ?? Backend.GPU;
    settings = {...settings, backend};

    if (backend === Backend.GPU || backend === Backend.GPU_ARTISAN) {
      await litertlm.setupDefaultWebGpuDevice();
    }

    const cleanup = new Cleanup();
    const modelStream = await modelToStream(settings.model);
    let engine: WasmEmbeddingEngine;
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

      let wasmSettings: WasmEmbeddingEngineSettings;
      if (settings.visionBackend !== undefined ||
          settings.audioBackend !== undefined) {
        wasmSettings = wasm.EmbeddingEngineSettings.createDefaultMultimodal(
            modelAssets, {value: backend},
            settings.visionBackend ? {value: settings.visionBackend} :
                                     undefined,
            settings.audioBackend ? {value: settings.audioBackend} : undefined);
      } else {
        wasmSettings = wasm.EmbeddingEngineSettings.createDefault(
            modelAssets, {value: backend});
      }
      cleanupModelAssets();

      const cleanupWasmSettings = cleanup.add(() => {
        wasmSettings.delete();
      });

      const resolvedBackend =
          wasmSettings.getMutableMainExecutorSettings().getBackend().value;

      fillWasmEmbeddingEngineSettingsFromEmbeddingEngineSettings(
          wasmSettings, settings, resolvedBackend);

      if (resolvedBackend === Backend.GPU) {
        setupStreamWeightsCallback(wasm);
      }

      try {
        engine = await wasm.EmbeddingEngine.createEngine(wasmSettings);
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
      cleanupWasmSettings();

      cleanup.add(() => {
        engine.delete();
      });
      return new EmbeddingEngine(
          engine, settings, () => cleanup.run());
    } catch (e) {
      cleanup.run();
      throw e;
    }
  }

  async computeEmbedding(
      input: EmbeddingInput,
      options: EmbeddingOptions = {},
  ): Promise<EmbeddingResponse> {
    const normalized = await normalizeInput(input);
    return this.mutex.acquireAndRun(async () => {
      const wasmResponse =
          await this.engine.computeEmbedding(normalized, toWasmEmbeddingOptions(options));
      const embedding = consumeEmscriptenVectorToArray(wasmResponse.embedding);
      return {
        embedding,
        inputLength: wasmResponse.inputLength,
        truncatedLength: wasmResponse.truncatedLength,
        numChunks: wasmResponse.numChunks,
      };
    });
  }

  async computeEmbeddingBatch(
      inputs: EmbeddingInput[],
      options: EmbeddingOptions = {},
  ): Promise<EmbeddingResponse[]> {
    const normalizedBatch = await Promise.all(inputs.map(normalizeInput));
    return this.mutex.acquireAndRun(async () => {
      const wasmResponses = await this.engine.computeEmbeddingBatch(
          normalizedBatch, toWasmEmbeddingOptions(options));
      try {
        const responses: EmbeddingResponse[] = [];
        const count = wasmResponses.size();
        for (let i = 0; i < count; i++) {
          const wasmResp = wasmResponses.get(i);
          const embedding =
              consumeEmscriptenVectorToArray(wasmResp.embedding);
          responses.push({
            embedding,
            inputLength: wasmResp.inputLength,
            truncatedLength: wasmResp.truncatedLength,
            numChunks: wasmResp.numChunks,
          });
        }
        return responses;
      } finally {
        wasmResponses.delete();
      }
    });
  }

  async delete(): Promise<void> {
    await this.mutex.acquireAndRun(() => {
      this.deleteCallback();
    });
  }
}
