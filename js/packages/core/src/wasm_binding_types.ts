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

import {WasmModule} from '@litertjs/wasm-utils';

import {type ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';

/**
 * Backend types for LiteRT-LM.
 */
export const Backend = {
  // Ideally, we'd pull these from the Wasm module, but we want them to be
  // available in JS before the Wasm module is loaded.
  UNSPECIFIED: 0,
  CPU_ARTISAN: 1,
  GPU_ARTISAN: 2,
  CPU: 3,
  GPU: 4,
  GOOGLE_TENSOR_ARTISAN: 5,
  NPU: 6,
};

/**
 * A LiteRT-LM Backend.
 */
export type Backend = (typeof Backend)[keyof typeof Backend];

/**
 * An object that must be manually deleted to free its memory.
 */
export declare interface Deletable {
  delete(): void;
}

/**
 * A C++ vector of elements.
 */
export declare interface EmscriptenVector<T> extends Deletable {
  size(): number;
  get(index: number): T;
  push_back(item: T): void;
}

/**
 * A C++ vector of uint32_t.
 */
export declare interface VectorUint32Constructor {
  new(): EmscriptenVector<number>;
}

/**
 * A C++ vector of int.
 */
export declare interface VectorIntConstructor {
  new(): EmscriptenVector<number>;
}

/**
 * A C++ vector of vector of int.
 */
export declare interface VectorVectorIntConstructor {
  new(): EmscriptenVector<EmscriptenVector<number>>;
}

/**
 * A C++ enum value in Wasm.
 */
export declare interface EmscriptenEnumElement<T> {
  value: T;
}

type EmscriptenEnum<T extends object> = {
  [K in keyof T]: EmscriptenEnumElement<T[K]>;
};

declare interface ModelAssetsConstructor {
  // This `new` signature is required to make TypeScript treat
  // `ModelAssetsConstructor` as the class from which `ModelAssets` are
  // constructed, even though we don't use it.
  new(...args: never[]): ModelAssets;  // Do not call.
  create(modelPath: string): ModelAssets;
  createStreaming(stream: ReadableStreamDataStream): ModelAssets;
}

declare const ModelAssetsBrand: unique symbol;

/**
 * LiteRT-LM ModelAssets
 */
export declare interface ModelAssets extends Deletable {
  // Prevent accidentally assigning other objects as ModelAssets.
  [ModelAssetsBrand]: void;
  getPath(): string;
}

/**
 * LiteRT-LM Backend enum values
 */
declare interface EngineSettingsConstructor {
  // This `new` signature is required to make TypeScript treat
  // `EngineSettingsConstructor` as the class from which `EngineSettings` are
  // constructed, even though we don't use it.
  new(...args: never[]): EngineSettings;  // Do not call.
  createDefault(
      modelAssets: ModelAssets,
      backend: EmscriptenEnumElement<Backend>): EngineSettings;
}

declare const EngineSettingsBrand: unique symbol;

/**
 * LiteRT-LM EngineSettings
 */
export declare interface EngineSettings extends Deletable {
  // Prevent accidentally assigning other objects as EngineSettings.
  [EngineSettingsBrand]: void;
  getMutableMainExecutorSettings(): LlmExecutorSettings;
  getParallelFileSectionLoading(): boolean;
  setParallelFileSectionLoading(parallelFileSectionLoading: boolean): void;
  getSingleThreadedExecution(): boolean;
  setSingleThreadedExecution(singleThreadedExecution: boolean): void;
  enableBenchmark(): void;
}

/**
 * LiteRT-LM CpuConfig
 */
export declare interface CpuConfig {
  kv_increment_size: number;
  prefill_chunk_size: number;
  number_of_threads: number;
}

/**
 * LiteRT-LM GpuConfig
 */
export declare interface GpuConfig {
  max_top_k: number;
  external_tensor_mode: boolean;
}

/**
 * LiteRT-LM GpuArtisanConfig
 */
export declare interface GpuArtisanConfig {
  // Number of output candidates.
  num_output_candidates: number;

  // Whether to wait for weight uploads before prefilling.
  wait_for_weight_uploads: boolean;

  // Number of decode steps per sync. Used by GPU only.
  num_decode_steps_per_sync: number;

  // Sequence batch size for encoding. Used by GPU only. Number of input
  // tokens to process at a time for batch processing. Setting this value to 1
  // means both the encoding and decoding share the same graph of sequence
  // length of 1. Setting this value to 0 means the batch size will be
  // optimized programmatically.
  sequence_batch_size: number;

  // The supported lora ranks for the base model. Used by GPU only. By default
  // it will be empty, meaning not supporting any lora ranks.
  supported_lora_ranks: EmscriptenVector<number>;

  // Maximum top k, which is the max Top-K value supported for all
  // sessions created with the engine, used by GPU only. If a session with
  // Top-K value larger than this is being asked to be created, it will be
  // rejected(throw error). The max top k will be 1, which means only greedy
  // decoding is supported for any sessions created with this engine.
  max_top_k: number;

  // Enables decode logits.
  // AiCore uses decode logits, so this is enabled for AiCore.
  // LLM Engine defaults to disabling decode logits.
  enable_decode_logits: boolean;

  // Enables external embeddings.
  // AiCore uses external embeddings, so this is enabled for AiCore.
  // LLM Engine defaults to disabling external embeddings.
  enable_external_embeddings: boolean;

  // Whether the submodel should be used if available.
  use_submodel: boolean;

  // For low GPU memory with long contexts, use minimally sized ringbuffers for
  // local attention. Prevents instantaneous rewinding, but saves a lot of
  // memory, especially for large models.
  use_autosized_ringbuffers: boolean;
}

/**
 * LiteRT-LM AdvancedSettings
 */
export declare interface AdvancedSettings {
  prefill_batch_sizes: number[];
  num_output_candidates: number;
  configure_magic_numbers: boolean;
  verify_magic_numbers: boolean;
  clear_kv_cache_before_prefill: boolean;
  num_logits_to_print_after_decode: number;
  gpu_madvise_original_shared_tensors: boolean;
  is_benchmark: boolean;
  preferred_device_substr: string;
  num_threads_to_upload: number;
  num_threads_to_compile: number;
  convert_weights_on_gpu: boolean;
  optimize_shader_compilation: boolean;
  share_constant_tensors: boolean;
}

/**
 * LiteRT-LM ExecutorSettingsBase constructor
 */
declare interface ExecutorSettingsBaseConstructor {
  new(...args: never[]): ExecutorSettingsBase;  // Do not call.
}

declare const ExecutorSettingsBaseBrand: unique symbol;

/**
 * LiteRT-LM ExecutorSettingsBase
 */
export declare interface ExecutorSettingsBase extends Deletable {
  // Prevent accidentally assigning other objects as ExecutorSettingsBase.
  [ExecutorSettingsBaseBrand]: void;
  getCacheDir(): string;
  setCacheDir(cacheDir: string): void;
}

/**
 * LiteRT-LM LlmExecutorSettings constructor
 */
declare interface LlmExecutorSettingsConstructor {
  new(...args: never[]): LlmExecutorSettings;  // Do not call.
}

declare const LlmExecutorSettingsBrand: unique symbol;

/**
 * LiteRT-LM LlmExecutorSettings
 */
export declare interface LlmExecutorSettings extends ExecutorSettingsBase {
  // Prevent accidentally assigning other objects as LlmExecutorSettings.
  [LlmExecutorSettingsBrand]: void;
  getMaxNumTokens(): number;
  setMaxNumTokens(maxNumTokens: number): void;
  setBackendConfigCpu(cpuConfig: CpuConfig): void;
  setBackendConfigGpu(gpuConfig: GpuConfig): void;
  setBackendConfigGpuArtisan(gpuArtisanConfig: GpuArtisanConfig): void;
  getBackendConfigCpu(): CpuConfig;
  getBackendConfigGpu(): GpuConfig;
  getBackendConfigGpuArtisan(): GpuArtisanConfig;
  setSamplerBackend(backend: EmscriptenEnumElement<Backend>): void;
  getSamplerBackend(): EmscriptenEnumElement<Backend>;
  setAdvancedSettings(advancedSettings: AdvancedSettings): void;
  getAdvancedSettings(): AdvancedSettings;
}

/**
 * LiteRT-LM Engine constructor
 */
declare interface EngineConstructor {
  // This `new` signature is required to make TypeScript treat
  // `EngineConstructor` as the class from which `Engine` are
  // constructed, even though we don't use it.
  new(...args: never[]): Engine;  // Do not call.
  createEngine(engineSettings: EngineSettings, inputPromptAsHint: string):
      Promise<Engine>;
  createStreaming(engineSettings: EngineSettings, inputPromptAsHint: string):
      Promise<Engine>;
}

declare const EngineBrand: unique symbol;

/**
 * LiteRT-LM Engine
 */
export declare interface Engine extends Deletable {
  // Prevent accidentally assigning other objects as Engine.
  [EngineBrand]: void;
  createSession(sessionConfig: SessionConfig): Session;
  getEngineSettings(): EngineSettings;
  waitUntilDone(): Promise<void>;
}

/**
 * LiteRT-LM SamplerType enum values
 */
export const SamplerType = {
  TYPE_UNSPECIFIED: 0,
  TOP_K: 1,
  TOP_P: 2,
  GREEDY: 3,
} as const;
/**
 * LiteRT-LM SamplerType
 */
export type SamplerType = typeof SamplerType[keyof typeof SamplerType];

/**
 * LiteRT-LM SamplerParameters
 */
export declare interface SamplerParameters {
  type(): EmscriptenEnumElement<SamplerType>;
  setType(type: EmscriptenEnumElement<SamplerType>): void;
  k(): number;
  setK(k: number): void;
  p(): number;
  setP(p: number): void;
  temperature(): number;
  setTemperature(temperature: number): void;
  seed(): number;
  setSeed(seed: number): void;
}

/**
 * LiteRT-LM SessionConfig constructor
 */
declare interface SessionConfigConstructor {
  // This `new` signature is required to make TypeScript treat
  // `SessionConfigConstructor` as the class from which `SessionConfig` are
  // constructed, even though we don't use it.
  new(...args: never[]): SessionConfig;  // Do not call.
  createDefault(): SessionConfig;
}

declare const SessionConfigBrand: unique symbol;

/**
 * LiteRT-LM SessionConfig
 */
export declare interface SessionConfig extends Deletable {
  // Prevent accidentally assigning other objects as SessionConfig.
  [SessionConfigBrand]: void;

  getAudioModalityEnabled(): boolean;
  setAudioModalityEnabled(audioModalityEnabled: boolean): void;
  getVisionModalityEnabled(): boolean;
  setVisionModalityEnabled(visionModalityEnabled: boolean): void;
  getMutableSamplerParams(): SamplerParameters;
  getStopTokenIds(): EmscriptenVector<EmscriptenVector<number>>;
  setStopTokenIds(stopTokenIds: EmscriptenVector<EmscriptenVector<number>>):
      void;
  getStartTokenId(): number;
  setStartTokenId(startTokenId: number): void;
  getNumOutputCandidates(): number;
  setNumOutputCandidates(numOutputCandidates: number): void;
  getSamplerBackend(): EmscriptenEnumElement<Backend>;
  setSamplerBackend(backend: EmscriptenEnumElement<Backend>): void;
  getApplyPromptTemplateInSession(): boolean;
  setApplyPromptTemplateInSession(applyPromptTemplateInSession: boolean): void;
  getUseExternalSampler(): boolean;
  setUseExternalSampler(useExternalSampler: boolean): void;
  getMaxOutputTokens(): number;
  setMaxOutputTokens(maxOutputTokens: number): void;
}

/**
 * LiteRT-LM Session constructor
 */
declare interface SessionConstructor {
  // This `new` signature is required to make TypeScript treat
  // `SessionConstructor` as the class from which `Session` are
  // constructed, even though we don't use it.
  new(...args: never[]): Session;  // Do not call.
}

declare const SessionBrand: unique symbol;

/**
 * LiteRT-LM Session
 */
export declare interface Session extends Deletable {
  // Prevent accidentally assigning other objects as Session.
  [SessionBrand]: void;
  getSessionConfig(): SessionConfig;
  runPrefill(inputs: string[]): Promise<void>;
  runDecode(): Promise<Responses>;
  cancelProcess(): void;
  clone(): Session;
}

/**
 * LiteRT-LM Responses constructor
 */
declare interface ResponsesConstructor {
  // This `new` signature is required to make TypeScript treat
  // `ResponsesConstructor` as the class from which `Responses` are
  // constructed, even though we don't use it.
  new(...args: never[]): Responses;  // Do not call.
}

declare const ResponsesBrand: unique symbol;

/**
 * LiteRT-LM Responses
 */
export declare interface Responses extends Deletable {
  // Prevent accidentally assigning other objects as Responses.
  [ResponsesBrand]: void;
  getTexts(): EmscriptenVector<string>;
}

/**
 * LiteRT-LM ConversationConfig constructor
 */
declare interface ConversationConfigConstructor {
  new(...args: never[]): ConversationConfig;  // Do not call.
  createDefault(engine: Engine): ConversationConfig;
  createCustom(
      engine: Engine, sessionConfig: SessionConfig,
      enableConstrainedDecoding: boolean, prefillPrefaceOnInit: boolean,
      filterChannelContentFromKvCache: boolean,
      prefaceJson: string): ConversationConfig;
}

declare const ConversationConfigBrand: unique symbol;

/**
 * LiteRT-LM ConversationConfig
 */
export declare interface ConversationConfig extends Deletable {
  [ConversationConfigBrand]: void;
}

/**
 * LiteRT-LM Conversation constructor
 */
declare interface ConversationConstructor {
  new(...args: never[]): Conversation;  // Do not call.
  create(engine: Engine, config: ConversationConfig): Promise<Conversation>;
}

declare const ConversationBrand: unique symbol;

/**
 * LiteRT-LM Conversation
 */
export declare interface Conversation extends Deletable {
  [ConversationBrand]: void;
  sendMessage(messageJson: string): Promise<string>;
  sendMessageAsync(
      messageJson: string,
      callback:
          (chunk: string|null, isDone: boolean, error: string|null) => void):
      Promise<void>;
  getHistory(): string;
  getTokenCount(): number;
  getBenchmarkInfo(): BenchmarkInfo;
  cancelProcess(): void;
  clone(): Conversation;
}

/** Benchmark metadata for tracking decoding efficiency. */
export declare interface BenchmarkInfo {
  lastPrefillTokensPerSecond: number;
  lastPrefillTokenCount: number;
  lastDecodeTokensPerSecond: number;
  lastDecodeTokenCount: number;
  timeToFirstTokenInSecond: number;
}


declare interface ReadableStreamDataStreamConstructor {
  new(...args: never[]): ReadableStreamDataStream;
  create(stream: ReadableStreamDataStreamWrapper): ReadableStreamDataStream;
}

declare const ReadableStreamDataStreamBrand: unique symbol;

/**
 * LiteRT-LM ReadableStreamDataStream.
 */
export declare interface ReadableStreamDataStream extends Deletable {
  // Prevent accidentally assigning other objects as ReadableStreamDataStream.
  [ReadableStreamDataStreamBrand]: void;
}

type BackendEnum = EmscriptenEnum<{
  UNSPECIFIED: Backend; CPU_ARTISAN: Backend; GPU_ARTISAN: Backend;
  CPU: Backend;
  GPU: Backend;
  GOOGLE_TENSOR_ARTISAN: Backend;
  NPU: Backend;
}>;

/**
 * Interface for the C++ LiteRt LM bindings.
 */
export declare interface LiteRtLmWasm extends WasmModule {
  preinitializedWebGPUDevice?: GPUDevice;
  VectorUint32: VectorUint32Constructor;
  VectorInt: VectorIntConstructor;
  VectorVectorInt: VectorVectorIntConstructor;
  FS: FileSystemApi;
  setupLogging(): void;
  Backend: BackendEnum;
  ModelAssets: ModelAssetsConstructor;
  EngineSettings: EngineSettingsConstructor;
  LlmExecutorSettings: LlmExecutorSettingsConstructor;
  Engine: EngineConstructor;
  SessionConfig: SessionConfigConstructor;
  ReadableStreamDataStream: ReadableStreamDataStreamConstructor;
  ConversationConfig: ConversationConfigConstructor;
  Conversation: ConversationConstructor;
}


// Minimal subset of Emscripten FS api needed in order to write a model file to
// the in-memory file system. This will likely be removed when we switch to
// streaming loading.

declare interface FileSystemApi {
  writeFile(path: string, data: string|ArrayBufferView): void;
  unlink(path: string): void;
}
