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

import {ReadableStreamDataStreamWrapper, ReadableStreamDataStreamWrapperInterface} from './readable_stream_data_stream_wrapper.js';

export {ReadableStreamDataStreamWrapper};

/**
 * Backend types for LiteRT-LM.
 */
export const Backend = {
  // Ideally, we'd pull these from the Wasm module, but we want them to be
  // available in JS before the Wasm module is loaded.
  UNSPECIFIED: 0,
  GPU_ARTISAN: 1,
  CPU: 2,
  GPU: 3,
  GOOGLE_TENSOR_ARTISAN: 4,
  NPU: 5,
};

/**
 * A LiteRT-LM Backend.
 */
export type Backend = (typeof Backend)[keyof typeof Backend];

/**
 * Activation data types for LiteRT-LM executors.
 */
export const ActivationDataType = {
  // Ideally, we'd pull these from the Wasm module, but we want them to be
  // available in JS before the Wasm module is loaded.
  FLOAT32: 0,
  FLOAT16: 1,
  INT16: 2,
  INT8: 3,
} as const;

/**
 * A LiteRT-LM ActivationDataType.
 */
export type ActivationDataType =
    (typeof ActivationDataType)[keyof typeof ActivationDataType];

/**
 * How speculative decoding (e.g. MTP) is used during decode.
 */
export const SpeculativeDecodingMode = {
  // Ideally, we'd pull these from the Wasm module, but we want them to be
  // available in JS before the Wasm module is loaded.
  OFF: 0,
  ON: 1,
  ADAPTIVE: 2,
} as const;

/**
 * A LiteRT-LM SpeculativeDecodingMode.
 */
export type SpeculativeDecodingMode =
    (typeof SpeculativeDecodingMode)[keyof typeof SpeculativeDecodingMode];

/**
 * LiteRT-LM SpeculativeDecodingConfig. Unset fields are undefined, but must be
 * present when passed to Wasm.
 */
export declare interface SpeculativeDecodingConfig {
  mode: EmscriptenEnumElement<SpeculativeDecodingMode>|undefined;
  lenience: number|undefined;
}

/**
 * An object that must be manually deleted to free its memory.
 */
export declare interface Deletable {
  delete(): void;
}

/**
 * The result of a Wasm binding that can fail: either the value or an `Error`.
 *
 * Bindings return errors instead of throwing them because throwing from C++
 * leaks memory and Wasm stack. Use `unwrapWasmResult` to get the value.
 */
export type WasmResult<T> = T|Error;

/**
 * Returns the value of a `WasmResult`, or throws its error.
 */
export function unwrapWasmResult<T>(result: WasmResult<T>): T {
  if (result instanceof Error) {
    throw result;
  }
  return result;
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
 * A C++ vector of float.
 */
export declare interface VectorFloatConstructor {
  new(): EmscriptenVector<number>;
}

/**
 * A C++ vector of EmbeddingResponse.
 */
export declare interface VectorEmbeddingResponseConstructor {
  new(): EmscriptenVector<WasmEmbeddingResponse>;
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
  createStreaming(stream: ReadableStreamDataStream): WasmResult<ModelAssets>;
}

declare const ModelAssetsBrand: unique symbol;

/**
 * LiteRT-LM ModelAssets
 */
export declare interface ModelAssets extends Deletable {
  // Prevent accidentally assigning other objects as ModelAssets.
  [ModelAssetsBrand]: void;
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
      modelAssets: ModelAssets, backend: EmscriptenEnumElement<Backend>):
      Promise<WasmResult<EngineSettings>>;
  // Embind requires all arguments to be passed; use `undefined` to disable a
  // modality.
  createDefaultMultimodal(
      modelAssets: ModelAssets, backend: EmscriptenEnumElement<Backend>,
      visionBackend: EmscriptenEnumElement<Backend>|undefined,
      audioBackend: EmscriptenEnumElement<Backend>|
      undefined): Promise<WasmResult<EngineSettings>>;
}

declare const EngineSettingsBrand: unique symbol;

/**
 * LiteRT-LM EngineSettings
 */
export declare interface EngineSettings extends Deletable {
  // Prevent accidentally assigning other objects as EngineSettings.
  [EngineSettingsBrand]: void;
  getMutableMainExecutorSettings(): LlmExecutorSettings;
  /** Returns undefined if vision is not enabled. */
  getVisionBackend(): EmscriptenEnumElement<Backend>|undefined;
  /** Returns null if vision is not enabled. */
  getMutableVisionExecutorSettings(): ExecutorSettingsBase|null;
  /** Returns undefined if audio is not enabled. */
  getAudioBackend(): EmscriptenEnumElement<Backend>|undefined;
  /** Returns null if audio is not enabled. */
  getMutableAudioExecutorSettings(): ExecutorSettingsBase|null;
  getMaxVisionTokensPerImage(): number|undefined;
  setMaxVisionTokensPerImage(maxVisionTokensPerImage: number): void;
  getMinVisionTokensPerImage(): number|undefined;
  setMinVisionTokensPerImage(minVisionTokensPerImage: number): void;
  getParallelFileSectionLoading(): boolean;
  setParallelFileSectionLoading(parallelFileSectionLoading: boolean): void;
  getSingleThreadedExecution(): boolean;
  setSingleThreadedExecution(singleThreadedExecution: boolean): void;
  enableBenchmark(): void;
  getSpeculativeDecodingConfig(): SpeculativeDecodingConfig;
  setSpeculativeDecodingConfig(config: SpeculativeDecodingConfig): void;
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
  getBackend(): EmscriptenEnumElement<Backend>;
  getActivationDataType(): EmscriptenEnumElement<ActivationDataType>|undefined;
  setActivationDataType(
      activationDataType: EmscriptenEnumElement<ActivationDataType>): void;
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
  getMaxNumImages(): number;
  setMaxNumImages(maxNumImages: number): void;
  setBackendConfigCpu(cpuConfig: CpuConfig): void;
  setBackendConfigGpu(gpuConfig: GpuConfig): void;
  setBackendConfigGpuArtisan(gpuArtisanConfig: GpuArtisanConfig): void;
  getBackendConfigCpu(): WasmResult<CpuConfig>;
  getBackendConfigGpu(): WasmResult<GpuConfig>;
  getBackendConfigGpuArtisan(): WasmResult<GpuArtisanConfig>;
  setSamplerBackend(backend: EmscriptenEnumElement<Backend>): void;
  getSamplerBackend(): EmscriptenEnumElement<Backend>;
  setAdvancedSettings(advancedSettings: AdvancedSettings): void;
  /** Returns undefined if the advanced settings have not been set. */
  getAdvancedSettings(): AdvancedSettings|undefined;
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
      Promise<WasmResult<Engine>>;
  createLegacyEngine(engineSettings: EngineSettings, inputPromptAsHint: string):
      Promise<WasmResult<Engine>>;
}

declare const EngineBrand: unique symbol;

/**
 * LiteRT-LM Engine
 */
export declare interface Engine extends Deletable {
  // Prevent accidentally assigning other objects as Engine.
  [EngineBrand]: void;
  createSession(sessionConfig: SessionConfig): WasmResult<Session>;
  getEngineSettings(): EngineSettings;
  waitUntilDone(): Promise<WasmResult<void>>;
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
  /** Unset fields inherit the engine's values. */
  getSpeculativeDecodingConfig(): SpeculativeDecodingConfig;
  setSpeculativeDecodingConfig(config: SpeculativeDecodingConfig): void;
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
  runPrefill(inputs: string[]): Promise<WasmResult<void>>;
  runDecode(): Promise<WasmResult<Responses>>;
  cancelProcess(): void;
  clone(): WasmResult<Session>;
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
  createDefault(engine: Engine): WasmResult<ConversationConfig>;
  createCustom(
      engine: Engine, sessionConfig: SessionConfig,
      enableConstrainedDecoding: boolean, prefillPrefaceOnInit: boolean,
      filterChannelContentFromKvCache: boolean,
      prefaceJson: string): WasmResult<ConversationConfig>;
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
  create(engine: Engine, config: ConversationConfig):
      Promise<WasmResult<Conversation>>;
}

declare const ConversationBrand: unique symbol;

/**
 * LiteRT-LM Conversation
 */
export declare interface Conversation extends Deletable {
  [ConversationBrand]: void;
  /**
   * @param blobs Binary payloads referenced by `blob_index` in `messageJson`.
   * @param visualTokenBudget Max vision tokens per image for this message, or
   *     <= 0 for no budget.
   */
  sendMessage(
      messageJson: string, blobs: Uint8Array[],
      visualTokenBudget: number): Promise<WasmResult<string>>;
  sendMessageAsync(
      messageJson: string, blobs: Uint8Array[],
      callback:
          (chunk: string|null, isDone: boolean, error: string|null) => void,
      visualTokenBudget: number): Promise<WasmResult<void>>;
  getHistory(): string;
  getTokenCount(): WasmResult<number>;
  getBenchmarkInfo(): BenchmarkInfo;
  cancelProcess(): void;
  clone(): WasmResult<Conversation>;
}

/** Benchmark metadata for tracking decoding efficiency. */
export declare interface BenchmarkInfo {
  lastPrefillTokensPerSecond: number;
  lastPrefillTokenCount: number;
  lastDecodeTokensPerSecond: number;
  lastDecodeTokenCount: number;
  timeToFirstTokenInSecond: number;
  /**
   * Number of speculative decoding (MTP) rounds, accumulated over all decode
   * turns since the first decode turn.
   */
  mtpRounds: number;
  /** Number of tokens drafted by the MTP drafter. */
  mtpDraftedTokens: number;
  /** Number of drafted tokens accepted by the verifier. */
  mtpAcceptedTokens: number;
  /** Number of tokens emitted by MTP rounds. */
  mtpEmittedTokens: number;
  /**
   * Number of plain (non-speculative) decode steps, including the steps that
   * re-prime the MTP drafter.
   */
  plainDecodeSteps: number;
}


declare interface ReadableStreamDataStreamConstructor {
  new(...args: never[]): ReadableStreamDataStream;
  create(stream: ReadableStreamDataStreamWrapperInterface):
      ReadableStreamDataStream;
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
  UNSPECIFIED: Backend; GPU_ARTISAN: Backend;
  CPU: Backend;
  GPU: Backend;
  GOOGLE_TENSOR_ARTISAN: Backend;
  NPU: Backend;
}>;

/**
 * Input overflow strategies for embedding computations.
 */
export const InputOverflowStrategy = {
  CHUNK_AND_AVERAGE: 0,
  TRUNCATE: 1,
  ERROR: 2,
} as const;

/**
 * An input overflow strategy.
 */
export type InputOverflowStrategy =
    (typeof InputOverflowStrategy)[keyof typeof InputOverflowStrategy];

/**
 * Options for configuring the embedding computation.
 */
export declare interface EmbeddingOptions {
  normalize?: boolean;
  insertSpecialTokens?: boolean;
  inputOverflowStrategy?: InputOverflowStrategy;
  visionTokensPerImage?: number;
  outputSize?: number;
}

/**
 * Wasm-level result of an embedding computation.
 */
export declare interface WasmEmbeddingResponse {
  embedding: EmscriptenVector<number>;
  inputLength: number;
  truncatedLength?: number;
  numChunks: number;
}

declare const EmbeddingExecutorSettingsBrand: unique symbol;

/**
 * LiteRT-LM EmbeddingExecutorSettings
 */
export declare interface EmbeddingExecutorSettings extends ExecutorSettingsBase {
  [EmbeddingExecutorSettingsBrand]: void;
  getNumThreads(): number;
  setNumThreads(numThreads: number): void;
}

declare interface EmbeddingEngineSettingsConstructor {
  new(...args: never[]): EmbeddingEngineSettings;  // Do not call.
  createDefault(
      modelAssets: ModelAssets, backend: EmscriptenEnumElement<Backend>):
      WasmResult<EmbeddingEngineSettings>;
  createDefaultMultimodal(
      modelAssets: ModelAssets, backend: EmscriptenEnumElement<Backend>,
      visionBackend?: EmscriptenEnumElement<Backend>,
      audioBackend?: EmscriptenEnumElement<Backend>):
      WasmResult<EmbeddingEngineSettings>;
}

declare const EmbeddingEngineSettingsBrand: unique symbol;

/**
 * LiteRT-LM EmbeddingEngineSettings
 */
export declare interface EmbeddingEngineSettings extends Deletable {
  [EmbeddingEngineSettingsBrand]: void;
  getMaxInputLength(): number | undefined;
  setMaxInputLength(maxInputLength?: number): void;
  getMinInputLength(): number | undefined;
  setMinInputLength(minInputLength?: number): void;
  getVisionTokensPerImage(): number | undefined;
  setVisionTokensPerImage(visionTokensPerImage?: number): void;
  getMinVisionTokensPerImage(): number | undefined;
  setMinVisionTokensPerImage(minVisionTokensPerImage?: number): void;
  getMutableMainExecutorSettings(): EmbeddingExecutorSettings;
}

declare interface EmbeddingEngineConstructor {
  new(...args: never[]): EmbeddingEngine;  // Do not call.
  createEngine(settings: EmbeddingEngineSettings):
      Promise<WasmResult<EmbeddingEngine>>;
}

declare const EmbeddingEngineBrand: unique symbol;

/**
 * LiteRT-LM EmbeddingEngine
 */
export declare interface EmbeddingEngine extends Deletable {
  [EmbeddingEngineBrand]: void;
  computeEmbedding(input: unknown, options: EmbeddingOptions):
      Promise<WasmResult<WasmEmbeddingResponse>>;
  computeEmbeddingBatch(inputs: unknown[], options: EmbeddingOptions):
      Promise<WasmResult<EmscriptenVector<WasmEmbeddingResponse>>>;
}

/**
 * Interface for the C++ LiteRt LM bindings.
 */
export declare interface LiteRtLmWasm extends WasmModule {
  preinitializedWebGPUDevice?: GPUDevice;
  VectorUint32: VectorUint32Constructor;
  VectorInt: VectorIntConstructor;
  VectorVectorInt: VectorVectorIntConstructor;
  VectorFloat: VectorFloatConstructor;
  VectorEmbeddingResponse: VectorEmbeddingResponseConstructor;
  setupLogging(): void;
  /** Returns an AdvancedSettings object with the C++ default values. */
  createDefaultAdvancedSettings(): AdvancedSettings;
  Backend: BackendEnum;
  ActivationDataType: EmscriptenEnum<{
    FLOAT32: ActivationDataType; FLOAT16: ActivationDataType;
    INT16: ActivationDataType;
    INT8: ActivationDataType;
  }>;
  SpeculativeDecodingMode: EmscriptenEnum<{
    OFF: SpeculativeDecodingMode; ON: SpeculativeDecodingMode;
    ADAPTIVE: SpeculativeDecodingMode;
  }>;
  ModelAssets: ModelAssetsConstructor;
  EngineSettings: EngineSettingsConstructor;
  LlmExecutorSettings: LlmExecutorSettingsConstructor;
  Engine: EngineConstructor;
  EmbeddingEngineSettings: EmbeddingEngineSettingsConstructor;
  EmbeddingEngine: EmbeddingEngineConstructor;
  SessionConfig: SessionConfigConstructor;
  ReadableStreamDataStream: ReadableStreamDataStreamConstructor;
  ConversationConfig: ConversationConfigConstructor;
  Conversation: ConversationConstructor;
}
