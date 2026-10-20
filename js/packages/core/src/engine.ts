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
import {EngineSettings, fillWasmEngineSettingsFromEngineSettings, GpuArtisanConfig, ResolvedEngineSettings, wasmEngineSettingsToEngineSettings} from './engine_settings.js';
import {getOrLoadGlobalLiteRtLm} from './load_litertlm.js';
import {Mutex} from './mutex.js';
import {Session} from './session.js';
import {SessionConfig, sessionConfigToWasmSessionConfig} from './session_config.js';
import {createStreamingModelAssets} from './stream_utils.js';
import {Backend, ConversationConfig as WasmConversationConfig, Deletable, Engine as WasmEngine, LiteRtLmWasm, SessionConfig as WasmSessionConfig} from './wasm_binding_types.js';

/**
 * LiteRT-LM Engine
 */
export class Engine implements Deletable {
  readonly settings: ResolvedEngineSettings;
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
    const visionBackend = engineSettings.visionBackend;
    if (backend === Backend.GPU || backend === Backend.GPU_ARTISAN ||
        samplerBackend === Backend.GPU ||
        samplerBackend === Backend.GPU_ARTISAN ||
        visionBackend === Backend.GPU) {
      await litertlm.setupDefaultWebGpuDevice();
    }

    const cleanup = new Cleanup();

    let engine: WasmEngine;
    try {
      const {modelAssets, cleanupModelAssets} =
          await createStreamingModelAssets(engineSettings.model, wasm, cleanup);

      const wasmEngineSettings = visionBackend === undefined ?
          await wasm.EngineSettings.createDefault(
              modelAssets, {value: backend}) :
          await wasm.EngineSettings.createDefaultMultimodal(
              modelAssets, {value: backend}, {value: visionBackend},
              /* audioBackend= */ undefined);
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

      try {
        if (resolvedBackend === Backend.GPU_ARTISAN) {
          engine = await wasm.Engine.createLegacyEngine(
              wasmEngineSettings, inputPromptAsHint);
        } else {
          engine = await wasm.Engine.createEngine(
              wasmEngineSettings, inputPromptAsHint);
        }
      } finally {
        if (resolvedBackend === Backend.GPU || visionBackend === Backend.GPU) {
          try {
            await wasm.clearStoredWeightsStreams();
          } catch (cleanupError) {
            console.error('Error clearing stored weights streams:', cleanupError);
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

  /**
   * Returns `sessionConfig` with modalities supported by this engine enabled
   * unless they were explicitly configured.
   */
  private withDefaultModalities(sessionConfig: SessionConfig): SessionConfig {
    if (this.settings.visionBackend !== undefined &&
        sessionConfig.visionModalityEnabled === undefined) {
      return {...sessionConfig, visionModalityEnabled: true};
    }
    return sessionConfig;
  }

  async createSession(sessionConfig: SessionConfig = {}): Promise<Session> {
    return this.mutexes.executor.acquireAndRun(() => {
      const wasmSessionConfig = sessionConfigToWasmSessionConfig(
          this.withDefaultModalities(sessionConfig), this.wasm);

      const wasmSession = this.engine.createSession(wasmSessionConfig);
      wasmSessionConfig.delete();
      return new Session(wasmSession, this.mutexes);
    });
  }

  async createConversation(config?: ConversationConfig): Promise<Conversation> {
    if (!config && this.settings.visionBackend !== undefined) {
      // A custom config is needed to enable the vision modality. Match the C++
      // ConversationConfig defaults.
      config = {filterChannelContentFromKvCache: true};
    }
    return this.mutexes.executor.acquireAndRun(async () => {
      let wasmConfig: WasmConversationConfig;
      let wasmSessionConfig: WasmSessionConfig|undefined;

      if (config) {
        wasmSessionConfig = sessionConfigToWasmSessionConfig(
            this.withDefaultModalities(config.sessionConfig || {}), this.wasm);
        const prefaceJson =
            config.preface ? JSON.stringify(config.preface) : '';
        const usesAutosizedRingbuffers =
            this.settings.backend === Backend.GPU_ARTISAN &&
            Boolean(
                (this.settings.mainExecutorSettings?.backendConfig as
                 GpuArtisanConfig | undefined)
                    ?.use_autosized_ringbuffers);
        const filterChannelContentFromKvCache =
            config.filterChannelContentFromKvCache ?? usesAutosizedRingbuffers;
        wasmConfig = this.wasm.ConversationConfig.createCustom(
            this.engine, wasmSessionConfig, !!config.enableConstrainedDecoding,
            !!config.prefillPrefaceOnInit, filterChannelContentFromKvCache,
            prefaceJson);
      } else {
        wasmConfig = this.wasm.ConversationConfig.createDefault(this.engine);
      }

      const wasmConversation =
          await this.wasm.Conversation.create(this.engine, wasmConfig);

      wasmConfig.delete();
      if (wasmSessionConfig) {
        wasmSessionConfig.delete();
      }

      return new Conversation(
          wasmConversation, this.engine, this.mutexes,
          this.settings.maxVisionTokensPerImage ?? 0);
    });
  }

  async delete(): Promise<void> {
    await this.mutexes.executor.acquireAndRun(() => {
      this.deleteCallback();
    });
  }
}

