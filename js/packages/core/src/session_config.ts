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

import {getGlobalLiteRtLm} from './global_litertlm.js';
import {RecursiveRequired} from './types.js';
import {Backend, SamplerType, SessionConfig as WasmSessionConfig} from './wasm_binding_types.js';
import {consumeEmscriptenVectorToArray, fillEmscriptenVector} from './wasm_utils.js';

// Re-export SamplerType as it's compatible with the TS API.
export {SamplerType};

/**
 * Configures how to sample the next token.
 */
export interface SamplerParameters {
  type?: SamplerType;
  k?: number;
  p?: number;
  temperature?: number;
  seed?: number;
}

/**
 * Configures a LiteRT-LM Session.
 */
export interface SessionConfig {
  audioModalityEnabled?: boolean;
  visionModalityEnabled?: boolean;
  samplerParams?: SamplerParameters;
  stopTokenIds?: number[][];
  startTokenId?: number;
  numOutputCandidates?: number;
  samplerBackend?: Backend;
  applyPromptTemplateInSession?: boolean;
  useExternalSampler?: boolean;
  maxOutputTokens?: number;
}

/**
 * Converts a SessionConfig to a WasmSessionConfig.
 */
export function sessionConfigToWasmSessionConfig(
    sessionConfig: SessionConfig,
    wasm = getGlobalLiteRtLm().liteRtLmWasm): WasmSessionConfig {
  const wasmSessionConfig = wasm.SessionConfig.createDefault();

  if (sessionConfig.audioModalityEnabled !== undefined) {
    wasmSessionConfig.setAudioModalityEnabled(
        sessionConfig.audioModalityEnabled);
  }
  if (sessionConfig.visionModalityEnabled !== undefined) {
    wasmSessionConfig.setVisionModalityEnabled(
        sessionConfig.visionModalityEnabled);
  }
  if (sessionConfig.samplerParams !== undefined) {
    const samplerParams = wasmSessionConfig.getMutableSamplerParams();
    if (sessionConfig.samplerParams.type !== undefined) {
      samplerParams.setType({value: sessionConfig.samplerParams.type});
    }
    if (sessionConfig.samplerParams.k !== undefined) {
      samplerParams.setK(sessionConfig.samplerParams.k);
    }
    if (sessionConfig.samplerParams.p !== undefined) {
      samplerParams.setP(sessionConfig.samplerParams.p);
    }
    if (sessionConfig.samplerParams.temperature !== undefined) {
      samplerParams.setTemperature(sessionConfig.samplerParams.temperature);
    }
    if (sessionConfig.samplerParams.seed !== undefined) {
      samplerParams.setSeed(sessionConfig.samplerParams.seed);
    }
  }
  if (sessionConfig.startTokenId !== undefined) {
    wasmSessionConfig.setStartTokenId(sessionConfig.startTokenId);
  }
  if (sessionConfig.numOutputCandidates !== undefined) {
    wasmSessionConfig.setNumOutputCandidates(sessionConfig.numOutputCandidates);
  }
  if (sessionConfig.samplerBackend !== undefined) {
    wasmSessionConfig.setSamplerBackend({value: sessionConfig.samplerBackend});
  }
  if (sessionConfig.applyPromptTemplateInSession !== undefined) {
    wasmSessionConfig.setApplyPromptTemplateInSession(
        sessionConfig.applyPromptTemplateInSession);
  }
  if (sessionConfig.useExternalSampler !== undefined) {
    wasmSessionConfig.setUseExternalSampler(sessionConfig.useExternalSampler);
  }
  if (sessionConfig.maxOutputTokens !== undefined) {
    wasmSessionConfig.setMaxOutputTokens(sessionConfig.maxOutputTokens);
  }
  if (sessionConfig.stopTokenIds !== undefined) {
    const outerVec = new wasm.VectorVectorInt();
    for (const innerArr of sessionConfig.stopTokenIds) {
      const innerVec = new wasm.VectorInt();
      fillEmscriptenVector(innerArr, innerVec);
      outerVec.push_back(innerVec);
      innerVec.delete();
    }
    wasmSessionConfig.setStopTokenIds(outerVec);
    outerVec.delete();
  }

  return wasmSessionConfig;
}

/**
 * Converts a WasmSessionConfig to a SessionConfig.
 *
 * All fields will be populated with the WASM object's values, which may be
 * default values if they were not set.
 */
export function wasmSessionConfigToSessionConfig(
    wasmSessionConfig: WasmSessionConfig): RecursiveRequired<SessionConfig> {
  const samplerParams = wasmSessionConfig.getMutableSamplerParams();

  return {
    audioModalityEnabled: wasmSessionConfig.getAudioModalityEnabled(),
    visionModalityEnabled: wasmSessionConfig.getVisionModalityEnabled(),
    samplerParams: {
      type: samplerParams.type().value,
      k: samplerParams.k(),
      p: samplerParams.p(),
      temperature: samplerParams.temperature(),
      seed: samplerParams.seed(),
    },
    stopTokenIds:
        consumeEmscriptenVectorToArray(wasmSessionConfig.getStopTokenIds())
            .map(consumeEmscriptenVectorToArray),
    startTokenId: wasmSessionConfig.getStartTokenId(),
    numOutputCandidates: wasmSessionConfig.getNumOutputCandidates(),
    samplerBackend: wasmSessionConfig.getSamplerBackend().value,
    applyPromptTemplateInSession:
        wasmSessionConfig.getApplyPromptTemplateInSession(),
    useExternalSampler: wasmSessionConfig.getUseExternalSampler(),
    maxOutputTokens: wasmSessionConfig.getMaxOutputTokens(),
  };
}
