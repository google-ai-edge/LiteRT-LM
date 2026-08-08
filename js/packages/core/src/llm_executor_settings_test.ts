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

import {LiteRtLm, loadLiteRtLm, unloadLiteRtLm, type Wasm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

describe('LlmExecutorSettings', () => {
  let liteRtLm: LiteRtLm;
  let modelAssets: Wasm.ModelAssets;
  let engineSettings: Wasm.EngineSettings;
  let llmExecutorSettings: Wasm.LlmExecutorSettings;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
  });

  beforeEach(() => {
    modelAssets = liteRtLm.liteRtLmWasm.ModelAssets.create('/path/to/model');
    engineSettings = liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
        modelAssets, liteRtLm.liteRtLmWasm.Backend.CPU);
    llmExecutorSettings = engineSettings.getMutableMainExecutorSettings();
  });

  afterEach(() => {
    // Do not delete llmExecutorSettings, as it is owned by engineSettings.
    engineSettings.delete();
    modelAssets.delete();
  });

  it('sets and gets max num tokens', () => {
    llmExecutorSettings.setMaxNumTokens(100);
    expect(llmExecutorSettings.getMaxNumTokens()).toBe(100);
  });

  it('sets and gets backend config CPU', () => {
    const cpuConfig = {
      kv_increment_size: 1,
      prefill_chunk_size: 2,
      number_of_threads: 3,
    };
    llmExecutorSettings.setBackendConfigCpu(cpuConfig);
    expect(llmExecutorSettings.getBackendConfigCpu()).toEqual(cpuConfig);
  });

  it('sets and gets backend config GPU', () => {
    const gpuConfig = {
      max_top_k: 42,
      external_tensor_mode: true,
    };
    llmExecutorSettings.setBackendConfigGpu(gpuConfig);
    expect(llmExecutorSettings.getBackendConfigGpu()).toEqual(gpuConfig);
  });

  it('throws error if getting CPU config when GPU config is set', () => {
    const gpuConfig = {
      max_top_k: 42,
      external_tensor_mode: true,
    };
    llmExecutorSettings.setBackendConfigGpu(gpuConfig);
    expect(() => {
      llmExecutorSettings.getBackendConfigCpu();
    }).toThrowError(/Backend config is not valid/);
  });

  it('sets and gets sampler backend', () => {
    llmExecutorSettings.setSamplerBackend(liteRtLm.liteRtLmWasm.Backend.CPU);
    expect(llmExecutorSettings.getSamplerBackend())
        .toBe(liteRtLm.liteRtLmWasm.Backend.CPU);

    llmExecutorSettings.setSamplerBackend(liteRtLm.liteRtLmWasm.Backend.GPU);
    expect(llmExecutorSettings.getSamplerBackend())
        .toBe(liteRtLm.liteRtLmWasm.Backend.GPU);
  });

  it('sets and gets advanced settings', () => {
    const advancedSettings = {
      prefill_batch_sizes: [1, 2, 3],
      num_output_candidates: 4,
      configure_magic_numbers: true,
      verify_magic_numbers: false,
      clear_kv_cache_before_prefill: true,
      num_logits_to_print_after_decode: 5,
      gpu_madvise_original_shared_tensors: true,
      is_benchmark: true,
      preferred_device_substr: 'test_device',
      num_threads_to_upload: 5,
      num_threads_to_compile: 6,
      convert_weights_on_gpu: true,
      optimize_shader_compilation: true,
      share_constant_tensors: true,
    };
    llmExecutorSettings.setAdvancedSettings(advancedSettings);
    expect(llmExecutorSettings.getAdvancedSettings()).toEqual(advancedSettings);
  });
});
