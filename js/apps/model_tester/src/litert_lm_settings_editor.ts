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

import {type AdvancedSettings, Backend, type CpuConfig, type EngineSettings, type GpuArtisanConfig, type GpuConfig, SamplerType, type SessionConfig} from '@litert-lm/core';
import {css, html, LitElement} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

// tslint:disable:no-new-decorators

function isChanged<T>(
    a: T|undefined, b: Partial<T>|undefined, property: keyof T): boolean {
  if (a == null || b == null) return false;
  const valA = a[property];
  const valB = b[property];

  if (valA instanceof Set && valB instanceof Set) {
    if (valA.size !== valB.size) return true;
    for (const item of valA) {
      if (!valB.has(item)) return true;
    }
    return false;
  }

  if (Array.isArray(valA) && Array.isArray(valB)) {
    if (valA.length !== valB.length) return true;
    return valA.some((item, index) => item !== valB[index]);
  }

  return valA !== valB;
}

/**
 * A component for editing LiteRT LM model settings.
 */
@customElement('litert-lm-settings-editor')
export class LitertLmSettingsEditor extends LitElement {
  static override styles = css`
    :host {
      display: block;
    }
    .config-section {
      margin-bottom: 24px;
    }
    .config-section h3 {
      margin-top: 0;
      padding-bottom: 8px;
      border-bottom: 1px solid #eee;
    }
    .config-field {
      display: flex;
      flex-direction: column;
      margin-bottom: 8px;
      font-size: 13px;
      padding: 4px;
      border-radius: 4px;
      transition: background-color 0.2s;
    }
    .config-field.changed {
      background-color: #fff3e0;
    }
    .config-field label {
      font-weight: bold;
      margin-bottom: 4px;
    }
    .config-field input[type="number"], .config-field input[type="text"], select {
      padding: 6px;
      border: 1px solid #aaa;
      border-radius: 4px;
      font-family: monospace;
      width: 100%;
      box-sizing: border-box;
    }
    .config-field input[type="checkbox"] {
      margin-top: 4px;
      align-self: flex-start;
    }
  `;

  // These are the settings that are currently loaded for the model.
  // They're used to determine whether a field has been changed.
  @property({type: Object}) loadedSettings?: Partial<EngineSettings>;
  @property({type: Object}) loadedSessionConfig?: SessionConfig;

  private cpuConfig: CpuConfig = {
    kv_increment_size: 16,
    prefill_chunk_size: -1,
    number_of_threads: 4,
  } satisfies CpuConfig;

  private gpuConfig: GpuConfig = {
    max_top_k: 64,
    external_tensor_mode: false,
  } satisfies GpuConfig;

  private gpuArtisanConfig: GpuArtisanConfig = {
    num_output_candidates: 1,
    wait_for_weight_uploads: false,
    num_decode_steps_per_sync: 1,
    sequence_batch_size: 0,
    supported_lora_ranks: [],
    max_top_k: 64,
    enable_decode_logits: false,
    enable_external_embeddings: false,
    use_submodel: false,
    use_autosized_ringbuffers: false,
  };

  @state()
  private editedSettings: Omit<Required<EngineSettings>, 'model'> = {
    backend: Backend.GPU_ARTISAN,
    mainExecutorSettings: {
      maxNumTokens: 128,
      samplerBackend: Backend.GPU_ARTISAN,
      backendConfig: this.cpuConfig,
      advancedSettings: {
        prefill_batch_sizes: [],
        num_output_candidates: 1,
        configure_magic_numbers: true,
        verify_magic_numbers: false,
        clear_kv_cache_before_prefill: true,
        num_logits_to_print_after_decode: 0,
        gpu_madvise_original_shared_tensors: true,
        is_benchmark: false,
        preferred_device_substr: '',
        num_threads_to_upload: -1,
        num_threads_to_compile: -1,
        convert_weights_on_gpu: true,
        optimize_shader_compilation: true,
        share_constant_tensors: true,
      },
    },
    benchmarkEnabled: false,
  };

  @state()
  private editedSessionConfig: SessionConfig = {
    samplerParams: {
      type: SamplerType.TYPE_UNSPECIFIED as SamplerType,
      k: 3,
      p: 1.0,
      temperature: 1.0,
      seed: 0,
    },
    maxOutputTokens: 120,
    applyPromptTemplateInSession: true,
    numOutputCandidates: 1,
    audioModalityEnabled: false,
    visionModalityEnabled: false,
  };

  getSessionConfig(): SessionConfig {
    return this.editedSessionConfig;
  }

  override updated(changedProperties: Map<string, unknown>) {
    super.updated(changedProperties);
    if (changedProperties.has('editedSessionConfig')) {
      this.dispatchEvent(new CustomEvent('session-config-changed', {
        detail: this.editedSessionConfig,
        bubbles: true,
        composed: true,
      }));
    }
  }

  getSettings(): Partial<EngineSettings> {
    return structuredClone(this.editedSettings);
  }

  override render() {
    return html`
      <div class="config-section">
        <h3>Session Config</h3>
        ${this.renderSessionConfig()}
      </div>
      <div class="config-section">
        <h3>Engine Settings</h3>
        ${this.renderEngineSettings()}
      </div>
    `;
  }

  private renderSessionConfig() {
    const config = this.editedSessionConfig;
    const loaded = this.loadedSessionConfig;

    return html`
      ${
        this.renderField(
            'Max Output Tokens', config, loaded, 'maxOutputTokens', 'number',
            (val) => {
              this.editedSessionConfig = {...config, maxOutputTokens: val};
            })}
      ${
        this.renderField(
            'Apply Prompt Template', config, loaded,
            'applyPromptTemplateInSession', 'checkbox',
            (val) => {
              this.editedSessionConfig = {
                ...config,
                applyPromptTemplateInSession: val
              };
            })}
      ${
        this.renderField(
            'Audio Modality', config, loaded, 'audioModalityEnabled',
            'checkbox',
            (val) => {
              this.editedSessionConfig = {...config, audioModalityEnabled: val};
            })}
      ${
        this.renderField(
            'Vision Modality', config, loaded, 'visionModalityEnabled',
            'checkbox', (val) => {
              this.editedSessionConfig = {
                ...config,
                visionModalityEnabled: val
              };
            })}
      
      <div class="config-section">
        <label>Sampler Parameters</label>
        ${this.renderSamplerParams()}
      </div>
    `;
  }

  private renderSamplerParams() {
    const params = this.editedSessionConfig.samplerParams!;
    const loadedParams = this.loadedSessionConfig?.samplerParams;

    return html`
      <div class="config-field ${
        isChanged(
            this.editedSessionConfig, this.loadedSessionConfig,
            'samplerParams') ?
            'changed' :
            ''}">
        <label>Type</label>
        <select .value=${
        params.type?.toString() ?? ''} @change=${(e: Event) => {
      const type = Number((e.target as HTMLSelectElement).value) as SamplerType;
      this.editedSessionConfig = {
        ...this.editedSessionConfig,
        samplerParams: {...params, type}
      };
    }}>
          <option value=${SamplerType.TYPE_UNSPECIFIED}>Unspecified</option>
          <option value=${SamplerType.GREEDY}>Greedy</option>
          <option value=${SamplerType.TOP_K}>Top K</option>
          <option value=${SamplerType.TOP_P}>Top P</option>
        </select>
      </div>
      ${
        this.renderField(
            'K', params, loadedParams, 'k', 'number',
            (val) => {
              this.editedSessionConfig = {
                ...this.editedSessionConfig,
                samplerParams: {...params, k: val}
              };
            })}
      ${
        this.renderField(
            'P', params, loadedParams, 'p', 'number',
            (val) => {
              this.editedSessionConfig = {
                ...this.editedSessionConfig,
                samplerParams: {...params, p: val}
              };
            })}
      ${
        this.renderField(
            'Temperature', params, loadedParams, 'temperature', 'number',
            (val) => {
              this.editedSessionConfig = {
                ...this.editedSessionConfig,
                samplerParams: {...params, temperature: val}
              };
            })}
      ${
        this.renderField(
            'Seed', params, loadedParams, 'seed', 'number', (val) => {
              this.editedSessionConfig = {
                ...this.editedSessionConfig,
                samplerParams: {...params, seed: val}
              };
            })}
    `;
  }

  private renderEngineSettings() {
    const settings = this.editedSettings;
    const loaded = this.loadedSettings;

    return html`
      <div class="config-field ${
        isChanged(settings, loaded, 'backend') ? 'changed' : ''}">
        <label>Backend</label>
        <select .value=${settings.backend.toString()} @change=${
        this.onBackendChange}>
          <option value=${Backend.CPU}>CPU</option>
          <option value=${Backend.GPU}>GPU</option>
          <option value=${Backend.GPU_ARTISAN}>GPU Artisan</option>
        </select>
      </div>

      ${
        this.renderField(
            'Max Num Tokens', settings.mainExecutorSettings,
            loaded?.mainExecutorSettings, 'maxNumTokens', 'number', (val) => {
              this.editedSettings = {
                ...settings,
                mainExecutorSettings:
                    {...settings.mainExecutorSettings, maxNumTokens: val}
              };
            })}

      <div class="config-section">
        <h4>Backend Config (${this.getBackendName(settings.backend)})</h4>
        ${this.renderBackendConfig()}
      </div>

      <div class="config-section">
        <h4>Advanced Settings</h4>
        ${this.renderAdvancedSettings()}
      </div>
    `;
  }

  private renderBackendConfig() {
    const backend = this.editedSettings.backend;
    switch (backend) {
      case Backend.CPU:
        return this.renderCpuConfig();
      case Backend.GPU:
        return this.renderGpuConfig();
      case Backend.GPU_ARTISAN:
        return this.renderGpuArtisanConfig();
      default:
        return html`<div>No config for this backend</div>`;
    }
  }

  private renderCpuConfig() {
    const config = this.cpuConfig;
    const loaded = this.getLoadedBackendConfig(Backend.CPU) as CpuConfig;
    return html`
      ${
        this.renderField(
            'KV Increment Size', config, loaded, 'kv_increment_size', 'number',
            (val) => {
              this.cpuConfig.kv_increment_size = val;
              this.requestUpdate();
            },
            'The increment size of the kv-cache. Each time during decode, the kv-cache size is increased by this size.')}
      ${
        this.renderField(
            'Prefill Chunk Size', config, loaded, 'prefill_chunk_size',
            'number',
            (val) => {
              this.cpuConfig.prefill_chunk_size = val;
              this.requestUpdate();
            },
            'The maximum number of tokens to process in a single prefill chunk. -1 indicates no chunking.')}
      ${
        this.renderField(
            'Number of Threads', config, loaded, 'number_of_threads',
            'number', (val) => {
              this.cpuConfig.number_of_threads = val;
              this.requestUpdate();
            }, 'Number of threads.')}
    `;
  }

  private renderGpuConfig() {
    const config = this.gpuConfig;
    const loaded = this.getLoadedBackendConfig(Backend.GPU) as GpuConfig;
    return html`
      ${
        this.renderField(
            'Max Top K', config, loaded, 'max_top_k', 'number',
            (val) => {
              this.gpuConfig.max_top_k = val;
              this.requestUpdate();
            },
            'Maximum Top-K value supported for all sessions created with the engine.')}
      ${
        this.renderField(
            'External Tensor Mode', config, loaded, 'external_tensor_mode',
            'checkbox', (val) => {
              this.gpuConfig.external_tensor_mode = val;
              this.requestUpdate();
            }, 'Whether to use external tensor mode.')}
    `;
  }

  private renderGpuArtisanConfig() {
    const config = this.gpuArtisanConfig;
    const loaded =
        this.getLoadedBackendConfig(Backend.GPU_ARTISAN) as GpuArtisanConfig;
    return html`
      ${
        this.renderField(
            'Num Output Candidates', config, loaded, 'num_output_candidates',
            'number',
            (val) => {
              this.gpuArtisanConfig.num_output_candidates = val;
              this.requestUpdate();
            },
            'Number of output candidates.')}
      ${
        this.renderField(
            'Wait For Weight Uploads', config, loaded,
            'wait_for_weight_uploads', 'checkbox',
            (val) => {
              this.gpuArtisanConfig.wait_for_weight_uploads = val;
              this.requestUpdate();
            },
            'Whether to wait for weight uploads before prefilling.')}
      ${
        this.renderField(
            'Num Decode Steps Per Sync', config, loaded,
            'num_decode_steps_per_sync', 'number',
            (val) => {
              this.gpuArtisanConfig.num_decode_steps_per_sync = val;
              this.requestUpdate();
            },
            'Number of decode steps per sync. Used by GPU only.')}
      ${
        this.renderField(
            'Sequence Batch Size', config, loaded, 'sequence_batch_size',
            'number',
            (val) => {
              this.gpuArtisanConfig.sequence_batch_size = val;
              this.requestUpdate();
            },
            'Sequence batch size for encoding. Used by GPU only.')}
      ${
        this.renderField(
            'Supported LoRA Ranks (comma separated)', config, loaded,
            'supported_lora_ranks', 'text',
            (val: string) => {
              this.gpuArtisanConfig.supported_lora_ranks =
                  val.split(',')
                      .map(s => Number(s.trim()))
                      .filter(n => !isNaN(n));
              this.requestUpdate();
            },
            'The supported lora ranks for the base model. Used by GPU only.')}
      ${
        this.renderField(
            'Max Top K', config, loaded, 'max_top_k', 'number',
            (val) => {
              this.gpuArtisanConfig.max_top_k = val;
              this.requestUpdate();
            },
            'Maximum Top-K value supported for all sessions created with the engine.')}
      ${
        this.renderField(
            'Enable Decode Logits', config, loaded, 'enable_decode_logits',
            'checkbox',
            (val) => {
              this.gpuArtisanConfig.enable_decode_logits = val;
              this.requestUpdate();
            },
            'Enables decode logits.')}
      ${
        this.renderField(
            'Enable External Embeddings', config, loaded,
            'enable_external_embeddings', 'checkbox',
            (val) => {
              this.gpuArtisanConfig.enable_external_embeddings = val;
              this.requestUpdate();
            },
            'Enables external embeddings.')}
      ${
        this.renderField(
            'Use Submodel', config, loaded, 'use_submodel',
            'checkbox', (val) => {
              this.gpuArtisanConfig.use_submodel = val;
              this.requestUpdate();
            }, 'Whether the submodel should be used if available.')}
      ${
        this.renderField(
            'Use Autosized Ringbuffers', config, loaded, 'use_autosized_ringbuffers',
            'checkbox', (val) => {
              this.gpuArtisanConfig.use_autosized_ringbuffers = val;
              this.requestUpdate();
            }, 'For low GPU memory with long contexts, use minimally sized ringbuffers for local attention. Prevents instantaneous rewinding, but saves a lot of memory, especially for large models.')}
    `;
  }

  private renderAdvancedSettings() {
    const config = this.editedSettings.mainExecutorSettings.advancedSettings!;
    const loaded = this.loadedSettings?.mainExecutorSettings?.advancedSettings;

    return html`
      ${
        this.renderField(
            'Prefill Batch Sizes (comma separated)', config, loaded,
            'prefill_batch_sizes', 'text',
            (val: string) => {
              const sizes = val.split(',')
                                .map(s => Number(s.trim()))
                                .filter(n => !isNaN(n));
              this.updateAdvanced({prefill_batch_sizes: sizes});
            },
            'Ordered set of the maximum number of prefill tokens processed at once when the graph has dynamic prefill lengths.')}
      ${
        this.renderField(
            'Num Output Candidates', config, loaded, 'num_output_candidates',
            'number',
            (val) => {
              this.updateAdvanced({num_output_candidates: val});
            },
            'The number of output candidates, or the decode batch size.')}
      ${
        this.renderField(
            'Configure Magic Numbers', config, loaded,
            'configure_magic_numbers', 'checkbox',
            (val) => {
              this.updateAdvanced({configure_magic_numbers: val});
            },
            'Whether to configure magic numbers when the model contains magic numbers.')}
      ${
        this.renderField(
            'Verify Magic Numbers', config, loaded, 'verify_magic_numbers',
            'checkbox',
            (val) => {
              this.updateAdvanced({verify_magic_numbers: val});
            },
            'Whether to verify magic numbers when the model contains magic numbers and test signatures.')}
      ${
        this.renderField(
            'Clear KV Cache Before Prefill', config, loaded,
            'clear_kv_cache_before_prefill', 'checkbox',
            (val) => {
              this.updateAdvanced({clear_kv_cache_before_prefill: val});
            },
            'Whether to clear kv cache before the first prefill step.')}
      ${
        this.renderField(
            'Num Logits To Print After Decode', config, loaded,
            'num_logits_to_print_after_decode', 'number',
            (val) => {
              this.updateAdvanced({num_logits_to_print_after_decode: val});
            },
            'For debugging purpose, the number of values to print after each decode step.')}
      ${
        this.renderField(
            'GPU Madvise Shared Tensors', config, loaded,
            'gpu_madvise_original_shared_tensors', 'checkbox',
            (val) => {
              this.updateAdvanced({gpu_madvise_original_shared_tensors: val});
            },
            'If true, the GPU backend will madvise the original shared tensors after use.')}
      ${
        this.renderField(
            'Is Benchmark', config, loaded, 'is_benchmark', 'checkbox',
            (val) => {
              this.updateAdvanced({is_benchmark: val});
            },
            'If true, the executor is running a benchmark.')}
      ${
        this.renderField(
            'Preferred Device Substr', config, loaded,
            'preferred_device_substr', 'text',
            (val) => {
              this.updateAdvanced({preferred_device_substr: val});
            },
            'Preferred WebGPU device name substring, case-insensitive.')}
      ${
        this.renderField(
            'Num Threads To Upload', config, loaded, 'num_threads_to_upload',
            'number',
            (val) => {
              this.updateAdvanced({num_threads_to_upload: val});
            },
            'Number of threads for WebGPU weight upload. -1 means determined by the runtime.')}
      ${
        this.renderField(
            'Num Threads To Compile', config, loaded, 'num_threads_to_compile',
            'number',
            (val) => {
              this.updateAdvanced({num_threads_to_compile: val});
            },
            'Number of threads for WebGPU kernel shader compilation. -1 means determined by the runtime.')}
      ${
        this.renderField(
            'Convert Weights On GPU', config, loaded, 'convert_weights_on_gpu',
            'checkbox',
            (val) => {
              this.updateAdvanced({convert_weights_on_gpu: val});
            },
            'If true, the executor will convert weights on GPU.')}
      ${
        this.renderField(
            'Optimize Shader Compilation', config, loaded,
            'optimize_shader_compilation', 'checkbox',
            (val) => {
              this.updateAdvanced({optimize_shader_compilation: val});
            },
            'If true, the executor enables Vulkan kernel shader optimization.')}
      ${
        this.renderField(
            'Share Constant Tensors', config, loaded, 'share_constant_tensors',
            'checkbox', (val) => {
              this.updateAdvanced({share_constant_tensors: val});
            }, 'If true, the executor enables constant tensor sharing.')}
    `;
  }

  private renderField<T, K extends keyof T,
                                   Type extends 'number'|'text'|'checkbox'>(
      label: string, obj: T, loadedObj: T|undefined, key: K, type: Type,
      onChange: (val: Type extends 'checkbox'? boolean: Type extends 'number'?
                 number: string) => void,
      description?: string) {
    const value = obj[key];
    let displayValue: string;
    if (value instanceof Set) {
      displayValue = Array.from(value).join(', ');
    } else if (Array.isArray(value)) {
      displayValue = value.join(', ');
    } else {
      displayValue = value?.toString() ?? '';
    }

    const changed = isChanged(obj, loadedObj, key);
    const isCollection = value instanceof Set || Array.isArray(value);

    return html`
      <div class="config-field ${changed ? 'changed' : ''}" title=${
        description ?? ''}>
        <label>${label}</label>
        ${
        type === 'checkbox' ?
            html`
          <input type="checkbox" ?checked=${value as boolean} @change=${
                (e: Event) => (onChange as (val: boolean) => void)(
                    (e.target as HTMLInputElement).checked)}>
        ` :
            html`
          <input type="${type}" .value=${displayValue} @input=${(e: Event) => {
              if (isCollection) return;  // Use @change for collections
              const val = type === 'number' ?
                  Number((e.target as HTMLInputElement).value) :
                  (e.target as HTMLInputElement).value;
              if (type === 'number') {
                (onChange as (val: number) => void)(val as number);
              } else {
                (onChange as (val: string) => void)(val as string);
              }
            }} @change=${(e: Event) => {
              if (!isCollection) return;
              (onChange as (val: string) =>
                   void)((e.target as HTMLInputElement).value);
            }}>
        `}
      </div>
    `;
  }

  private updateAdvanced(patch: Partial<AdvancedSettings>) {
    this.editedSettings = {
      ...this.editedSettings,
      mainExecutorSettings: {
        ...this.editedSettings.mainExecutorSettings,
        advancedSettings: {
          ...this.editedSettings.mainExecutorSettings.advancedSettings!,
          ...patch
        },
      }
    };
  }

  private onBackendChange(e: Event) {
    const backend = Number((e.target as HTMLSelectElement).value) as Backend;
    this.editedSettings = {
      ...this.editedSettings,
      backend,
      mainExecutorSettings: {
        ...this.editedSettings.mainExecutorSettings,
        backendConfig: this.getBackendConfig(backend),
      }
    };
  }

  private getBackendConfig(backend: Backend) {
    switch (backend) {
      case Backend.GPU:
        return this.gpuConfig;
      case Backend.GPU_ARTISAN:
        return this.gpuArtisanConfig;
      case Backend.CPU:
      default:
        return this.cpuConfig;
    }
  }

  private getLoadedBackendConfig(backend: Backend) {
    if (!this.loadedSettings?.mainExecutorSettings?.backendConfig) {
      return undefined;
    }
    // We only return it if the loaded backend matches the requested one.
    if (this.loadedSettings.backend === backend) {
      return this.loadedSettings.mainExecutorSettings.backendConfig;
    }
    return undefined;
  }

  private getBackendName(backend: Backend) {
    switch (backend) {
      case Backend.CPU:
        return 'CPU';
      case Backend.GPU:
        return 'GPU';
      case Backend.GPU_ARTISAN:
        return 'GPU Artisan';
      default:
        return 'Unknown';
    }
  }
}
