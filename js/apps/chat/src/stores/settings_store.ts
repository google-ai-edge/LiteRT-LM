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

import {z} from 'zod';

/** Schema for custom model stored in the file system. */
export const CustomModelSchema = z.object({
  name: z.string(),
  filename: z.string(),
  path: z.string(),
  size: z.string(),
});

/** Type for custom model stored in the file system. */
export type CustomModel = z.infer<typeof CustomModelSchema>;

/** Compute backend choice for the text modality (`'gpu'` or `'cpu'`). */
export const ModalityBackendSchema = z.enum(['gpu', 'cpu']);

/** Compute backend choice (`'gpu'` or `'cpu'`). */
export type ModalityBackend = z.infer<typeof ModalityBackendSchema>;

/** Compute backend choice for optional modalities (`'gpu'`, `'cpu'`, or `'off'`). */
export const OptionalModalityBackendSchema = z.enum(['gpu', 'cpu', 'off']);

/** Compute backend choice for Vision and Audio (`'gpu'`, `'cpu'`, or `'off'`). */
export type OptionalModalityBackend =
    z.infer<typeof OptionalModalityBackendSchema>;

/** Activation precision override for GPU modalities. */
export const ActivationPrecisionSchema = z.enum(['default', 'fp16', 'fp32']);

/** Activation precision override (`'default'`, `'fp16'`, or `'fp32'`). */
export type ActivationPrecision = z.infer<typeof ActivationPrecisionSchema>;

/** Schema for inference configuration settings. */
export const SettingsSchema = z.object({
  selectedModelPath: z.string(),
  contextLength: z.number().int().positive(),
  maxOutputTokens: z.number().int().positive(),
  samplerType: z.string(),
  temperature: z.number().min(0),
  topP: z.number().min(0).max(1),
  topK: z.number().int().nonnegative(),
  enableThinking: z.boolean(),
  /**
   * Whether to use speculative decoding (e.g. MTP) with models that support
   * it.
   */
  enableSpeculativeDecoding: z.boolean(),
  systemPrompt: z.string(),
  /** Names of the tools in `CHAT_TOOLS` that the model can't use. */
  disabledTools: z.array(z.string()),
  textBackend: ModalityBackendSchema.default('gpu'),
  visionBackend: OptionalModalityBackendSchema.default('gpu'),
  audioBackend: OptionalModalityBackendSchema.default('gpu'),
  textPrecision: ActivationPrecisionSchema.default('default'),
  visionPrecision: ActivationPrecisionSchema.default('default'),
  audioPrecision: ActivationPrecisionSchema.default('default'),
  customModels: z.array(CustomModelSchema).default([]),
  localDirModels: z.array(CustomModelSchema).default([]),
});

/** Schema for partial settings, used for parsing saved settings. */
export const PartialSettingsSchema = SettingsSchema.partial();

/** Type for inference configuration settings. */
export type Settings = z.infer<typeof SettingsSchema>;

/** Model-specific settings subset. */
export type ModelSettings = Pick<
    Settings,
    'selectedModelPath' | 'contextLength' | 'maxOutputTokens' | 'samplerType' |
    'temperature' | 'topP' | 'topK' | 'enableThinking' |
    'enableSpeculativeDecoding' | 'systemPrompt' | 'disabledTools' |
    ExperimentalSettingKey>;

/** The settings in the "Model Config" tab, which can be reset together. */
export const DEFAULT_MODEL_CONFIG = {
  contextLength: 4096,
  maxOutputTokens: 2048,
  samplerType: 'greedy',
  temperature: 1.0,
  topP: 0.95,
  topK: 64,
  enableThinking: true,
  enableSpeculativeDecoding: false,
};

/** The settings in the "System Prompt" tab, which can be reset together. */
export const DEFAULT_SYSTEM_PROMPT_CONFIG = {
  systemPrompt: '',
  disabledTools: [] as string[],
};

/** The settings in the "Experimental" tab, which can be reset together. */
export const EXPERIMENTAL_SETTING_KEYS = [
  'textBackend',
  'visionBackend',
  'audioBackend',
  'textPrecision',
  'visionPrecision',
  'audioPrecision',
] as const;

type ExperimentalSettingKey = (typeof EXPERIMENTAL_SETTING_KEYS)[number];

/** Default values for experimental modality backend and precision settings. */
export const DEFAULT_EXPERIMENTAL_CONFIG: Pick<Settings, ExperimentalSettingKey> = {
  textBackend: 'gpu',
  visionBackend: 'gpu',
  audioBackend: 'gpu',
  textPrecision: 'default',
  visionPrecision: 'default',
  audioPrecision: 'default',
};

/** A group of settings that can be reset to their defaults together. */
export type SettingsGroup = 'modelConfig'|'systemPrompt'|'experimental';

/**
 * List of supported default models available for selection.
 *
 * They're `hf://` URIs, so they load the file that each repo's web model
 * currently is: the `-gpu.litertlm` file, on the repo's `web` branch while it
 * has one. The sizes are approximate.
 */
export const MODELS = [
  {
    name: 'Gemma 4 E2B',
    filename: 'gemma-4-E2B-it-gpu.litertlm',
    path: 'hf://litert-community/gemma-4-E2B-it-litert-lm',
    size: '2.3 GB'
  },
  {
    name: 'Gemma 4 E4B',
    filename: 'gemma-4-E4B-it-gpu.litertlm',
    path: 'hf://litert-community/gemma-4-E4B-it-litert-lm',
    size: '3.3 GB'
  },
  {
    name: 'Gemma 4 12B',
    filename: 'gemma-4-12B-it-gpu.litertlm',
    path: 'hf://litert-community/gemma-4-12B-it-litert-lm',
    size: '6.2 GB'
  },
  {
    name: 'Gemma 4 26B A4B',
    filename: 'gemma-4-26B-A4B-it-gpu.litertlm',
    path: 'hf://litert-community/gemma-4-26B-A4B-it-litert-lm',
    size: '16.4 GB'
  },
  {
    name: 'Gemma 4 31B',
    filename: 'gemma-4-31B-it-gpu.litertlm',
    path: 'hf://litert-community/gemma-4-31B-it-litert-lm',
    size: '19.8 GB'
  }
];

/**
 * The default models' old paths, which loaded their `-web.litertlm` files from
 * `main`, and the paths that replaced them.
 */
export const LEGACY_MODEL_PATHS = new Map(
    MODELS.map(model => {
      const repo = model.path.slice('hf://'.length);
      const oldFile =
          model.filename.replace(/-gpu\.litertlm$/, '-web.litertlm');
      return [
        `https://huggingface.co/${repo}/resolve/main/${oldFile}`,
        model.path,
      ];
    }));

/** Returns the path that replaced a default model's old path, if it is one. */
export function migrateModelPath(path: string): string {
  return LEGACY_MODEL_PATHS.get(path) ?? path;
}

/** Finds a built-in, custom, or local-directory model by path. */
export function findModel(
    models: Partial<Pick<Settings, 'customModels'|'localDirModels'>>,
    path: string): CustomModel|undefined {
  return [
    ...MODELS,
    ...(models.customModels ?? []),
    ...(models.localDirModels ?? []),
  ].find(m => m.path === path);
}

/**
 * Returns true if `modelPath` points to a `-gpu.litertlm` WebGPU Artisan model
 * package, which cannot execute its text decoder on `Backend.CPU`.
 */
export function isGpuOnlyTextModel(
    modelPath: string,
    models?: Partial<Pick<Settings, 'customModels'|'localDirModels'>>):
    boolean {
  const matched = findModel(models ?? {}, modelPath);
  if (matched && /-gpu\.litertlm$/i.test(matched.filename)) {
    return true;
  }
  const cleanPath = modelPath.split(/[?#]/)[0] ?? modelPath;
  return /-gpu\.litertlm$/i.test(cleanPath);
}

/**
 * Practical model size threshold (1.5 GB decimal) at or above which loading
 * text weights into 32-bit WASM linear memory on CPU is likely to fail.
 */
export const CPU_TEXT_WASM_WARNING_BYTES = 1.5e9;

const SIZE_UNIT_MULTIPLIERS: Readonly<Record<string, number>> = {
  'B': 1,
  'KB': 1e3,
  'MB': 1e6,
  'GB': 1e9,
  'TB': 1e12,
  'KIB': 1024,
  'MIB': 1024 ** 2,
  'GIB': 1024 ** 3,
};

/** Parses a human-readable model size (e.g. `'2.3 GB'`, `'850 MB'`) to bytes. */
export function parseModelSizeBytes(sizeLabel: string|undefined): number|null {
  if (!sizeLabel) return null;
  const match = /^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB)\s*$/i.exec(
      sizeLabel);
  if (!match) return null;
  const value = Number(match[1]);
  if (!Number.isFinite(value) || value < 0) return null;
  const mult = SIZE_UNIT_MULTIPLIERS[match[2]!.toUpperCase()];
  return mult ? Math.round(value * mult) : null;
}

/**
 * Returns the known byte size of `modelPath` from `cachedSizeBytes` or its
 * `CustomModel.size` metadata, or `null` when unknown.
 */
export function getKnownModelSizeBytes(
    modelPath: string,
    models?: Partial<Pick<Settings, 'customModels'|'localDirModels'>>,
    cachedSizeBytes?: number): number|null {
  if (cachedSizeBytes !== undefined && Number.isFinite(cachedSizeBytes) &&
      cachedSizeBytes > 0) {
    return cachedSizeBytes;
  }
  const matched = findModel(models ?? {}, modelPath);
  return parseModelSizeBytes(matched?.size);
}

/**
 * Returns a non-blocking warning message when a non-`-gpu.litertlm` model's
 * known size is at or above the ~1.5 GB practical WASM memory limit, or `null`
 * when smaller, unknown, or WebGPU-only (`-gpu.litertlm`).
 */
export function getCpuTextMemoryWarning(
    modelPath: string,
    models?: Partial<Pick<Settings, 'customModels'|'localDirModels'>>,
    cachedSizeBytes?: number): string|null {
  if (isGpuOnlyTextModel(modelPath, models)) {
    return null;
  }
  const bytes = getKnownModelSizeBytes(modelPath, models, cachedSizeBytes);
  if (bytes === null || bytes < CPU_TEXT_WASM_WARNING_BYTES) {
    return null;
  }
  const matched = findModel(models ?? {}, modelPath);
  const sizeLabel = matched?.size?.trim() || `${(bytes / 1e9).toFixed(2)} GB`;
  return `${sizeLabel} may not fit in WASM memory (~1.5 GB practical limit); loading may fail.`;
}

/** Modalities with their own experimental backend and precision settings. */
export const MODALITIES = ['text', 'vision', 'audio'] as const;

/** A modality with its own experimental settings. */
export type Modality = (typeof MODALITIES)[number];

/** The backend and activation precision that a modality runs with. */
export interface ModalityConfig {
  backend: OptionalModalityBackend;
  precision: ActivationPrecision;
}

/** Resolved modality configurations across Text, Vision, and Audio. */
export interface ModalityConfigs {
  text: {backend: ModalityBackend; precision: ActivationPrecision};
  vision: ModalityConfig;
  audio: ModalityConfig;
}

/**
 * Returns the backend and precision each modality runs with. Text always runs
 * on GPU for WebGPU-only (`-gpu.litertlm`) models.
 */
export function getModalityConfigs(
    settings: Pick<Settings, 'selectedModelPath'>&
    Partial<Pick<Settings, ExperimentalSettingKey|'customModels'|'localDirModels'>>):
    ModalityConfigs {
  const gpuOnlyText =
      isGpuOnlyTextModel(settings.selectedModelPath ?? '', settings);
  return {
    text: {
      backend: gpuOnlyText ? 'gpu' : (settings.textBackend ?? 'gpu'),
      precision: settings.textPrecision ?? 'default',
    },
    vision: {
      backend: settings.visionBackend ?? 'gpu',
      precision: settings.visionPrecision ?? 'default',
    },
    audio: {
      backend: settings.audioBackend ?? 'gpu',
      precision: settings.audioPrecision ?? 'default',
    },
  };
}

/** Switches text back to GPU if the selected model can't run it on CPU. */
export function coerceGpuOnlyTextBackend(
    settings: Pick<Settings, 'selectedModelPath'|'textBackend'>&
    Partial<Pick<Settings, 'customModels'|'localDirModels'>>) {
  if (settings.textBackend === 'cpu' &&
      isGpuOnlyTextModel(settings.selectedModelPath, settings)) {
    settings.textBackend = 'gpu';
  }
}

/** Summary of non-default modality backend or disabled-modality overrides. */
export interface NonDefaultBackendSummary {
  label: 'WebGPU · CPU'|'CPU'|'Text only'|'No Vision'|'No Audio';
  detail: string;
}

/**
 * Returns a compact backend summary when any modality is configured off the
 * default all-WebGPU path, or `null` when all modalities run on WebGPU.
 */
export function getNonDefaultBackendSummary(
    settings: Parameters<typeof getModalityConfigs>[0]):
    NonDefaultBackendSummary|null {
  const configs = getModalityConfigs(settings);
  const cap = (m: Modality) => m[0].toUpperCase() + m.slice(1);
  const enabledModalities =
      MODALITIES.filter(m => configs[m].backend !== 'off');
  const offModalities =
      MODALITIES.filter(m => configs[m].backend === 'off').map(cap);
  const cpuModalities =
      enabledModalities.filter(m => configs[m].backend === 'cpu').map(cap);

  if (cpuModalities.length === 0 && offModalities.length === 0) {
    return null;
  }
  if (cpuModalities.length === 0) {
    if (offModalities.length === 2) {
      return {
        label: 'Text only',
        detail: 'Modalities: Text only on WebGPU (Vision, Audio off)',
      };
    }
    const offName = offModalities[0] as 'Vision' | 'Audio';
    const onNames = enabledModalities.map(cap).join(', ');
    return {
      label: offName === 'Vision' ? 'No Vision' : 'No Audio',
      detail: `Modalities: ${offName} off (${onNames} on WebGPU)`,
    };
  }
  const offSuffix =
      offModalities.length > 0 ? `; ${offModalities.join(', ')} off` : '';
  if (cpuModalities.length === enabledModalities.length) {
    return {
      label: 'CPU',
      detail: `Backend: CPU (${enabledModalities.map(cap).join(', ')} in WASM FP32${offSuffix})`,
    };
  }
  return {
    label: 'WebGPU · CPU',
    detail: `Backend: WebGPU · CPU (${cpuModalities.join(', ')} on CPU${offSuffix})`,
  };
}

/** Store for managing inference configuration and persistence. */
export class SettingsStore implements Settings {
  // Active model configuration parameters
  selectedModelPath = MODELS[0]!.path;
  contextLength = DEFAULT_MODEL_CONFIG.contextLength;
  maxOutputTokens = DEFAULT_MODEL_CONFIG.maxOutputTokens;
  samplerType = DEFAULT_MODEL_CONFIG.samplerType;
  temperature = DEFAULT_MODEL_CONFIG.temperature;
  topP = DEFAULT_MODEL_CONFIG.topP;
  topK = DEFAULT_MODEL_CONFIG.topK;
  enableThinking = DEFAULT_MODEL_CONFIG.enableThinking;
  enableSpeculativeDecoding = DEFAULT_MODEL_CONFIG.enableSpeculativeDecoding;
  systemPrompt = DEFAULT_SYSTEM_PROMPT_CONFIG.systemPrompt;
  disabledTools: string[] = [...DEFAULT_SYSTEM_PROMPT_CONFIG.disabledTools];
  textBackend: ModalityBackend = DEFAULT_EXPERIMENTAL_CONFIG.textBackend;
  visionBackend: OptionalModalityBackend =
      DEFAULT_EXPERIMENTAL_CONFIG.visionBackend;
  audioBackend: OptionalModalityBackend =
      DEFAULT_EXPERIMENTAL_CONFIG.audioBackend;
  textPrecision: ActivationPrecision =
      DEFAULT_EXPERIMENTAL_CONFIG.textPrecision;
  visionPrecision: ActivationPrecision =
      DEFAULT_EXPERIMENTAL_CONFIG.visionPrecision;
  audioPrecision: ActivationPrecision =
      DEFAULT_EXPERIMENTAL_CONFIG.audioPrecision;
  customModels: CustomModel[] = [];
  localDirModels: CustomModel[] = [];

  get modelSettings(): ModelSettings {
    return {
      selectedModelPath: this.selectedModelPath,
      contextLength: this.contextLength,
      maxOutputTokens: this.maxOutputTokens,
      samplerType: this.samplerType,
      temperature: this.temperature,
      topP: this.topP,
      topK: this.topK,
      enableThinking: this.enableThinking,
      enableSpeculativeDecoding: this.enableSpeculativeDecoding,
      systemPrompt: this.systemPrompt,
      disabledTools: [...this.disabledTools],
      textBackend: this.textBackend,
      visionBackend: this.visionBackend,
      audioBackend: this.audioBackend,
      textPrecision: this.textPrecision,
      visionPrecision: this.visionPrecision,
      audioPrecision: this.audioPrecision,
    };
  }

  private readonly SETTINGS_KEY = 'litertlm-chat-settings';

  constructor(
      private readonly updateCallback: () => void,
      private readonly storage: Storage = window.localStorage) {
    this.loadSettings();
  }

  loadSettings() {
    try {
      const data = this.storage.getItem(this.SETTINGS_KEY);
      if (data) {
        const parsed = JSON.parse(data);
        const result = PartialSettingsSchema.safeParse(parsed);
        if (result.success) {
          const validated = result.data;
          this.selectedModelPath = migrateModelPath(
              validated.selectedModelPath ?? this.selectedModelPath);
          this.contextLength = validated.contextLength ?? this.contextLength;
          this.maxOutputTokens =
              validated.maxOutputTokens ?? this.maxOutputTokens;
          this.samplerType = validated.samplerType ?? this.samplerType;
          this.temperature = validated.temperature ?? this.temperature;
          this.topP = validated.topP ?? this.topP;
          this.topK = validated.topK ?? this.topK;
          this.enableThinking = validated.enableThinking ?? this.enableThinking;
          this.enableSpeculativeDecoding = validated.enableSpeculativeDecoding ??
              this.enableSpeculativeDecoding;
          this.systemPrompt = validated.systemPrompt ?? this.systemPrompt;
          this.disabledTools = validated.disabledTools ?? this.disabledTools;
          this.textBackend = validated.textBackend ?? this.textBackend;
          this.visionBackend = validated.visionBackend ?? this.visionBackend;
          this.audioBackend = validated.audioBackend ?? this.audioBackend;
          this.textPrecision = validated.textPrecision ?? this.textPrecision;
          this.visionPrecision =
              validated.visionPrecision ?? this.visionPrecision;
          this.audioPrecision = validated.audioPrecision ?? this.audioPrecision;
          this.customModels = validated.customModels ?? [];
          this.localDirModels = validated.localDirModels ?? [];
          coerceGpuOnlyTextBackend(this);
        } else {
          console.warn(
              '[LiteRT-LM] Invalid settings in LocalStorage, using defaults:',
              result.error);
        }
      }
    } catch (e) {
      console.error('[LiteRT-LM] Failed to load settings:', e);
    }
  }

  saveSettings() {
    try {
      const payload = {
        selectedModelPath: this.selectedModelPath,
        contextLength: this.contextLength,
        maxOutputTokens: this.maxOutputTokens,
        samplerType: this.samplerType,
        temperature: this.temperature,
        topP: this.topP,
        topK: this.topK,
        enableThinking: this.enableThinking,
        enableSpeculativeDecoding: this.enableSpeculativeDecoding,
        systemPrompt: this.systemPrompt,
        disabledTools: this.disabledTools,
        textBackend: this.textBackend,
        visionBackend: this.visionBackend,
        audioBackend: this.audioBackend,
        textPrecision: this.textPrecision,
        visionPrecision: this.visionPrecision,
        audioPrecision: this.audioPrecision,
        customModels: this.customModels,
        localDirModels: this.localDirModels,
      };
      this.storage.setItem(this.SETTINGS_KEY, JSON.stringify(payload));
      this.updateCallback();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to save settings:', e);
    }
  }

  /**
   * Resets a group of settings to their defaults. Callers confirm with the user
   * first.
   */
  resetDefaults(group: SettingsGroup = 'modelConfig') {
    switch (group) {
      case 'modelConfig':
        Object.assign(this, DEFAULT_MODEL_CONFIG);
        break;
      case 'systemPrompt':
        this.systemPrompt = DEFAULT_SYSTEM_PROMPT_CONFIG.systemPrompt;
        this.disabledTools = [...DEFAULT_SYSTEM_PROMPT_CONFIG.disabledTools];
        break;
      case 'experimental':
        Object.assign(this, DEFAULT_EXPERIMENTAL_CONFIG);
        break;
      default: {
        // Fails to compile if a group isn't handled.
        const unhandled: never = group;
        throw new Error(`Unknown settings group: ${unhandled}`);
      }
    }
    this.saveSettings();
  }
}

/**
 * Adds a model to the custom models, unless it's already there, and selects
 * it.
 */
export function selectCustomModel(
    settings: Pick<
        SettingsStore,
        'customModels'|'selectedModelPath'|'saveSettings'|'textBackend'>&
    Partial<Pick<SettingsStore, 'localDirModels'>>,
    model: CustomModel) {
  if (!MODELS.some(m => m.path === model.path) &&
      !settings.customModels.some(m => m.path === model.path)) {
    settings.customModels = [...settings.customModels, model];
  }
  settings.selectedModelPath = model.path;
  coerceGpuOnlyTextBackend(settings);
  settings.saveSettings();
}
