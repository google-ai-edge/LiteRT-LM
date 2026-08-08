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
    'temperature' | 'topP' | 'topK' | 'enableThinking'>;

/** List of supported default models available for selection. */
export const MODELS = [
  {
    name: 'Gemma 4 E2B',
    filename: 'gemma-4-E2B-it-web.litertlm',
    path:
        'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm',
    size: '1.9 GB'
  },
  {
    name: 'Gemma 4 E4B',
    filename: 'gemma-4-E4B-it-web.litertlm',
    path:
        'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm',
    size: '2.8 GB'
  },
  {
    name: 'Gemma 4 12B',
    filename: 'gemma-4-12B-it-web.litertlm',
    path:
        'https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/resolve/main/gemma-4-12B-it-web.litertlm',
    size: '5.6 GB'
  },
  {
    name: 'Gemma 4 26B A4B',
    filename: 'gemma-4-26B-A4B-it-web.litertlm',
    path:
        'https://huggingface.co/litert-community/gemma-4-26B-A4B-it-litert-lm/resolve/main/gemma-4-26B-A4B-it-web.litertlm',
    size: '14.7 GB'
  },
  {
    name: 'Gemma 4 31B',
    filename: 'gemma-4-31B-it-web.litertlm',
    path:
        'https://huggingface.co/litert-community/gemma-4-31B-it-litert-lm/resolve/main/gemma-4-31B-it-web.litertlm',
    size: '17.9 GB'
  }
];


/** Store for managing inference configuration and persistence. */
export class SettingsStore implements Settings {
  // Active model configuration parameters
  selectedModelPath = MODELS[0]!.path;
  contextLength = 4096;
  maxOutputTokens = 2048;
  samplerType = 'greedy';
  temperature = 1.0;
  topP = 0.95;
  topK = 64;
  enableThinking = true;
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
    };
  }

  private readonly SETTINGS_KEY = 'litertlm-chat-settings';

  constructor(private readonly updateCallback: () => void) {
    this.loadSettings();
  }

  loadSettings() {
    try {
      const data = window.localStorage.getItem(this.SETTINGS_KEY);
      if (data) {
        const parsed = JSON.parse(data);
        const result = PartialSettingsSchema.safeParse(parsed);
        if (result.success) {
          const validated = result.data;
          this.selectedModelPath =
              validated.selectedModelPath ?? this.selectedModelPath;
          this.contextLength = validated.contextLength ?? this.contextLength;
          this.maxOutputTokens =
              validated.maxOutputTokens ?? this.maxOutputTokens;
          this.samplerType = validated.samplerType ?? this.samplerType;
          this.temperature = validated.temperature ?? this.temperature;
          this.topP = validated.topP ?? this.topP;
          this.topK = validated.topK ?? this.topK;
          this.enableThinking = validated.enableThinking ?? this.enableThinking;
          this.customModels = validated.customModels ?? [];
          this.localDirModels = validated.localDirModels ?? [];
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
        customModels: this.customModels,
        localDirModels: this.localDirModels,
      };
      window.localStorage.setItem(this.SETTINGS_KEY, JSON.stringify(payload));
      this.updateCallback();
    } catch (e) {
      console.error('[LiteRT-LM] Failed to save settings:', e);
    }
  }

  resetDefaults() {
    const confirmReset = confirm(
        'Are you sure you want to reset all inference settings to their defaults?');
    if (!confirmReset) return false;

    this.contextLength = 4096;
    this.maxOutputTokens = 2048;
    this.samplerType = 'greedy';
    this.temperature = 1.0;
    this.topP = 0.95;
    this.topK = 64;
    this.enableThinking = true;

    this.saveSettings();
    return true;
  }
}
