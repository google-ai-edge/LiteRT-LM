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

import type {ChatSessionStore} from './chat_session_store.js';
import {parseHuggingFaceModelReference} from './model_reference.js';
import type {ModelLoaderService} from './model_loader_service.js';
import {CustomModel, selectCustomModel, SettingsStore} from './settings_store.js';

/**
 * The URL parameter that loads a Hugging Face model when the demo opens, e.g.
 * `?model=litert-community/SmolLM3-3B`. It takes a repo ID (as copied by the
 * "Copy model name" button), a repo page URL or a `hf://` URI, so that model
 * cards can link to the demo with their model.
 */
export const MODEL_URL_PARAMETER = 'model';

/** What `loadModelFromUrl()` uses. */
export declare interface ModelFromUrlDependencies {
  settings: SettingsStore;
  modelLoader: ModelLoaderService;
  chatSession: ChatSessionStore;
  updateStatus: (message: string) => void;
}

/**
 * Loads the Hugging Face model given by the page's `model` URL parameter, if
 * there is one: starts a new conversation, adds the model to the custom models
 * and selects it, and loads it, which downloads and caches it. Returns whether
 * there was a `model` parameter; the model loads in the background.
 *
 * Only Hugging Face models are accepted, so that a link can't make the demo
 * download a file from an arbitrary URL.
 */
export function loadModelFromUrl(
    search: string, dependencies: ModelFromUrlDependencies): boolean {
  const text = new URLSearchParams(search).get(MODEL_URL_PARAMETER)?.trim();
  if (!text) return false;
  void loadHuggingFaceModel(text, dependencies);
  return true;
}

async function loadHuggingFaceModel(
    text: string,
    {settings, modelLoader, chatSession, updateStatus}:
        ModelFromUrlDependencies): Promise<void> {
  // A restored conversation would select its own model instead.
  await chatSession.startNewConversation();
  const reference = parseHuggingFaceModelReference(text);
  if (!reference) {
    updateStatus(`The "${MODEL_URL_PARAMETER}" URL parameter must be a ` +
                 `Hugging Face repo (org/repo) or hf:// URI, not "${text}".`);
    return;
  }
  let model: CustomModel;
  try {
    model = await modelLoader.addRemoteModel(reference);
  } catch {
    // The model loader shows the error in the status.
    return;
  }
  selectCustomModel(settings, model);
  await modelLoader.loadModelWeights(settings.modelSettings, async () => {
    await chatSession.createConversationSession();
  });
}
