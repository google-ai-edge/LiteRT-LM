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

import {ChatSessionStore} from './chat_session_store.js';
import {ModelLoaderService} from './model_loader_service.js';
import {loadModelFromUrl, ModelFromUrlDependencies} from './model_url_parameter.js';
import {CustomModel, MODELS, SettingsStore} from './settings_store.js';

describe('loadModelFromUrl', () => {
  const MODEL: CustomModel = {
    name: 'SmolLM3-3B',
    filename: 'SmolLM3-3B.litertlm',
    path:
        'https://huggingface.co/litert-community/SmolLM3-3B/resolve/c0ffee/SmolLM3-3B.litertlm',
    size: '3.13 GB',
  };
  let settings: SettingsStore;
  let modelLoader: jasmine.SpyObj<ModelLoaderService>;
  let chatSession: jasmine.SpyObj<ChatSessionStore>;
  let updateStatus: jasmine.Spy<(message: string) => void>;
  let dependencies: ModelFromUrlDependencies;

  beforeEach(() => {
    settings = new SettingsStore(() => {});
    modelLoader = jasmine.createSpyObj<ModelLoaderService>(
        'ModelLoaderService', ['addRemoteModel', 'loadModelWeights']);
    modelLoader.addRemoteModel.and.resolveTo(MODEL);
    modelLoader.loadModelWeights.and.resolveTo();
    chatSession = jasmine.createSpyObj<ChatSessionStore>(
        'ChatSessionStore',
        ['startNewConversation', 'createConversationSession']);
    chatSession.startNewConversation.and.resolveTo();
    updateStatus = jasmine.createSpy('updateStatus');
    dependencies = {settings, modelLoader, chatSession, updateStatus};
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  /** Waits for the model to be loaded in the background. */
  async function flush() {
    await new Promise(resolve => setTimeout(resolve));
  }

  it('does nothing without a model parameter', async () => {
    expect(loadModelFromUrl('?other=1', dependencies)).toBeFalse();
    expect(loadModelFromUrl('?model=', dependencies)).toBeFalse();
    await flush();

    expect(chatSession.startNewConversation).not.toHaveBeenCalled();
    expect(modelLoader.addRemoteModel).not.toHaveBeenCalled();
  });

  it('adds, selects and loads a Hugging Face model in a new conversation',
     async () => {
       expect(loadModelFromUrl(
                  '?model=litert-community/SmolLM3-3B', dependencies))
           .toBeTrue();
       await flush();

       expect(chatSession.startNewConversation).toHaveBeenCalled();
       expect(modelLoader.addRemoteModel)
           .toHaveBeenCalledOnceWith('hf://litert-community/SmolLM3-3B');
       expect(settings.customModels).toEqual([MODEL]);
       expect(settings.selectedModelPath).toBe(MODEL.path);
       expect(modelLoader.loadModelWeights)
           .toHaveBeenCalledOnceWith(
               settings.modelSettings, jasmine.any(Function));
     });

  it('accepts hf:// URIs and repo pages', async () => {
    loadModelFromUrl(
        '?model=hf%3A%2F%2Forg%2Frepo%40web%2Fmodel.litertlm', dependencies);
    loadModelFromUrl(
        '?model=https%3A%2F%2Fhuggingface.co%2Forg%2Frepo', dependencies);
    await flush();

    expect(modelLoader.addRemoteModel.calls.allArgs()).toEqual([
      ['hf://org/repo@web/model.litertlm'],
      ['hf://org/repo'],
    ]);
  });

  it('does not add a model twice', async () => {
    settings.customModels = [MODEL];

    loadModelFromUrl('?model=litert-community/SmolLM3-3B', dependencies);
    await flush();

    expect(settings.customModels).toEqual([MODEL]);
    expect(settings.selectedModelPath).toBe(MODEL.path);
  });

  it('rejects models that are not on Hugging Face', async () => {
    loadModelFromUrl(
        '?model=https%3A%2F%2Fexample.com%2Fmodel.litertlm', dependencies);
    await flush();

    expect(modelLoader.addRemoteModel).not.toHaveBeenCalled();
    expect(updateStatus)
        .toHaveBeenCalledOnceWith(jasmine.stringMatching(
            /"model" URL parameter must be a Hugging Face repo/));
    expect(settings.selectedModelPath).toBe(MODELS[0]!.path);
  });

  it('does not load the model if it cannot be added', async () => {
    modelLoader.addRemoteModel.and.rejectWith(new Error('Not found'));

    loadModelFromUrl('?model=org/missing', dependencies);
    await flush();

    expect(settings.customModels).toEqual([]);
    expect(modelLoader.loadModelWeights).not.toHaveBeenCalled();
  });

  it('passes cpu backend to addRemoteModel when textBackend is cpu and preserves cpu for standard .litertlm',
     async () => {
       settings.textBackend = 'cpu';

       loadModelFromUrl('?model=litert-community/SmolLM3-3B', dependencies);
       await flush();

       expect(modelLoader.addRemoteModel)
           .toHaveBeenCalledOnceWith('hf://litert-community/SmolLM3-3B', 'cpu');
       expect(settings.textBackend).toBe('cpu');
       expect(modelLoader.loadModelWeights)
           .toHaveBeenCalledOnceWith(
               jasmine.objectContaining({textBackend: 'cpu'}),
               jasmine.any(Function));
     });

  it('selects a built-in model directly without calling addRemoteModel or adding to customModels',
     async () => {
       settings.textBackend = 'cpu';

       loadModelFromUrl(
           '?model=litert-community/gemma-4-E4B-it-litert-lm', dependencies);
       await flush();

       expect(modelLoader.addRemoteModel).not.toHaveBeenCalled();
       expect(settings.customModels).toEqual([]);
       expect(settings.selectedModelPath)
           .toBe('hf://litert-community/gemma-4-E4B-it-litert-lm');
       expect(settings.textBackend).toBe('gpu');
       expect(modelLoader.loadModelWeights)
           .toHaveBeenCalledOnceWith(
               jasmine.objectContaining({
                 selectedModelPath:
                     'hf://litert-community/gemma-4-E4B-it-litert-lm',
                 textBackend: 'gpu',
               }),
               jasmine.any(Function));

       modelLoader.loadModelWeights.calls.reset();
       settings.textBackend = 'cpu';
       loadModelFromUrl(
           '?model=https%3A%2F%2Fhuggingface.co%2Flitert-community%2Fgemma-4-E2B-it-litert-lm',
           dependencies);
       await flush();

       expect(modelLoader.addRemoteModel).not.toHaveBeenCalled();
       expect(settings.customModels).toEqual([]);
       expect(settings.selectedModelPath)
           .toBe('hf://litert-community/gemma-4-E2B-it-litert-lm');
       expect(settings.textBackend).toBe('gpu');
     });

  it('coerces textBackend back to gpu when a custom ?model= resolves to a -gpu.litertlm package',
     async () => {
       settings.textBackend = 'cpu';
       const gpuOnlyModel: CustomModel = {
         name: 'Custom GPU Model',
         filename: 'custom-model-it-gpu.litertlm',
         path: 'hf://org/custom-gpu-model',
         size: '2.3 GB',
       };
       modelLoader.addRemoteModel.and.resolveTo(gpuOnlyModel);

       loadModelFromUrl('?model=org/custom-gpu-model', dependencies);
       await flush();

       expect(modelLoader.addRemoteModel)
           .toHaveBeenCalledOnceWith('hf://org/custom-gpu-model', 'cpu');
       expect(settings.customModels).toEqual([gpuOnlyModel]);
       expect(settings.textBackend).toBe('gpu');
       expect(modelLoader.loadModelWeights)
           .toHaveBeenCalledOnceWith(
               jasmine.objectContaining({textBackend: 'gpu'}),
               jasmine.any(Function));
     });
});
