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

import './sidebar_drawer';

import {LlmChatStateController} from '../state_controller.js';
import {ChatSessionStore} from '../stores/chat_session_store.js';
import {ModelLoaderService} from '../stores/model_loader_service.js';
import {LocalDirectoryService} from '../stores/local_directory_service.js';
import {MODELS, SettingsStore} from '../stores/settings_store.js';
import {LitertSidebar} from './sidebar_drawer.js';

describe('litert-sidebar', () => {
  let element: LitertSidebar;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
  let mockLocalDirService: jasmine.SpyObj<LocalDirectoryService>;
  let cachedModelsMap: Map<string, number>;
  let downloadProgressMap: Map<string, number>;
  let downloadSpeedsMap: Map<string, string>;

  beforeEach(async () => {
    mockChatSession = jasmine.createSpyObj('ChatSessionStore', [
      'startNewConversation',
      'selectConversation',
      'deleteConversation',
      'createConversationSession',
    ]);
    mockChatSession.activeSavedConvId = null;
    mockChatSession.conversationsList = [
      {id: 'conv1', title: 'Conversation 1', createdAt: 123, modelPath: 'path1'},
      {id: 'conv2', title: 'Conversation 2', createdAt: 456, modelPath: 'path2'},
    ];

    mockSettings = jasmine.createSpyObj('SettingsStore', [
      'saveSettings',
      'resetDefaults',
    ]);
    mockSettings.selectedModelPath = MODELS[0]!.path;
    mockSettings.contextLength = 2048;
    mockSettings.maxOutputTokens = 512;
    mockSettings.temperature = 0.7;
    mockSettings.topP = 0.9;
    mockSettings.topK = 40;
    mockSettings.samplerType = 'top_k';
    mockSettings.enableThinking = false;
    mockSettings.customModels = [];
    mockSettings.localDirModels = [];

    mockLocalDirService = jasmine.createSpyObj('LocalDirectoryService', [
      'mountDirectory',
      'scanDirectory',
      'getFile',
    ]);
    mockLocalDirService.isAuthorized = false;
    (mockLocalDirService as any).isSupported = false;

    cachedModelsMap = new Map<string, number>();
    downloadProgressMap = new Map<string, number>();
    downloadSpeedsMap = new Map<string, string>();

    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
      'deleteModelFromCache',
      'clearAllCache',
      'cancelDownload',
      'addRemoteModel',
    ]);
    mockModelLoader.cachedModels = cachedModelsMap;
    mockModelLoader.downloadProgresses = downloadProgressMap;
    mockModelLoader.downloadSpeeds = downloadSpeedsMap;
    mockModelLoader.isModelLoading = false;
    mockModelLoader.engine = null;
    mockModelLoader.metricLoadTime = '-';
    mockModelLoader.downloadAbortController = null;

    mockState = {
      chatSession: mockChatSession,
      settings: mockSettings,
      modelLoader: mockModelLoader,
      localDirService: mockLocalDirService,
      statusText: 'Ready',
      addHost: jasmine.createSpy('addHost'),
      removeHost: jasmine.createSpy('removeHost'),
      requestUpdate: jasmine.createSpy('requestUpdate'),
    } as unknown as jasmine.SpyObj<LlmChatStateController>;

    element = document.createElement('litert-sidebar');
    element.state = mockState;
    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    element.remove();
  });

  it('renders model list in dropdown', () => {
    const dropdown = element.shadowRoot!.querySelector('custom-dropdown');
    expect(dropdown).toBeTruthy();

    const items = element.shadowRoot!.querySelectorAll('.dropdown-item');
    // The models, the upload item and the add-from-URL item.
    expect(items.length).toBe(MODELS.length + 2);
    expect(items[0]!.querySelector('.model-name')!.textContent!.trim()).toBe(MODELS[0]!.name);
  });

  it('triggers model load on model change', async () => {
    const dropdown = element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;
    
    // Simulate selection change event from dropdown
    dropdown.dispatchEvent(new CustomEvent('change', {
      detail: MODELS[1]!.path,
    }));
    await element.updateComplete;

    expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
    expect(mockSettings.saveSettings).toHaveBeenCalled();
    expect(mockModelLoader.loadModelWeights).toHaveBeenCalled();
  });

  it('deletes model from cache when clicking delete cache button', async () => {
    // Make the first model cached
    cachedModelsMap.set(MODELS[0]!.path, 1024 * 1024 * 1024); // 1 GB
    element.requestUpdate();
    await element.updateComplete;

    const deleteBtn = element.shadowRoot!.querySelector('.delete-cache-btn') as HTMLButtonElement;
    expect(deleteBtn).toBeTruthy();

    deleteBtn.click();

    expect(mockModelLoader.deleteModelFromCache)
        .toHaveBeenCalledWith(MODELS[0]!.path);
  });

  it('resets selectedModelPath to default when the active standard model is deleted from cache',
     async () => {
       mockModelLoader.deleteModelFromCache.and.returnValue(
           Promise.resolve(true));
       mockSettings.selectedModelPath = MODELS[1]!.path;

       // Make the selected standard model cached
       cachedModelsMap.set(MODELS[1]!.path, 1024 * 1024 * 1024);
       element.requestUpdate();
       await element.updateComplete;

       const deleteBtn = element.shadowRoot!.querySelector(
                             '.delete-cache-btn') as HTMLButtonElement;
       expect(deleteBtn).toBeTruthy();

       deleteBtn.click();
       await new Promise(resolve => setTimeout(resolve));

       expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
       expect(mockSettings.saveSettings).toHaveBeenCalled();
     });

  it('does not reset selectedModelPath if a non-selected standard model is deleted from cache',
     async () => {
       mockModelLoader.deleteModelFromCache.and.returnValue(
           Promise.resolve(true));
       mockSettings.selectedModelPath = MODELS[1]!.path;

       // Make the first model cached (which is not selected)
       cachedModelsMap.set(MODELS[0]!.path, 1024 * 1024 * 1024);
       element.requestUpdate();
       await element.updateComplete;

       const deleteBtn = element.shadowRoot!.querySelector(
                             '.delete-cache-btn') as HTMLButtonElement;
       expect(deleteBtn).toBeTruthy();

       deleteBtn.click();
       await new Promise(resolve => setTimeout(resolve));

       expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
       expect(mockSettings.saveSettings).not.toHaveBeenCalled();
     });

  it('resets selectedModelPath to default and filters when selected custom model is deleted from cache',
     async () => {
       mockModelLoader.deleteModelFromCache.and.returnValue(
           Promise.resolve(true));
       const customModelPath = 'https://local-model/my-custom-model.litertlm';
       mockSettings.customModels = [{
         name: 'Custom Model',
         filename: 'my-custom-model.litertlm',
         path: customModelPath,
         size: '1.0 GB'
       }];
       mockSettings.selectedModelPath = customModelPath;

       // Make the custom model cached
       cachedModelsMap.set(customModelPath, 1024 * 1024 * 1024);
       element.requestUpdate();
       await element.updateComplete;

       const deleteBtn = element.shadowRoot!.querySelector(
                             '.delete-cache-btn') as HTMLButtonElement;
       expect(deleteBtn).toBeTruthy();

       deleteBtn.click();
       await new Promise(resolve => setTimeout(resolve));

       expect(mockSettings.customModels.length).toBe(0);
       expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
       expect(mockSettings.saveSettings).toHaveBeenCalled();
     });

  describe('adding a model by URL', () => {
    const remoteModel = {
      name: 'SmolLM3-3B',
      filename: 'SmolLM3-3B.litertlm',
      path:
          'https://huggingface.co/litert-community/SmolLM3-3B/resolve/c0ffee/SmolLM3-3B.litertlm',
      size: '3.13 GB',
    };

    async function openForm(): Promise<HTMLFormElement> {
      const dropdown =
          element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;
      dropdown.dispatchEvent(new CustomEvent('change', {detail: 'add-url'}));
      await element.updateComplete;
      return element.shadowRoot!.querySelector('#add-model-form')!;
    }

    async function submit(form: HTMLFormElement, text: string) {
      form.querySelector<HTMLInputElement>('#add-model-input')!.value = text;
      form.requestSubmit();
      await new Promise(resolve => setTimeout(resolve));
      await element.updateComplete;
    }

    it('has an item at the end of the model list', () => {
      const items = element.shadowRoot!.querySelectorAll('.dropdown-item');
      expect((items[items.length - 1] as HTMLElement).dataset['value'])
          .toBe('add-url');
    });

    it('shows a form without changing the selected model', async () => {
      const form = await openForm();

      expect(form).toBeTruthy();
      expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
      expect(mockModelLoader.loadModelWeights).not.toHaveBeenCalled();
    });

    it('adds, selects and loads the model', async () => {
      mockModelLoader.addRemoteModel.and.resolveTo(remoteModel);
      const form = await openForm();

      await submit(form, 'litert-community/SmolLM3-3B');

      expect(mockModelLoader.addRemoteModel)
          .toHaveBeenCalledOnceWith('litert-community/SmolLM3-3B');
      expect(mockSettings.customModels).toEqual([remoteModel]);
      expect(mockSettings.selectedModelPath).toBe(remoteModel.path);
      expect(mockSettings.saveSettings).toHaveBeenCalled();
      expect(mockModelLoader.loadModelWeights).toHaveBeenCalled();
      expect(element.shadowRoot!.querySelector('#add-model-form')).toBeNull();
    });

    it('does not add a model twice', async () => {
      mockSettings.customModels = [remoteModel];
      mockModelLoader.addRemoteModel.and.resolveTo({...remoteModel});

      await submit(await openForm(), 'litert-community/SmolLM3-3B');

      expect(mockSettings.customModels).toEqual([remoteModel]);
      expect(mockSettings.selectedModelPath).toBe(remoteModel.path);
    });

    it('keeps the form open if the model cannot be added', async () => {
      mockModelLoader.addRemoteModel.and.rejectWith(new Error('Not found'));

      await submit(await openForm(), 'org/missing');

      expect(mockSettings.customModels).toEqual([]);
      expect(mockModelLoader.loadModelWeights).not.toHaveBeenCalled();
      expect(element.shadowRoot!.querySelector('#add-model-form'))
          .toBeTruthy();
    });

    it('hides the form when another model is chosen', async () => {
      await openForm();
      const dropdown =
          element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;

      dropdown.dispatchEvent(
          new CustomEvent('change', {detail: MODELS[1]!.path}));
      await element.updateComplete;

      expect(element.shadowRoot!.querySelector('#add-model-form')).toBeNull();
      expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
    });

    for (const target of ['#add-model-input', '.btn-add-model']) {
      it(`hides the form on Escape in ${target}`, async () => {
        const form = await openForm();

        form.querySelector(target)!.dispatchEvent(
            new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        await element.updateComplete;

        expect(element.shadowRoot!.querySelector('#add-model-form'))
            .toBeNull();
      });
    }

    it('does not add a model while another is loading', async () => {
      mockModelLoader.isModelLoading = true;
      const form = await openForm();

      expect(form.querySelector<HTMLInputElement>('#add-model-input')!.disabled)
          .toBeTrue();
      expect(form.querySelector<HTMLButtonElement>('.btn-add-model')!.disabled)
          .toBeTrue();
      await submit(form, 'litert-community/SmolLM3-3B');

      expect(mockModelLoader.addRemoteModel).not.toHaveBeenCalled();
      expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
    });

    it('removes an added model that is not cached from the list', async () => {
      spyOn(window, 'confirm').and.returnValue(true);
      mockSettings.customModels = [remoteModel];
      mockSettings.selectedModelPath = remoteModel.path;
      element.requestUpdate();
      await element.updateComplete;

      const removeBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
          '.delete-cache-btn[title="Remove from list"]')!;
      removeBtn.click();

      expect(mockModelLoader.deleteModelFromCache).not.toHaveBeenCalled();
      expect(mockSettings.customModels).toEqual([]);
      expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
      expect(mockSettings.saveSettings).toHaveBeenCalled();
    });
  });

  it('updates settings on slider/number input changes', async () => {
    // 1. Context Length
    const contextInput = element.shadowRoot!.querySelector('#context-length') as HTMLInputElement;
    contextInput.value = '4096';
    contextInput.dispatchEvent(new Event('input'));
    expect(mockSettings.contextLength).toBe(4096);
    expect(mockSettings.saveSettings).toHaveBeenCalled();

    // 2. Max Gen
    const maxGenInput = element.shadowRoot!.querySelector('#max-output-tokens') as HTMLInputElement;
    maxGenInput.value = '1024';
    maxGenInput.dispatchEvent(new Event('input'));
    expect(mockSettings.maxOutputTokens).toBe(1024);

    // 3. Sampler Select
    const samplerSelect = element.shadowRoot!.querySelector('#sampler-type') as HTMLSelectElement;
    samplerSelect.value = 'greedy';
    samplerSelect.dispatchEvent(new Event('change'));
    expect(mockSettings.samplerType).toBe('greedy');

    // 4. Temp
    const tempInput = element.shadowRoot!.querySelector('#temperature') as HTMLInputElement;
    tempInput.value = '1.2';
    tempInput.dispatchEvent(new Event('input'));
    expect(mockSettings.temperature).toBe(1.2);

    // 5. Thinking Checkbox
    const thinkingCheckbox = element.shadowRoot!.querySelector('#enable-thinking') as HTMLInputElement;
    thinkingCheckbox.checked = true;
    thinkingCheckbox.dispatchEvent(new Event('change'));
    expect(mockSettings.enableThinking).toBeTrue();
  });

  it('resets defaults on reset button click', () => {
    const resetBtn = element.shadowRoot!.querySelector('#btn-reset-settings') as HTMLButtonElement;
    resetBtn.click();
    expect(mockSettings.resetDefaults).toHaveBeenCalled();
  });

  it('renders conversations list and triggers selection', async () => {
    const list = element.shadowRoot!.querySelector('#conversations-list');
    expect(list).toBeTruthy();

    const items = list!.querySelectorAll('.conv-item');
    // 1 header (New Chat) + 2 saved conversations
    expect(items.length).toBe(3);

    // Click New Chat
    (items[0] as HTMLElement).click();
    expect(mockChatSession.startNewConversation).toHaveBeenCalled();

    // Click Conversation 1
    (items[1] as HTMLElement).click();
    expect(mockChatSession.selectConversation).toHaveBeenCalledWith('conv1');
  });

  it('deletes conversation on click ✕', async () => {
    const list = element.shadowRoot!.querySelector('#conversations-list');
    const deleteBtn = list!.querySelector('.btn-delete-conv') as HTMLButtonElement;
    
    deleteBtn.click();

    expect(mockChatSession.deleteConversation).toHaveBeenCalledWith('conv1');
  });

  it('renders progress bar when model is downloading', async () => {
    mockModelLoader.isModelLoading = true;
    const modelKey = MODELS[0]!.path;
    downloadProgressMap.set(modelKey, 45);
    downloadSpeedsMap.set(modelKey, '5.2 MB/s');
    mockModelLoader.downloadAbortController = {} as AbortController;
    
    element.requestUpdate();
    await element.updateComplete;

    const progressContainer = element.shadowRoot!.querySelector('#progress-container');
    expect(progressContainer).toBeTruthy();

    const percentText = progressContainer!.querySelector('#progress-percent');
    expect(percentText!.textContent!.trim()).toBe('45%');

    const speedText = progressContainer!.querySelector('#progress-speed');
    expect(speedText!.textContent!.trim()).toBe('5.2 MB/s');

    const cancelBtn = progressContainer!.querySelector('#btn-cancel-download') as HTMLButtonElement;
    expect(cancelBtn).toBeTruthy();

    cancelBtn.click();
    expect(mockModelLoader.cancelDownload).toHaveBeenCalled();
  });

  it('triggers mountDirectory when select-dir is chosen (isSupported=true)', async () => {
    (mockLocalDirService as any).isSupported = true;
    element.requestUpdate();
    await element.updateComplete;

    const dropdown = element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;
    dropdown.dispatchEvent(new CustomEvent('change', {
      detail: 'select-dir',
    }));
    await element.updateComplete;

    expect(mockLocalDirService.mountDirectory).toHaveBeenCalled();
  });

  it('renders upload option when isSupported is false', async () => {
    (mockLocalDirService as any).isSupported = false;
    element.requestUpdate();
    await element.updateComplete;

    const dropdown = element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;
    const uploadItem = dropdown.querySelector('[data-value="upload"]');
    const selectDirItem = dropdown.querySelector('[data-value="select-dir"]');

    expect(uploadItem).toBeTruthy();
    expect(selectDirItem).toBeFalsy();
  });

  it('renders select-dir option when isSupported is true', async () => {
    (mockLocalDirService as any).isSupported = true;
    element.requestUpdate();
    await element.updateComplete;

    const dropdown = element.shadowRoot!.querySelector('custom-dropdown') as HTMLElement;
    const uploadItem = dropdown.querySelector('[data-value="upload"]');
    const selectDirItem = dropdown.querySelector('[data-value="select-dir"]');

    expect(uploadItem).toBeFalsy();
    expect(selectDirItem).toBeTruthy();
  });
});
