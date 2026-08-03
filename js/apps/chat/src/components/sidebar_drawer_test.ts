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
import {MODELS, SettingsStore} from '../stores/settings_store.js';
import {LitertSidebar} from './sidebar_drawer.js';

describe('litert-sidebar', () => {
  let element: LitertSidebar;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
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

    cachedModelsMap = new Map<string, number>();
    downloadProgressMap = new Map<string, number>();
    downloadSpeedsMap = new Map<string, string>();

    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
      'deleteModelFromCache',
      'clearAllCache',
      'cancelDownload',
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
    expect(items.length).toBe(MODELS.length);
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
    cachedModelsMap.set(MODELS[0]!.filename, 1024 * 1024 * 1024); // 1 GB
    element.requestUpdate();
    await element.updateComplete;

    const deleteBtn = element.shadowRoot!.querySelector('.delete-cache-btn') as HTMLButtonElement;
    expect(deleteBtn).toBeTruthy();

    deleteBtn.click();

    expect(mockModelLoader.deleteModelFromCache)
        .toHaveBeenCalledWith(MODELS[0]!.path);
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
    const modelFilename = MODELS[0]!.filename;
    downloadProgressMap.set(modelFilename, 45);
    downloadSpeedsMap.set(modelFilename, '5.2 MB/s');
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
});
