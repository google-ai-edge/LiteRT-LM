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

import './chat_app';

import {LlmChatStateController} from '../state_controller.js';
import {ChatSessionStore} from '../stores/chat_session_store.js';
import {LocalDirectoryService} from '../stores/local_directory_service.js';
import {ModelLoaderService} from '../stores/model_loader_service.js';
import {SettingsStore} from '../stores/settings_store.js';
import {LitertLmChatApp} from './chat_app.js';

interface TestableChatApp {
  state: LlmChatStateController;
}

describe('litert-lm-chat-app', () => {
  let element: LitertLmChatApp;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
  let mockLocalDirService: jasmine.SpyObj<LocalDirectoryService>;

  let mockClipboardWriteText: jasmine.Spy;

  beforeEach(async () => {
    // Mock clipboard
    if (!navigator.clipboard) {
      Object.assign(navigator, {
        clipboard: {
          writeText: () => Promise.resolve(),
        },
      });
    }
    mockClipboardWriteText = spyOn(navigator.clipboard, 'writeText')
                                 .and.returnValue(Promise.resolve());

    // Minimal state mock to prevent child components from crashing during render
    mockChatSession = jasmine.createSpyObj('ChatSessionStore', [
      'startNewConversation',
      'selectConversation',
      'deleteConversation',
    ]);
    mockChatSession.isGenerating = false;
    mockChatSession.messages = [];
    mockChatSession.conversationsList = [];
    mockChatSession.activeSavedConvId = null;

    mockSettings = jasmine.createSpyObj('SettingsStore', ['saveSettings']);
    mockSettings.selectedModelPath = 'path/to/model.bin';
    mockSettings.contextLength = 2048;
    mockSettings.maxOutputTokens = 512;
    mockSettings.temperature = 0.7;
    mockSettings.topP = 0.9;
    mockSettings.topK = 40;
    mockSettings.samplerType = 'greedy';
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

    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
    ]);
    mockModelLoader.cachedModels = new Map<string, number>();
    mockModelLoader.downloadProgresses = new Map<string, number>();
    mockModelLoader.downloadSpeeds = new Map<string, string>();
    mockModelLoader.isModelLoading = false;
    mockModelLoader.engine = null;
    mockModelLoader.metricLoadTime = '-';

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

    element = document.createElement('litert-lm-chat-app');
    // Inject mock state
    ((element as unknown) as TestableChatApp).state = mockState;

    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    if (element) {
      element.remove();
    }
  });

  it('renders header and container', () => {
    const header = element.shadowRoot!.querySelector('header');
    expect(header).toBeTruthy();
    expect(header!.querySelector('h1')!.textContent!.trim()).toBe('LiteRT-LM.js Chat');

    const appContainer = element.shadowRoot!.querySelector('.app-container');
    expect(appContainer).toBeTruthy();
  });

  it('toggles sidebar drawer', async () => {
    expect(element.isSidebarOpen).toBeFalse();

    const toggleBtn = element.shadowRoot!.querySelector('#btn-toggle-sidebar') as HTMLButtonElement;
    toggleBtn.click();
    await element.updateComplete;

    expect(element.isSidebarOpen).toBeTrue();

    // Click overlay to close
    const overlay = element.shadowRoot!.querySelector('.sidebar-overlay') as HTMLElement;
    overlay.click();
    await element.updateComplete;

    expect(element.isSidebarOpen).toBeFalse();
  });

  it('toggles learn more drawer', async () => {
    expect(element.isLearnMoreOpen).toBeFalse();

    const toggleBtn = element.shadowRoot!.querySelector('#btn-toggle-learn-more') as HTMLButtonElement;
    toggleBtn.click();
    await element.updateComplete;

    expect(element.isLearnMoreOpen).toBeTrue();

    // Click overlay to close
    const overlay = element.shadowRoot!.querySelector('.sidebar-right-overlay') as HTMLElement;
    overlay.click();
    await element.updateComplete;

    expect(element.isLearnMoreOpen).toBeFalse();
  });

  it('handles preview-html event and opens preview overlay', async () => {
    expect(element.isPreviewOpen).toBeFalse();

    const chatWindow = element.shadowRoot!.querySelector('litert-chat-window')!;
    expect(chatWindow).toBeTruthy();

    // Dispatch the custom preview event
    chatWindow.dispatchEvent(new CustomEvent('preview-html', {
      detail: { base64Code: 'PGgxPkhlbGxvPC9oMT4=' }, // "<h1>Hello</h1>"
      bubbles: true,
      composed: true,
    }));
    await element.updateComplete;

    expect(element.isPreviewOpen).toBeTrue();

    const iframe = element.shadowRoot!.querySelector('#preview-iframe') as HTMLIFrameElement;
    expect(iframe).toBeTruthy();
    expect(iframe.srcdoc).toContain('<h1>Hello</h1>');

    // Close preview
    const closeBtn = element.shadowRoot!.querySelector('#btn-close-preview') as HTMLButtonElement;
    closeBtn.click();
    await element.updateComplete;

    expect(element.isPreviewOpen).toBeFalse();
    expect(iframe.srcdoc).toBe('');
  });
});
