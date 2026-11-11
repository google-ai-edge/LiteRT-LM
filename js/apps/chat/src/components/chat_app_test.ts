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
import {MODELS, SettingsStore} from '../stores/settings_store.js';
import {LitertLmChatApp} from './chat_app.js';
import {CustomDropdown} from './custom_dropdown.js';

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
      'undoDeleteConversation',
      'dismissDeletedConversation',
    ]);
    mockChatSession.isGenerating = false;
    mockChatSession.messages = [];
    mockChatSession.conversationsList = [];
    mockChatSession.activeSavedConvId = null;
    mockChatSession.deletedConversation = null;

    mockSettings = jasmine.createSpyObj('SettingsStore', ['saveSettings']);
    mockSettings.selectedModelPath = 'path/to/model.bin';
    mockSettings.contextLength = 2048;
    mockSettings.maxOutputTokens = 512;
    mockSettings.temperature = 0.7;
    mockSettings.topP = 0.9;
    mockSettings.topK = 40;
    mockSettings.samplerType = 'greedy';
    mockSettings.enableThinking = false;
    mockSettings.enableSpeculativeDecoding = false;
    mockSettings.systemPrompt = '';
    mockSettings.disabledTools = [];
    mockSettings.textBackend = 'gpu';
    mockSettings.visionBackend = 'gpu';
    mockSettings.audioBackend = 'gpu';
    mockSettings.textPrecision = 'default';
    mockSettings.visionPrecision = 'default';
    mockSettings.audioPrecision = 'default';
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
      storage: window.localStorage,
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

  it('toggles sidebar drawer and manages inert/aria-expanded state', async () => {
    const initiallyOpen = window.innerWidth > 768;
    expect(element.isSidebarOpen).toBe(initiallyOpen);

    const toggleBtn = element.shadowRoot!.querySelector('#btn-toggle-sidebar') as HTMLButtonElement;
    const sidebar = element.shadowRoot!.querySelector('.sidebar') as HTMLElement;
    expect(toggleBtn.getAttribute('aria-expanded')).toBe(String(initiallyOpen));

    toggleBtn.click();
    await element.updateComplete;

    expect(element.isSidebarOpen).toBe(!initiallyOpen);
    expect(toggleBtn.getAttribute('aria-expanded')).toBe(String(!initiallyOpen));
    expect(sidebar.hasAttribute('inert')).toBe(initiallyOpen);

    toggleBtn.click();
    await element.updateComplete;
    expect(element.isSidebarOpen).toBe(initiallyOpen);

    // Click overlay to close when open
    if (element.isSidebarOpen) {
      const overlay = element.shadowRoot!.querySelector('.sidebar-overlay') as HTMLElement;
      overlay.click();
      await element.updateComplete;
      expect(element.isSidebarOpen).toBeFalse();
      expect(sidebar.hasAttribute('inert')).toBeTrue();
    }
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
    expect(iframe.getAttribute('sandbox')).toContain('allow-scripts');
    expect((iframe as HTMLIFrameElement & {csp?: string}).csp).toBeFalsy();

    // Close preview
    const closeBtn = element.shadowRoot!.querySelector('#btn-close-preview') as HTMLButtonElement;
    closeBtn.click();
    await element.updateComplete;

    expect(element.isPreviewOpen).toBeFalse();
    expect(iframe.srcdoc).toBe('');
    expect(iframe.getAttribute('sandbox')).toContain('allow-scripts');
    expect((iframe as HTMLIFrameElement & {csp?: string}).csp).toBeFalsy();
  });

  it('traps Tab focus across shadow roots inside Learn More drawer and restores inner shadow trigger focus on Escape',
     async () => {
       const toggleBtn = element.shadowRoot!.querySelector(
                             '#btn-toggle-learn-more') as HTMLButtonElement;
       toggleBtn.click();
       await element.updateComplete;
       const learnMore =
           element.shadowRoot!.querySelector('litert-learn-more')!;
       await learnMore.updateComplete;

       const closeBtn = learnMore.shadowRoot!.querySelector(
                            '.btn-dismiss-right-drawer') as HTMLButtonElement;
       await new Promise<void>(
           resolve => requestAnimationFrame(() => resolve()));
       expect(learnMore.shadowRoot!.activeElement).toBe(closeBtn);
       const links = Array.from(
           learnMore.shadowRoot!.querySelectorAll<HTMLAnchorElement>('a[href]'));
       const lastLink = links[links.length - 1]!;

       lastLink.focus();
       lastLink.dispatchEvent(new KeyboardEvent(
           'keydown', {key: 'Tab', bubbles: true, composed: true}));
       expect(learnMore.shadowRoot!.activeElement).toBe(closeBtn);

       closeBtn.dispatchEvent(new KeyboardEvent(
           'keydown',
           {key: 'Tab', shiftKey: true, bubbles: true, composed: true}));
       expect(learnMore.shadowRoot!.activeElement).toBe(lastLink);

       window.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
       await element.updateComplete;
       expect(element.isLearnMoreOpen).toBeFalse();
       expect(element.shadowRoot!.activeElement).toBe(toggleBtn);
     });

  it('cancels an open confirmation dialog when removed', async () => {
    const confirmPromise = mockState.confirmHandler!({
      title: 'Clear all cached models?',
      message: 'Delete all downloaded weights?',
    });
    await element.updateComplete;

    element.remove();

    expect(await confirmPromise).toBeFalse();
  });

  it('focuses Cancel by default in confirmation dialog and renders Undo toast on conversation delete',
     async () => {
       const confirmPromise = mockState.confirmHandler!({
         title: 'Reset Model Config?',
         message: 'Reset parameters?',
         confirmLabel: 'Reset Defaults',
         danger: true,
       });
       await element.updateComplete;
       await new Promise<void>(
           resolve => requestAnimationFrame(() => resolve()));

       const cancelBtn = element.shadowRoot!.querySelector(
                             '#btn-confirm-cancel') as HTMLButtonElement;
       expect(cancelBtn).toBeTruthy();
       expect(element.shadowRoot!.activeElement).toBe(cancelBtn);

       cancelBtn.click();
       const confirmed = await confirmPromise;
       expect(confirmed).toBeFalse();

       mockChatSession.deletedConversation = {
         meta: {
           id: 'c1',
           title: 'Deleted Chat',
           createdAt: 1,
           modelPath: 'path/to/model.bin',
         },
         index: 0,
         messages: [],
         wasActive: false,
       };
       element.requestUpdate();
       await element.updateComplete;

       const toastEl =
           element.shadowRoot!.querySelector('.undo-toast') as HTMLElement;
       expect(toastEl).toBeTruthy();
       expect(toastEl.getAttribute('role')).toBe('status');
       expect(toastEl.getAttribute('aria-live')).toBe('polite');
       const undoBtn = element.shadowRoot!.querySelector(
                           '#btn-undo-delete') as HTMLButtonElement;
       expect(undoBtn).toBeTruthy();
       const sidebarEl = element.shadowRoot!.querySelector('litert-sidebar')!;
       sidebarEl.dispatchEvent(new CustomEvent(
           'conversation-deleted', {bubbles: true, composed: true}));
       await element.updateComplete;
       await new Promise<void>(
           resolve => requestAnimationFrame(() => resolve()));
       expect(element.shadowRoot!.activeElement).toBe(undoBtn);

       undoBtn.click();
       expect(mockChatSession.undoDeleteConversation).toHaveBeenCalled();

       // Verify auto-dismiss restores focus out of the removed toast
       mockChatSession.deletedConversation = {
         meta: {
           id: 'c2',
           title: 'Auto Dismiss Chat',
           createdAt: 2,
           modelPath: 'path/to/model.bin',
         },
         index: 0,
         messages: [],
         wasActive: false,
       };
       element.requestUpdate();
       await element.updateComplete;
       const undoBtn2 = element.shadowRoot!.querySelector(
                            '#btn-undo-delete') as HTMLButtonElement;
       undoBtn2.focus();
       expect(element.shadowRoot!.activeElement).toBe(undoBtn2);

       mockChatSession.deletedConversation = null;
       element.requestUpdate();
       await element.updateComplete;
       const newChatBtn = sidebarEl.shadowRoot!.querySelector(
           '.new-chat-item .conv-select-btn') as HTMLElement;
       if (element.isSidebarOpen && newChatBtn) {
         expect(sidebarEl.shadowRoot!.activeElement).toBe(newChatBtn);
       }
     });

  it('displays Generating state in unified composer model chip while streaming',
     async () => {
       mockChatSession.isGenerating = true;
       element.requestUpdate();
       const chatWindow =
           element.shadowRoot!.querySelector('litert-chat-window')!;
       chatWindow.requestUpdate();
       await element.updateComplete;
       await chatWindow.updateComplete;

       const stateEl = chatWindow.shadowRoot!.querySelector(
                           '.model-pill-state') as HTMLElement;
       const dotEl = chatWindow.shadowRoot!.querySelector(
                         '.model-pill-dot') as HTMLElement;
       expect(stateEl.textContent!.trim()).toBe('Generating');
       expect(dotEl.classList).toContain('generating');
     });

  it('defaults theme mode to system and cycles through manual overrides back to system',
     async () => {
       window.localStorage.removeItem('litertlm-theme-mode');
       window.localStorage.removeItem('litertlm-theme');
       element.remove();
       element = document.createElement('litert-lm-chat-app');
       ((element as unknown) as TestableChatApp).state = mockState;
       document.body.appendChild(element);
       await element.updateComplete;

       const themeBtn = element.shadowRoot!.querySelector(
                            '#btn-toggle-theme') as HTMLButtonElement;
       expect(themeBtn.getAttribute('title')).toContain('Theme: System');

       themeBtn.click();
       await element.updateComplete;
       const firstOverride = window.localStorage.getItem('litertlm-theme-mode');
       expect(firstOverride === 'light' || firstOverride === 'dark').toBeTrue();
       expect(document.documentElement.getAttribute('data-theme'))
           .toBe(firstOverride);

       themeBtn.click();
       await element.updateComplete;
       const secondOverride =
           window.localStorage.getItem('litertlm-theme-mode');
       expect(secondOverride === 'light' || secondOverride === 'dark')
           .toBeTrue();
       expect(secondOverride).not.toBe(firstOverride);
       expect(document.documentElement.getAttribute('data-theme'))
           .toBe(secondOverride);

       themeBtn.click();
       await element.updateComplete;
       expect(window.localStorage.getItem('litertlm-theme-mode')).toBeNull();
       expect(themeBtn.getAttribute('title')).toContain('Theme: System');
       window.localStorage.removeItem('litertlm-theme-mode');
     });

  it('renders clean brand header without redundant chip and sets aria-selected on composer options',
     async () => {
       expect(element.shadowRoot!.querySelector('.header-badge')).toBeNull();
       expect(element.shadowRoot!.querySelector('header h1')!.textContent!.trim())
           .toBe('LiteRT-LM.js Chat');

       const chatWindow =
           element.shadowRoot!.querySelector('litert-chat-window')!;
       const sidebar =
           element.shadowRoot!.querySelector('litert-sidebar')!;
       await chatWindow.updateComplete;
       await sidebar.updateComplete;
       expect(sidebar.shadowRoot!.querySelector('#runtime-backend-badge'))
           .toBeNull();
       expect(
           chatWindow.shadowRoot!.querySelector('#composer-model-backend-note'))
           .toBeNull();

       mockSettings.visionBackend = 'cpu';
       chatWindow.requestUpdate();
       sidebar.requestUpdate();
       await chatWindow.updateComplete;
       await sidebar.updateComplete;
       expect(
           sidebar.shadowRoot!.querySelector('#runtime-backend-badge')!
               .textContent!.trim())
           .toBe('WebGPU · CPU');
       expect(
           chatWindow.shadowRoot!.querySelector('#composer-model-backend-note')!
               .textContent)
           .toContain('WebGPU · CPU');
       expect(
           chatWindow.shadowRoot!
               .querySelector<HTMLButtonElement>('#btn-composer-model')!
               .getAttribute('title'))
           .toContain('WebGPU · CPU');

       mockSettings.selectedModelPath =
           'https://example.com/gemma-3-1b-it.litertlm';
       mockSettings.textBackend = 'cpu';
       mockSettings.audioBackend = 'cpu';
       chatWindow.requestUpdate();
       sidebar.requestUpdate();
       await chatWindow.updateComplete;
       await sidebar.updateComplete;
       expect(
           sidebar.shadowRoot!.querySelector('#runtime-backend-badge')!
               .textContent!.trim())
           .toBe('CPU');

       // Switching back to a -gpu.litertlm model treats Text as GPU -> mixed
       mockSettings.selectedModelPath = MODELS[0]!.path;
       mockSettings.visionBackend = 'gpu';
       mockSettings.audioBackend = 'gpu';
       mockSettings.textBackend = 'gpu';
       element.requestUpdate();
       chatWindow.requestUpdate();
       sidebar.requestUpdate();
       await element.updateComplete;
       await chatWindow.updateComplete;
       await sidebar.updateComplete;

       const composerDropdown = chatWindow.shadowRoot!.querySelector(
           '#composer-model-dropdown') as CustomDropdown;
       await composerDropdown.updateComplete;
       const options = Array.from(
           composerDropdown.querySelectorAll<HTMLElement>('.dropdown-item'));
       expect(options[0]!.getAttribute('role')).toBe('option');
       expect(options[0]!.getAttribute('aria-selected')).toBe('true');
       expect(options[1]!.getAttribute('aria-selected')).toBe('false');
     });

  it('exposes a first-focusable Skip to composer button and / shortcut without hijacking editable inputs',
     async () => {
       const skipBtn = element.shadowRoot!.querySelector(
                           '#skip-to-composer') as HTMLButtonElement;
       expect(skipBtn).toBeTruthy();
       expect(skipBtn.textContent!.trim()).toBe('Skip to composer');

       const chatWindow =
           element.shadowRoot!.querySelector('litert-chat-window')!;
       await chatWindow.updateComplete;
       const chatInput =
           chatWindow.shadowRoot!.querySelector('#chat-input') as HTMLElement;

       // Clicking / activating Skip to composer focuses #chat-input
       skipBtn.click();
       expect(chatWindow.shadowRoot!.activeElement).toBe(chatInput);

       // Pressing '/' when focus is on a non-editable button focuses #chat-input
       const themeBtn = element.shadowRoot!.querySelector(
                            '#btn-toggle-theme') as HTMLButtonElement;
       themeBtn.focus();
       expect(element.shadowRoot!.activeElement).toBe(themeBtn);
       const slashEvent = new KeyboardEvent(
           'keydown', {key: '/', bubbles: true, cancelable: true});
       window.dispatchEvent(slashEvent);
       expect(slashEvent.defaultPrevented).toBeTrue();
       expect(chatWindow.shadowRoot!.activeElement).toBe(chatInput);

       // Pressing '/' while focused inside an editable input is not intercepted
       const input = document.createElement('input');
       document.body.appendChild(input);
       input.focus();
       const typedSlash = new KeyboardEvent(
           'keydown', {key: '/', bubbles: true, cancelable: true});
       input.dispatchEvent(typedSlash);
       expect(typedSlash.defaultPrevented).toBeFalse();
       input.remove();
     });

  it('uses official Teal symbol on dark surfaces and official Blue symbol on light surfaces',
     async () => {
       const chatWindow =
           element.shadowRoot!.querySelector('litert-chat-window')!;
       element.theme = 'dark';
       element.requestUpdate();
       await element.updateComplete;
       await chatWindow.updateComplete;

       const headerLogo =
           element.shadowRoot!.querySelector('.brand-logo') as HTMLImageElement;
       const heroLogo = chatWindow.shadowRoot!.querySelector(
           '.hero-mark img') as HTMLImageElement;
       expect(headerLogo.getAttribute('src'))
           .toBe('./assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg');
       expect(heroLogo.getAttribute('src'))
           .toBe('./assets/LiteRT_Logo_Symbol-only_RGB_Color_Teal.svg');

       element.theme = 'light';
       element.requestUpdate();
       await element.updateComplete;
       await chatWindow.updateComplete;

       expect(headerLogo.getAttribute('src'))
           .toBe('./assets/LiteRT_Logo_Symbol-only_RGB_Color_Blue.svg');
       expect(heroLogo.getAttribute('src'))
           .toBe('./assets/LiteRT_Logo_Symbol-only_RGB_Color_Blue.svg');
     });
});
