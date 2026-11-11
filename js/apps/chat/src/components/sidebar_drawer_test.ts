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
    mockSettings.enableSpeculativeDecoding = false;
    mockSettings.systemPrompt = '';
    mockSettings.disabledTools = [];
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
      storage: window.localStorage,
      requestConfirm: jasmine.createSpy('requestConfirm').and.resolveTo(true),
    } as unknown as jasmine.SpyObj<LlmChatStateController>;

    element = document.createElement('litert-sidebar');
    element.state = mockState;
    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    element.remove();
  });

  async function toggleSettingsPage(id: string) {
    element.shadowRoot!.querySelector<HTMLButtonElement>(`#tab-${id}`)!.click();
    await element.updateComplete;
  }

  describe('settings pages', () => {
    it('collapse settings disclosures below conversations by default', () => {
      expect(element.shadowRoot!.querySelector('#conversations-list'))
          .toBeTruthy();
      expect(element.shadowRoot!.querySelector('#panel-model-config'))
          .toBeNull();
      expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
          .toBeNull();
      const modelToggle =
          element.shadowRoot!.querySelector('#tab-model-config')!;
      const promptToggle =
          element.shadowRoot!.querySelector('#tab-system-prompt')!;
      expect(modelToggle.getAttribute('aria-expanded')).toBe('false');
      expect(promptToggle.getAttribute('aria-expanded')).toBe('false');
    });

    it('switches between disclosures so at most one panel is open', async () => {
      await toggleSettingsPage('model-config');
      expect(element.shadowRoot!.querySelector('#panel-model-config'))
          .toBeTruthy();
      expect(element.shadowRoot!.querySelector('#tab-model-config')!.classList)
          .toContain('active');
      expect(element.shadowRoot!.querySelector('#tab-model-config')!
                 .getAttribute('aria-expanded'))
          .toBe('true');

      await toggleSettingsPage('system-prompt');

      expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
          .toBeTruthy();
      expect(element.shadowRoot!.querySelector('#panel-model-config'))
          .toBeNull();
      expect(element.shadowRoot!.querySelector('#conversations-list'))
          .toBeTruthy();
    });

    it('hides when the expanded disclosure is clicked again', async () => {
         await toggleSettingsPage('model-config');
         expect(element.shadowRoot!.querySelector('#panel-model-config'))
             .toBeTruthy();

         await toggleSettingsPage('system-prompt');
         expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
             .toBeTruthy();

         await toggleSettingsPage('system-prompt');

         expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
             .toBeNull();
         expect(element.shadowRoot!.querySelector('#tab-system-prompt')!
                    .getAttribute('aria-expanded'))
             .toBe('false');
         expect(element.shadowRoot!.querySelector('#conversations-list'))
             .toBeTruthy();
       });

    it('routes Tab/Shift+Tab naturally through an expanded upward settings tray and restores focus on close or Escape',
       async () => {
         const modelBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
             '#tab-model-config')!;
         const sysBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
             '#tab-system-prompt')!;
         await toggleSettingsPage('model-config');

         modelBtn.focus();
         expect(element.shadowRoot!.activeElement).toBe(modelBtn);

         // Forward Tab from expanded #tab-model-config enters .settings-tray
         modelBtn.dispatchEvent(new KeyboardEvent(
             'keydown', {key: 'Tab', bubbles: true, cancelable: true}));
         const closeTrayBtn =
             element.shadowRoot!.querySelector<HTMLButtonElement>(
                 '.btn-close-tray')!;
         expect(element.shadowRoot!.activeElement).toBe(closeTrayBtn);

         // Shift+Tab from first tray element returns to #tab-model-config
         closeTrayBtn.dispatchEvent(new KeyboardEvent(
             'keydown',
             {key: 'Tab', shiftKey: true, bubbles: true, cancelable: true}));
         expect(element.shadowRoot!.activeElement).toBe(modelBtn);

         // Forward Tab from last tray element (#btn-reset-settings) goes to #tab-system-prompt
         const resetBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
             '#btn-reset-settings')!;
         resetBtn.focus();
         resetBtn.dispatchEvent(new KeyboardEvent(
             'keydown', {key: 'Tab', bubbles: true, cancelable: true}));
         expect(element.shadowRoot!.activeElement).toBe(sysBtn);

         // Shift+Tab from #tab-system-prompt goes back into #btn-reset-settings
         sysBtn.dispatchEvent(new KeyboardEvent(
             'keydown',
             {key: 'Tab', shiftKey: true, bubbles: true, cancelable: true}));
         expect(element.shadowRoot!.activeElement).toBe(resetBtn);

         // Clicking .btn-close-tray closes panel and restores focus to #tab-model-config
         closeTrayBtn.focus();
         closeTrayBtn.click();
         await element.updateComplete;
         expect(element.shadowRoot!.querySelector('#panel-model-config'))
             .toBeNull();
         expect(element.shadowRoot!.activeElement).toBe(modelBtn);

         // Pressing Escape inside #panel-system-prompt restores focus to #tab-system-prompt
         await toggleSettingsPage('system-prompt');
         const sysTextarea =
             element.shadowRoot!.querySelector<HTMLTextAreaElement>(
                 '#system-prompt')!;
         sysTextarea.focus();
         sysTextarea.dispatchEvent(new KeyboardEvent(
             'keydown', {key: 'Escape', bubbles: true, cancelable: true}));
         await element.updateComplete;
         expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
             .toBeNull();
         expect(element.shadowRoot!.activeElement).toBe(sysBtn);
       });
  });

  describe('model config page', () => {
    beforeEach(async () => {
      await toggleSettingsPage('model-config');
    });

    it('updates settings on number input changes and describes disabled Greedy fields via tooltip and aria-describedby',
       async () => {
         // 1. Context Length
         const contextInput = element.shadowRoot!.querySelector(
             '#context-length') as HTMLInputElement;
         contextInput.value = '4096';
         contextInput.dispatchEvent(new Event('input'));
         expect(mockSettings.contextLength).toBe(4096);
         expect(mockSettings.saveSettings).toHaveBeenCalled();

         // 2. Max Gen
         const maxGenInput = element.shadowRoot!.querySelector(
             '#max-output-tokens') as HTMLInputElement;
         maxGenInput.value = '1024';
         maxGenInput.dispatchEvent(new Event('input'));
         expect(mockSettings.maxOutputTokens).toBe(1024);

         // 3. Temp & Top-P while in Top-K / Top-P before switching to Greedy
         const tempInput = element.shadowRoot!.querySelector(
             '#temperature') as HTMLInputElement;
         tempInput.value = '1.2';
         tempInput.dispatchEvent(new Event('input'));
         expect(mockSettings.temperature).toBe(1.2);

         const thinkingCheckbox = element.shadowRoot!.querySelector(
             '#enable-thinking') as HTMLInputElement;
         thinkingCheckbox.checked = true;
         thinkingCheckbox.dispatchEvent(new Event('change'));
         expect(mockSettings.enableThinking).toBeTrue();

         const topPInput =
             element.shadowRoot!.querySelector('#top-p') as HTMLInputElement;
         topPInput.value = '0.5';
         topPInput.dispatchEvent(new Event('input'));
         expect(mockSettings.topP).toBe(0.5);

         // 4. Sampler Select -> Greedy removes visible hint box and attaches tooltip + aria-describedby
         const samplerSelect = element.shadowRoot!.querySelector(
             '#sampler-type') as HTMLSelectElement;
         samplerSelect.value = 'greedy';
         samplerSelect.dispatchEvent(new Event('change'));
         expect(mockSettings.samplerType).toBe('greedy');
         element.requestUpdate();
         await element.updateComplete;

         expect(element.shadowRoot!.querySelector('#greedy-sampler-hint'))
             .toBeNull();
         const greedyDesc =
             element.shadowRoot!.querySelector('#greedy-disabled-desc');
         expect(greedyDesc?.textContent).toContain('Not used with Greedy');
         for (const selector of ['#temperature', '#top-p', '#top-k']) {
           const field = element.shadowRoot!.querySelector(
               selector) as HTMLInputElement;
           expect(field.disabled).toBeTrue();
           expect(field.getAttribute('title')).toContain('Not used with Greedy');
           expect(field.getAttribute('aria-describedby'))
               .toBe('greedy-disabled-desc');
         }
       });

    it('toggles speculative decoding with concise drafter hint', () => {
      const toggle = element.shadowRoot!.querySelector<HTMLInputElement>(
          '#enable-speculative-decoding')!;
      const hint =
          toggle.closest('.toggle-row')!.querySelector('.setting-hint')!;
      expect(hint.textContent!.trim())
          .toBe('Adaptive drafting on models with a drafter (e.g. MTP).');
      expect(toggle.checked).toBeFalse();

      toggle.checked = true;
      toggle.dispatchEvent(new Event('change'));

      expect(mockSettings.enableSpeculativeDecoding).toBeTrue();
      expect(mockSettings.saveSettings).toHaveBeenCalled();
    });

    it('makes full checkbox cards clickable (<label>) without double-toggling and preserves exact a11y name and description',
       async () => {
         const thinkingInput = element.shadowRoot!.querySelector<
             HTMLInputElement>('#enable-thinking')!;
         const card = thinkingInput.closest('.toggle-row') as HTMLLabelElement;
         const title = card.querySelector('.toggle-title') as HTMLElement;
         const hint = card.querySelector('.setting-hint') as HTMLElement;

         expect(card.tagName).toBe('LABEL');
         expect(card.getAttribute('for')).toBe('enable-thinking');
         expect(thinkingInput.getAttribute('aria-labelledby'))
             .toBe('enable-thinking-label');
         expect(thinkingInput.getAttribute('aria-describedby'))
             .toBe('enable-thinking-desc');
         expect(title.textContent!.trim()).toBe('Thinking mode');
         expect(hint.textContent!.trim())
             .toBe('Include step-by-step reasoning before answering.');

         expect(thinkingInput.checked).toBeFalse();

         // Clicking the card container toggles once
         card.click();
         await element.updateComplete;
         expect(thinkingInput.checked).toBeTrue();
         expect(mockSettings.enableThinking).toBeTrue();

         // Clicking the description hint toggles once
         hint.click();
         await element.updateComplete;
         expect(thinkingInput.checked).toBeFalse();
         expect(mockSettings.enableThinking).toBeFalse();

         // Clicking the title toggles once
         title.click();
         await element.updateComplete;
         expect(thinkingInput.checked).toBeTrue();
         expect(mockSettings.enableThinking).toBeTrue();

         // Clicking the checkbox input directly toggles once (no double-toggle)
         thinkingInput.click();
         await element.updateComplete;
         expect(thinkingInput.checked).toBeFalse();
         expect(mockSettings.enableThinking).toBeFalse();
       });

    it('resets defaults on reset button click', async () => {
      const resetBtn = element.shadowRoot!.querySelector('#btn-reset-settings') as HTMLButtonElement;
      resetBtn.click();
      await Promise.resolve();
      expect(mockSettings.resetDefaults).toHaveBeenCalledWith('modelConfig');
    });
  });

  describe('system prompt page', () => {
    beforeEach(async () => {
      await toggleSettingsPage('system-prompt');
    });

    it('is shown when selected', () => {
      expect(element.shadowRoot!.querySelector('#panel-system-prompt'))
          .toBeTruthy();
      expect(element.shadowRoot!.querySelector('#panel-model-config'))
          .toBeNull();
      expect(element.shadowRoot!.querySelector('#tab-system-prompt')!.classList)
          .toContain('active');
    });

    it('updates the system prompt', () => {
      const textarea =
          element.shadowRoot!.querySelector<HTMLTextAreaElement>(
              '#system-prompt')!;
      textarea.value = 'You are a pirate.';
      textarea.dispatchEvent(new Event('change'));

      expect(mockSettings.systemPrompt).toBe('You are a pirate.');
      expect(mockSettings.saveSettings).toHaveBeenCalled();
    });

    it('toggles tools via full-card click, description click, and input click without double-toggling',
       async () => {
         const toggle = element.shadowRoot!.querySelector<HTMLInputElement>(
             '#tool-run_javascript')!;
         const card = toggle.closest('.toggle-row') as HTMLLabelElement;
         const hint = card.querySelector('.setting-hint') as HTMLElement;
         expect(card.tagName).toBe('LABEL');
         expect(toggle.getAttribute('aria-labelledby'))
             .toBe('tool-run_javascript-label');
         expect(toggle.getAttribute('aria-describedby'))
             .toBe('tool-run_javascript-desc');
         expect(toggle.checked).toBeTrue();

         // Clicking description text toggles off once
         hint.click();
         await element.updateComplete;
         expect(toggle.checked).toBeFalse();
         expect(mockSettings.disabledTools).toEqual(['run_javascript']);

         // Clicking outer card toggles back on once
         card.click();
         await element.updateComplete;
         expect(toggle.checked).toBeTrue();
         expect(mockSettings.disabledTools).toEqual([]);

         // Clicking checkbox input directly toggles off once
         toggle.click();
         await element.updateComplete;
         expect(toggle.checked).toBeFalse();
         expect(mockSettings.disabledTools).toEqual(['run_javascript']);
       });

    it('resets the system prompt and tools', async () => {
      element.shadowRoot!
          .querySelector<HTMLButtonElement>('#btn-reset-system-prompt')!
          .click();
      await Promise.resolve();
      expect(mockSettings.resetDefaults).toHaveBeenCalledWith('systemPrompt');
    });
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

  it('deletes conversation on click ✕ and dispatches conversation-deleted',
     async () => {
       const list = element.shadowRoot!.querySelector('#conversations-list');
       const deleteBtn =
           list!.querySelector('.btn-delete-conv') as HTMLButtonElement;
       let deletedFired = false;
       element.addEventListener('conversation-deleted', () => {
         deletedFired = true;
       });

       deleteBtn.click();

       expect(mockChatSession.deleteConversation).toHaveBeenCalledWith('conv1');
       expect(deletedFired).toBeTrue();
     });

  it('renders compact summary badges for speculative decoding and custom system prompt',
     async () => {
       mockSettings.samplerType = 'greedy';
       mockSettings.contextLength = 4096;
       mockSettings.enableSpeculativeDecoding = true;
       mockSettings.systemPrompt = 'Be concise.';
       element.requestUpdate();
       await element.updateComplete;

       const modelBadge =
           element.shadowRoot!
               .querySelector('#tab-model-config .tuning-pill')!
               .textContent!.trim();
       const promptBadge =
           element.shadowRoot!
               .querySelector('#tab-system-prompt .tuning-pill')!
               .textContent!.trim();

       expect(modelBadge).toBe('Greedy · 4k+');
       expect(promptBadge).toBe('Custom · 1/1');
     });

  it('renders compact single-line download status in Runtime card without duplicating composer progress bar',
     async () => {
       mockModelLoader.isModelLoading = true;
       mockState.statusText = 'Downloading weights (gemma-4-E2B)...';
       const modelKey = MODELS[0]!.path;
       downloadProgressMap.set(modelKey, 45);
       downloadSpeedsMap.set(modelKey, '5.2 MB/s');
       mockModelLoader.downloadAbortController = {} as AbortController;

       element.requestUpdate();
       await element.updateComplete;

       expect(element.shadowRoot!.querySelector('#progress-container'))
           .toBeNull();
       expect(element.shadowRoot!.querySelector('#btn-cancel-download'))
           .toBeNull();

       const statusText =
           element.shadowRoot!.querySelector('#status-text') as HTMLElement;
       expect(statusText).toBeTruthy();
       expect(statusText.textContent!.trim()).toBe('Downloading weights · 45%');

       const statusDot =
           element.shadowRoot!.querySelector('#status-dot') as HTMLElement;
       expect(statusDot.classList.contains('loading')).toBeTrue();
     });

  it('summarizes reload, download, and error states in the Runtime status line',
     async () => {
       mockModelLoader.engine = {} as unknown as ModelLoaderService['engine'];
       mockModelLoader.needsModelReload =
           jasmine.createSpy('needsModelReload').and.returnValue(true);
       element.requestUpdate();
       await element.updateComplete;
       const statusText = () => element.shadowRoot!.querySelector<HTMLElement>(
                                           '#status-text')!.textContent!.trim();
       expect(statusText()).toBe('Reloads on next message');

       mockModelLoader.isModelLoading = true;
       mockState.statusText = 'Downloading weights (gemma-4-E2B)...';
       mockModelLoader.downloadProgresses.set(MODELS[0]!.path, 64);
       element.requestUpdate();
       await element.updateComplete;
       expect(statusText()).toBe('Downloading weights · 64%');
       expect(element.shadowRoot!.querySelector('#progress-container'))
           .toBeNull();

       mockModelLoader.isModelLoading = false;
       mockState.errorMessage = 'Failed to run inference';
       element.requestUpdate();
       await element.updateComplete;
       expect(statusText()).toBe('Error · see message in chat');
     });

  it('labels the mobile drawer header as Menu with Close / Close menu button',
     () => {
       const title = element.shadowRoot!.querySelector(
                         '.mobile-sidebar-header .section-title') as HTMLElement;
       const closeBtn = element.shadowRoot!.querySelector(
                            '#btn-dismiss-sidebar') as HTMLButtonElement;
       expect(title.textContent!.trim()).toBe('Menu');
       expect(closeBtn.textContent!.trim()).toBe('Close');
       expect(closeBtn.getAttribute('aria-label')).toBe('Close menu');
     });
});
