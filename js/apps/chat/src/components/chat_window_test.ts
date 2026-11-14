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

import './chat_window';

import {Engine} from '@litert-lm/core';

import {LlmChatStateController} from '../state_controller.js';
import {ChatSessionStore, PromptMedia, PromptPart} from '../stores/chat_session_store.js';
import {LocalDirectoryService} from '../stores/local_directory_service.js';
import {getModelKey, ModelLoaderService} from '../stores/model_loader_service.js';
import {MODELS, SettingsStore} from '../stores/settings_store.js';
import {LitertChatWindow} from './chat_window.js';
import {CustomDropdown} from './custom_dropdown.js';

describe('litert-chat-window', () => {
  let element: LitertChatWindow;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
  let mockLocalDirService: jasmine.SpyObj<LocalDirectoryService>;
  let cachedModelsMap: Map<string, number>;
  let createdMedia: Map<string, PromptMedia>;
  const fakeImage: PromptMedia = {
    id: 'img1',
    kind: 'image',
    name: 'cat.png',
    blob: new Blob(),
    url: 'blob:fake-url',
  };

  function getEditor(): HTMLDivElement {
    return element.shadowRoot!.querySelector('#chat-input') as HTMLDivElement;
  }

  /** Adds a file through the hidden file input, as the '+' button does. */
  async function addFileViaPicker(file: File): Promise<void> {
    const input =
        element.shadowRoot!.querySelector('#media-input') as HTMLInputElement;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    input.files = dataTransfer.files;
    input.dispatchEvent(new Event('change'));
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  /** Adds an image through the hidden file input. */
  async function addImageViaPicker(): Promise<File> {
    const file =
        new File([new Uint8Array([1])], 'cat.png', {type: 'image/png'});
    await addFileViaPicker(file);
    return file;
  }

  beforeEach(async () => {
    mockChatSession = jasmine.createSpyObj('ChatSessionStore', [
      'sendMessage',
      'cancelGeneration',
      'createMedia',
      'getMedia',
      'redoResponse',
      'rewindAndEdit',
    ]);
    // Initialize properties used during render
    mockChatSession.isGenerating = false;
    mockChatSession.messages = [];
    createdMedia = new Map([[fakeImage.id, fakeImage]]);
    mockChatSession.createMedia.and.callFake((kind, name, blob) => {
      if (kind === 'image') return fakeImage;
      const media: PromptMedia =
          {id: `media${createdMedia.size}`, kind, name, blob, url: 'blob:x'};
      createdMedia.set(media.id, media);
      return media;
    });
    mockChatSession.getMedia.and.callFake((id: string) => createdMedia.get(id));

    mockSettings = jasmine.createSpyObj('SettingsStore', ['saveSettings']);
    mockSettings.selectedModelPath =
        'https://example.com/path/to/model_file.litertlm';
    mockSettings.customModels = [];
    mockSettings.localDirModels = [];
    mockSettings.textBackend = 'gpu';
    mockSettings.visionBackend = 'gpu';
    mockSettings.audioBackend = 'gpu';
    mockSettings.textPrecision = 'default';
    mockSettings.visionPrecision = 'default';
    mockSettings.audioPrecision = 'default';
    mockSettings.contextLength = 4096;
    mockSettings.maxOutputTokens = 2048;
    mockSettings.samplerType = 'greedy';
    mockSettings.temperature = 1.0;
    mockSettings.topP = 0.95;
    mockSettings.topK = 64;
    mockSettings.enableThinking = true;
    mockSettings.enableSpeculativeDecoding = false;
    mockSettings.systemPrompt = '';
    mockSettings.disabledTools = [];

    mockLocalDirService = jasmine.createSpyObj('LocalDirectoryService', [
      'mountDirectory',
      'scanDirectory',
      'getFile',
    ]);
    mockLocalDirService.isAuthorized = false;
    (mockLocalDirService as unknown as {isSupported: boolean}).isSupported =
        false;
    (mockLocalDirService as unknown as {directoryName: string | null})
        .directoryName = null;

    cachedModelsMap = new Map<string, number>();
    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
      'deleteModelFromCache',
      'clearAllCache',
      'cancelDownload',
      'addRemoteModel',
    ]);
    mockModelLoader.isModelLoading = false;
    mockModelLoader.cachedModels = cachedModelsMap;
    mockModelLoader.downloadProgresses = new Map<string, number>();
    mockModelLoader.downloadSpeeds = new Map<string, string>();
    mockModelLoader.downloadAbortController = null;

    mockState = {
      chatSession: mockChatSession,
      settings: mockSettings,
      modelLoader: mockModelLoader,
      localDirService: mockLocalDirService,
      statusText: 'Idle',
      addHost: jasmine.createSpy('addHost'),
      removeHost: jasmine.createSpy('removeHost'),
      requestUpdate: jasmine.createSpy('requestUpdate'),
      storage: window.localStorage,
      requestConfirm: jasmine.createSpy('requestConfirm').and.resolveTo(true),
    } as unknown as jasmine.SpyObj<LlmChatStateController>;

    element = document.createElement('litert-chat-window');
    element.state = mockState;
    document.body.appendChild(element);
    await element.updateComplete;
  });

  afterEach(() => {
    element.remove();
  });

  it('renders quick starters when messages are empty', () => {
    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    expect(starters.length).toBe(3);
    expect(starters[0]!.querySelector('.starter-title')?.textContent?.trim())
        .toBe('Explain WebGPU');
    expect(starters[0]!.querySelector('.starter-subtitle')?.textContent?.trim())
        .toBeTruthy();
  });

  it('does not render quick starters when messages exist', async () => {
    mockChatSession.messages = [
      {role: 'user', text: 'Hi', senderName: 'User'},
    ];
    element.requestUpdate();
    await element.updateComplete;

    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    expect(starters.length).toBe(0);

    const bubbles = element.shadowRoot!.querySelectorAll('litert-chat-bubble');
    expect(bubbles.length).toBe(1);
  });

  it('disables input when model is loading', async () => {
    mockModelLoader.isModelLoading = true;
    mockModelLoader.downloadProgresses =
        new Map([[mockSettings.selectedModelPath, 64]]);
    element.requestUpdate();
    await element.updateComplete;

    const editor = getEditor();
    expect(editor.getAttribute('contenteditable')).toBe('false');
    expect(editor.dataset['placeholder'])
        .toBe('Model is preparing, please wait...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeTrue();
    expect(sendBtn.textContent!.trim()).toBe('Loading 64%…');
  });

  it('enables input when model is not loading and enables Send once prompt has content', async () => {
    const editor = getEditor();
    expect(editor.getAttribute('contenteditable')).toBe('true');
    expect(editor.dataset['placeholder']).toContain('Message LiteRT-LM...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeTrue();

    editor.textContent = 'Hello';
    editor.dispatchEvent(new Event('input'));
    await element.updateComplete;
    expect(sendBtn.disabled).toBeFalse();
  });

  it('shows "Send" when model is cached or already loaded in memory', async () => {
    cachedModelsMap.set(mockSettings.selectedModelPath, 1000);
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Send');

    cachedModelsMap.clear();
    (mockModelLoader as unknown as {engine: object}).engine = {};
    element.requestUpdate();
    await element.updateComplete;
    expect(sendBtn.textContent!.trim()).toBe('Send');
  });

  it('shows "Load & Send" when local-dir model is not yet loaded', async () => {
    mockSettings.selectedModelPath = 'local-dir://gemma-4-e2b';
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Load & Send');
  });

  it('shows "Download & Send" when model is not cached', () => {
    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Download & Send');
  });

  it('shows "Stop" button during generation', async () => {
    mockChatSession.isGenerating = true;
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send');
    expect(sendBtn).toBeNull();

    const stopBtn = element.shadowRoot!.querySelector('.btn-stop') as HTMLButtonElement;
    expect(stopBtn).toBeTruthy();
    expect(stopBtn.textContent!.trim()).toBe('Stop');

    stopBtn.focus();
    stopBtn.click();
    expect(mockChatSession.cancelGeneration).toHaveBeenCalled();

    mockChatSession.isGenerating = false;
    element.requestUpdate();
    await element.updateComplete;
    expect(element.shadowRoot!.activeElement).toBe(getEditor());
  });

  it('renders Try again and Choose another model recovery buttons on error banner',
     async () => {
       mockState.errorMessage = 'WebGPU unavailable';
       mockState.dismissError = jasmine.createSpy('dismissError');
       element.requestUpdate();
       await element.updateComplete;

       const actions =
           element.shadowRoot!.querySelectorAll<HTMLButtonElement>(
               '.btn-alert-action');
       expect(actions.length).toBe(2);
       expect(actions[0]!.textContent!.trim()).toBe('Try again');
       expect(actions[1]!.textContent!.trim()).toBe('Choose another model');

       let chooseModelFired = false;
       element.addEventListener('choose-model', () => {
         chooseModelFired = true;
       });
       actions[1]!.click();
       expect(chooseModelFired).toBeTrue();

       actions[0]!.click();
       await Promise.resolve();
       expect(mockState.dismissError).toHaveBeenCalled();
       expect(mockModelLoader.loadModelWeights).toHaveBeenCalled();
     });

  it('calls redoResponse when Try again is clicked after a failed assistant turn',
     async () => {
       mockChatSession.redoResponse.and.resolveTo();
       mockChatSession.messages = [
         {role: 'user', text: 'Hi', senderName: 'User'},
         {
           role: 'assistant',
           text: '',
           thoughtText: 'Partial thought',
           senderName: 'Assistant',
         },
       ];
       mockState.errorMessage = 'Stream interrupted';
       mockState.dismissError = jasmine.createSpy('dismissError');
       element.requestUpdate();
       await element.updateComplete;

       const retryBtn = element.shadowRoot!.querySelector(
                            '#btn-retry-error') as HTMLButtonElement;
       retryBtn.click();
       await Promise.resolve();

       expect(mockState.dismissError).toHaveBeenCalled();
       expect(mockChatSession.redoResponse).toHaveBeenCalledWith(1);
       expect(mockModelLoader.loadModelWeights).not.toHaveBeenCalled();
     });

  it('keeps the streaming bubble from shrinking until generation ends',
     async () => {
       const nextFrame = () =>
           new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
       const renderReply = async (text: string, isGenerating: boolean) => {
         mockChatSession.isGenerating = isGenerating;
         mockChatSession.messages = [
           {role: 'user', text: 'Hi', senderName: 'User'},
           {role: 'assistant', text, senderName: 'Model'},
         ];
         element.requestUpdate();
         await element.updateComplete;
         await nextFrame();
       };
       const getReplyBubble = () =>
           element.shadowRoot!.querySelectorAll('litert-chat-bubble')[1] as
           HTMLElement;

       await renderReply('Paragraph\n\n'.repeat(20), true);
       const tallHeight = getReplyBubble().offsetHeight;

       // Re-rendered markdown that lays out shorter must not shrink the bubble.
       await renderReply('Paragraph', true);
       expect(getReplyBubble().offsetHeight).toBeGreaterThanOrEqual(tallHeight);

       // Once generation ends, the bubble returns to its natural height.
       await renderReply('Paragraph', false);
       expect(getReplyBubble().style.minHeight).toBe('');
       expect(getReplyBubble().offsetHeight).toBeLessThan(tallHeight);
     });

  it('sends message on send button click', async () => {
    const editor = getEditor();
    editor.textContent = 'Test prompt';
    editor.dispatchEvent(new Event('input'));
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    sendBtn.click();
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith('Test prompt');
    expect(editor.textContent).toBe('');
  });

  it('sends message on Enter key press', async () => {
    const editor = getEditor();
    editor.textContent = 'Test prompt Enter';
    
    const event = new KeyboardEvent('keydown', {key: 'Enter', shiftKey: false});
    editor.dispatchEvent(event);
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith('Test prompt Enter');
    expect(editor.textContent).toBe('');
  });

  it('does not send message on Shift+Enter key press', async () => {
    const editor = getEditor();
    editor.textContent = 'Test prompt Shift Enter';
    
    const event = new KeyboardEvent('keydown', {key: 'Enter', shiftKey: true});
    editor.dispatchEvent(event);
    await element.updateComplete;

    expect(mockChatSession.sendMessage).not.toHaveBeenCalled();
    expect(editor.textContent).toBe('Test prompt Shift Enter');
  });

  it('inserts newline on Ctrl+J key press', async () => {
    const editor = getEditor();
    editor.textContent = 'Line 1Line 2';
    editor.focus();
    const range = document.createRange();
    range.setStart(editor.firstChild!, 6);
    range.collapse(true);
    const selection = document.getSelection()!;
    selection.removeAllRanges();
    selection.addRange(range);

    editor.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'j', ctrlKey: true}));
    editor.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith('Line 1\nLine 2');
  });

  it('serializes <div> line wrappers as newlines', async () => {
    const editor = getEditor();
    editor.innerHTML = 'Line 1<div>Line 2</div><div><br></div><div>Line 4</div>';

    editor.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
    await element.updateComplete;

    expect(mockChatSession.sendMessage)
        .toHaveBeenCalledWith('Line 1\nLine 2\n\nLine 4');
  });

  it('does not send an empty message', async () => {
    getEditor().textContent = '   ';
    getEditor().dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
    await element.updateComplete;

    expect(mockChatSession.sendMessage).not.toHaveBeenCalled();
  });

  it('adds picked images inline and sends them in order', async () => {
    const editor = getEditor();
    editor.textContent = 'Describe ';

    const file = await addImageViaPicker();

    const chip = editor.querySelector('.media-chip') as HTMLElement;
    expect(chip).toBeTruthy();
    expect(chip.dataset['mediaId']).toBe(fakeImage.id);
    expect(chip.querySelector('img.chip-thumb')).toBeTruthy();
    expect(mockChatSession.createMedia)
        .toHaveBeenCalledWith('image', 'cat.png', file);

    editor.appendChild(document.createTextNode('please'));
    editor.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
    await element.updateComplete;

    expect(mockChatSession.sendMessage).toHaveBeenCalledWith([
      {type: 'text', text: 'Describe '},
      {type: 'media', media: fakeImage},
      // The non-breaking spacer after the chip is sent as a normal space.
      {type: 'text', text: ' please'},
    ]);
    expect(editor.querySelector('.media-chip')).toBeNull();
  });

  it('adds picked audio files as playable chips', async () => {
    const file =
        new File([new Uint8Array([1])], 'meow.wav', {type: 'audio/wav'});
    await addFileViaPicker(file);

    expect(mockChatSession.createMedia)
        .toHaveBeenCalledWith('audio', 'meow.wav', file);
    const chip = getEditor().querySelector('.media-chip') as HTMLElement;
    expect(chip).toBeTruthy();
    expect(chip.querySelector('.chip-play-btn')).toBeTruthy();
    expect(chip.querySelector('img')).toBeNull();

    getEditor().dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
    await element.updateComplete;

    const media = createdMedia.get(chip.dataset['mediaId']!)!;
    expect(mockChatSession.sendMessage).toHaveBeenCalledWith([
      {type: 'media', media},
      {type: 'text', text: ' '},
    ]);
  });

  it('rejects files that are not images or audio', async () => {
    await addFileViaPicker(
        new File(['text'], 'notes.txt', {type: 'text/plain'}));

    expect(mockChatSession.createMedia).not.toHaveBeenCalled();
    expect(getEditor().querySelector('.media-chip')).toBeNull();
  });

  it('removes media when its chip delete button is clicked', async () => {
    await addImageViaPicker();
    const editor = getEditor();
    const deleteBtn =
        editor.querySelector('.chip-del-btn') as HTMLButtonElement;
    deleteBtn.click();

    expect(editor.querySelector('.media-chip')).toBeNull();
  });

  it('populates input when clicking quick starter', async () => {
    const starters = element.shadowRoot!.querySelectorAll('.btn-starter');
    const firstStarter = starters[0] as HTMLButtonElement;
    
    firstStarter.click();
    await element.updateComplete;

    const editor = getEditor();
    expect(editor.textContent).toBe('Explain WebGPU in simple terms.');
    expect(element.shadowRoot!.activeElement).toBe(editor);
  });

  it('handles edit-prompt event', async () => {
    // Add a message to render a bubble
    mockChatSession.messages = [
      {role: 'assistant', text: 'Some response', senderName: 'Assistant'},
    ];
    element.requestUpdate();
    await element.updateComplete;

    const bubble = element.shadowRoot!.querySelector('litert-chat-bubble');
    expect(bubble).toBeTruthy();

    const editor = getEditor();
    expect(editor.textContent).toBe('');

    const parts: PromptPart[] = [
      {type: 'text', text: 'Edited '},
      {type: 'media', media: fakeImage},
      {type: 'text', text: ' Prompt'},
    ];
    bubble!.dispatchEvent(new CustomEvent('edit-prompt', {
      detail: {parts},
      bubbles: true,
      composed: true,
    }));
    await element.updateComplete;

    expect(editor.textContent).toContain('Edited ');
    expect(editor.textContent).toContain(' Prompt');
    const chip = editor.querySelector('.media-chip') as HTMLElement;
    expect(chip.dataset['mediaId']).toBe(fakeImage.id);
    expect(element.shadowRoot!.activeElement).toBe(editor);
  });

  it('immediately releases auto-scroll on upward wheel during streaming and re-pins via Latest button',
     async () => {
       const warnSpy = spyOn(console, 'warn').and.callThrough();
       element.style.height = '320px';
       element.style.display = 'flex';
       const nextFrame = () =>
           new Promise<void>(resolve => requestAnimationFrame(() => resolve()));

       const longText = Array.from({length: 50}, (_, i) => `Line ${i + 1}`)
                            .join('\n\n');
       mockChatSession.isGenerating = true;
       mockChatSession.messages = [
         {role: 'user', text: 'Stream lines', senderName: 'User'},
         {role: 'assistant', text: longText, senderName: 'Assistant'},
       ];
       element.requestUpdate();
       await element.updateComplete;
       await nextFrame();

       const msgBox =
           element.shadowRoot!.querySelector('#chat-messages') as HTMLDivElement;
       expect(msgBox.scrollHeight).toBeGreaterThan(msgBox.clientHeight);
       expect(element.shadowRoot!.querySelector('.btn-scroll-bottom'))
           .toBeNull();

       // Subtle upward trackpad wheel event (deltaY = -4) immediately unpins.
       msgBox.dispatchEvent(
           new WheelEvent('wheel', {deltaY: -4, bubbles: true}));
       msgBox.scrollTop = Math.max(0, msgBox.scrollTop - 20);
       msgBox.dispatchEvent(new Event('scroll'));
       await element.updateComplete;

       const pinnedUserTop = msgBox.scrollTop;
       const latestBtn = element.shadowRoot!.querySelector(
                             '.btn-scroll-bottom') as HTMLButtonElement;
       expect(latestBtn).toBeTruthy();

       // Subsequent streamed tokens must NOT fight or pull scrollTop back down.
       mockChatSession.messages = [
         {role: 'user', text: 'Stream lines', senderName: 'User'},
         {
           role: 'assistant',
           text: `${longText}\n\nMore streamed output`,
           senderName: 'Assistant',
         },
       ];
       element.requestUpdate();
       await element.updateComplete;
       await nextFrame();
       expect(msgBox.scrollTop).toBe(pinnedUserTop);

       // Clicking Latest re-pins to the bottom immediately.
       latestBtn.click();
       await element.updateComplete;
       expect(element.shadowRoot!.querySelector('.btn-scroll-bottom'))
           .toBeNull();

       const litScheduleWarnings = warnSpy.calls.allArgs().filter(
           args => String(args[0] ?? '').includes('scheduled an update'));
       expect(litScheduleWarnings).toEqual([]);
     });

  describe('unified bottom-left composer model picker', () => {
    beforeEach(async () => {
      mockSettings.selectedModelPath = MODELS[0]!.path;
      element.requestUpdate();
      await element.updateComplete;
    });

    it('renders upward model dropdown in bottom-left of composer without Shift+Enter hint',
       () => {
         expect(element.shadowRoot!.querySelector('.composer-hint')).toBeNull();
         const dropdown = element.shadowRoot!.querySelector(
             '#composer-model-dropdown') as CustomDropdown;
         expect(dropdown).toBeTruthy();

         const items = dropdown.querySelectorAll('.dropdown-item');
         // Built-in models + upload + add-url
         expect(items.length).toBe(MODELS.length + 2);
         expect(items[0]!.querySelector('.model-name')!.textContent!.trim())
             .toBe(MODELS[0]!.name);
       });

    it('triggers model load on model change and supports Clear All cached weights in menu footer',
       async () => {
         cachedModelsMap.set(MODELS[0]!.path, 2_300_000_000);
         element.requestUpdate();
         await element.updateComplete;

         const dropdown = element.shadowRoot!.querySelector(
             '#composer-model-dropdown') as CustomDropdown;
         const footerValue = dropdown.querySelector(
             '.cache-footer-row .cache-bar-value') as HTMLElement;
         expect(footerValue.textContent!.trim()).toBe('2.30 GB');

         const clearBtn = dropdown.querySelector(
             '#composer-model-clear-cache') as HTMLButtonElement;
         expect(clearBtn.disabled).toBeFalse();
         clearBtn.click();
         expect(mockModelLoader.clearAllCache).toHaveBeenCalled();

         dropdown.dispatchEvent(
             new CustomEvent('change', {detail: MODELS[1]!.path}));
         await element.updateComplete;
         expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
         expect(mockSettings.saveSettings).toHaveBeenCalled();
         expect(mockModelLoader.loadModelWeights).toHaveBeenCalled();
       });

    it('deletes cached models via ✕ and resets selectedModelPath only when active model is deleted',
       async () => {
         mockModelLoader.deleteModelFromCache.and.returnValue(
             Promise.resolve(true));
         mockSettings.selectedModelPath = MODELS[1]!.path;
         cachedModelsMap.set(MODELS[0]!.path, 1024 * 1024 * 1024);
         element.requestUpdate();
         await element.updateComplete;

         let deleteBtn = element.shadowRoot!.querySelector(
             '.delete-cache-btn') as HTMLButtonElement;
         deleteBtn.click();
         await new Promise(resolve => setTimeout(resolve));
         expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
         expect(mockSettings.saveSettings).not.toHaveBeenCalled();

         // Now cache and delete the active model
         cachedModelsMap.clear();
         cachedModelsMap.set(MODELS[1]!.path, 1024 * 1024 * 1024);
         element.requestUpdate();
         await element.updateComplete;

         deleteBtn = element.shadowRoot!.querySelector(
             '.delete-cache-btn') as HTMLButtonElement;
         deleteBtn.click();
         await new Promise(resolve => setTimeout(resolve));
         expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);
         expect(mockSettings.saveSettings).toHaveBeenCalled();
       });

    it('supports select-dir and mounted local directory Rescan in menu footer',
       async () => {
         (mockLocalDirService as unknown as {isSupported: boolean})
             .isSupported = true;
         mockLocalDirService.isAuthorized = true;
         (mockLocalDirService as unknown as {directoryName: string | null})
             .directoryName = 'my-models';
         element.requestUpdate();
         await element.updateComplete;

         const dropdown = element.shadowRoot!.querySelector(
             '#composer-model-dropdown') as CustomDropdown;
         const selectDirItem =
             dropdown.querySelector('[data-value="select-dir"]');
         const uploadItem = dropdown.querySelector('[data-value="upload"]');
         expect(selectDirItem).toBeTruthy();
         expect(selectDirItem!.textContent).toContain('Change local directory');
         expect(uploadItem).toBeTruthy();

         dropdown.dispatchEvent(
             new CustomEvent('change', {detail: 'select-dir'}));
         await element.updateComplete;
         expect(mockLocalDirService.mountDirectory).toHaveBeenCalled();

         const rescanBtn =
             dropdown.querySelector('.btn-rescan-dir') as HTMLButtonElement;
         expect(rescanBtn).toBeTruthy();
         rescanBtn.click();
         expect(mockLocalDirService.scanDirectory).toHaveBeenCalled();
       });

    it('renders inline composer download progress strip with Cancel button while loading',
       async () => {
         mockModelLoader.isModelLoading = true;
         mockState.statusText = 'Downloading Gemma 4 E2B...';
         mockModelLoader.downloadProgresses.set(MODELS[0]!.path, 45);
         mockModelLoader.downloadSpeeds.set(MODELS[0]!.path, '5.2 MB/s');
         mockModelLoader.downloadAbortController = {} as AbortController;
         element.requestUpdate();
         await element.updateComplete;

         const bar = element.shadowRoot!.querySelector(
             '#composer-loading-bar') as HTMLElement;
         expect(bar).toBeTruthy();
         expect(bar.textContent).toContain('Downloading Gemma 4 E2B...');
         expect(bar.textContent).toContain('5.2 MB/s · 45%');

         const cancelBtn = element.shadowRoot!.querySelector(
             '#btn-composer-cancel-download') as HTMLButtonElement;
         expect(cancelBtn).toBeTruthy();
         cancelBtn.click();
         expect(mockModelLoader.cancelDownload).toHaveBeenCalled();

         const sendBtn = element.shadowRoot!.querySelector(
             '#btn-send') as HTMLButtonElement;
         expect(sendBtn.getAttribute('aria-label')).toContain('Loading 45%');
       });

    it('renders compact mobile state label alongside full state label for Reload on send',
       async () => {
         mockModelLoader.engine = {} as unknown as Engine;
         mockModelLoader.needsModelReload = jasmine.createSpy().and.returnValue(true);
         element.requestUpdate();
         await element.updateComplete;

         const fullState = element.shadowRoot!.querySelector(
             '#btn-composer-model .state-full') as HTMLElement;
         const compactState = element.shadowRoot!.querySelector(
             '#btn-composer-model .state-compact') as HTMLElement;
         expect(fullState?.textContent?.trim()).toBe('Reload on send');
         expect(compactState?.textContent?.trim()).toBe('Reload');
       });

    it('renders mobile download glyph on #btn-send while weights are uncached and switches back to up-arrow once cached or loaded',
       async () => {
         mockSettings.selectedModelPath = MODELS[0]!.path;
         element.requestUpdate();
         await element.updateComplete;

         let sendBtn = element.shadowRoot!.querySelector(
             '#btn-send') as HTMLButtonElement;
         expect(sendBtn.classList.contains('needs-download')).toBeTrue();
         expect(sendBtn.querySelector('.send-icon-download')).toBeTruthy();
         expect(sendBtn.getAttribute('aria-label')).toBe('Download & Send');

         cachedModelsMap.set(MODELS[0]!.path, 2_300_000_000);
         element.requestUpdate();
         await element.updateComplete;

         sendBtn = element.shadowRoot!.querySelector(
             '#btn-send') as HTMLButtonElement;
         expect(sendBtn.classList.contains('needs-download')).toBeFalse();
         expect(sendBtn.querySelector('.send-icon-download')).toBeNull();
         expect(sendBtn.querySelector('.send-icon-up')).toBeTruthy();
         expect(sendBtn.getAttribute('aria-label')).toBe('Send');
       });

    describe('adding a model by URL via upward composer popover', () => {
      const remoteModel = {
        name: 'SmolLM3-3B',
        filename: 'SmolLM3-3B.litertlm',
        path:
            'https://huggingface.co/litert-community/SmolLM3-3B/resolve/c0ffee/SmolLM3-3B.litertlm',
        size: '3.13 GB',
      };

      async function openForm(): Promise<HTMLFormElement> {
        const dropdown = element.shadowRoot!.querySelector(
            '#composer-model-dropdown') as HTMLElement;
        dropdown.dispatchEvent(new CustomEvent('change', {detail: 'add-url'}));
        await element.updateComplete;
        return element.shadowRoot!.querySelector('#composer-add-model-form')!;
      }

      async function submit(form: HTMLFormElement, text: string) {
        form.querySelector<HTMLInputElement>(
            '#composer-add-model-input')!.value = text;
        form.requestSubmit();
        await new Promise(resolve => setTimeout(resolve));
        await element.updateComplete;
      }

      it('adds, selects, and loads a remote model without duplicates, and removes uncached custom models',
         async () => {
           mockModelLoader.addRemoteModel.and.resolveTo(remoteModel);
           const form = await openForm();
           expect(form).toBeTruthy();
           expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);

           await submit(form, 'litert-community/SmolLM3-3B');
           expect(mockModelLoader.addRemoteModel)
               .toHaveBeenCalledOnceWith('litert-community/SmolLM3-3B', 'gpu');
           expect(mockSettings.customModels).toEqual([remoteModel]);
           expect(mockSettings.selectedModelPath).toBe(remoteModel.path);
           expect(
               element.shadowRoot!.querySelector('#composer-add-model-form'))
               .toBeNull();

           // Submitting same model again does not duplicate
           mockModelLoader.addRemoteModel.and.resolveTo({...remoteModel});
           await submit(await openForm(), 'litert-community/SmolLM3-3B');
           expect(mockSettings.customModels).toEqual([remoteModel]);

           // Removing uncached custom model from list
           element.requestUpdate();
           await element.updateComplete;
           const removeBtn = element.shadowRoot!.querySelector<
               HTMLButtonElement>('.delete-cache-btn[title="Remove from list"]')!;
           removeBtn.click();
           await Promise.resolve();
           expect(mockSettings.customModels).toEqual([]);
           expect(mockSettings.selectedModelPath).toBe(MODELS[0]!.path);

           // Entering a built-in Hugging Face repo selects the built-in entry
           // directly without calling addRemoteModel or appending to customModels
           mockModelLoader.addRemoteModel.calls.reset();
           await submit(
               await openForm(),
               'litert-community/gemma-4-E4B-it-litert-lm');
           expect(mockModelLoader.addRemoteModel).not.toHaveBeenCalled();
           expect(mockSettings.customModels).toEqual([]);
           expect(mockSettings.selectedModelPath).toBe(MODELS[1]!.path);
         });

      it('keeps popover open on lookup error, disables while loading, and closes on Escape restoring focus',
         async () => {
           mockModelLoader.addRemoteModel.and.rejectWith(
               new Error('Not found'));
           const form = await openForm();
           await submit(form, 'org/missing');
           expect(
               element.shadowRoot!.querySelector('#composer-add-model-form'))
               .toBeTruthy();

           const input = form.querySelector<HTMLInputElement>(
               '#composer-add-model-input')!;
           input.dispatchEvent(
               new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
           await element.updateComplete;
           expect(
               element.shadowRoot!.querySelector('#composer-add-model-form'))
               .toBeNull();
           const triggerBtn = element.shadowRoot!.querySelector(
               '#btn-composer-model') as HTMLButtonElement;
           expect(element.shadowRoot!.activeElement).toBe(triggerBtn);

           mockModelLoader.isModelLoading = true;
           element.requestUpdate();
           await element.updateComplete;
           const disabledForm = await openForm();
           expect(
               disabledForm
                   .querySelector<HTMLInputElement>('#composer-add-model-input')!
                   .disabled)
               .toBeTrue();
         });
    });

    it('names the selected model and size in the high-VRAM confirmation dialog',
       async () => {
         window.localStorage.removeItem(
             'litertlm-large-model-warning-dismissed');
         const largeModel = MODELS.find(m => m.path.includes('gemma-4-26B'))!;
         let capturedOptions: {title: string; message: string}|null = null;
         mockState.requestConfirm.and.callFake(async (opts) => {
           capturedOptions = {title: opts.title, message: opts.message};
           return false;
         });

         const dropdown = element.shadowRoot!.querySelector(
             '#composer-model-dropdown') as HTMLElement;
         dropdown.dispatchEvent(
             new CustomEvent('change', {detail: largeModel.path}));
         await new Promise(resolve => setTimeout(resolve));
         await element.updateComplete;

         expect(capturedOptions).not.toBeNull();
         expect(capturedOptions!.title)
             .toContain(`${largeModel.name} (${largeModel.size})`);
         expect(capturedOptions!.message)
             .toContain(`${largeModel.name} (${largeModel.size}) needs`);
       });

    it('hides starter cards when the composer draft is non-empty and restores them when cleared',
       async () => {
         expect(element.shadowRoot!.querySelector('.starters-container'))
             .toBeTruthy();

         element.dispatchEvent(new CustomEvent('edit-prompt', {
           detail: {parts: [{type: 'text', text: 'Draft prompt'}]},
         }));
         // Also trigger via bubble listener path: simulate typing in #chat-input
         const chatInput =
             element.shadowRoot!.querySelector('#chat-input') as HTMLElement;
         chatInput.textContent = 'Draft prompt';
         chatInput.dispatchEvent(new Event('input'));
         await element.updateComplete;

         expect(element.shadowRoot!.querySelector('.starters-container'))
             .toBeNull();

         chatInput.textContent = '';
         chatInput.dispatchEvent(new Event('input'));
         await element.updateComplete;

         expect(element.shadowRoot!.querySelector('.starters-container'))
             .toBeTruthy();
       });

    describe('offline weight availability guard', () => {
      it('disables Send, Enter-to-send, and starter cards with an inline hint when offline and uncached',
         async () => {
           element.isOffline = true;
           await element.updateComplete;

           const starters = Array.from(
               element.shadowRoot!.querySelectorAll<HTMLButtonElement>(
                   '.btn-starter'));
           expect(starters.length).toBe(3);
           expect(starters.every(btn => btn.disabled)).toBeTrue();

           const hint = element.shadowRoot!.querySelector(
               '#composer-offline-hint');
           expect(hint).toBeTruthy();
           expect(hint!.textContent).toContain('choose a cached or local model');

           const editor = getEditor();
           editor.textContent = 'Can I send offline?';
           editor.dispatchEvent(new Event('input'));
           await element.updateComplete;

           const sendBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
               '#btn-send')!;
           expect(sendBtn.disabled).toBeTrue();
           expect(sendBtn.textContent?.trim()).toBe('Offline · Not cached');

           editor.dispatchEvent(new KeyboardEvent(
               'keydown', {key: 'Enter', bubbles: true, cancelable: true}));
           await element.updateComplete;
           expect(mockChatSession.sendMessage).not.toHaveBeenCalled();
           expect(editor.textContent).toBe('Can I send offline?');

           const dropdown = element.shadowRoot!.querySelector(
               '#composer-model-dropdown') as CustomDropdown;
           const openSpy = spyOn(dropdown, 'open');
           const chooseBtn =
               element.shadowRoot!.querySelector<HTMLButtonElement>(
                   '#btn-offline-choose-model')!;
           chooseBtn.click();
           expect(openSpy).toHaveBeenCalled();
         });

      it('keeps Send and starters enabled when offline with cached, local-dir, or loaded weights',
         async () => {
           element.isOffline = true;
           cachedModelsMap.set(
               getModelKey(mockSettings.selectedModelPath), 1024);
           element.requestUpdate();
           await element.updateComplete;

           expect(element.shadowRoot!.querySelector('#composer-offline-hint'))
               .toBeNull();
           const starters = Array.from(
               element.shadowRoot!.querySelectorAll<HTMLButtonElement>(
                   '.btn-starter'));
           expect(starters.every(btn => !btn.disabled)).toBeTrue();

           const editor = getEditor();
           editor.textContent = 'Cached prompt';
           editor.dispatchEvent(new Event('input'));
           await element.updateComplete;

           const sendBtn = element.shadowRoot!.querySelector<HTMLButtonElement>(
               '#btn-send')!;
           expect(sendBtn.disabled).toBeFalse();
           expect(sendBtn.textContent?.trim()).toBe('Send');

           // Local directory model while offline
           cachedModelsMap.clear();
           mockSettings.selectedModelPath = 'local-dir://Gemma4.litertlm';
           element.requestUpdate();
           await element.updateComplete;
           expect(element.shadowRoot!.querySelector('#composer-offline-hint'))
               .toBeNull();
           expect(sendBtn.disabled).toBeFalse();
           expect(sendBtn.textContent?.trim()).toBe('Load & Send');

           // Already-loaded engine while offline
           mockSettings.selectedModelPath =
               'https://example.com/path/to/model_file.litertlm';
           mockModelLoader.engine = {} as Engine;
           element.requestUpdate();
           await element.updateComplete;
           expect(element.shadowRoot!.querySelector('#composer-offline-hint'))
               .toBeNull();
           expect(sendBtn.disabled).toBeFalse();
           expect(sendBtn.textContent?.trim()).toBe('Send');
         });
    });

    describe('modality disabling', () => {
      it('disables camera, mic, and media attach buttons with explanations when Vision and/or Audio are off',
         async () => {
           const addMediaBtn =
               element.shadowRoot!.querySelector<HTMLButtonElement>(
                   '#btn-add-media')!;
           const cameraBtn =
               element.shadowRoot!.querySelector<HTMLButtonElement>(
                   '#btn-camera')!;
           const recordBtn =
               element.shadowRoot!.querySelector<HTMLButtonElement>(
                   '#btn-record')!;
           const mediaInput =
               element.shadowRoot!.querySelector<HTMLInputElement>(
                   '#media-input')!;

           const heroSubtitle = () =>
               element.shadowRoot!
                   .querySelector<HTMLElement>('.hero-subtitle')!
                   .textContent!.replace(/\s+/g, ' ')
                   .trim();
           const firstCapTag = () =>
               element.shadowRoot!
                   .querySelector<HTMLElement>('.capability-tag')!
                   .textContent!.replace(/\s+/g, ' ')
                   .trim();

           expect(heroSubtitle()).toContain('Attach images or audio inline');
           expect(firstCapTag()).toBe('Vision & Audio');

           mockSettings.visionBackend = 'off';
           mockSettings.audioBackend = 'gpu';
           element.requestUpdate();
           await element.updateComplete;

           expect(cameraBtn.disabled).toBeTrue();
           expect(cameraBtn.title).toContain('Vision is turned off');
           expect(recordBtn.disabled).toBeFalse();
           expect(addMediaBtn.disabled).toBeFalse();
           expect(addMediaBtn.title).toContain('Vision is turned off');
           expect(mediaInput.accept).toBe('audio/*');
           expect(heroSubtitle()).toContain('Attach audio inline');
           expect(heroSubtitle()).not.toContain('images');
           expect(firstCapTag()).toBe('Audio');

           // Rejects dropped/picked images when Vision is off
           await addImageViaPicker();
           expect(mockChatSession.createMedia).not.toHaveBeenCalled();
           expect(mockState.statusText).toContain('Vision is turned off');

           mockSettings.visionBackend = 'gpu';
           mockSettings.audioBackend = 'off';
           element.requestUpdate();
           await element.updateComplete;
           expect(heroSubtitle()).toContain('Attach images inline');
           expect(heroSubtitle()).not.toContain('audio');
           expect(firstCapTag()).toBe('Vision');

           mockSettings.visionBackend = 'off';
           mockSettings.audioBackend = 'off';
           element.requestUpdate();
           await element.updateComplete;

           expect(cameraBtn.disabled).toBeTrue();
           expect(recordBtn.disabled).toBeTrue();
           expect(recordBtn.title).toContain('Audio is turned off');
           expect(addMediaBtn.disabled).toBeTrue();
           expect(addMediaBtn.title)
               .toContain('Vision and Audio are turned off');
           expect(heroSubtitle()).toContain('in text-only mode');
           expect(heroSubtitle()).not.toContain('Attach');
           expect(firstCapTag()).toBe('Text-Only Mode');
         });
    });
  });
});
