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

import {LlmChatStateController} from '../state_controller.js';
import {ChatSessionStore, PromptMedia, PromptPart} from '../stores/chat_session_store.js';
import {ModelLoaderService} from '../stores/model_loader_service.js';
import {SettingsStore} from '../stores/settings_store.js';
import {LitertChatWindow} from './chat_window.js';

describe('litert-chat-window', () => {
  let element: LitertChatWindow;
  let mockState: jasmine.SpyObj<LlmChatStateController>;
  let mockChatSession: jasmine.SpyObj<ChatSessionStore>;
  let mockSettings: jasmine.SpyObj<SettingsStore>;
  let mockModelLoader: jasmine.SpyObj<ModelLoaderService>;
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
    mockSettings.selectedModelPath = 'path/to/model_file.litertlm';

    cachedModelsMap = new Map<string, number>();
    mockModelLoader = jasmine.createSpyObj('ModelLoaderService', [
      'loadModelWeights',
    ]);
    mockModelLoader.isModelLoading = false;
    mockModelLoader.cachedModels = cachedModelsMap;

    mockState = {
      chatSession: mockChatSession,
      settings: mockSettings,
      modelLoader: mockModelLoader,
      statusText: 'Idle',
      addHost: jasmine.createSpy('addHost'),
      removeHost: jasmine.createSpy('removeHost'),
      requestUpdate: jasmine.createSpy('requestUpdate'),
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
    expect(starters[0]!.textContent!.trim()).toBe('Explain WebGPU');
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
    element.requestUpdate();
    await element.updateComplete;

    const editor = getEditor();
    expect(editor.getAttribute('contenteditable')).toBe('false');
    expect(editor.dataset['placeholder'])
        .toBe('Model is preparing, please wait...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeTrue();
  });

  it('enables input when model is not loading', () => {
    const editor = getEditor();
    expect(editor.getAttribute('contenteditable')).toBe('true');
    expect(editor.dataset['placeholder']).toContain('Message LiteRT-LM...');

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBeFalse();
  });

  it('shows "Send" when model is cached', async () => {
    cachedModelsMap.set('model_file.litertlm', 1000);
    element.requestUpdate();
    await element.updateComplete;

    const sendBtn = element.shadowRoot!.querySelector('#btn-send') as HTMLButtonElement;
    expect(sendBtn.textContent!.trim()).toBe('Send');
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

    stopBtn.click();
    expect(mockChatSession.cancelGeneration).toHaveBeenCalled();
  });

  it('sends message on send button click', async () => {
    const editor = getEditor();
    editor.textContent = 'Test prompt';
    
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
});
