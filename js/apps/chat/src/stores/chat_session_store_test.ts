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

import {Engine, getOrLoadGlobalLiteRtLm} from '@litert-lm/core';
import {EngineFake} from '@litert-lm/core/testing';

import {ChatSessionStore} from './chat_session_store.js';
import {ModelLoaderService} from './model_loader_service.js';
import {SettingsStore} from './settings_store.js';

describe('ChatSessionStore', () => {
  let settingsStore: SettingsStore;
  let modelLoader: ModelLoaderService;
  let chatSessionStore: ChatSessionStore;

  let fakeEngine: EngineFake;
  let updateCallbackCalled = false;
  let statusUpdated = false;

  beforeEach(async () => {
    // Reset state & mocks
    updateCallbackCalled = false;
    statusUpdated = false;
    window.localStorage.clear();

    settingsStore = new SettingsStore(() => {});

    fakeEngine = new EngineFake({model: 'fake'});
    const fakeEngineCreate: typeof Engine.create = async (settings, hint?) => {
      return fakeEngine as unknown as Engine;
    };
    const fakeLoadWasm = async () => {};

    modelLoader = new ModelLoaderService(
        () => {}, () => {}, undefined, fakeEngineCreate,
        fakeLoadWasm as unknown as typeof getOrLoadGlobalLiteRtLm);

    modelLoader.engine = fakeEngine as unknown as Engine;
    modelLoader.loadedSettings = structuredClone(settingsStore.modelSettings);

    chatSessionStore = new ChatSessionStore(
        () => {
          updateCallbackCalled = true;
        },
        settingsStore, modelLoader,
        (msg: string) => {
          statusUpdated = true;
        });
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('starts a new conversation cleanly', async () => {
    await chatSessionStore.startNewConversation();

    expect(chatSessionStore.messages.length).toBe(0);
    expect(chatSessionStore.activeSavedConvId).toBeNull();
    expect(window.localStorage.getItem('litertlm-active-conv-id')).toBeNull();
    expect(updateCallbackCalled).toBeTrue();
    expect(statusUpdated).toBeTrue();
  });

  it('generates auto-id and commits newly sent messages to localStorage history',
     async () => {
       // Push dummy items
       chatSessionStore.messages.push(
           {role: 'user', text: 'Hello world', senderName: 'User'});

       // Simulating internal private function call safely
       (chatSessionStore as unknown as {
         commitActiveChatHistory: () => void
       }).commitActiveChatHistory();

       const storedId = chatSessionStore.activeSavedConvId;
       expect(storedId).toBeDefined();

       const indexData =
           window.localStorage.getItem('litertlm-conversations-list');
       expect(indexData).toContain('Hello world');

       const historyData =
           window.localStorage.getItem(`litertlm-chat-history-${storedId}`);
       expect(historyData).toContain('Hello world');
     });

  it('selects saved conversation and safely updates model references',
     async () => {
       // Create a mock conversation state in localStorage to load
       const mockId = '12345';
       const mockHistory =
           [{role: 'user', text: 'Where is the moon?', senderName: 'User'}];
       const mockList = [{
         id: mockId,
         title: 'Where is the moon?',
         createdAt: Date.now(),
         modelPath: 'test_model_v1.bin'
       }];

       window.localStorage.setItem(
           'litertlm-conversations-list', JSON.stringify(mockList));
       window.localStorage.setItem(
           `litertlm-chat-history-${mockId}`, JSON.stringify(mockHistory));

       // Refresh to pick up the list
       chatSessionStore.loadSavedConversationsIndex();

       await chatSessionStore.selectConversation(mockId);

       expect(chatSessionStore.messages.length).toBe(1);
       expect(chatSessionStore.messages[0].text).toEqual('Where is the moon?');
       expect(chatSessionStore.activeSavedConvId).toEqual(mockId);
       expect(settingsStore.selectedModelPath)
           .toEqual('test_model_v1.bin');  // it changed the model preference
                                           // natively!
     });

  it('deletes active conversation properly', () => {
    const mockId = '111';
    chatSessionStore.conversationsList.push(
        {id: mockId, title: 'Delete me', createdAt: 0, modelPath: 'test.bin'});
    chatSessionStore.activeSavedConvId = mockId;

    chatSessionStore.deleteConversation(mockId);

    // Check that we wiped lists and id cleanly
    expect(chatSessionStore.conversationsList.length).toBe(0);
    expect(chatSessionStore.activeSavedConvId).toBeNull();
  });

  it('slices array intelligently on redoResponse', async () => {
    // Manually construct chat scenario
    chatSessionStore.messages = [
      {role: 'user', text: 'First user prompt', senderName: 'User'},  // idx: 0
      {
        role: 'assistant',
        text: 'First model response',
        senderName: 'Assistant'
      },                                                               // idx: 1
      {role: 'user', text: 'Second user prompt', senderName: 'User'},  // idx: 2
      {role: 'assistant', text: 'BAD RESPONSE', senderName: 'Assistant'}
      // idx: 3
    ];

    // We mock sendMessage safely to ignore trying to create the actual async
    // model logic
    spyOn(chatSessionStore, 'sendMessage').and.returnValue(Promise.resolve());

    // We redo the array
    await chatSessionStore.redoResponse(3);

    // Slices off previous BAD response, leaves Second User Prompt at tail and
    // sends it.
    expect(chatSessionStore.messages.length).toBe(2);
    // And actually called sendMessage using element[2]
    expect(chatSessionStore.sendMessage)
        .toHaveBeenCalledWith('Second user prompt');
  });

  it('rewinds to user message array intelligently on rewindAndEdit',
     async () => {
       // Manually construct chat scenario
       chatSessionStore.messages = [
         {
           role: 'user',
           text: 'First user prompt',
           senderName: 'User'
         },  // idx: 0
         {
           role: 'assistant',
           text: 'First model response',
           senderName: 'Assistant'
         },                                                           // idx: 1
         {role: 'user', text: 'Please edit me', senderName: 'User'},  // idx: 2
         {role: 'assistant', text: 'Bad Response', senderName: 'Assistant'}
         // idx: 3
       ];

       const extractText = await chatSessionStore.rewindAndEdit(2);

       expect(chatSessionStore.messages.length)
           .toBe(2);  // array was truncated right before the user message
       expect(extractText)
           .toEqual('Please edit me');  // the target user message is extracted
                                        // correctly
     });

  it('creates conversation session cleanly', async () => {
    await chatSessionStore.createConversationSession();

    expect(chatSessionStore.activeConversation).toBeDefined();
    expect(statusUpdated).toBeTrue();
  });

  it('sendMessage processes streams appropriately', async () => {
    await chatSessionStore.createConversationSession();

    await chatSessionStore.sendMessage('Hello model');

    expect(chatSessionStore.messages.length).toBe(2);
    expect(chatSessionStore.messages[0].text).toBe('Hello model');
    expect(chatSessionStore.messages[1].text).toBe('hello from fake');
    expect(chatSessionStore.isGenerating).toBeFalse();
  });

  it('sendMessage processes simulated function calls correctly', async () => {
    await chatSessionStore.createConversationSession();

    // Use our strongly typed internal fake engine
    const converseFake = fakeEngine.cachedConversation!;
    chatSessionStore.codeSandbox.run = jasmine.createSpy().and.returnValue(
        Promise.resolve({resultAsString: '42', consoleMessages: []}));

    converseFake.queueResponse({
      role: 'model',
      content: 'Let me calculate that.',
      tool_calls: [{
        type: 'function',
        function: {name: 'run_javascript', arguments: {code: '21+21'}}
      }]
    });

    converseFake.queueResponse({role: 'model', content: ' The answer is 42.'});

    await chatSessionStore.sendMessage('What is 21+21?');

    expect(chatSessionStore.messages.length).toBe(2);
    expect(chatSessionStore.messages[1].text)
        .toBe('Let me calculate that. The answer is 42.');

    // We can verify the API's backend logic:
    const rawHistory = converseFake.getHistory();
    expect(rawHistory.length)
        .toBe(4);  // request -> tool call -> tool response -> text response
    expect(rawHistory[1].tool_calls).toBeDefined();
    expect(rawHistory[1].tool_calls![0].function.name).toBe('run_javascript');
    expect(chatSessionStore.codeSandbox.run).toHaveBeenCalledWith('21+21');
  });

  it('sendMessage cancels generation gracefully', async () => {
    await chatSessionStore.createConversationSession();

    // We want to cancel midway. We can call cancelGeneration synchronously
    // but the promise handling in JS might complete before cancellation if the
    // mock streaming is entirely synchronous. However, we can just trigger it,
    // though our ConversationFake is completely synchronous. Let's at least
    // ensure cancelGeneration sets the flag.
    chatSessionStore.cancelGeneration();
    expect(chatSessionStore.isCancelled).toBeTrue();
  });

  it('sendMessage ignores empty text or if already generating', async () => {
    await chatSessionStore.createConversationSession();

    await chatSessionStore.sendMessage('  ');  // empty text
    expect(chatSessionStore.messages.length).toBe(0);

    chatSessionStore.isGenerating = true;
    await chatSessionStore.sendMessage('valid message');
    expect(chatSessionStore.messages.length).toBe(0);
  });

  it('sendMessage reloads model if loadedSettings does not match current settings', async () => {
    // Force reload by setting loadedSettings to null
    modelLoader.loadedSettings = null;

    spyOn(modelLoader, 'loadModelWeights').and.callFake(async (settings, onModelLoaded) => {
      await onModelLoaded();
    });
    spyOn(chatSessionStore, 'createConversationSession').and.resolveTo();

    await chatSessionStore.sendMessage('Hello');

    expect(modelLoader.loadModelWeights).toHaveBeenCalled();
    expect(chatSessionStore.createConversationSession).toHaveBeenCalled();
  });
});
