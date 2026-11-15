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

import {Engine, getOrLoadGlobalLiteRtLm, Message} from '@litert-lm/core';
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
  let lastStatus = '';

  beforeEach(async () => {
    // Reset state & mocks
    updateCallbackCalled = false;
    statusUpdated = false;
    lastStatus = '';
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
          lastStatus = msg;
        });
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('starts a new conversation cleanly', async () => {
    window.localStorage.setItem('litertlm-active-conv-id', 'previous-conv');

    await chatSessionStore.startNewConversation();

    expect(chatSessionStore.messages.length).toBe(0);
    expect(chatSessionStore.activeSavedConvId).toBeNull();
    expect(window.localStorage.getItem('litertlm-active-conv-id')).toBeNull();
    expect(updateCallbackCalled).toBeTrue();
    expect(lastStatus).toBe('Model loaded and ready.');
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

       const extractParts = await chatSessionStore.rewindAndEdit(2);

       expect(chatSessionStore.messages.length)
           .toBe(2);  // array was truncated right before the user message
       expect(extractParts).toEqual([
         {type: 'text', text: 'Please edit me'}
       ]);  // the target user message is extracted correctly
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
    expect(chatSessionStore.messages[0].parts).toBeUndefined();
    expect(chatSessionStore.messages[1].text).toBe('hello from fake');
    expect(chatSessionStore.isGenerating).toBeFalse();
  });

  it('sendMessage sends media in the order they were placed', async () => {
    await chatSessionStore.createConversationSession();
    const imageBlob =
        new Blob([new Uint8Array([1, 2, 3])], {type: 'image/png'});
    const audioBlob =
        new Blob([new Uint8Array([4, 5, 6])], {type: 'audio/wav'});
    const image = chatSessionStore.createMedia('image', 'cat.png', imageBlob);
    const audio = chatSessionStore.createMedia('audio', 'meow.wav', audioBlob);
    const sendSpy =
        spyOn(chatSessionStore.activeConversation!, 'sendMessageStreaming')
            .and.callThrough();

    await chatSessionStore.sendMessage([
      {type: 'text', text: '  Is '},
      {type: 'media', media: image},
      {type: 'text', text: ' making this sound: '},
      {type: 'media', media: audio},
      {type: 'text', text: '?  '},
    ]);

    const expectedMessage: Message = {
      role: 'user',
      content: [
        {type: 'text', text: 'Is '},
        {type: 'image', data: imageBlob},
        {type: 'text', text: ' making this sound: '},
        {type: 'audio', data: audioBlob},
        {type: 'text', text: '?'},
      ],
    };
    expect(sendSpy.calls.mostRecent().args[0]).toEqual(expectedMessage);
    const userMsg = chatSessionStore.messages[0];
    expect(userMsg.text)
        .toBe('Is [image: cat.png] making this sound: [audio: meow.wav]?');
    expect(userMsg.parts).toEqual([
      {type: 'text', text: 'Is '},
      {type: 'image', id: image.id, name: 'cat.png'},
      {type: 'text', text: ' making this sound: '},
      {type: 'audio', id: audio.id, name: 'meow.wav'},
      {type: 'text', text: '?'},
    ]);

    // Only media references are persisted, not media bytes.
    const history = window.localStorage.getItem(
        `litertlm-chat-history-${chatSessionStore.activeSavedConvId}`);
    expect(history).toContain(image.id);
    expect(history).toContain(audio.id);
  });

  it('restores media from memory and placeholders otherwise', () => {
    const image = chatSessionStore.createMedia(
        'image', 'cat.png', new Blob([], {type: 'image/png'}));
    const parts = chatSessionStore.getMessageParts({
      role: 'user',
      text: '',
      senderName: 'User',
      parts: [
        {type: 'image', id: image.id, name: 'cat.png'},
        {type: 'text', text: ' and '},
        {type: 'image', id: 'gone', name: 'dog.png'},
        {type: 'audio', id: 'gone too', name: 'bark.wav'},
      ],
    });

    expect(parts).toEqual([
      {type: 'media', media: image},
      {type: 'text', text: ' and '},
      {type: 'text', text: '[image: dog.png]'},
      {type: 'text', text: '[audio: bark.wav]'},
    ]);
  });

  it('redoResponse resends media', async () => {
    const audio = chatSessionStore.createMedia(
        'audio', 'meow.wav', new Blob([], {type: 'audio/wav'}));
    chatSessionStore.messages = [
      {
        role: 'user',
        text: '[audio: meow.wav]',
        parts: [{type: 'audio', id: audio.id, name: 'meow.wav'}],
        senderName: 'User'
      },
      {role: 'assistant', text: 'A cat', senderName: 'Assistant'},
    ];
    spyOn(chatSessionStore, 'sendMessage').and.returnValue(Promise.resolve());

    await chatSessionStore.redoResponse(1);

    expect(chatSessionStore.sendMessage).toHaveBeenCalledWith([
      {type: 'media', media: audio}
    ]);
  });

  it('sendMessage processes simulated function calls correctly', async () => {
    await chatSessionStore.createConversationSession();

    // Use our strongly typed internal fake engine
    const converseFake = fakeEngine.cachedConversation!;
    chatSessionStore.codeSandbox.run = jasmine.createSpy().and.returnValue(
        Promise.resolve({resultAsString: '42', consoleMessages: []}));

    converseFake.queueResponse({
      role: 'assistant',
      content: 'Let me calculate that.',
      tool_calls: [{
        type: 'function',
        function: {name: 'run_javascript', arguments: {code: '21+21'}}
      }]
    });

    converseFake.queueResponse({role: 'assistant', content: ' The answer is 42.'});

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

  it('sendMessage cancels generation gracefully mid-thought and mid-answer',
     async () => {
       await chatSessionStore.createConversationSession();

       // 1. Mid-thought cancellation: no answer text emitted yet.
       let thoughtPulls = 0;
       spyOn(chatSessionStore.activeConversation!, 'sendMessageStreaming')
           .and.callFake(() => new ReadableStream<Message>({
             async pull(controller) {
               thoughtPulls++;
               if (thoughtPulls === 1) {
                 controller.enqueue({
                   role: 'assistant',
                   channels: {thought: 'Analyzing WebGPU shader...' as string},
                 } as unknown as Message);
                 return;
               }
               chatSessionStore.cancelGeneration();
               await Promise.resolve();
               controller.close();
             },
           }));

       await chatSessionStore.sendMessage('Stop during thinking');
       expect(chatSessionStore.isGenerating).toBeFalse();
       expect(lastStatus).toBe('Generation stopped.');
       expect(chatSessionStore.messages[1].thoughtText)
           .toContain('Analyzing WebGPU shader...');
       expect(chatSessionStore.messages[1].text).toBe('');
       expect(chatSessionStore.messages[1].stopped).toBeTrue();

       // 2. Mid-answer cancellation (activeConversation was rebuilt in finally).
       let answerPulls = 0;
       spyOn(chatSessionStore.activeConversation!, 'sendMessageStreaming')
           .and.callFake(() => new ReadableStream<Message>({
             async pull(controller) {
               answerPulls++;
               if (answerPulls === 1) {
                 controller.enqueue({
                   role: 'assistant',
                   content: 'Partial answer',
                 });
                 return;
               }
               chatSessionStore.cancelGeneration();
               await Promise.resolve();
               controller.close();
             },
           }));

       await chatSessionStore.sendMessage('Stop during answer');
       expect(chatSessionStore.isGenerating).toBeFalse();
       expect(lastStatus).toBe('Generation stopped.');
       expect(chatSessionStore.messages[3].text).toBe('Partial answer');
       expect(chatSessionStore.messages[3].stopped).toBeTrue();
       expect(chatSessionStore.messages[3].tokensCount).toBeTruthy();
     });

  it('removes blank assistant turn when runtime error occurs before first token',
     async () => {
       await chatSessionStore.createConversationSession();
       spyOn(chatSessionStore.activeConversation!, 'sendMessageStreaming')
           .and.callFake(() => new ReadableStream<Message>({
             pull(controller) {
               controller.error(new Error('WebGPU device lost'));
             },
           }));

       await chatSessionStore.sendMessage('Trigger error');

       expect(chatSessionStore.isGenerating).toBeFalse();
       expect(lastStatus).toContain('Generation failed: WebGPU device lost');
       expect(chatSessionStore.messages.length).toBe(1);
       expect(chatSessionStore.messages[0].role).toBe('user');
       expect(chatSessionStore.messages[0].text).toBe('Trigger error');
     });

  it('shows tool results and errors in the tool trace', async () => {
    await chatSessionStore.createConversationSession();
    const converseFake = fakeEngine.cachedConversation!;
    chatSessionStore.codeSandbox.run = jasmine.createSpy().and.returnValues(
        Promise.resolve({
          resultAsString: '42',
          consoleMessages: [{type: 'log', text: 'computing'}],
        }),
        Promise.resolve({error: 'boom', consoleMessages: []}));
    const toolCall = {
      role: 'assistant' as const,
      content: '',
      tool_calls: [{
        type: 'function' as const,
        function: {name: 'run_javascript', arguments: {code: '21+21'}}
      }]
    };
    converseFake.queueResponse(toolCall);
    converseFake.queueResponse({role: 'assistant', content: 'Done.'});
    converseFake.queueResponse(toolCall);
    converseFake.queueResponse({role: 'assistant', content: 'Failed.'});

    await chatSessionStore.sendMessage('Run it');
    await chatSessionStore.sendMessage('Run it again');

    expect(chatSessionStore.messages[1].toolCalls?.[0]?.outputText)
        .toBe('computing\n→ 42');
    expect(chatSessionStore.messages[3].toolCalls?.[0]?.outputText)
        .toBe('Error: boom');
  });

  it('keeps tool calls when an error interrupts generation', async () => {
    await chatSessionStore.createConversationSession();
    const converseFake = fakeEngine.cachedConversation!;
    chatSessionStore.codeSandbox.run = jasmine.createSpy().and.returnValue(
        Promise.resolve({resultAsString: '42', consoleMessages: []}));
    converseFake.queueResponse({
      role: 'assistant',
      content: '',
      tool_calls: [{
        type: 'function',
        function: {name: 'run_javascript', arguments: {code: '21+21'}}
      }]
    });
    const sendMessageStreaming =
        converseFake.sendMessageStreaming.bind(converseFake);
    let calls = 0;
    spyOn(converseFake, 'sendMessageStreaming')
        .and.callFake(message => calls++ === 0 ?
                          sendMessageStreaming(message) :
                          new ReadableStream<Message>({
                            pull(controller) {
                              controller.error(new Error('WebGPU device lost'));
                            },
                          }));

    await chatSessionStore.sendMessage('What is 21+21?');

    expect(lastStatus).toContain('Generation failed: WebGPU device lost');
    expect(chatSessionStore.messages.length).toBe(2);
    expect(chatSessionStore.messages[1].toolCalls?.map(call => call.name))
        .toEqual(['run_javascript']);
    expect(chatSessionStore.messages[1].text)
        .toBe('*[Generation interrupted by an error]*');
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

  describe('system prompt and tools', () => {
    async function sendAndGetConfig() {
      const createConversation =
          spyOn(fakeEngine, 'createConversation').and.callThrough();
      await chatSessionStore.sendMessage('Hello');
      return createConversation.calls.mostRecent().args[0]!;
    }

    it('does not add a system message if the system prompt is empty',
       async () => {
         const config = await sendAndGetConfig();

         expect(config.preface?.messages).toBeUndefined();
       });

    it('starts the conversation with the system prompt', async () => {
      settingsStore.systemPrompt = '  You are a pirate.  ';
      chatSessionStore.messages = [
        {role: 'user', text: 'Hi', senderName: 'User'},
        {role: 'assistant', text: 'Ahoy', senderName: 'Assistant'},
      ];

      const config = await sendAndGetConfig();

      const expectedMessages: Message[] = [
        {role: 'system', content: 'You are a pirate.'},
        {role: 'user', content: 'Hi'},
        {role: 'assistant', content: 'Ahoy'},
      ];
      expect(config.preface?.messages as unknown).toEqual(expectedMessages);
    });

    it('gives the model the enabled tools', async () => {
      const config = await sendAndGetConfig();

      expect(config.preface?.tools?.map(
                 tool => (tool as {function: {name: string}}).function.name))
          .toEqual(['run_javascript']);
    });

    it('does not give the model disabled tools', async () => {
      settingsStore.disabledTools = ['run_javascript'];

      const config = await sendAndGetConfig();

      expect(config.preface?.tools).toBeUndefined();
    });

    it('creates a new session when the system prompt or tools change',
       async () => {
         await chatSessionStore.createConversationSession();
         const session = chatSessionStore.activeConversation;

         settingsStore.systemPrompt = 'Be concise.';
         await chatSessionStore.sendMessage('Hello');
         const sessionWithPrompt = chatSessionStore.activeConversation;
         expect(sessionWithPrompt).not.toBe(session);

         settingsStore.disabledTools = ['run_javascript'];
         await chatSessionStore.sendMessage('Hello again');
         expect(chatSessionStore.activeConversation).not.toBe(sessionWithPrompt);
       });

    it('uses Top-K 1 for greedy sampling', async () => {
      settingsStore.samplerType = 'greedy';
      settingsStore.topK = 8;

      const config = await sendAndGetConfig();

      expect(config.sessionConfig?.samplerParams?.k).toBe(1);
    });

    it('keeps the session if Top-K changes under greedy sampling', async () => {
      settingsStore.samplerType = 'greedy';
      await chatSessionStore.createConversationSession();
      const session = chatSessionStore.activeConversation;

      settingsStore.topK = 8;
      await chatSessionStore.sendMessage('Hello');

      expect(chatSessionStore.activeConversation).toBe(session);
    });

    it('keeps the session if the disabled tools are only reordered',
       async () => {
         settingsStore.disabledTools = ['tool_a', 'tool_b'];
         await chatSessionStore.createConversationSession();
         const session = chatSessionStore.activeConversation;

         settingsStore.disabledTools = ['tool_b', 'tool_a'];
         await chatSessionStore.sendMessage('Hello');

         expect(chatSessionStore.activeConversation).toBe(session);
       });
  });
});
