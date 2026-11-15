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

import {Conversation} from './conversation.js';
import {Engine} from './engine.js';
import {Mutex} from './mutex.js';
import {Backend, Conversation as WasmConversation, Engine as WasmEngine} from './wasm_binding_types.js';

describe('Conversation', () => {
  let mockWasmConversation: jasmine.SpyObj<WasmConversation>;
  let mockWasmEngine: jasmine.SpyObj<WasmEngine>;
  let mockMutexes: {executor: Mutex};

  beforeEach(() => {
    mockWasmConversation = jasmine.createSpyObj('WasmConversation', [
      'sendMessage', 'sendMessageAsync', 'getHistory', 'getTokenCount',
      'getBenchmarkInfo', 'cancelProcess', 'delete', 'clone'
    ]);

    mockWasmEngine = jasmine.createSpyObj('WasmEngine', ['waitUntilDone']);
    mockWasmEngine.waitUntilDone.and.returnValue(Promise.resolve());

    mockMutexes = {executor: new Mutex()};
  });
  it('forwards regular messages properly', async () => {
    const conversation =
        new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);

    mockWasmConversation.sendMessage.and.returnValue(
        Promise.resolve(JSON.stringify({role: 'assistant', content: 'hello'})));

    const response = await conversation.sendMessage('hi');

    expect(response.content).toBe('hello');
    expect(mockWasmConversation.sendMessage)
        .toHaveBeenCalledWith(
            JSON.stringify({role: 'user', content: 'hi'}), [], 0);
  });

  it('forwards the visual token budget', async () => {
    const conversation = new Conversation(
        mockWasmConversation, mockWasmEngine, mockMutexes,
        /* visualTokenBudget= */ 140);

    mockWasmConversation.sendMessage.and.returnValue(
        Promise.resolve(JSON.stringify({role: 'assistant', content: 'hello'})));

    await conversation.sendMessage('hi');

    expect(mockWasmConversation.sendMessage)
        .toHaveBeenCalledWith(
            JSON.stringify({role: 'user', content: 'hi'}), [], 140);
  });

  it('extracts binary image data into blobs array and forwards visualTokenBudget in streaming',
     async () => {
       const conversation = new Conversation(
           mockWasmConversation, mockWasmEngine, mockMutexes,
           /* visualTokenBudget= */ 280);

       mockWasmConversation.sendMessageAsync.and.callFake(
           (messageJson, blobs, callback, visualTokenBudget) => {
             callback(
                 JSON.stringify({role: 'assistant', content: 'apple'}), false,
                 null);
             callback(null, true, null);
             return Promise.resolve();
           });

       const imageBytes = new Uint8Array([104, 105]);
       const imageBlob = new Blob([new Uint8Array([1, 2, 3])]);
       const stream = conversation.sendMessageStreaming({
         role: 'user',
         content: [
           {type: 'image', data: imageBytes},
           {type: 'image', data: imageBlob},
           {type: 'image', data: 'aGk='},
           {type: 'text', text: 'What is this?'},
         ],
       });

       const reader = stream.getReader();
       const chunk = await reader.read();
       expect(chunk.done).toBeFalse();
       expect(chunk.value?.content).toBe('apple');
       const done = await reader.read();
       expect(done.done).toBeTrue();

       expect(mockWasmConversation.sendMessageAsync)
           .toHaveBeenCalledWith(
               JSON.stringify({
                 role: 'user',
                 content: [
                   {type: 'image', blob_index: 0},
                   {type: 'image', blob_index: 1},
                   {type: 'image', blob: 'aGk='},
                   {type: 'text', text: 'What is this?'},
                 ],
               }),
               [imageBytes, new Uint8Array([1, 2, 3])],
               jasmine.any(Function), 280);
     });

  it('handles stream error and does not throw TypeError on close', async () => {
    const conversation =
        new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);

    mockWasmConversation.sendMessageAsync.and.callFake(
        (messageJson, blobs, callback) => {
          callback(null, false, 'Some WASM error');
          return Promise.resolve();
        });

    const stream = conversation.sendMessageStreaming('hi');
    const reader = stream.getReader();

    await expectAsync(reader.read()).toBeRejectedWithError('Some WASM error');

    // Wait a bit to allow async executeGeneration to finish and potentially throw.
    await new Promise(resolve => setTimeout(resolve, 50));
  });

  it('clones conversation properly and allows independent message sending',
     async () => {
       const conversation = new Conversation(
           mockWasmConversation, mockWasmEngine, mockMutexes,
           /* visualTokenBudget= */ 70);

       const mockClonedWasmConversation = jasmine.createSpyObj(
           'WasmConversation', ['sendMessage', 'delete']);
       mockWasmConversation.clone.and.returnValue(mockClonedWasmConversation);
       mockClonedWasmConversation.sendMessage.and.returnValue(Promise.resolve(
           JSON.stringify({role: 'assistant', content: 'cloned response'})));

       const clonedConversation = await conversation.clone();
       expect(clonedConversation).toBeDefined();
       expect(mockWasmConversation.clone).toHaveBeenCalled();

       const response = await clonedConversation.sendMessage('test');
       expect(response.content).toBe('cloned response');
       expect(mockClonedWasmConversation.sendMessage)
           .toHaveBeenCalledWith(
               JSON.stringify({role: 'user', content: 'test'}), [], 70);
     });

  it('throws an error when cloning a busy conversation', async () => {
    const conversation =
        new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);
    let finishSend: () => void = () => {};
    mockWasmConversation.sendMessage.and.returnValue(
        new Promise<string>(resolve => {
          finishSend = () =>
              resolve(JSON.stringify({role: 'assistant', content: 'done'}));
        }));

    const p = conversation.sendMessage('hi');
    await expectAsync(conversation.clone())
        .toBeRejectedWithError(/Conversation is busy/);
    finishSend();
    await p;
  });

  describe('Engine.createConversation filterChannelContentFromKvCache', () => {
    let createCustomSpy: jasmine.Spy;
    let mockSessionConfig: {delete: jasmine.Spy};
    let mockConversationConfig: {delete: jasmine.Spy};

    function createTestEngine(
        backend: Backend, useAutosizedRingbuffers = false): Engine {
      mockSessionConfig = {delete: jasmine.createSpy('sessionConfig.delete')};
      mockConversationConfig = {
        delete: jasmine.createSpy('conversationConfig.delete')
      };
      createCustomSpy = jasmine.createSpy('createCustom')
                            .and.returnValue(mockConversationConfig);
      const mockWasm = {
        SessionConfig: {
          createDefault: () => mockSessionConfig,
        },
        ConversationConfig: {
          createCustom: createCustomSpy,
          createDefault:
              jasmine.createSpy('createDefault').and.returnValue(
                  mockConversationConfig),
        },
        Conversation: {
          create: jasmine.createSpy('Conversation.create')
                      .and.resolveTo(mockWasmConversation),
        },
      };
      return Object.assign(Object.create(Engine.prototype), {
        wasm: mockWasm,
        engine: mockWasmEngine,
        settings: {
          backend,
          mainExecutorSettings: {
            backendConfig: {
              use_autosized_ringbuffers: useAutosizedRingbuffers,
            },
          },
        },
        mutexes: mockMutexes,
      }) as Engine;
    }

    it('defaults filterChannelContentFromKvCache to true for GPU_ARTISAN with use_autosized_ringbuffers',
       async () => {
         const engine = createTestEngine(Backend.GPU_ARTISAN, true);
         await engine.createConversation({});
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, false, false, true, '');
       });

    it('defaults filterChannelContentFromKvCache to false for GPU_ARTISAN without use_autosized_ringbuffers',
       async () => {
         const engine = createTestEngine(Backend.GPU_ARTISAN, false);
         await engine.createConversation({});
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, false, false, false, '');
       });

    it('defaults filterChannelContentFromKvCache to false for non-GPU_ARTISAN',
       async () => {
         const engine = createTestEngine(Backend.CPU, true);
         await engine.createConversation({});
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, false, false, false, '');
       });

    it('respects explicit filterChannelContentFromKvCache override',
       async () => {
         const gpuArtisanEngine = createTestEngine(Backend.GPU_ARTISAN, true);
         await gpuArtisanEngine.createConversation(
             {filterChannelContentFromKvCache: false});
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, false, false, false, '');

         const cpuEngine = createTestEngine(Backend.CPU, false);
         await cpuEngine.createConversation(
             {filterChannelContentFromKvCache: true});
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, false, false, true, '');
       });
  });
});

