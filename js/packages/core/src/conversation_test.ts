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
import {Mutex} from './mutex.js';
import {Conversation as WasmConversation, Engine as WasmEngine} from './wasm_binding_types.js';

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
        Promise.resolve(JSON.stringify({role: 'model', content: 'hello'})));

    const response = await conversation.sendMessage('hi');

    expect(response.content).toBe('hello');
    expect(mockWasmConversation.sendMessage)
        .toHaveBeenCalledWith(JSON.stringify({role: 'user', content: 'hi'}));
  });

  it('handles stream error and does not throw TypeError on close', async () => {
    const conversation =
        new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);

    mockWasmConversation.sendMessageAsync.and.callFake(
        (messageJson, callback) => {
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
       const conversation =
           new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);

       const mockClonedWasmConversation = jasmine.createSpyObj(
           'WasmConversation', ['sendMessage', 'delete']);
       mockWasmConversation.clone.and.returnValue(mockClonedWasmConversation);
       mockClonedWasmConversation.sendMessage.and.returnValue(Promise.resolve(
           JSON.stringify({role: 'model', content: 'cloned response'})));

       const clonedConversation = await conversation.clone();
       expect(clonedConversation).toBeDefined();
       expect(mockWasmConversation.clone).toHaveBeenCalled();

       const response = await clonedConversation.sendMessage('test');
       expect(response.content).toBe('cloned response');
       expect(mockClonedWasmConversation.sendMessage)
           .toHaveBeenCalledWith(
               JSON.stringify({role: 'user', content: 'test'}));
     });

  it('throws an error when cloning a busy conversation', async () => {
    const conversation =
        new Conversation(mockWasmConversation, mockWasmEngine, mockMutexes);
    let finishSend: () => void = () => {};
    mockWasmConversation.sendMessage.and.returnValue(
        new Promise<string>(resolve => {
          finishSend = () =>
              resolve(JSON.stringify({role: 'model', content: 'done'}));
        }));

    const p = conversation.sendMessage('hi');
    await expectAsync(conversation.clone())
        .toBeRejectedWithError(/Conversation is busy/);
    finishSend();
    await p;
  });
});
