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

// Placeholder for internal dependency on trusted resource url

import {Conversation} from './conversation.js';
import {Engine} from './engine.js';
import {loadLiteRtLm, unloadLiteRtLm} from './load_litertlm.js';
import {Mutex} from './mutex.js';
import {waitForWebGpuAdapter} from './testing/webgpu.js';
import {Backend, Conversation as WasmConversation, Engine as WasmEngine, unwrapWasmResult} from './wasm_binding_types.js';

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

  describe('Engine.createConversation ConversationConfig defaults', () => {
    let createCustomSpy: jasmine.Spy;
    let createDefaultSpy: jasmine.Spy;
    let mockSessionConfig: {
      delete: jasmine.Spy; setVisionModalityEnabled: jasmine.Spy;
      setAudioModalityEnabled: jasmine.Spy;
      setMaxOutputTokens: jasmine.Spy;
    };
    let mockConversationConfig: {delete: jasmine.Spy};

    function createTestEngine(backend: Backend, options: {
      useAutosizedRingbuffers?: boolean;
      visionBackend?: Backend;
      audioBackend?: Backend;
    } = {}): Engine {
      mockSessionConfig = {
        delete: jasmine.createSpy('sessionConfig.delete'),
        setVisionModalityEnabled:
            jasmine.createSpy('sessionConfig.setVisionModalityEnabled'),
        setAudioModalityEnabled:
            jasmine.createSpy('sessionConfig.setAudioModalityEnabled'),
        setMaxOutputTokens:
            jasmine.createSpy('sessionConfig.setMaxOutputTokens'),
      };
      mockConversationConfig = {
        delete: jasmine.createSpy('conversationConfig.delete')
      };
      createCustomSpy = jasmine.createSpy('createCustom')
                            .and.returnValue(mockConversationConfig);
      createDefaultSpy = jasmine.createSpy('createDefault')
                             .and.returnValue(mockConversationConfig);
      const mockWasm = {
        SessionConfig: {
          createDefault: () => mockSessionConfig,
        },
        ConversationConfig: {
          createCustom: createCustomSpy,
          createDefault: createDefaultSpy,
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
          visionBackend: options.visionBackend,
          audioBackend: options.audioBackend,
          mainExecutorSettings: {
            backendConfig: {
              use_autosized_ringbuffers:
                  options.useAutosizedRingbuffers ?? false,
            },
          },
        },
        mutexes: mockMutexes,
      }) as Engine;
    }

    it('uses ConversationConfig.createDefault when no config is passed on text-only engines',
       async () => {
         const engine = createTestEngine(Backend.GPU_ARTISAN);
         await engine.createConversation();
         expect(createDefaultSpy).toHaveBeenCalledWith(mockWasmEngine);
         expect(createCustomSpy).not.toHaveBeenCalled();
       });

    it('leaves unset booleans undefined when no config is passed on multimodal engines',
       async () => {
         const engine = createTestEngine(
             Backend.GPU_ARTISAN, {visionBackend: Backend.GPU});
         await engine.createConversation();
         expect(createDefaultSpy).not.toHaveBeenCalled();
         expect(mockSessionConfig.setVisionModalityEnabled)
             .toHaveBeenCalledWith(true);
         expect(createCustomSpy)
             .toHaveBeenCalledWith(
                 mockWasmEngine, mockSessionConfig, undefined, undefined,
                 undefined, '');
       });

    it('leaves unset booleans undefined for empty and partial configs across backends',
       async () => {
         for (const [backend, useAutosizedRingbuffers] of [
                  [Backend.GPU_ARTISAN, true],
                  [Backend.GPU_ARTISAN, false],
                  [Backend.GPU, false],
                  [Backend.CPU, false],
         ] as const) {
           const emptyEngine =
               createTestEngine(backend, {useAutosizedRingbuffers});
           await emptyEngine.createConversation({});
           expect(createCustomSpy)
               .toHaveBeenCalledWith(
                   mockWasmEngine, mockSessionConfig, undefined, undefined,
                   undefined, '');

           const partialEngine =
               createTestEngine(backend, {useAutosizedRingbuffers});
           await partialEngine.createConversation({
             sessionConfig: {maxOutputTokens: 64},
             preface: {messages: [{role: 'system', content: 'Be brief.'}]},
           });
           expect(mockSessionConfig.setMaxOutputTokens)
               .toHaveBeenCalledWith(64);
           expect(createCustomSpy)
               .toHaveBeenCalledWith(
                   mockWasmEngine, mockSessionConfig, undefined, undefined,
                   undefined,
                   JSON.stringify(
                       {messages: [{role: 'system', content: 'Be brief.'}]}));
         }
       });

    it('forwards explicit ConversationConfig boolean overrides', async () => {
      const gpuArtisanEngine = createTestEngine(
          Backend.GPU_ARTISAN, {useAutosizedRingbuffers: true});
      await gpuArtisanEngine.createConversation({
        enableConstrainedDecoding: true,
        prefillPrefaceOnInit: true,
        filterChannelContentFromKvCache: false,
      });
      expect(createCustomSpy)
          .toHaveBeenCalledWith(
              mockWasmEngine, mockSessionConfig, true, true, false, '');

      const cpuEngine = createTestEngine(Backend.CPU);
      await cpuEngine.createConversation({
        enableConstrainedDecoding: false,
        prefillPrefaceOnInit: false,
        filterChannelContentFromKvCache: true,
      });
      expect(createCustomSpy)
          .toHaveBeenCalledWith(
              mockWasmEngine, mockSessionConfig, false, false, true, '');
    });
  });

  describe('ConversationConfig WASM C++ defaults', () => {
    const CPU_MODEL_PATH = '/models/tiny_gemma.litertlm';
    const GPU_ARTISAN_MODEL_PATH = '/models/gemma3-1b-hw.litertlm';

    async function servesLiteRtLmFile(path: string): Promise<boolean> {
      try {
        const response = await fetch(path);
        if (!response.ok || !response.body) {
          return false;
        }
        const reader = response.body.getReader();
        const {value} = await reader.read();
        await reader.cancel();
        return new TextDecoder().decode(value?.subarray(0, 8)) === 'LITERTLM';
      } catch {
        return false;
      }
    }

    it('produces identical effective C++ ConversationConfig values for no args, {}, and partial configs',
       async () => {
         if (!(await servesLiteRtLmFile(CPU_MODEL_PATH))) {
           pending('The local models are not served at /models.');
           return;
         }
         unloadLiteRtLm();
         const liteRtLm = await loadLiteRtLm('/wasm');
         const wasm = liteRtLm.liteRtLmWasm;

         const cpuEngine = await Engine.create({
           model: CPU_MODEL_PATH,
           backend: Backend.CPU,
           mainExecutorSettings: {maxNumTokens: 128},
         });
         try {
           const internalEngine =
               (cpuEngine as unknown as {engine: WasmEngine}).engine;
           const defaultCfg = unwrapWasmResult(
               wasm.ConversationConfig.createDefault(internalEngine));
           const sessionCfg = wasm.SessionConfig.createDefault();
           const emptyCustomCfg =
               unwrapWasmResult(wasm.ConversationConfig.createCustom(
                   internalEngine, sessionCfg, undefined, undefined, undefined,
                   ''));
           const partialSessionCfg = wasm.SessionConfig.createDefault();
           partialSessionCfg.setMaxOutputTokens(32);
           const partialCustomCfg =
               unwrapWasmResult(wasm.ConversationConfig.createCustom(
                   internalEngine, partialSessionCfg, undefined, undefined,
                   undefined,
                   JSON.stringify(
                       {messages: [{role: 'system', content: 'Hi'}]})));
           const overriddenCustomCfg =
               unwrapWasmResult(wasm.ConversationConfig.createCustom(
                   internalEngine, sessionCfg, true, true, false, ''));
           try {
             const snapshot = (cfg: typeof defaultCfg) => ({
               constrainedDecodingEnabled: cfg.constrainedDecodingEnabled(),
               prefillPrefaceOnInit: cfg.prefillPrefaceOnInit(),
               filterChannelContentFromKvCache:
                   cfg.filterChannelContentFromKvCache(),
               enableRewinding: cfg.enableRewinding(),
             });
             expect(snapshot(defaultCfg)).toEqual({
               constrainedDecodingEnabled: false,
               prefillPrefaceOnInit: false,
               filterChannelContentFromKvCache: true,
               enableRewinding: false,
             });
             expect(snapshot(emptyCustomCfg)).toEqual(snapshot(defaultCfg));
             expect(snapshot(partialCustomCfg)).toEqual(snapshot(defaultCfg));
             expect(snapshot(overriddenCustomCfg)).toEqual({
               constrainedDecodingEnabled: true,
               prefillPrefaceOnInit: true,
               filterChannelContentFromKvCache: false,
               enableRewinding: false,
             });
           } finally {
             defaultCfg.delete();
             emptyCustomCfg.delete();
             partialCustomCfg.delete();
             overriddenCustomCfg.delete();
             sessionCfg.delete();
             partialSessionCfg.delete();
           }
         } finally {
           await cpuEngine.delete();
         }

         if ((await servesLiteRtLmFile(GPU_ARTISAN_MODEL_PATH)) &&
             (await waitForWebGpuAdapter())) {
           for (const [useAutosizedRingbuffers, expectedRewinding] of [
                    [false, true],
                    [true, false],
           ] as const) {
             const gpuEngine = await Engine.create({
               model: GPU_ARTISAN_MODEL_PATH,
               mainExecutorSettings: {
                 maxNumTokens: 128,
                 backendConfig: {
                   use_autosized_ringbuffers: useAutosizedRingbuffers,
                 },
               },
             });
             try {
               const internalEngine =
                   (gpuEngine as unknown as {engine: WasmEngine}).engine;
               const defaultCfg = unwrapWasmResult(
                   wasm.ConversationConfig.createDefault(internalEngine));
               const sessionCfg = wasm.SessionConfig.createDefault();
               const customCfg =
                   unwrapWasmResult(wasm.ConversationConfig.createCustom(
                       internalEngine, sessionCfg, undefined, undefined,
                       undefined, ''));
               try {
                 expect(defaultCfg.filterChannelContentFromKvCache())
                     .toBe(true);
                 expect(customCfg.filterChannelContentFromKvCache()).toBe(true);
                 expect(defaultCfg.enableRewinding()).toBe(expectedRewinding);
                 expect(customCfg.enableRewinding()).toBe(expectedRewinding);
               } finally {
                 defaultCfg.delete();
                 customCfg.delete();
                 sessionCfg.delete();
               }
             } finally {
               await gpuEngine.delete();
             }
           }
         }
       },
       120_000);
  });
});
