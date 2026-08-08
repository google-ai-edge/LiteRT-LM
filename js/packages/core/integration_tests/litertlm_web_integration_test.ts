/**
 * Copyright 2026 Google LLC
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

import {Backend, Conversation, ConversationConfig, Engine, getOrLoadGlobalLiteRtLm, GpuArtisanConfig, LiteRtLm, loadLiteRtLm, Message, SamplerType, Session, SessionConfig, unloadLiteRtLm, type Wasm} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

jasmine.DEFAULT_TIMEOUT_INTERVAL = 300_000;  // 300 seconds

const MODEL_PATH = '/models/tiny_gemma.litertlm';
const GPU_ARTISAN_MODEL_PATH = '/models/gemma3-1b-hw.litertlm';

describe('LiteRtLm tests', () => {
  let liteRtLm: LiteRtLm;
  async function resetLiteRtLm() {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm(trustedResourceUrl`/wasm`);
    await liteRtLm.setupDefaultWebGpuDevice();
  }

  beforeAll(async () => {
    await resetLiteRtLm();
    // Wait for the WASM module to be loaded. Sometimes, this takes a while.
    // especially when using DWARF debugging.
  }, 300_000);

  it('loads the WASM module', () => {
    expect(liteRtLm).toBeDefined();
  });

  it('loads or gets the global instance with getOrLoadGlobalLiteRtLm',
     async () => {
       unloadLiteRtLm();
       const path1 = trustedResourceUrl`/wasm`;
       const path2 = trustedResourceUrl`/other/wasm`;

       // First call loads it
       const p1 = getOrLoadGlobalLiteRtLm(path1);
       const p2 = getOrLoadGlobalLiteRtLm(path1);
       expect(p1).toBe(p2);  // Should return the exact same promise

       // Calling with undefined uses the existing path instead of default when
       // already loading/loaded
       const p3 = getOrLoadGlobalLiteRtLm();
       expect(p3).toBe(p1);

       // Calling with a different path throws an error
       expect(() => getOrLoadGlobalLiteRtLm(path2))
           .toThrowError(
               /LiteRT-LM is already loading \/ loaded with a different path/);

       await p1;
     });

  it('automatically loads the WASM module when loading a model', async () => {
    unloadLiteRtLm();
    const engine = await Engine.create({
      model: MODEL_PATH,
      backend: Backend.CPU,
      mainExecutorSettings: {
        maxNumTokens: 128,
      },
    });
    expect(engine).toBeDefined();
    await engine.delete();
  });

  it('streaming-loads an Artisan GPU model', async () => {
    const gpuArtisanConfig: GpuArtisanConfig = {
      num_output_candidates: 2,
      wait_for_weight_uploads: true,
      num_decode_steps_per_sync: 4,
      sequence_batch_size: 8,
      supported_lora_ranks: [],
      max_top_k: 5,
      enable_decode_logits: true,
      enable_external_embeddings: false,
      use_submodel: true,
      use_autosized_ringbuffers: false,
    };

    const engine = await Engine.create({
      model: GPU_ARTISAN_MODEL_PATH,
      backend: Backend.GPU_ARTISAN,
      mainExecutorSettings: {
        maxNumTokens: 128,
        backendConfig: gpuArtisanConfig,
      },
    });

    expect(engine).toBeDefined();

    // Run a decode step to ensure the model is working as expected.
    const sessionConfig: SessionConfig = {};
    const session = await engine.createSession(sessionConfig);
    await session.runPrefill(['test input']);
    const responses = await session.runDecode();
    expect(responses).toBeDefined();
    responses.delete();
    await session.delete();

    await engine.delete();
  });

  it('defaults to GPU_ARTISAN backend if not specified', async () => {
    const engine = await Engine.create({
      model: GPU_ARTISAN_MODEL_PATH,
    });

    expect(engine).toBeDefined();
    expect(engine.settings.backend).toBe(Backend.GPU_ARTISAN);

    await engine.delete();
  });

  for (const [name, backend] of [
           ['CPU', Backend.CPU],
           ['GPU_ARTISAN', Backend.GPU_ARTISAN],
  ] as const) {
    describe(`Engine (${name})`, () => {
      let engine: Engine;
      const modelPath =
          backend === Backend.CPU ? MODEL_PATH : GPU_ARTISAN_MODEL_PATH;
      beforeAll(async () => {
        try {
          const backendConfig = backend === Backend.CPU ? undefined : {
            num_output_candidates: 2,
            wait_for_weight_uploads: true,
            num_decode_steps_per_sync: 4,
            sequence_batch_size: 8,
            supported_lora_ranks: [],
            max_top_k: 5,
            enable_decode_logits: true,
            enable_external_embeddings: false,
            use_submodel: true,
            use_autosized_ringbuffers: false,
          };
          engine = await Engine.create({
            model: modelPath,
            backend,
            mainExecutorSettings: {
              maxNumTokens: 128,
              backendConfig,
            },
            benchmarkEnabled: true,
          });
        } catch (e) {
          console.error(e);
          throw e;
        }
      });

      afterAll(() => {
        if (engine) engine.delete();
      });

      it('creates Engine', () => {
        expect(engine).toBeDefined();
      });

      it('engine.settings values match constructor arguments', () => {
        expect(engine.settings.model).toBe(modelPath);
        expect(engine.settings.backend).toBe(backend);
        expect(engine.settings.mainExecutorSettings.maxNumTokens).toBe(128);
      });

      it('engine.settings has defaults for unspecified fields', () => {
        expect(engine.settings.mainExecutorSettings.samplerBackend)
            .toBeDefined();
        const advancedSettings =
            engine.settings.mainExecutorSettings.advancedSettings;
        expect(advancedSettings.prefill_batch_sizes).toBeDefined();
        expect(advancedSettings.num_output_candidates).toBeDefined();
        expect(advancedSettings.configure_magic_numbers).toBeDefined();
        expect(advancedSettings.verify_magic_numbers).toBeDefined();
        expect(advancedSettings.clear_kv_cache_before_prefill).toBeDefined();
        expect(advancedSettings.num_logits_to_print_after_decode).toBeDefined();
        expect(advancedSettings.gpu_madvise_original_shared_tensors)
            .toBeDefined();
        expect(advancedSettings.is_benchmark).toBeDefined();
        expect(advancedSettings.preferred_device_substr).toBeDefined();
        expect(advancedSettings.num_threads_to_upload).toBeDefined();
        expect(advancedSettings.num_threads_to_compile).toBeDefined();
        expect(advancedSettings.convert_weights_on_gpu).toBeDefined();
        expect(advancedSettings.optimize_shader_compilation).toBeDefined();
        expect(advancedSettings.share_constant_tensors).toBeDefined();
      });

      describe('Session', () => {
        let sessionConfig: SessionConfig;
        let session: Session;

        beforeEach(async () => {
          sessionConfig = {
            maxOutputTokens: 128,
            samplerParams: {
              type: SamplerType.TOP_P,
              k: 1,
              p: 0.9,
            },
          };
          session = await engine.createSession(sessionConfig);
        });
        afterEach(() => {
          session.delete();
        });

        it('creates Session', () => {
          expect(session).toBeDefined();
        });

        it('session.config values match constructor arguments', () => {
          expect(session.config.maxOutputTokens).toBe(128);
          expect(session.config.samplerParams.type).toBe(SamplerType.TOP_P);
          expect(session.config.samplerParams.p).toBeCloseTo(0.9, 1e-6);
        });

        it('session.config has defaults for unspecified fields', () => {
          expect(session.config.applyPromptTemplateInSession).toBeDefined();
          expect(session.config.numOutputCandidates).toBeDefined();
          expect(session.config.audioModalityEnabled).toBeDefined();
          expect(session.config.visionModalityEnabled).toBeDefined();
          expect(session.config.samplerParams).toBeDefined();
          expect(session.config.stopTokenIds).toBeDefined();
          expect(session.config.startTokenId).toBeDefined();
          expect(session.config.samplerBackend).toBeDefined();
          expect(session.config.useExternalSampler).toBeDefined();
        });

        it('runs prefill', async () => {
          await expectAsync(session.runPrefill(['test input'])).toBeResolved();
        });

        it('runs decode', async () => {
          await session.runPrefill(['test input']);
          const responses = await session.runDecode();
          expect(responses).toBeDefined();
          responses.delete();
        });

        it('runs two turns of prefill and decode', async () => {
          await session.runPrefill(['test input']);
          const responses = await session.runDecode();
          const texts = responses.getTexts();
          expect(texts[0]).toBeDefined();
          responses.delete();
          await session.runPrefill(['test input']);
          const responses2 = await session.runDecode();
          const texts2 = responses2.getTexts();
          expect(texts2[0]).toBeDefined();
          responses2.delete();
        });

        it('throws an error on overlapping session operations', async () => {
          const p1 = session.runPrefill(['test input']);
          await expectAsync(session.runPrefill(['test input 2']))
              .toBeRejectedWithError(/Session is busy/);
          await expectAsync(p1).toBeResolved();
        });

        it('clones session and allows continuing both independently', async () => {
          await session.runPrefill(['test input']);
          const clonedSession = await session.clone();
          try {
            expect(clonedSession).toBeDefined();

            // Continue with original
            const responses1 = await session.runDecode();
            expect(responses1.getTexts()[0]).toBeDefined();
            responses1.delete();

            // Continue with clone from the point of cloning
            const responses2 = await clonedSession.runDecode();
            expect(responses2.getTexts()[0]).toBeDefined();
            responses2.delete();
          } finally {
            await clonedSession.delete();
          }
        });

        it('clones session after decode and continues prefill independently', async () => {
          await session.runPrefill(['test input']);
          const responses = await session.runDecode();
          responses.delete();

          const clonedSession = await session.clone();
          try {
            expect(clonedSession).toBeDefined();

            // Continue with clone
            await clonedSession.runPrefill(['follow up on clone']);
            const clonedResponses = await clonedSession.runDecode();
            expect(clonedResponses.getTexts()[0]).toBeDefined();
            clonedResponses.delete();

            // Continue with original
            await session.runPrefill(['follow up on original']);
            const origResponses = await session.runDecode();
            expect(origResponses.getTexts()[0]).toBeDefined();
            origResponses.delete();
          } finally {
            await clonedSession.delete();
          }
        });

        it('clones session and outputs identical responses for the same prompt under greedy sampling', async () => {
          await session.runPrefill(['test input']);
          const clonedSession = await session.clone();
          try {
            expect(clonedSession).toBeDefined();

            await session.runPrefill(['Tell me a joke.']);
            const origResponses = await session.runDecode();
            const origText = origResponses.getTexts()[0];
            origResponses.delete();

            await clonedSession.runPrefill(['Tell me a joke.']);
            const cloneResponses = await clonedSession.runDecode();
            const cloneText = cloneResponses.getTexts()[0];
            cloneResponses.delete();

            expect(origText).toBeDefined();
            expect(cloneText).toBeDefined();
            expect(cloneText).toEqual(origText);
          } finally {
            await clonedSession.delete();
          }
        });
      });

      describe('Conversation', () => {
        let conversation: Conversation;

        beforeEach(async () => {
          conversation = await engine.createConversation();
        });

        afterEach(() => {
          conversation.delete();
        });

        it('creates Conversation', () => {
          expect(conversation).toBeDefined();
        });

        it('gets token count initially and after sending message', async () => {
          let tokenCount = await conversation.getTokenCount();
          expect(tokenCount).toBe(0);

          const message = {
            role: 'user',
            content: 'Hello',
          };
          await conversation.sendMessage(message);

          tokenCount = await conversation.getTokenCount();
          expect(tokenCount).toBeGreaterThan(0);
        });

        it('gets benchmark info after sending message', async () => {
          const message = {
            role: 'user',
            content: 'Hello benchmark test',
          };
          await conversation.sendMessage(message);

          const benchmark = await conversation.getBenchmarkInfo();
          expect(benchmark).toBeDefined();

          expect(benchmark.lastPrefillTokenCount).toBeGreaterThan(0);
          expect(benchmark.lastDecodeTokenCount).toBeGreaterThan(0);
          expect(benchmark.lastPrefillTokensPerSecond).toBeGreaterThan(0);
          expect(benchmark.lastDecodeTokensPerSecond).toBeDefined();
          expect(benchmark.timeToFirstTokenInSecond).toBeDefined();
        });

        it('sends message and gets response', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
          };
          const response = await conversation.sendMessage(message);
          expect(response).toBeDefined();
        });

        it('sends message and streams response', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
          };
          const stream = conversation.sendMessageStreaming(message);
          const reader = stream.getReader();
          let chunkCount = 0;
          let done = false;
          while (!done) {
            const {value, done: streamDone} = await reader.read();
            if (streamDone) {
              done = true;
            } else {
              expect(value).toBeDefined();
              chunkCount++;
            }
          }
          expect(chunkCount).toBeGreaterThan(0);
        });

        it('sends string message and gets response', async () => {
          const response = await conversation.sendMessage('Hello');
          expect(response).toBeDefined();
        });

        it('sends string message and streams response', async () => {
          const stream = conversation.sendMessageStreaming('Hello');
          const reader = stream.getReader();
          let chunkCount = 0;
          let done = false;
          while (!done) {
            const {value, done: streamDone} = await reader.read();
            if (streamDone) {
              done = true;
            } else {
              expect(value).toBeDefined();
              chunkCount++;
            }
          }
          expect(chunkCount).toBeGreaterThan(0);
        });

        it('cancels stream and generation gracefully', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
          };
          const stream = conversation.sendMessageStreaming(message);
          const reader = stream.getReader();

          // Read start chunks.
          await reader.read();

          // Cancel the stream, which should trigger cancelProcess under the
          // hood.
          await reader.cancel();

          // Can start a new message after cancelling.
          const newConversation = await engine.createConversation();
          const response = await newConversation.sendMessage(message);
          expect(response).toBeDefined();
          await newConversation.delete();
        });

        it('sends message with structured content array', async () => {
          const message: Message = {
            role: 'user',
            content: [{type: 'text', text: 'Hello'}]
          };
          const response = await conversation.sendMessage(message);
          expect(response).toBeDefined();
        });

        it('sends message with channels', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
            channels: {'reasoning': 'think'}
          };
          const response = await conversation.sendMessage(message);
          expect(response).toBeDefined();
        });

        it('gets history', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
          };
          await conversation.sendMessage(message);
          const history = await conversation.getHistory();
          expect(history.length).toBeGreaterThan(0);
        });

        it('throws an error on overlapping sendMessage calls', async () => {
          const message1 = {
            role: 'user',
            content: 'First message',
          };
          const message2 = {
            role: 'user',
            content: 'Second message',
          };
          const p1 = conversation.sendMessage(message1);
          await expectAsync(conversation.sendMessage(message2))
              .toBeRejectedWithError(/Conversation is busy/);
          await expectAsync(p1).toBeResolved();
        });

        it('clones conversation and allows continuing both independently', async () => {
          await conversation.sendMessage('Hello');
          const clonedConversation = await conversation.clone();
          try {
            expect(clonedConversation).toBeDefined();

            const origHistoryBefore = await conversation.getHistory();
            const cloneHistoryBefore = await clonedConversation.getHistory();
            expect(cloneHistoryBefore.length).toBe(origHistoryBefore.length);

            // Continue with original
            const resp1 = await conversation.sendMessage('How are you?');
            expect(resp1).toBeDefined();
            const origHistoryAfter = await conversation.getHistory();

            // Continue with clone from the same point at which it was cloned
            const resp2 = await clonedConversation.sendMessage('What is 2+2?');
            expect(resp2).toBeDefined();
            const cloneHistoryAfter = await clonedConversation.getHistory();

            expect(cloneHistoryAfter.length).toBe(origHistoryAfter.length);
            expect(cloneHistoryAfter[cloneHistoryAfter.length - 2].content).toBe('What is 2+2?');
            expect(origHistoryAfter[origHistoryAfter.length - 2].content).toBe('How are you?');
          } finally {
            await clonedConversation.delete();
          }
        });

        it('clones conversation and outputs identical responses for the same prompt under greedy sampling', async () => {
          await conversation.sendMessage('Hello');
          const clonedConversation = await conversation.clone();
          try {
            expect(clonedConversation).toBeDefined();

            const prompt = 'What is the capital of France?';
            const origResp = await conversation.sendMessage(prompt);
            const cloneResp = await clonedConversation.sendMessage(prompt);

            expect(origResp).toBeDefined();
            expect(cloneResp).toBeDefined();
            expect(cloneResp).toEqual(origResp);

            const origHistory = await conversation.getHistory();
            const cloneHistory = await clonedConversation.getHistory();
            expect(cloneHistory.length).toBe(origHistory.length);
            expect(cloneHistory).toEqual(origHistory);
          } finally {
            await clonedConversation.delete();
          }
        });
      });

      describe('Conversation with Custom Config', () => {
        let conversation: Conversation;

        beforeEach(async () => {
          const config: ConversationConfig = {
            enableConstrainedDecoding: true,
            preface: {
              messages: [{role: 'system', content: 'Pretend you are a pirate.'}]
            }
          };
          conversation = await engine.createConversation(config);
        });

        afterEach(() => {
          conversation.delete();
        });

        it('creates Conversation', () => {
          expect(conversation).toBeDefined();
        });

        it('sends message and gets response', async () => {
          const message = {
            role: 'user',
            content: 'Hello',
          };
          const response = await conversation.sendMessage(message);
          expect(response).toBeDefined();
        });
      });
    });
  }

  it('creates and runs a session from a ReadableStream', async () => {
    const response = await fetch(MODEL_PATH, {
      credentials: 'same-origin',
    });
    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch model file from ${MODEL_PATH}`);
    }
    const engine = await Engine.create({
      model: response.body,
      backend: Backend.CPU,
      mainExecutorSettings: {
        maxNumTokens: 128,
      },
    });
    const sessionConfig: SessionConfig = {};
    const session = await engine.createSession(sessionConfig);
    await session.runPrefill(['test input']);
    const responses = await session.runDecode();
    expect(responses).toBeDefined();
    responses.delete();
    await session.delete();
    if (engine) await engine.delete();
  });

  it('creates and runs a session from a Blob', async () => {
    const response = await fetch(MODEL_PATH, {
      credentials: 'same-origin',
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch model file from ${MODEL_PATH}`);
    }
    const engine = await Engine.create({
      model: await response.blob(),
      backend: Backend.CPU,
      mainExecutorSettings: {
        maxNumTokens: 128,
      },
    });
    const sessionConfig: SessionConfig = {};
    const session = await engine.createSession(sessionConfig);
    await session.runPrefill(['test input']);
    const responses = await session.runDecode();
    expect(responses).toBeDefined();
    responses.delete();
    await session.delete();
    if (engine) await engine.delete();
  });

  describe('Wasm tests', () => {
    describe('Engine', () => {
      beforeAll(async () => {
        await loadModelToVfs(
            liteRtLm.liteRtLmWasm,
            MODEL_PATH,
            'model.litertlm',
        );
      }, 120_000);

      // TODO: b/474424353 - Add converted model GPU tests back once we have a
      // model with static signatures.
      for (const backendName of ['CPU']) {
        describe(backendName, () => {
          let modelAssets: Wasm.ModelAssets;
          let engineSettings: Wasm.EngineSettings;
          let engine: Wasm.Engine;
          let backend: Wasm.EmscriptenEnumElement<Backend>;
          beforeAll(async () => {
            backend = backendName === 'CPU' ?
                liteRtLm.liteRtLmWasm.Backend.CPU :
                liteRtLm.liteRtLmWasm.Backend.GPU;
          });

          beforeEach(async () => {
            modelAssets =
                liteRtLm.liteRtLmWasm.ModelAssets.create('model.litertlm');
            engineSettings = liteRtLm.liteRtLmWasm.EngineSettings.createDefault(
                modelAssets, backend);
            engineSettings.setSingleThreadedExecution(true);
            engineSettings.setParallelFileSectionLoading(false);
            const executorSettings =
                engineSettings.getMutableMainExecutorSettings();

            if (backendName === 'CPU') {
              executorSettings.setBackendConfigCpu({
                kv_increment_size: 16,
                prefill_chunk_size: -1,
                number_of_threads: 1,
              });
            } else {
              executorSettings.setBackendConfigGpu({
                max_top_k: 3,
                external_tensor_mode: false,
              });
            }

            executorSettings.setSamplerBackend(backend);
            executorSettings.setMaxNumTokens(128);
            executorSettings.setCacheDir(':nocache');
            const advancedSettings = {
              prefill_batch_sizes: [128],
              num_output_candidates: 1,
              configure_magic_numbers: true,
              verify_magic_numbers: true,
              clear_kv_cache_before_prefill: true,
              num_logits_to_print_after_decode: 3,
              gpu_madvise_original_shared_tensors: true,
              is_benchmark: false,
              preferred_device_substr: '',
              num_threads_to_upload: 1,
              num_threads_to_compile: 1,
              convert_weights_on_gpu: true,
              optimize_shader_compilation: true,
              share_constant_tensors: true,
            };
            executorSettings.setAdvancedSettings(advancedSettings);

            engine = await liteRtLm.liteRtLmWasm.Engine.createEngine(
                engineSettings, '');
          });

          afterEach(() => {
            if (engine) engine.delete();
            engineSettings.delete();
            modelAssets.delete();
          });

          it('creates Engine', () => {
            expect(engine).toBeDefined();
          });

          describe('Session', () => {
            let session: Wasm.Session;
            beforeEach(() => {
              try {
                console.log('Creating session');
                session = engine.createSession(
                    liteRtLm.liteRtLmWasm.SessionConfig.createDefault());
                console.log('Session created', session);
              } catch (e) {
                console.error('Error creating session', e);
                throw e;
              }
            });

            afterEach(() => {
              session.delete();
            });

            it('runs prefill', async () => {
              await expectAsync(session.runPrefill(['Hello World']))
                  .not.toBeRejected();
            });

            it('runs prefill and decode', async () => {
              await session.runPrefill(['Hello World']);
              const responses = await session.runDecode();
              const texts = responses.getTexts();
              expect(responses.getTexts().size()).toBe(1);
              expect(responses.getTexts().get(0).length).toBeGreaterThan(0);
              texts.delete();
              responses.delete();
            });

            it('outputs a reasonable response', async () => {
              pending(
                  'We need a larger, more predictable model if we want to test output values.');
            });

            it('clones session', async () => {
              await session.runPrefill(['Hello World']);
              const clonedSession = await session.clone();
              try {
                expect(clonedSession).toBeDefined();
                const responses = await clonedSession.runDecode();
                expect(responses.getTexts().size()).toBe(1);
                responses.delete();
              } finally {
                clonedSession.delete();
              }
            });
          });
        });
      }
    });
  });
});


async function loadModelToVfs(
    module: Wasm.LiteRtLmWasm,
    modelUrl: string,
    dstPath: string,
) {
  const response = await fetch(modelUrl, {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch model file from ${modelUrl}`);
  }
  const blob = await response.blob();
  const fileContentBuffer = await blob.arrayBuffer();
  const fileContent = new Uint8Array(fileContentBuffer);

  try {
    module.FS.writeFile(dstPath, fileContent);
  } catch (e) {
    console.error(`Error writing file to VFS:`, e);
    throw e;
  }
}
