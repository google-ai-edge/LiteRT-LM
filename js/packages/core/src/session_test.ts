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

import {Mutex} from './mutex.js';
import {Session} from './session.js';
import {Backend, EmscriptenVector, SamplerParameters, SessionConfig as WasmSessionConfig, Session as WasmSession} from './wasm_binding_types.js';

describe('Session', () => {
  let mockWasmSession: jasmine.SpyObj<WasmSession>;
  let mockWasmSessionConfig: jasmine.SpyObj<WasmSessionConfig>;
  let mockMutexes: {executor: Mutex};

  beforeEach(() => {
    mockWasmSessionConfig = jasmine.createSpyObj('WasmSessionConfig', [
      'delete', 'getAudioModalityEnabled', 'getVisionModalityEnabled',
      'getMutableSamplerParams', 'getStopTokenIds', 'getStartTokenId',
      'getNumOutputCandidates', 'getSamplerBackend',
      'getApplyPromptTemplateInSession', 'getUseExternalSampler',
      'getMaxOutputTokens'
    ]);
    mockWasmSessionConfig.getAudioModalityEnabled.and.returnValue(false);
    mockWasmSessionConfig.getVisionModalityEnabled.and.returnValue(false);

    const dummySamplerParams: SamplerParameters = {
      type: () => ({value: 0}),
      k: () => 40,
      p: () => 0.95,
      temperature: () => 0.8,
      seed: () => 0,
      setType: () => {},
      setK: () => {},
      setP: () => {},
      setTemperature: () => {},
      setSeed: () => {},
    };
    mockWasmSessionConfig.getMutableSamplerParams.and.returnValue(dummySamplerParams);

    const dummyStopTokenIds: EmscriptenVector<EmscriptenVector<number>> = {
      size: () => 0,
      get: () => ({
        size: () => 0,
        get: () => 0,
        push_back: () => {},
        delete: () => {}
      } as EmscriptenVector<number>),
      push_back: () => {},
      delete: () => {},
    };
    mockWasmSessionConfig.getStopTokenIds.and.returnValue(dummyStopTokenIds);

    mockWasmSessionConfig.getStartTokenId.and.returnValue(1);
    mockWasmSessionConfig.getNumOutputCandidates.and.returnValue(1);
    mockWasmSessionConfig.getSamplerBackend.and.returnValue({value: Backend.CPU});
    mockWasmSessionConfig.getApplyPromptTemplateInSession.and.returnValue(false);
    mockWasmSessionConfig.getUseExternalSampler.and.returnValue(false);
    mockWasmSessionConfig.getMaxOutputTokens.and.returnValue(1024);

    mockWasmSession = jasmine.createSpyObj('WasmSession', [
      'getSessionConfig', 'runPrefill', 'runDecode', 'cancelProcess',
      'delete', 'clone'
    ]);
    mockWasmSession.getSessionConfig.and.returnValue(mockWasmSessionConfig);

    mockMutexes = {executor: new Mutex()};
  });

  it('clones session properly and allows independent prefill and decode execution', async () => {
    const session = new Session(mockWasmSession, mockMutexes);
    const mockClonedWasmSession = jasmine.createSpyObj('WasmSession', [
      'getSessionConfig', 'runPrefill', 'runDecode', 'cancelProcess',
      'delete', 'clone'
    ]);
    mockClonedWasmSession.getSessionConfig.and.returnValue(
        mockWasmSessionConfig);
    mockWasmSession.clone.and.returnValue(mockClonedWasmSession);

    const cloned = await session.clone();
    expect(cloned).toBeDefined();
    expect(mockWasmSession.clone).toHaveBeenCalled();

    mockClonedWasmSession.runPrefill.and.returnValue(Promise.resolve());
    await cloned.runPrefill(['test prefill on clone']);
    expect(mockClonedWasmSession.runPrefill)
        .toHaveBeenCalledWith(['test prefill on clone']);

    const mockResponses = jasmine.createSpyObj('WasmResponses', ['getTexts', 'delete']);
    const dummyTexts: EmscriptenVector<string> = {
      size: () => 1,
      get: (i: number) => 'cloned decoded text',
      push_back: () => {},
      delete: () => {},
    };
    mockResponses.getTexts.and.returnValue(dummyTexts);
    mockClonedWasmSession.runDecode.and.returnValue(Promise.resolve(mockResponses));

    const responses = await cloned.runDecode();
    expect(responses.getTexts()).toEqual(['cloned decoded text']);
    expect(mockClonedWasmSession.runDecode).toHaveBeenCalled();
  });

  it('throws an error when cloning a busy session', async () => {
    const session = new Session(mockWasmSession, mockMutexes);
    let finishPrefill: () => void = () => {};
    mockWasmSession.runPrefill.and.returnValue(new Promise<void>(resolve => {
      finishPrefill = resolve;
    }));

    const p = session.runPrefill(['long prefill']);
    await expectAsync(session.clone()).toBeRejectedWithError(/Session is busy/);
    finishPrefill();
    await p;
  });
});
