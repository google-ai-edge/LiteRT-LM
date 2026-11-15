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

import {LiteRtLm, loadLiteRtLm, sessionConfigToWasmSessionConfig, SpeculativeDecodingMode, unloadLiteRtLm, wasmSessionConfigToSessionConfig} from '@litert-lm/core';
// Placeholder for internal dependency on trusted resource url

describe('SessionConfig', () => {
  let liteRtLm: LiteRtLm;

  beforeAll(async () => {
    unloadLiteRtLm();
    liteRtLm = await loadLiteRtLm('/wasm');
  });

  it('round-trips maxOutputTokens', () => {
    const wasmSessionConfig = sessionConfigToWasmSessionConfig(
        {maxOutputTokens: 42}, liteRtLm.liteRtLmWasm);
    expect(wasmSessionConfigToSessionConfig(wasmSessionConfig).maxOutputTokens)
        .toBe(42);
    wasmSessionConfig.delete();
  });

  it('inherits the speculative decoding config by default', () => {
    const wasmSessionConfig =
        sessionConfigToWasmSessionConfig({}, liteRtLm.liteRtLmWasm);
    expect(wasmSessionConfig.getSpeculativeDecodingConfig())
        .toEqual({mode: undefined, lenience: undefined});
    expect(wasmSessionConfigToSessionConfig(wasmSessionConfig)
               .speculativeDecodingConfig)
        .toEqual({});
    wasmSessionConfig.delete();
  });

  for (const mode
           of [SpeculativeDecodingMode.OFF, SpeculativeDecodingMode.ON,
               SpeculativeDecodingMode.ADAPTIVE]) {
    it(`round-trips speculative decoding mode ${mode}`, () => {
      const wasmSessionConfig = sessionConfigToWasmSessionConfig(
          {speculativeDecodingConfig: {mode}}, liteRtLm.liteRtLmWasm);
      expect(wasmSessionConfigToSessionConfig(wasmSessionConfig)
                 .speculativeDecodingConfig)
          .toEqual({mode});
      wasmSessionConfig.delete();
    });
  }

  it('round-trips only the lenience, inheriting the mode', () => {
    const wasmSessionConfig = sessionConfigToWasmSessionConfig(
        {speculativeDecodingConfig: {lenience: 0.25}}, liteRtLm.liteRtLmWasm);
    expect(wasmSessionConfigToSessionConfig(wasmSessionConfig)
               .speculativeDecodingConfig)
        .toEqual({lenience: 0.25});
    wasmSessionConfig.delete();
  });

  it('can reset the speculative decoding config to inherit the engine',
     () => {
       const wasmSessionConfig = sessionConfigToWasmSessionConfig(
           {
             speculativeDecodingConfig:
                 {mode: SpeculativeDecodingMode.ON, lenience: 1},
           },
           liteRtLm.liteRtLmWasm);
       wasmSessionConfig.setSpeculativeDecodingConfig(
           {mode: undefined, lenience: undefined});
       expect(wasmSessionConfigToSessionConfig(wasmSessionConfig)
                  .speculativeDecodingConfig)
           .toEqual({});
       wasmSessionConfig.delete();
     });

  it('rejects an out-of-range lenience', () => {
    expect(
        () => sessionConfigToWasmSessionConfig(
            {speculativeDecodingConfig: {lenience: -0.1}},
            liteRtLm.liteRtLmWasm))
        .toThrowError(/lenience must be in \[0, 1\]/);
  });
});
