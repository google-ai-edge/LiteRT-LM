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

import {WasmModule} from '@litertjs/wasm-utils';

import {getGlobalLiteRtLm, hasGlobalLiteRtLm, setGlobalLiteRtLm} from './global_litertlm.js';
import {LiteRtLm, setupDefaultWebGpuDevice} from './litertlm_web.js';
import {LiteRtLmWasm} from './wasm_binding_types.js';

const ANGLE_M5_MAX_RENDERER =
    'ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Max, Unspecified Version)';

function createFakeDevice(
    vendor = 'apple', architecture = 'metal-3', device = '',
    description = ''): GPUDevice {
  return {
    adapterInfo: {vendor, architecture, device, description},
  } as unknown as GPUDevice;
}

function createFakeGpu(device: GPUDevice) {
  const requestDevice =
      jasmine.createSpy('requestDevice').and.resolveTo(device);
  const fakeAdapter = {
    limits: {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 1024,
      maxStorageBuffersPerShaderStage: 8,
      maxTextureDimension2D: 8192,
    },
    features: new Set<string>(['shader-f16']),
    requestDevice,
  };
  const requestAdapter =
      jasmine.createSpy('requestAdapter').and.resolveTo(fakeAdapter);
  return {gpu: {requestAdapter}, requestAdapter, requestDevice};
}

function createFakeWasm(): LiteRtLmWasm {
  return {
    setupLogging: jasmine.createSpy('setupLogging'),
    preinitializedWebGPUDevice: undefined,
  } as unknown as LiteRtLmWasm;
}

describe('setupDefaultWebGpuDevice', () => {
  let originalGpuDescriptor: PropertyDescriptor|undefined;
  let hadPreviousGlobal: boolean;
  let previousGlobal: LiteRtLm|undefined;
  let getContextSpy: jasmine.Spy;

  beforeEach(() => {
    originalGpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');
    hadPreviousGlobal = hasGlobalLiteRtLm();
    previousGlobal = hadPreviousGlobal ? getGlobalLiteRtLm() : undefined;

    const fakeGl = {
      RENDERER: 0x1F01,
      getParameter: (pname: number) =>
          pname === 0x1F01 ? ANGLE_M5_MAX_RENDERER : null,
      getExtension: () => null,
    };
    getContextSpy =
        spyOn(OffscreenCanvas.prototype, 'getContext')
            .and.returnValue(fakeGl as unknown as WebGL2RenderingContext);
  });

  afterEach(() => {
    if (originalGpuDescriptor) {
      Object.defineProperty(navigator, 'gpu', originalGpuDescriptor);
    } else {
      delete (navigator as unknown as {gpu?: unknown}).gpu;
    }
    setGlobalLiteRtLm(hadPreviousGlobal ? previousGlobal : undefined);
  });

  it('identifies a bucketed Apple GPU with WebGL by default', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});
    const liteRtLm = new LiteRtLm(createFakeWasm() as unknown as WasmModule);

    await liteRtLm.setupDefaultWebGpuDevice();

    const preinitialized = liteRtLm.liteRtLmWasm.preinitializedWebGPUDevice!;
    expect(preinitialized).toBe(device);
    expect(preinitialized.adapterInfo.device).toBe('Apple M5 Max');
    expect(preinitialized.adapterInfo.description).toBe('Apple M5 Max');
    expect(preinitialized.adapterInfo.vendor).toBe('apple');
    expect(preinitialized.adapterInfo.architecture).toBe('metal-3');
  });

  it('skips WebGL probing when identifyGpuWithWebGl is false', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});
    const liteRtLm = new LiteRtLm(createFakeWasm() as unknown as WasmModule);

    await liteRtLm.setupDefaultWebGpuDevice({identifyGpuWithWebGl: false});

    const preinitialized = liteRtLm.liteRtLmWasm.preinitializedWebGPUDevice!;
    expect(preinitialized).toBe(device);
    expect(preinitialized.adapterInfo.device).toBe('');
    expect(preinitialized.adapterInfo.description).toBe('');
    expect(getContextSpy).not.toHaveBeenCalled();
  });

  it('forwards WebGpuDeviceOptions from the top-level helper', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});
    const liteRtLm = new LiteRtLm(createFakeWasm() as unknown as WasmModule);
    setGlobalLiteRtLm(liteRtLm);

    await setupDefaultWebGpuDevice({identifyGpuWithWebGl: false});

    expect(liteRtLm.liteRtLmWasm.preinitializedWebGPUDevice).toBe(device);
    expect(device.adapterInfo.device).toBe('');
    expect(getContextSpy).not.toHaveBeenCalled();
  });

  it('does not recreate the device once preinitializedWebGPUDevice is set',
     async () => {
       const device = createFakeDevice();
       const {gpu, requestAdapter} = createFakeGpu(device);
       Object.defineProperty(
           navigator, 'gpu', {value: gpu, configurable: true});
       const liteRtLm = new LiteRtLm(createFakeWasm() as unknown as WasmModule);

       await liteRtLm.setupDefaultWebGpuDevice({identifyGpuWithWebGl: false});
       await liteRtLm.setupDefaultWebGpuDevice({identifyGpuWithWebGl: true});

       expect(requestAdapter).toHaveBeenCalledTimes(1);
       expect(device.adapterInfo.device).toBe('');
     });
});
