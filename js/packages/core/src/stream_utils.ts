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

import {LiteRtLmWasm} from './wasm_binding_types.js';

/**
 * Converts a model source (URL string, Blob, or ReadableStream) into a ReadableStream.
 */
export async function modelToStream(
    model: string | Blob | ReadableStream<Uint8Array>):
    Promise<ReadableStream<Uint8Array>> {
  if (model instanceof ReadableStream) {
    return model;
  }
  if (model instanceof Blob) {
    return model.stream();
  }

  const modelUrl = model;
  const response = await fetch(modelUrl, {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch model file from ${modelUrl}`);
  }
  return response.body!;
}

/**
 * Sets up the callback that streams model weights directly into WebGPU buffers.
 */
export function setupStreamWeightsCallback(wasm: LiteRtLmWasm): void {
  const gpuDevice = wasm.preinitializedWebGPUDevice;
  if (!gpuDevice) {
    throw new Error('WebGPU device not initialized');
  }

  wasm.registerStreamWeightsCallback(async (
      tflIds: Int32Array,
      wgpuBufferIds: Uint32Array,
      offsets: Float64Array,
      lengths: Float64Array,
  ) => {
    const requests = [];
    if (tflIds.length !== wgpuBufferIds.length) {
      throw new Error(
          `Stream weights callback received arrays of different lengths: ` +
          `tflIds=${tflIds.length}, wgpuBufferIds=${wgpuBufferIds.length}`);
    }
    for (let i = 0; i < tflIds.length; i++) {
      requests.push({
        id: tflIds[i],
        wgpuBufferId: wgpuBufferIds[i],
        offset: offsets[i],
        length: lengths[i],
      });
    }
    requests.sort((a, b) => a.offset - b.offset);

    const CHUNK_SIZE = 4 * 1024 * 1024;
    const tempPtr = wasm._malloc(CHUNK_SIZE);
    if (tempPtr === 0) {
      throw new Error('Failed to allocate Wasm memory for streaming weights');
    }
    try {
      for (const req of requests) {
        const gpuBuffer = wasm.WebGPU.getJsObject(req.wgpuBufferId) as GPUBuffer;
        if (!gpuBuffer) {
          throw new Error(`Failed to find GPUBuffer for ID: ${req.wgpuBufferId}`);
        }
        const modelType = wasm.getCurrentlyCompilingModel();
        let bytesUploaded = 0;
        while (bytesUploaded < req.length) {
          const chunkSize = Math.min(CHUNK_SIZE, req.length - bytesUploaded);
          await wasm.readStoredWeights(
              modelType, req.offset + bytesUploaded, chunkSize, tempPtr);

          let chunkData = new Uint8Array(wasm.HEAPU8.buffer, tempPtr, chunkSize);
          if (chunkData.byteLength % 4 !== 0 &&
              bytesUploaded + chunkSize === req.length) {
            const paddedSize = (chunkData.byteLength + 3) & ~3;
            const paddedData = new Uint8Array(paddedSize);
            paddedData.set(chunkData);
            chunkData = paddedData;
          }

          gpuDevice.queue.writeBuffer(
              gpuBuffer, bytesUploaded,
              chunkData as unknown as BufferSource);
          bytesUploaded += chunkSize;
        }
      }
    } finally {
      wasm._free(tempPtr);
    }
  });
}
