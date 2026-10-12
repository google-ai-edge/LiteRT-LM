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

import {Cleanup} from './cleanup.js';
import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';
import {LiteRtLmWasm, ModelAssets} from './wasm_binding_types.js';

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
 * Creates a streaming Wasm `ModelAssets` from a model source (URL string, Blob,
 * or ReadableStream) and registers cleanup callbacks for the underlying
 * `ReadableStreamDataStream` and `ModelAssets`.
 */
export async function createStreamingModelAssets(
    model: string | Blob | ReadableStream<Uint8Array>,
    wasm: LiteRtLmWasm,
    cleanup: Cleanup,
): Promise<{modelAssets: ModelAssets; cleanupModelAssets: () => void}> {
  const modelStream = await modelToStream(model);
  const streamWrapper =
      new ReadableStreamDataStreamWrapper(modelStream, () => wasm.HEAPU8);
  const dataStream = wasm.ReadableStreamDataStream.create(streamWrapper);
  cleanup.add(() => {
    dataStream.delete();
  });
  const modelAssets = wasm.ModelAssets.createStreaming(dataStream);
  const cleanupModelAssets = cleanup.add(() => {
    modelAssets.delete();
  });
  return {modelAssets, cleanupModelAssets};
}

