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

import {BlobDataStreamWrapper} from './blob_data_stream_wrapper.js';
import {Cleanup} from './cleanup.js';
import {resolveModelUrl} from './model-cache/model_source.js';
import {ResolvedModel} from './model_input.js';
import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';
import {LiteRtLmWasm, ModelAssets} from './wasm_binding_types.js';

/**
 * Converts a model source (URL string, `ModelSource`, Blob, or ReadableStream)
 * into a ReadableStream. `hf://` URIs must be resolved first, with
 * `resolveModelInput()`.
 */
export async function modelToStream(model: ResolvedModel):
    Promise<ReadableStream<Uint8Array>> {
  if (model instanceof ReadableStream) {
    return model;
  }
  if (model instanceof Blob) {
    return model.stream();
  }

  const source = typeof model === 'string' ? {url: model} : model;
  const init: RequestInit = {credentials: 'same-origin'};
  if (source.accessToken) {
    init.headers = {'Authorization': `Bearer ${source.accessToken}`};
  }
  const response = await fetch(resolveModelUrl(source.url), init);
  if (!response.ok) {
    throw new Error(`Failed to fetch model file from ${source.url}`);
  }
  return response.body!;
}

/**
 * Creates a streaming Wasm `ModelAssets` from a model source (URL string,
 * `ModelSource`, Blob, or ReadableStream) and registers cleanup callbacks for
 * the underlying `ReadableStreamDataStream` and `ModelAssets`.
 */
export async function createStreamingModelAssets(
    model: ResolvedModel,
    wasm: LiteRtLmWasm,
    cleanup: Cleanup,
    ): Promise<{modelAssets: ModelAssets; cleanupModelAssets: () => void}> {
  const streamWrapper = model instanceof Blob ?
      new BlobDataStreamWrapper(model, () => wasm.HEAPU8) :
      new ReadableStreamDataStreamWrapper(
          await modelToStream(model), () => wasm.HEAPU8);
  const dataStream = wasm.ReadableStreamDataStream.create(streamWrapper);
  cleanup.add(() => {
    dataStream.delete();
    // C++ also closes the wrapper when it's done with the stream, but it may
    // still be holding a reference here.
    streamWrapper.close();
  });
  const modelAssets = wasm.ModelAssets.createStreaming(dataStream);
  const cleanupModelAssets = cleanup.add(() => {
    modelAssets.delete();
  });
  return {modelAssets, cleanupModelAssets};
}
