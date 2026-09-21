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

import {modelToStream} from './stream_utils.js';

describe('stream_utils', () => {
  it('converts a Blob to a ReadableStream', async () => {
    const blob = new Blob([new Uint8Array([1, 2, 3, 4])]);
    const stream = await modelToStream(blob);
    expect(stream).toBeDefined();
    const reader = stream.getReader();
    const result = await reader.read();
    expect(result.value).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  it('returns an existing ReadableStream unchanged', async () => {
    const existingStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([9, 8, 7]));
        controller.close();
      },
    });
    const stream = await modelToStream(existingStream);
    expect(stream).toBe(existingStream);
  });
});
