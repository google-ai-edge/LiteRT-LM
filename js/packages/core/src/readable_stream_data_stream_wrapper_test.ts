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

// taze: BigInt from //third_party/javascript/node_modules/typescript:es2020.bigint

import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';

describe('ReadableStreamDataStreamWrapper', () => {
  let wasmHeap: {heap: Uint8Array};

  beforeEach(() => {
    wasmHeap = {heap: new Uint8Array(2 * 1024 * 1024)};  // 2MB heap
  });

  function createWrapper(
      data: Uint8Array, bytesPerSlice: number,
      options: {shouldError?: boolean, disableByob?: boolean} = {}) {
    let position = 0;
    const underlyingSource: UnderlyingSource = {
      type: options.disableByob ? undefined : 'bytes',
      pull(controller: any) {
        if (options.shouldError) {
          controller.error(new Error('Simulated stream error'));
          return;
        }

        if (position >= data.length) {
          controller.close();
          if (controller.byobRequest) {
            controller.byobRequest.respond(0);
          }
          return;
        }

        const byobRequest = controller.byobRequest;
        if (byobRequest) {
          const view = byobRequest.view;
          if (view) {
            const copyLen = Math.min(view.byteLength, data.length - position);
            if (copyLen > 0) {
              const dest =
                  new Uint8Array(view.buffer, view.byteOffset, copyLen);
              dest.set(data.subarray(position, position + copyLen));
              position += copyLen;
              byobRequest.respond(copyLen);
            }
          }
        } else {
          controller.enqueue(
              data.subarray(position) as ArrayBufferView<ArrayBuffer>);
          position = data.length;
        }
      }
    };
    const stream = new ReadableStream<Uint8Array>(underlyingSource);
    return new ReadableStreamDataStreamWrapper(
        stream, () => wasmHeap.heap, bytesPerSlice);
  }

  for (const disableByob of [false, true]) {
    describe(`(disableByob=${disableByob})`, () => {
      function createTestWrapper(
          data: Uint8Array, bytesPerSlice: number,
          options: {shouldError?: boolean} = {}) {
        return createWrapper(data, bytesPerSlice, {...options, disableByob});
      }

      it('readAndPreserve across slice boundaries', async () => {
        const data = new Uint8Array(20).fill(0).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
        const ptr = 100;

        // Read 12 bytes starting at offset 4 (spans slice 0 and 1)
        const status = await wrapper.readAndPreserve(ptr, 4, 12);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr, ptr + 12))
            .toEqual(data.subarray(4, 16));
      });

      it('readAndDiscard across slice boundaries', async () => {
        const data = new Uint8Array(20).fill(0).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
        const ptr = 100;

        const status = await wrapper.readAndDiscard(ptr, 4, 12);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr, ptr + 12))
            .toEqual(data.subarray(4, 16));

        // Subsequent read of already discarded data should fail
        const status2 = await wrapper.readAndPreserve(ptr + 20, 4, 4);
        expect(status2.error).toBeDefined();
        expect(status2.error?.message)
            .toContain('overlaps discarded regions: [[4, 8]]');
      });

      it('discards non-contiguous regions then the whole range', async () => {
        const data = new Uint8Array(20).fill(0).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 20);

        // Discard [0, 5], [10, 15], then [0, 20]
        await wrapper.discard(0, 5);
        await wrapper.discard(10, 5);
        await wrapper.discard(0, 20);

        // This checks if the slice is fully discarded.
        // If we try to read, it should throw because it's discarded.
        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error).toBeDefined();
        expect(status.error?.message)
            .toContain('overlaps discarded regions: [[0, 20]]');
      });

      it('handles unaligned EOF correctly', async () => {
        const data = new Uint8Array(12).fill(7);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
        const ptr = 0;

        // First slice is full (8 bytes), second is partial (4 bytes)
        const status = await wrapper.readAndPreserve(ptr, 0, 12);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(0, 12)).toEqual(data);
      });

      it('fails on early EOF', async () => {
        const data = new Uint8Array(10).fill(1);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 20);

        // Request 15 bytes but only 10 available
        const status = await wrapper.readAndPreserve(0, 0, 15);
        expect(status.error).toBeDefined();
        expect(status.error?.message).toContain('Expected 5 more bytes');
      });

      it('fails when BigInt exceeds MAX_SAFE_INTEGER', async () => {
        const data = new Uint8Array(10);
        const wrapper = createTestWrapper(data, 10);
        const largeValue = BigInt(Number.MAX_SAFE_INTEGER) + 1n;

        const status = await wrapper.readAndPreserve(0, largeValue, 1n);
        expect(status.error).toBeDefined();
        expect(status.error?.message).toContain('too large');
      });

      it('fails on stream error', async () => {
        const data = new Uint8Array(10);
        const wrapper = createTestWrapper(data, 10, {shouldError: true});

        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error).toBeDefined();
        expect(status.error?.message).toContain('Simulated stream error');
      });

      it('handles zero-length read correctly', async () => {
        const data = new Uint8Array(10).fill(5);
        const wrapper = createTestWrapper(data, 10);

        const status = await wrapper.readAndPreserve(0, 0, 0);
        expect(status.error).toBeUndefined();
      });

      it('fails on overlapping discards', async () => {
        const data = new Uint8Array(10).fill(1);
        const wrapper = createTestWrapper(data, 10);

        // Discard [0, 2], then [4, 6], then [2, 4]. [0, 6] should be merged.
        await wrapper.discard(0, 2);
        await wrapper.discard(4, 2);
        await wrapper.discard(2, 2);

        // Then discard [6, 10]. The whole slice [0, 10] should be discarded.
        await wrapper.discard(6, 4);

        const status = await wrapper.readAndPreserve(0, 0, 1);
        expect(status.error).toBeDefined();
        expect(status.error?.message)
            .toContain('overlaps discarded regions: [[0, 10]]');
      });

      it('fails on overlapping reads', async () => {
        const data = new Uint8Array(100).fill(1);
        const wrapper = createTestWrapper(data, 50);

        // Start a read but don't await immediately
        const p1 = wrapper.readAndPreserve(0, 0, 10);
        const p2 = wrapper.readAndPreserve(20, 0, 10);

        const [res1, res2] = await Promise.all([p1, p2]);

        // One of them (likely second) should fail because the reader is locked
        // or because they are both trying to pull from the same stream/reader.
        expect(res1.error || res2.error).toBeDefined();
      });

      it('can discard an already partially discarded region', async () => {
        const data = new Uint8Array(10).fill(1);
        const wrapper = createTestWrapper(data, 10);
        await wrapper.discard(2, 5);
        const status = await wrapper.discard(3, 7);
        expect(status.error).toBeUndefined();
        const status2 = await wrapper.readAndPreserve(0, 6, 4);
        expect(status2.error).toBeDefined();
        expect(status2.error?.message)
            .toContain('overlaps discarded regions: [[2, 10]]');
      });

      it('does not crash when wasm heap is detached and replaced', async () => {
        const data = new Uint8Array(10).map((_, i) => i);
        const wrapper = createTestWrapper(data, 10);
        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error).toBeUndefined();

        // Detach and replace the wasm heap.
        // This happens when Emscripten resizes the heap.
        const newHeapBuffer = (wasmHeap.heap.buffer as unknown as {
                                transfer(): ArrayBuffer
                              }).transfer();
        wasmHeap.heap = new Uint8Array(newHeapBuffer);

        const status2 = await wrapper.readAndPreserve(5, 5, 5);
        expect(status2.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(0, 10)).toEqual(data);
      });
    });
  }
});