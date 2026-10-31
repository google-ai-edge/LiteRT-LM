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

// Tests for behavior shared by all data stream wrappers, which must all follow
// the contract of the C++ `DataStream` they back.

import {BlobDataStreamWrapper} from './blob_data_stream_wrapper.js';
import {ReadableStreamDataStreamResult, ReadableStreamDataStreamWrapper, ReadableStreamDataStreamWrapperInterface} from './readable_stream_data_stream_wrapper.js';

/**
 * Returns the pending BYOB request of a byte stream's controller, or null for
 * a default stream.
 */
function getByobRequest(controller: ReadableStreamController<Uint8Array>):
    ReadableStreamBYOBRequest|null {
  return controller instanceof ReadableByteStreamController ?
      controller.byobRequest :
      null;
}

/** Creates a byte stream of `data`, optionally supporting BYOB readers. */
function createStream(
    data: Uint8Array, byob: boolean): ReadableStream<Uint8Array> {
  let position = 0;
  return new ReadableStream<Uint8Array>({
    type: byob ? 'bytes' : undefined,
    pull(controller: ReadableStreamController<Uint8Array>) {
      if (position >= data.length) {
        controller.close();
        getByobRequest(controller)?.respond(0);
        return;
      }
      const byobRequest = getByobRequest(controller);
      if (byobRequest?.view) {
        const view = byobRequest.view;
        const length = Math.min(view.byteLength, data.length - position);
        new Uint8Array(view.buffer, view.byteOffset, length)
            .set(data.subarray(position, position + length));
        position += length;
        byobRequest.respond(length);
      } else {
        controller.enqueue(data.slice(position));
        position = data.length;
      }
    },
  } as UnderlyingSource<Uint8Array>);
}

const IMPLEMENTATIONS: Array<{
  name: string,
  create: (data: Uint8Array<ArrayBuffer>, getWasmHeap: () => Uint8Array) =>
      ReadableStreamDataStreamWrapperInterface,
}> = [
  {
    name: 'ReadableStreamDataStreamWrapper with a BYOB reader',
    create: (data, getWasmHeap) => new ReadableStreamDataStreamWrapper(
        createStream(data, /* byob= */ true), getWasmHeap,
        /* bytesPerSlice= */ 8),
  },
  {
    name: 'ReadableStreamDataStreamWrapper with a default reader',
    create: (data, getWasmHeap) => new ReadableStreamDataStreamWrapper(
        createStream(data, /* byob= */ false), getWasmHeap,
        /* bytesPerSlice= */ 8),
  },
  {
    name: 'BlobDataStreamWrapper',
    create: (data, getWasmHeap) => new BlobDataStreamWrapper(
        new Blob([data]), getWasmHeap, {maxBytesPerRead: 7}),
  },
];

for (const {name, create} of IMPLEMENTATIONS) {
  describe(`${name} (DataStream contract)`, () => {
    let wasmHeap: Uint8Array;
    const ptr = 100;

    beforeEach(() => {
      wasmHeap = new Uint8Array(4096);
    });

    function createWrapper(data: Uint8Array<ArrayBuffer>) {
      return create(data, () => wasmHeap);
    }

    function sequence(length: number): Uint8Array<ArrayBuffer> {
      return new Uint8Array(length).map((_, i) => i);
    }

    it('reads across chunk boundaries', async () => {
      const data = sequence(20);
      const wrapper = createWrapper(data);

      expect((await wrapper.readAndPreserve(ptr, 4, 12)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 12)).toEqual(data.subarray(4, 16));
    });

    it('reads up to exactly the end', async () => {
      const data = sequence(12);
      const wrapper = createWrapper(data);

      expect((await wrapper.readAndPreserve(ptr, 0, 12)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 12)).toEqual(data);
    });

    it('reads in any order', async () => {
      const data = sequence(20);
      const wrapper = createWrapper(data);

      expect((await wrapper.readAndPreserve(ptr, 10, 10)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 10)).toEqual(data.subarray(10, 20));
      expect((await wrapper.readAndDiscard(ptr, 0, 5)).error).toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 5)).toEqual(data.subarray(0, 5));
      expect((await wrapper.readAndDiscard(ptr, 5, 15)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 15)).toEqual(data.subarray(5, 20));
    });

    it('fails reads past the end', async () => {
      const status = await createWrapper(sequence(10)).readAndPreserve(0, 0, 15);
      expect(status.error).toBeDefined();
    });

    it('fails reads that start at the end', async () => {
      const wrapper = createWrapper(sequence(10));
      expect((await wrapper.readAndDiscard(ptr, 0, 10)).error).toBeUndefined();

      const status = await wrapper.readAndPreserve(ptr, 10, 5);
      expect(status.error).toBeDefined();
    });

    it('handles zero-length reads', async () => {
      expect((await createWrapper(sequence(10)).readAndPreserve(0, 5, 0)).error)
          .toBeUndefined();
      expect((await createWrapper(sequence(0)).readAndDiscard(0, 0, 0)).error)
          .toBeUndefined();
    });

    it('fails to read data discarded after it was read', async () => {
      const data = sequence(20);
      const wrapper = createWrapper(data);

      expect((await wrapper.readAndDiscard(ptr, 4, 12)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 12)).toEqual(data.subarray(4, 16));

      const status = await wrapper.readAndPreserve(ptr, 4, 4);
      expect(status.error?.message)
          .toContain('overlaps discarded regions: [[4, 16]]');
    });

    it('fails to read data discarded before it was read', async () => {
      const wrapper = createWrapper(sequence(40));
      expect(wrapper.discard(10, 10).error).toBeUndefined();

      const status = await wrapper.readAndPreserve(ptr, 12, 4);
      expect(status.error?.message)
          .toContain('overlaps discarded regions: [[10, 20]]');
    });

    it('reads around discarded data', async () => {
      const data = sequence(100);
      const wrapper = createWrapper(data);
      expect(wrapper.discard(10, 50).error).toBeUndefined();

      expect((await wrapper.readAndPreserve(ptr, 70, 10)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 10)).toEqual(data.subarray(70, 80));
      expect((await wrapper.readAndPreserve(ptr, 0, 10)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 10)).toEqual(data.subarray(0, 10));
      expect((await wrapper.readAndPreserve(ptr, 60, 10)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 10)).toEqual(data.subarray(60, 70));
      expect((await wrapper.readAndDiscard(ptr, 80, 20)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 20)).toEqual(data.subarray(80, 100));
    });

    it('merges overlapping discards', async () => {
      const wrapper = createWrapper(sequence(20));
      expect(wrapper.discard(0, 2).error).toBeUndefined();
      expect(wrapper.discard(4, 2).error).toBeUndefined();
      expect(wrapper.discard(2, 2).error).toBeUndefined();
      expect(wrapper.discard(5, 5).error).toBeUndefined();
      // Discarding an already discarded region is allowed.
      expect(wrapper.discard(3, 4).error).toBeUndefined();

      const status = await wrapper.readAndPreserve(ptr, 9, 2);
      expect(status.error?.message)
          .toContain('overlaps discarded regions: [[0, 10]]');
    });

    it('ignores zero and negative discards', async () => {
      const data = sequence(10);
      const wrapper = createWrapper(data);
      expect(wrapper.discard(0, 0).error).toBeUndefined();
      expect(wrapper.discard(0, -5).error).toBeUndefined();

      expect((await wrapper.readAndPreserve(ptr, 0, 10)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 10)).toEqual(data);
    });

    it('rejects concurrent reads', async () => {
      const wrapper = createWrapper(sequence(100));

      const [status1, status2] = await Promise.all([
        wrapper.readAndPreserve(0, 0, 10),
        wrapper.readAndPreserve(20, 0, 10),
      ]);
      expect(status1.error).toBeUndefined();
      expect(status2.error?.message)
          .toContain('Concurrent reads are not supported');
    });

    it('gets the wasm heap for each read', async () => {
      const data = sequence(10);
      const wrapper = createWrapper(data);
      expect((await wrapper.readAndPreserve(0, 0, 5)).error).toBeUndefined();

      // Emscripten detaches and replaces the heap when it grows.
      wasmHeap = new Uint8Array(
          (wasmHeap.buffer as unknown as {transfer(): ArrayBuffer})
              .transfer());

      expect((await wrapper.readAndPreserve(5, 5, 5)).error).toBeUndefined();
      expect(wasmHeap.subarray(0, 10)).toEqual(data);
    });

    it('accepts BigInt arguments', async () => {
      const data = sequence(20);
      const wrapper = createWrapper(data);

      const preserveStatus: ReadableStreamDataStreamResult =
          await wrapper.readAndPreserve(ptr, 4n, 12n);
      expect(preserveStatus.error).toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 12)).toEqual(data.subarray(4, 16));
      expect(wrapper.discard(0n, 4n).error).toBeUndefined();
      expect((await wrapper.readAndDiscard(ptr, 16n, 4n)).error)
          .toBeUndefined();
      expect(wasmHeap.subarray(ptr, ptr + 4)).toEqual(data.subarray(16, 20));
    });

    it('fails when BigInt exceeds MAX_SAFE_INTEGER', async () => {
      const wrapper = createWrapper(sequence(10));
      const largeValue = BigInt(Number.MAX_SAFE_INTEGER) + 1n;

      expect((await wrapper.readAndPreserve(0, largeValue, 1n)).error?.message)
          .toContain('too large');
      expect((await wrapper.readAndDiscard(0, largeValue, 1n)).error?.message)
          .toContain('too large');
      expect(wrapper.discard(largeValue, 1n).error?.message)
          .toContain('too large');
    });

    it('matches a reference model for random operations', async () => {
      for (let seed = 1; seed <= 40; seed++) {
        const random = seededRandom(seed);
        const randomInt = (n: number) => Math.floor(random() * n);
        const length = 50 + randomInt(200);
        const data = new Uint8Array(length).map(() => randomInt(256));
        const wrapper = createWrapper(data);

        // The reference model.
        const discarded = new Array<boolean>(length).fill(false);
        let furthestRead = 0;

        for (let op = 0; op < 80; op++) {
          // Mostly operate near the furthest read, like model loading.
          const offset = Math.max(
              0, Math.min(length - 1, furthestRead - 30 + randomInt(80)));
          const count = 1 + randomInt(Math.min(40, length - offset));
          const kind = randomInt(3);
          const context = `seed ${seed}, op ${op}: ${
              ['readAndDiscard', 'readAndPreserve', 'discard'][kind]}(${
              offset}, ${count})`;

          if (kind === 2) {
            expect(wrapper.discard(offset, count).error)
                .withContext(context)
                .toBeUndefined();
            discarded.fill(true, offset, offset + count);
            continue;
          }

          wasmHeap.fill(0, ptr, ptr + count);
          const status = kind === 0 ?
              await wrapper.readAndDiscard(ptr, offset, count) :
              await wrapper.readAndPreserve(ptr, offset, count);
          if (discarded.slice(offset, offset + count).includes(true)) {
            expect(status.error?.message)
                .withContext(context)
                .toContain('overlaps discarded regions');
            continue;
          }
          expect(status.error).withContext(context).toBeUndefined();
          expect(wasmHeap.subarray(ptr, ptr + count))
              .withContext(context)
              .toEqual(data.subarray(offset, offset + count));
          if (kind === 0) {
            discarded.fill(true, offset, offset + count);
          }
          furthestRead = Math.max(furthestRead, offset + count);
        }
      }
    });
  });
}

/** Returns a deterministic pseudo-random number generator (mulberry32). */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
