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

// Tests for behavior specific to ReadableStreamDataStreamWrapper. Behavior
// shared with BlobDataStreamWrapper is tested in data_stream_wrappers_test.ts.

import {ReadableStreamDataStreamWrapper} from './readable_stream_data_stream_wrapper.js';

describe('ReadableStreamDataStreamWrapper', () => {
  let wasmHeap: {heap: Uint8Array};

  beforeEach(() => {
    wasmHeap = {heap: new Uint8Array(2 * 1024 * 1024)};  // 2MB heap
  });

  /**
   * Returns the pending BYOB request of a byte stream's controller, or null
   * for a default stream.
   */
  function getByobRequest(controller: ReadableStreamController<Uint8Array>):
      ReadableStreamBYOBRequest|null {
    return controller instanceof ReadableByteStreamController ?
        controller.byobRequest :
        null;
  }

  function createWrapper(
      data: Uint8Array, bytesPerSlice: number, options: {
        shouldError?: boolean,
        disableByob?: boolean,
        maxCachedBytes?: number,
      } = {}) {
    let position = 0;
    const underlyingSource: UnderlyingSource<Uint8Array> = {
      type: options.disableByob ? undefined : 'bytes',
      pull(controller: ReadableStreamController<Uint8Array>) {
        if (options.shouldError) {
          controller.error(new Error('Simulated stream error'));
          return;
        }

        if (position >= data.length) {
          controller.close();
          getByobRequest(controller)?.respond(0);
          return;
        }

        const byobRequest = getByobRequest(controller);
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
              data.subarray(position) as Uint8Array<ArrayBuffer>);
          position = data.length;
        }
      }
    };
    const stream = new ReadableStream<Uint8Array>(underlyingSource);
    return new ReadableStreamDataStreamWrapper(
        stream, () => wasmHeap.heap, bytesPerSlice,
        {maxCachedBytes: options.maxCachedBytes});
  }

  for (const disableByob of [false, true]) {
    describe(`(disableByob=${disableByob})`, () => {
      function createTestWrapper(
          data: Uint8Array, bytesPerSlice: number,
          options: {shouldError?: boolean, maxCachedBytes?: number} = {}) {
        return createWrapper(data, bytesPerSlice, {...options, disableByob});
      }

      it('fails on stream error', async () => {
        const data = new Uint8Array(10);
        const wrapper = createTestWrapper(data, 10, {shouldError: true});

        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error).toBeDefined();
        expect(status.error?.message).toContain('Simulated stream error');
      });

      it('caches undiscarded data when reading past discarded data',
         async () => {
           const data = new Uint8Array(30).fill(0).map((_, i) => i);
           const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
           const ptr = 100;

           // Discard unfetched data between two regions that are read later.
           expect(wrapper.discard(4, 14).error).toBeUndefined();
           const status = await wrapper.readAndPreserve(ptr, 20, 6);
           expect(status.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr, ptr + 6))
               .toEqual(data.subarray(20, 26));

           // Data before and after the discarded region was cached.
           const status2 = await wrapper.readAndPreserve(ptr, 0, 4);
           expect(status2.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr, ptr + 4))
               .toEqual(data.subarray(0, 4));
           const status3 = await wrapper.readAndPreserve(ptr, 18, 2);
           expect(status3.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr, ptr + 2))
               .toEqual(data.subarray(18, 20));
         });

      it('readAndDiscard fails gracefully on stream error', async () => {
        const data = new Uint8Array(10);
        const wrapper = createTestWrapper(data, 10, {shouldError: true});

        const status = await wrapper.readAndDiscard(0, 0, 5);
        expect(status.error).toBeDefined();
        expect(status.error?.message).toContain('Simulated stream error');
      });

      it('reads past a region discarded before it was read', async () => {
        const data = new Uint8Array(40).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 10);
        const ptr = 100;

        // Discard slice 1 entirely before it has been read from the stream.
        expect(wrapper.discard(10, 10).error).toBeUndefined();

        const status = await wrapper.readAndDiscard(ptr, 25, 5);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr, ptr + 5))
            .toEqual(data.subarray(25, 30));

        // Data before and after the discarded region was cached.
        const status2 = await wrapper.readAndPreserve(ptr + 10, 0, 10);
        expect(status2.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr + 10, ptr + 20))
            .toEqual(data.subarray(0, 10));
        const status3 = await wrapper.readAndPreserve(ptr + 20, 20, 5);
        expect(status3.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr + 20, ptr + 25))
            .toEqual(data.subarray(20, 25));

        // The discarded data can not be read.
        const status4 = await wrapper.readAndPreserve(ptr, 15, 1);
        expect(status4.error?.message)
            .toContain('overlaps discarded regions: [[10, 20]]');
      });

      it('reads past a multi-slice region discarded before it was read',
         async () => {
           const data = new Uint8Array(100).map((_, i) => i);
           const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
           const ptr = 100;

           // Discard [10, 60), which partially covers slices 1 and 7 and fully
           // covers slices 2 to 6.
           expect(wrapper.discard(10, 50).error).toBeUndefined();

           const status = await wrapper.readAndPreserve(ptr, 70, 10);
           expect(status.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr, ptr + 10))
               .toEqual(data.subarray(70, 80));

           // Only the slices with undiscarded data were cached: 0 and 1 for
           // [0, 10), and 7, 8 and 9 for [60, 80).
           expect(wrapper.getStats().cachedBytes).toBe(5 * 8);

           const status2 = await wrapper.readAndPreserve(ptr + 10, 0, 10);
           expect(status2.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr + 10, ptr + 20))
               .toEqual(data.subarray(0, 10));
           const status3 = await wrapper.readAndPreserve(ptr + 20, 60, 10);
           expect(status3.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr + 20, ptr + 30))
               .toEqual(data.subarray(60, 70));
           const status4 = await wrapper.readAndDiscard(ptr + 30, 80, 20);
           expect(status4.error).toBeUndefined();
           expect(wasmHeap.heap.subarray(ptr + 30, ptr + 50))
               .toEqual(data.subarray(80, 100));
         });

      it('reads past a partially discarded slice', async () => {
        const data = new Uint8Array(40).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 20);
        const ptr = 100;

        // Discard [5, 10) in slice 0 before it has been read.
        expect(wrapper.discard(5, 5).error).toBeUndefined();

        const status = await wrapper.readAndPreserve(ptr, 12, 4);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr, ptr + 4))
            .toEqual(data.subarray(12, 16));

        // The undiscarded data in the same slice was cached.
        const status2 = await wrapper.readAndPreserve(ptr + 10, 0, 5);
        expect(status2.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr + 10, ptr + 15))
            .toEqual(data.subarray(0, 5));
        const status3 = await wrapper.readAndPreserve(ptr + 20, 10, 2);
        expect(status3.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr + 20, ptr + 22))
            .toEqual(data.subarray(10, 12));

        const status4 = await wrapper.readAndPreserve(ptr + 30, 5, 5);
        expect(status4.error?.message)
            .toContain('overlaps discarded regions: [[5, 10]]');
      });

      it('reads past unfetched data that was discarded', async () => {
        const data = new Uint8Array(40).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
        const ptr = 100;

        // Read part of a section, then discard the whole section, including
        // its unread end, which spans partial and whole slices.
        expect((await wrapper.readAndDiscard(ptr, 0, 6)).error)
            .toBeUndefined();
        expect(wrapper.discard(0, 27).error).toBeUndefined();

        const status = await wrapper.readAndDiscard(ptr, 27, 10);
        expect(status.error).toBeUndefined();
        expect(wasmHeap.heap.subarray(ptr, ptr + 10))
            .toEqual(data.subarray(27, 37));
        expect(wrapper.getStats().peakCachedBytes).toBe(0);

        const status2 = await wrapper.readAndPreserve(ptr, 20, 4);
        expect(status2.error?.message)
            .toContain('overlaps discarded regions: [[0, 37]]');
      });

      it('fails to read a discarded region ahead of the stream', async () => {
        const data = new Uint8Array(40).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 10);

        expect(wrapper.discard(10, 10).error).toBeUndefined();

        const status = await wrapper.readAndPreserve(0, 12, 4);
        expect(status.error?.message)
            .toContain('overlaps discarded regions: [[10, 20]]');
        // The failed read did not consume the stream.
        expect(wrapper.getStats().streamPosition).toBe(0);
      });

      it('caches nothing when reading in order with readAndDiscard',
         async () => {
           const data = new Uint8Array(100).map((_, i) => i);
           const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 8);
           const ptr = 100;

           for (let offset = 0; offset < 100; offset += 7) {
             const count = Math.min(7, 100 - offset);
             const status = await wrapper.readAndDiscard(ptr, offset, count);
             expect(status.error).toBeUndefined();
             expect(wasmHeap.heap.subarray(ptr, ptr + count))
                 .toEqual(data.subarray(offset, offset + count));
           }
           expect(wrapper.getStats()).toEqual({
             streamPosition: 100,
             cachedBytes: 0,
             peakCachedBytes: 0,
           });
         });

      it('releases cached slices once they are discarded', async () => {
        const data = new Uint8Array(40).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 10);

        // Reading ahead caches slices 0 and 1.
        expect((await wrapper.readAndDiscard(0, 20, 5)).error)
            .toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(20);

        // Slice 0 is released once all of it is discarded.
        expect((await wrapper.readAndDiscard(0, 0, 5)).error)
            .toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(20);
        expect(wrapper.discard(5, 5).error).toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(10);

        expect((await wrapper.readAndDiscard(0, 10, 10)).error)
            .toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(0);
        expect(wrapper.getStats().peakCachedBytes).toBe(20);
      });

      it('fails reads that would cache more than maxCachedBytes', async () => {
        const data = new Uint8Array(100).map((_, i) => i);
        const wrapper = createTestWrapper(
            data, /* bytesPerSlice= */ 10, {maxCachedBytes: 30});

        // Caching 3 slices is allowed.
        expect((await wrapper.readAndDiscard(0, 30, 1)).error)
            .toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(30);

        const status = await wrapper.readAndPreserve(0, 50, 1);
        expect(status.error?.message).toContain('more than 30 allowed');
      });

      it('does not report success for reads past the end', async () => {
        const data = new Uint8Array(10).map((_, i) => i);
        const wrapper = createTestWrapper(data, /* bytesPerSlice= */ 20);

        expect((await wrapper.readAndPreserve(0, 0, 10)).error)
            .toBeUndefined();

        // The stream is exactly at its end, but it hasn't reported that yet.
        const status = await wrapper.readAndPreserve(0, 10, 5);
        expect(status.error?.message).toContain('Expected 5 more bytes');
        const status2 = await wrapper.readAndPreserve(0, 10, 5);
        expect(status2.error).toBeDefined();
      });

      it('fails all reads after the stream fails', async () => {
        const data = new Uint8Array(10).map((_, i) => i);
        const wrapper = createTestWrapper(data, 10, {shouldError: true});

        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error?.message).toContain('Simulated stream error');

        const status2 = await wrapper.readAndPreserve(0, 0, 5);
        expect(status2.error?.message)
            .toContain(
                'Reading from the stream previously failed: ' +
                'Simulated stream error');
      });

      it('cancels the stream and fails reads when closed', async () => {
        // An endless stream, so it is still open when the wrapper is closed.
        let cancelled = false;
        const stream = new ReadableStream<Uint8Array>({
          type: disableByob ? undefined : 'bytes',
          pull(controller: ReadableStreamController<Uint8Array>) {
            const byobRequest = getByobRequest(controller);
            if (byobRequest?.view) {
              byobRequest.respond(Math.min(byobRequest.view.byteLength, 5));
            } else {
              controller.enqueue(new Uint8Array(5));
            }
          },
          cancel() {
            cancelled = true;
          },
        } as UnderlyingSource<Uint8Array>);
        const wrapper = new ReadableStreamDataStreamWrapper(
            stream, () => wasmHeap.heap, /* bytesPerSlice= */ 10);
        expect((await wrapper.readAndPreserve(0, 0, 15)).error)
            .toBeUndefined();
        expect(wrapper.getStats().cachedBytes).toBe(20);

        wrapper.close();
        wrapper.close();  // Closing twice is fine.
        await Promise.resolve();

        expect(cancelled).toBeTrue();
        expect(wrapper.getStats().cachedBytes).toBe(0);
        const status = await wrapper.readAndPreserve(0, 0, 5);
        expect(status.error?.message).toContain('closed');
      });

      it('matches a reference model for random operations', async () => {
        for (let seed = 1; seed <= 40; seed++) {
          const random = seededRandom(seed);
          const randomInt = (n: number) => Math.floor(random() * n);
          const length = 50 + randomInt(200);
          const data = new Uint8Array(length).map(() => randomInt(256));
          const bytesPerSlice = 1 + randomInt(40);
          const wrapper = createTestWrapper(data, bytesPerSlice);
          const ptr = 1000;

          // The reference model.
          const discarded = new Array<boolean>(length).fill(false);
          // Slices that have had data cached in them.
          const cachedSlices = new Set<number>();
          let streamPosition = 0;

          for (let op = 0; op < 80; op++) {
            // Mostly operate near the stream position, like model loading.
            const offset = Math.max(
                0, Math.min(length - 1, streamPosition - 30 + randomInt(80)));
            const count = 1 + randomInt(Math.min(40, length - offset));
            const kind = randomInt(3);
            const context = `seed ${seed}, bytesPerSlice ${bytesPerSlice}, op ${
                op}: ${['readAndDiscard', 'readAndPreserve', 'discard'][kind]}(${
                offset}, ${count})`;

            if (kind === 2) {
              expect(wrapper.discard(offset, count).error)
                  .withContext(context)
                  .toBeUndefined();
              discarded.fill(true, offset, offset + count);
            } else {
              wasmHeap.heap.fill(0, ptr, ptr + count);
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
              expect(wasmHeap.heap.subarray(ptr, ptr + count))
                  .withContext(context)
                  .toEqual(data.subarray(offset, offset + count));
              // Bytes read from the stream are cached if they are skipped
              // over and not discarded, or if they are preserved.
              for (let i = streamPosition; i < offset + count; i++) {
                if (i < offset ? !discarded[i] : kind === 1) {
                  cachedSlices.add(Math.floor(i / bytesPerSlice));
                }
              }
              if (kind === 0) {
                discarded.fill(true, offset, offset + count);
              }
              streamPosition = Math.max(streamPosition, offset + count);
            }

            // A slice stays cached until all of it is discarded.
            let expectedCachedSlices = 0;
            for (const index of cachedSlices) {
              for (let i = index * bytesPerSlice;
                   i < (index + 1) * bytesPerSlice; i++) {
                if (i >= length || !discarded[i]) {
                  expectedCachedSlices++;
                  break;
                }
              }
            }
            expect(wrapper.getStats().cachedBytes)
                .withContext(context)
                .toBe(expectedCachedSlices * bytesPerSlice);
          }
        }
      });
    });
  }
});

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