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

// Tests for behavior specific to BlobDataStreamWrapper. Behavior shared with
// ReadableStreamDataStreamWrapper is tested in data_stream_wrappers_test.ts.

import {BlobDataStreamWrapper} from './blob_data_stream_wrapper.js';

describe('BlobDataStreamWrapper', () => {
  let heap: Uint8Array;
  const data = new Uint8Array(100).map((_, i) => i);

  beforeEach(() => {
    heap = new Uint8Array(1024);
  });

  it('splits large reads', async () => {
    const wrapper = new BlobDataStreamWrapper(
        new Blob([data]), () => heap, {maxBytesPerRead: 7});
    const sliceSpy = spyOn(Blob.prototype, 'slice').and.callThrough();

    expect((await wrapper.readAndPreserve(0, 3, 90)).error).toBeUndefined();
    expect(heap.subarray(0, 90)).toEqual(data.subarray(3, 93));
    expect(sliceSpy).toHaveBeenCalledTimes(13);
  });

  it('fails reads of discarded data that a Blob could serve', async () => {
    const wrapper = new BlobDataStreamWrapper(new Blob([data]), () => heap);
    expect((await wrapper.readAndDiscard(0, 0, 50)).error).toBeUndefined();

    // Reading behind the discarded range would work for a Blob, but not for a
    // stream, so it fails here too.
    expect((await wrapper.readAndPreserve(0, 40, 20)).error?.message)
        .toContain('overlaps discarded regions: [[0, 50]]');
    expect((await wrapper.readAndPreserve(0, 50, 10)).error).toBeUndefined();
    expect(heap.subarray(0, 10)).toEqual(data.subarray(50, 60));
  });

  it('rejects an invalid maxBytesPerRead', () => {
    expect(
        () => new BlobDataStreamWrapper(
            new Blob([data]), () => heap, {maxBytesPerRead: 0}))
        .toThrowError(/maxBytesPerRead/);
  });
});
