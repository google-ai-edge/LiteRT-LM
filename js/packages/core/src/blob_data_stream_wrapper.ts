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

import {getDiscardedReadError, RangeSet, ReadableStreamDataStreamResult, ReadableStreamDataStreamWrapperInterface, toError, toNumber} from './readable_stream_data_stream_wrapper.js';

/** Options for BlobDataStreamWrapper. */
export declare interface BlobDataStreamWrapperOptions {
  /**
   * Maximum number of bytes read from the Blob at once, which bounds the
   * temporary memory used by large reads. Defaults to 16 MB.
   */
  maxBytesPerRead?: number;
}

const DEFAULT_MAX_BYTES_PER_READ = 16 * 1024 * 1024;

/**
 * Serves LiteRT LM C++ streamed model loading from a Blob (e.g. a File).
 *
 * Unlike a ReadableStream, a Blob supports random access, so nothing is
 * cached and reads can be in any order. Discarded ranges are still tracked,
 * and reading them fails like it does for a ReadableStream, so that a model
 * that loads from a Blob also loads from a stream.
 */
export class BlobDataStreamWrapper implements
    ReadableStreamDataStreamWrapperInterface {
  private readonly discarded = new RangeSet();
  private readonly maxBytesPerRead: number;
  private isReading = false;

  constructor(
      private readonly blob: Blob,
      // Heap is detached whenever it's resized, so we get it each time.
      private readonly getWasmHeap: () => Uint8Array,
      options: BlobDataStreamWrapperOptions = {}) {
    this.maxBytesPerRead =
        options.maxBytesPerRead ?? DEFAULT_MAX_BYTES_PER_READ;
    if (!Number.isSafeInteger(this.maxBytesPerRead) ||
        this.maxBytesPerRead <= 0) {
      throw new Error(`maxBytesPerRead must be a positive integer, got ${
          this.maxBytesPerRead}`);
    }
  }

  readAndDiscard(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult> {
    return this.read(destAddress, offset, count, /* discard= */ true);
  }

  readAndPreserve(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult> {
    return this.read(destAddress, offset, count, /* discard= */ false);
  }

  discard(offset: number|bigint, count: number|bigint):
      ReadableStreamDataStreamResult {
    try {
      const start = toNumber(offset);
      this.discarded.add(start, start + toNumber(count));
      return {};
    } catch (e) {
      return {error: toError(e)};
    }
  }

  close(): void {}

  private async read(
      destAddress: number, offset: number|bigint, count: number|bigint,
      discard: boolean): Promise<ReadableStreamDataStreamResult> {
    try {
      const dest = destAddress >>> 0;
      const start = toNumber(offset);
      const end = start + toNumber(count);
      if (this.isReading) {
        throw new Error('Concurrent reads are not supported');
      }
      if (end <= start) return {};
      if (end > this.blob.size) {
        throw new Error(`Read of [${start}, ${end}) exceeds the model size of ${
            this.blob.size} bytes`);
      }
      this.checkNotDiscarded(start, end);

      this.isReading = true;
      try {
        for (let position = start; position < end;
             position += this.maxBytesPerRead) {
          const chunkEnd = Math.min(position + this.maxBytesPerRead, end);
          const bytes = new Uint8Array(
              await this.blob.slice(position, chunkEnd).arrayBuffer());
          if (bytes.length !== chunkEnd - position) {
            throw new Error(`Read of [${position}, ${chunkEnd}) returned ${
                bytes.length} bytes`);
          }
          this.getWasmHeap().set(bytes, dest + position - start);
        }
      } finally {
        this.isReading = false;
      }

      if (discard) {
        this.discarded.add(start, end);
      }
      return {};
    } catch (e) {
      return {error: toError(e)};
    }
  }

  private checkNotDiscarded(start: number, end: number): void {
    const error = getDiscardedReadError(this.discarded, start, end);
    if (error) throw error;
  }
}
