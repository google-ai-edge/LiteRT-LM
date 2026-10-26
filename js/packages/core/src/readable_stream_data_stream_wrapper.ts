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

/**
 * Result returned by ReadableStreamDataStreamWrapper methods to C++ Emval.
 */
export declare interface ReadableStreamDataStreamResult {
  error?: Error;
}

/**
 * Extern interface for the data stream wrapper methods invoked by C++
 * (`ReadableStreamDataStream` in readable_stream_data_stream.cc).
 */
export declare interface ReadableStreamDataStreamWrapperInterface {
  readAndDiscard(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult>;
  readAndPreserve(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult>;
  discard(offset: number|bigint, count: number|bigint):
      ReadableStreamDataStreamResult;
  /**
   * Releases the underlying source. Called when the C++ stream is destroyed.
   * Must be idempotent.
   */
  close(): void;
}

/** Options for ReadableStreamDataStreamWrapper. */
export declare interface ReadableStreamDataStreamWrapperOptions {
  /**
   * Maximum number of bytes that may be cached for later reads. Reads that
   * would cache more fail with an error instead of growing memory without
   * bound. Defaults to no limit.
   */
  maxCachedBytes?: number;
  /**
   * Logs a warning once the cache grows past this many bytes. Defaults to
   * 1 GB.
   */
  warnCachedBytes?: number;
}

/** Statistics about a ReadableStreamDataStreamWrapper's memory use. */
export declare interface ReadableStreamDataStreamWrapperStats {
  /** Number of bytes read from the underlying stream so far. */
  streamPosition: number;
  /** Bytes currently held in cache slices. */
  cachedBytes: number;
  /** Largest value `cachedBytes` has reached. */
  peakCachedBytes: number;
}

const DEFAULT_BYTES_PER_SLICE = 10_000_000;
const DEFAULT_WARN_CACHED_BYTES = 1_000_000_000;
/** Size of the scratch buffer used for BYOB reads. */
const BYOB_SCRATCH_BYTES = 4 * 1024 * 1024;
/** Number of released slice buffers kept for reuse. */
const MAX_FREE_BUFFERS = 2;

/** A set of half-open [start, end) ranges, kept sorted and merged. */
export class RangeSet {
  // Sorted, disjoint and non-adjacent, so both starts and ends are strictly
  // increasing.
  private readonly ranges: Array<[number, number]> = [];

  add(start: number, end: number): void {
    if (start >= end) return;
    // Merge with every range that overlaps or touches [start, end).
    const first = this.firstIndexWhereEndIsAtLeast(start);
    let last = first;
    while (last < this.ranges.length && this.ranges[last][0] <= end) {
      start = Math.min(start, this.ranges[last][0]);
      end = Math.max(end, this.ranges[last][1]);
      last++;
    }
    this.ranges.splice(first, last - first, [start, end]);
  }

  /** Returns the ranges that overlap [start, end). */
  overlapping(start: number, end: number): Array<[number, number]> {
    const result: Array<[number, number]> = [];
    for (let i = this.firstIndexWhereEndIsAtLeast(start + 1);
         i < this.ranges.length && this.ranges[i][0] < end; i++) {
      result.push(this.ranges[i]);
    }
    return result;
  }

  /** Returns true if every position in [start, end) is in the set. */
  covers(start: number, end: number): boolean {
    if (start >= end) return true;
    const i = this.firstIndexWhereEndIsAtLeast(end);
    return i < this.ranges.length && this.ranges[i][0] <= start;
  }

  /** Returns the parts of [start, end) that are not in the set. */
  gaps(start: number, end: number): Array<[number, number]> {
    const result: Array<[number, number]> = [];
    let position = start;
    for (const [rStart, rEnd] of this.overlapping(start, end)) {
      if (rStart > position) result.push([position, rStart]);
      position = Math.max(position, rEnd);
    }
    if (position < end) result.push([position, end]);
    return result;
  }

  private firstIndexWhereEndIsAtLeast(position: number): number {
    let low = 0;
    let high = this.ranges.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (this.ranges[mid][1] < position) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low;
  }
}

/**
 * Wraps a ReadableStream for use in LiteRT LM C++ streamed model loading.
 *
 * C++ reads the model through a random access interface, but a ReadableStream
 * can only be read forward. Reads are served as follows:
 *
 * - Bytes the stream has not reached yet are read from the stream. If the read
 *   discards them, they are copied straight to the wasm heap without being
 *   cached, so loading a model in file order caches nothing.
 * - Reading ahead of the stream caches the skipped bytes in fixed size slices
 *   so they can be read later, except for bytes that have already been
 *   discarded, which are dropped.
 * - Reading bytes behind the stream copies them from the cache.
 *
 * Discarded bytes can not be read again. A slice's memory is released once
 * all of its bytes have been discarded.
 *
 * If reading from the stream fails, the wrapper fails all later reads since
 * its state no longer matches the stream.
 */
export class ReadableStreamDataStreamWrapper implements
    ReadableStreamDataStreamWrapperInterface {
  /** Cached data, by slice index. Each buffer is `bytesPerSlice` long. */
  private readonly slices = new Map<number, Uint8Array<ArrayBuffer>>();
  private readonly discarded = new RangeSet();
  /** Released slice buffers that can be reused. */
  private readonly freeBuffers: Array<Uint8Array<ArrayBuffer>> = [];

  private readonly reader:
      {byobReader: ReadableStreamBYOBReader, defaultReader?: undefined}|{
        defaultReader: ReadableStreamDefaultReader<Uint8Array>,
        byobReader?: undefined
      };
  /** Unconsumed part of the last chunk from the default reader. */
  private leftover?: Uint8Array;
  /** Buffer for BYOB reads. Replaced after each read, which transfers it. */
  private byobScratch?: ArrayBuffer;

  private streamPosition = 0;
  /** The stream's length, once its end has been reached. */
  private streamLength?: number;
  private cachedBytes = 0;
  private peakCachedBytes = 0;
  private warnedAboutCacheSize = false;
  private readonly maxCachedBytes: number;
  private readonly warnCachedBytes: number;

  private isReading = false;
  private closed = false;
  /** The error that put the wrapper in a failed state, if any. */
  private failure?: Error;

  constructor(
      stream: ReadableStream<Uint8Array>,
      // Heap is detached whenever it's resized, so we get it each time.
      private readonly getWasmHeap: () => Uint8Array,
      readonly bytesPerSlice = DEFAULT_BYTES_PER_SLICE,
      options: ReadableStreamDataStreamWrapperOptions = {}) {
    if (!Number.isSafeInteger(bytesPerSlice) || bytesPerSlice <= 0) {
      throw new Error(
          `bytesPerSlice must be a positive integer, got ${bytesPerSlice}`);
    }
    this.maxCachedBytes = options.maxCachedBytes ?? Infinity;
    this.warnCachedBytes = options.warnCachedBytes ?? DEFAULT_WARN_CACHED_BYTES;
    try {
      this.reader = {byobReader: stream.getReader({mode: 'byob'})};
    } catch (e) {
      this.reader = {defaultReader: stream.getReader()};
    }
  }

  async readAndDiscard(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult> {
    try {
      await this.read(
          destAddress >>> 0, toNumber(offset), toNumber(count),
          /* discard= */ true);
      return {};
    } catch (e) {
      return {error: toError(e)};
    }
  }

  async readAndPreserve(
      destAddress: number, offset: number|bigint,
      count: number|bigint): Promise<ReadableStreamDataStreamResult> {
    try {
      await this.read(
          destAddress >>> 0, toNumber(offset), toNumber(count),
          /* discard= */ false);
      return {};
    } catch (e) {
      return {error: toError(e)};
    }
  }

  discard(offset: number|bigint, count: number|bigint):
      ReadableStreamDataStreamResult {
    try {
      const start = toNumber(offset);
      const length = toNumber(count);
      if (length > 0) {
        this.discardRange(start, start + length);
      }
      return {};
    } catch (e) {
      return {error: toError(e)};
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.slices.clear();
    this.freeBuffers.length = 0;
    this.cachedBytes = 0;
    this.leftover = undefined;
    this.byobScratch = undefined;
    const reader = this.reader.byobReader ?? this.reader.defaultReader;
    // Cancelling stops any in-progress download. It rejects if the stream has
    // already failed, in which case there is nothing to release.
    reader.cancel().catch(() => {});
  }

  getStats(): ReadableStreamDataStreamWrapperStats {
    return {
      streamPosition: this.streamPosition,
      cachedBytes: this.cachedBytes,
      peakCachedBytes: this.peakCachedBytes,
    };
  }

  private async read(
      destAddress: number, offset: number, count: number,
      discard: boolean): Promise<void> {
    this.checkUsable();
    if (this.isReading) {
      throw new Error('Concurrent reads are not supported');
    }
    if (count <= 0) return;
    const end = offset + count;
    this.checkNotDiscarded(offset, end);

    this.isReading = true;
    try {
      // 1. Advance the stream to the requested range, caching the bytes in
      // between unless they have already been discarded.
      if (this.streamPosition < offset) {
        await this.consume(offset - this.streamPosition, (bytes, position) => {
          this.cache(bytes, position);
        });
      }

      // 2. Copy the part of the range that is already cached.
      const cachedEnd = Math.min(this.streamPosition, end);
      for (const {index, start, end: viewEnd} of this.sliceViews(
               offset, cachedEnd)) {
        const data = this.slices.get(index);
        if (!data) {
          throw new Error(`No cached data for [${start}, ${viewEnd})`);
        }
        const sliceStart = index * this.bytesPerSlice;
        this.getWasmHeap().set(
            data.subarray(start - sliceStart, viewEnd - sliceStart),
            destAddress + start - offset);
      }

      // 3. Read the rest of the range from the stream, caching it only if it
      // is being preserved.
      const uncachedStart = Math.max(offset, this.streamPosition);
      if (uncachedStart < end) {
        await this.consume(end - uncachedStart, (bytes, position) => {
          this.getWasmHeap().set(bytes, destAddress + position - offset);
          if (!discard) this.cache(bytes, position);
        });
      }

      if (discard) {
        this.discardRange(offset, end);
      }
    } finally {
      this.isReading = false;
    }
  }

  /**
   * Reads exactly `length` bytes from the stream, passing them to `sink` in
   * order along with their position in the stream. The bytes passed to `sink`
   * are only valid until it returns.
   *
   * Puts the wrapper in a failed state on error, since the bytes read so far
   * are lost.
   */
  private async consume(
      length: number,
      sink: (bytes: Uint8Array, position: number) => void): Promise<void> {
    try {
      let remaining = length;
      while (remaining > 0) {
        const bytes = await this.readChunk(remaining);
        if (this.closed) {
          throw new Error('ReadableStreamDataStreamWrapper was closed');
        }
        if (!bytes) {
          this.streamLength = this.streamPosition;
          throw new Error(`Read from stream returned early EOF at position ${
              this.streamPosition}. Expected ${remaining} more bytes.`);
        }
        sink(bytes, this.streamPosition);
        this.streamPosition += bytes.length;
        remaining -= bytes.length;
      }
    } catch (e) {
      this.failure ??= toError(e);
      throw e;
    }
  }

  /**
   * Reads between 1 and `maxLength` bytes from the stream, or returns
   * undefined at the end of the stream.
   */
  private async readChunk(maxLength: number): Promise<Uint8Array|undefined> {
    while (true) {
      if (this.reader.byobReader) {
        this.byobScratch ??= new ArrayBuffer(BYOB_SCRATCH_BYTES);
        const view = new Uint8Array(
            this.byobScratch, 0,
            Math.min(maxLength, this.byobScratch.byteLength));
        const {done, value} = await this.reader.byobReader.read(view);
        // The read transfers the scratch buffer to `value`.
        this.byobScratch = value?.buffer;
        if (value && value.byteLength > 0) return value;
        if (done) return undefined;
      } else {
        let chunk = this.leftover;
        this.leftover = undefined;
        if (!chunk) {
          const {done, value} = await this.reader.defaultReader.read();
          if (done) return undefined;
          chunk = value;
        }
        if (chunk.length > maxLength) {
          this.leftover = chunk.subarray(maxLength);
          return chunk.subarray(0, maxLength);
        }
        if (chunk.length > 0) return chunk;
      }
    }
  }

  /**
   * Stores the bytes read from the stream at `position` in the cache, except
   * for those that have been discarded.
   */
  private cache(bytes: Uint8Array, position: number): void {
    const end = position + bytes.length;
    for (const [start, gapEnd] of this.discarded.gaps(position, end)) {
      for (const {index, start: viewStart, end: viewEnd} of this.sliceViews(
               start, gapEnd)) {
        const data = this.slices.get(index) ?? this.allocateSlice(index);
        data.set(
            bytes.subarray(viewStart - position, viewEnd - position),
            viewStart - index * this.bytesPerSlice);
      }
    }
  }

  private allocateSlice(index: number): Uint8Array<ArrayBuffer> {
    const cachedBytes = this.cachedBytes + this.bytesPerSlice;
    if (cachedBytes > this.maxCachedBytes) {
      throw new Error(`Reading position ${
          this.streamPosition} would cache ${cachedBytes} bytes, more than ${
          this.maxCachedBytes} allowed. The model is being read too far out ` +
                      'of order.');
    }
    const data = this.freeBuffers.pop() ?? new Uint8Array(this.bytesPerSlice);
    this.slices.set(index, data);
    this.cachedBytes = cachedBytes;
    this.peakCachedBytes = Math.max(this.peakCachedBytes, cachedBytes);
    if (!this.warnedAboutCacheSize && cachedBytes > this.warnCachedBytes) {
      this.warnedAboutCacheSize = true;
      console.warn(`Streamed model loading has cached ${
          cachedBytes} bytes because the model is being read out of order.`);
    }
    return data;
  }

  private discardRange(start: number, end: number): void {
    this.discarded.add(start, end);
    for (const {index} of this.sliceViews(start, end)) {
      const data = this.slices.get(index);
      if (!data) continue;
      const sliceStart = index * this.bytesPerSlice;
      const sliceEnd = Math.min(
          sliceStart + this.bytesPerSlice, this.streamLength ?? Infinity);
      if (this.discarded.covers(sliceStart, sliceEnd)) {
        this.slices.delete(index);
        this.cachedBytes -= this.bytesPerSlice;
        if (this.freeBuffers.length < MAX_FREE_BUFFERS) {
          this.freeBuffers.push(data);
        }
      }
    }
  }

  private checkUsable(): void {
    if (this.closed) {
      throw new Error('ReadableStreamDataStreamWrapper is closed');
    }
    if (this.failure) {
      throw new Error(`Reading from the stream previously failed: ${
          this.failure.message}`);
    }
  }

  private checkNotDiscarded(start: number, end: number): void {
    const error = getDiscardedReadError(this.discarded, start, end);
    if (error) throw error;
  }

  /** Splits [start, end) at slice boundaries. */
  private *
      sliceViews(start: number, end: number):
          IterableIterator<{index: number, start: number, end: number}> {
    let position = start;
    while (position < end) {
      const index = Math.floor(position / this.bytesPerSlice);
      const viewEnd = Math.min((index + 1) * this.bytesPerSlice, end);
      yield {index, start: position, end: viewEnd};
      position = viewEnd;
    }
  }
}

/** Converts an offset or size from C++ to a number. */
export function toNumber(val: number|bigint): number {
  if (typeof val === 'number') {
    return val;
  }
  if (val > Number.MAX_SAFE_INTEGER) {
    throw new Error(
        `BigInt value ${val} too large to be represented as a number`);
  }
  return Number(val);
}

/** Converts a caught value to an Error, which C++ expects. */
export function toError(e: unknown): Error {
  return e instanceof Error ? e : new Error(String(e));
}

/**
 * Returns an error for a read of [start, end) if it overlaps `discarded`, or
 * undefined if it doesn't.
 */
export function getDiscardedReadError(
    discarded: RangeSet, start: number, end: number): Error|undefined {
  const regions = discarded.overlapping(start, end);
  if (regions.length === 0) return undefined;
  return new Error(`Read of [${start}, ${end}) overlaps discarded regions: [${
      regions.map(([rStart, rEnd]) => `[${rStart}, ${rEnd}]`).join(', ')}]`);
}
