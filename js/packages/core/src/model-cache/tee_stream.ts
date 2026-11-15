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
 * A pre-allocated circular RingBuffer class with strict capacity constraints
 * and zero-allocation reads.
 */
export class RingBuffer {
  private readonly buffer: Uint8Array;
  private writePos = 0;
  private readPos = 0;
  private closed = false;  // Whether the source stream has reached EOF.
  private active = true;   // false if this stream cancelled or aborted.
  private errorVal: unknown = null;
  private dataListener: (() => void) | null = null;
  private spaceListener: (() => void) | null = null;

  constructor(capacity: number) {
    this.buffer = new Uint8Array(capacity);
  }

  get capacity(): number {
    return this.buffer.length;
  }

  get unreadBytes(): number {
    return this.writePos - this.readPos;
  }

  get freeBytes(): number {
    return this.capacity - this.unreadBytes;
  }

  get isClosed(): boolean {
    return this.closed;
  }

  get isActive(): boolean {
    return this.active;
  }

  deactivate() {
    this.active = false;
  }

  get error(): unknown {
    return this.errorVal;
  }

  /**
   * Writes raw bytes into the circular buffer.
   * Throws an error if the chunk size exceeds current available freeBytes.
   */
  write(chunk: Uint8Array) {
    if (chunk.byteLength > this.freeBytes) {
      throw new Error(`[RingBuffer] Buffer overflow! Free space: ${this.freeBytes} bytes, requested: ${chunk.byteLength} bytes.`);
    }

    const pos = this.writePos % this.capacity;
    if (pos + chunk.byteLength <= this.capacity) {
      this.buffer.set(chunk, pos);
    } else {
      const part1 = this.capacity - pos;
      this.buffer.set(chunk.subarray(0, part1), pos);
      this.buffer.set(chunk.subarray(part1), 0);
    }

    this.writePos += chunk.byteLength;
    this.notifyData();
  }

  /**
   * Reads the largest contiguous unread block currently held in the circular
   * buffer.
   */
  private readNextChunk(): { chunk: Uint8Array; readLen: number } {
    const pos = this.readPos % this.capacity;
    const avail = this.unreadBytes;
    const readLen = Math.min(avail, this.capacity - pos);

    const chunk = this.buffer.subarray(pos, pos + readLen);

    return { chunk, readLen };
  }

  /**
   * An Async Generator that yields incoming circular buffer chunks
   * sequentially,automatically and statefully reclaiming buffer space (deferred
   * pointer advancement) when the downstream consumer requests the next chunk
   * via generator.next().
   */
  async *readAll(): AsyncGenerator<Uint8Array> {
    let lastReadLen = 0;
    while (true) {
      // Defer pointer advancement: reclaim space from the previous chunk on
      // the next pull iteration request.
      if (lastReadLen > 0) {
        this.readPos += lastReadLen;
        lastReadLen = 0;
        this.notifySpace();
      }

      if (this.errorVal) {
        throw this.errorVal;
      }
      if (this.unreadBytes === 0) {
        if (this.closed) {
          break;
        }
        // Wait for bytes to arrive in the queue asynchronously
        await new Promise<void>(resolve => this.waitData(resolve));
        continue;
      }

      // Read contiguous chunk using the private helper
      const { chunk, readLen } = this.readNextChunk();
      lastReadLen = readLen; // Defer pointer advancement size
      yield chunk;
    }
  }

  close() {
    this.closed = true;
    this.notifyData();
  }

  setError(err: unknown) {
    this.errorVal = err;
    this.notifyData();
  }

  /**
   * Registers a listener to be notified when new bytes are written or the
   * stream closes/errors.
   */
  waitData(cb: () => void) {
    this.dataListener = cb;
  }

  /**
   * Registers a listener to be notified when space is freed up in the buffer.
   * Pass null to remove the listener.
   */
  waitSpace(cb: (() => void)|null) {
    this.spaceListener = cb;
  }

  private notifyData() {
    if (this.dataListener) {
      const cb = this.dataListener;
      this.dataListener = null;
      cb();
    }
  }

  private notifySpace() {
    if (this.spaceListener) {
      const cb = this.spaceListener;
      this.spaceListener = null;
      cb();
    }
  }
}

/**
 * Split a single ReadableStream byte stream into two identical streams.
 *
 * This differs from the builtin `tee()` method:
 *   - `teeStream()` supports backpressure. `tee()` will buffer indefinitely.
 *   - `teeStream()` stops buffering when one of the output streams is closed.
 *     `tee()` continues to buffer even after one output stream is closed.
 *
 * Each chunk returned by one of the substreams is valid until the next chunk is
 * pulled from that same substream. After that, the underlying ArrayBuffer may
 * be overwritten.
 */
export function teeStream(
    sourceStream: ReadableStream<Uint8Array>,
    cacheSize = 20 * 1024 * 1024  // 20MB capacity per queue by default
    ): [ReadableStream<Uint8Array>, ReadableStream<Uint8Array>] {
  const buf1 = new RingBuffer(cacheSize);
  const buf2 = new RingBuffer(cacheSize);

  const abortController = new AbortController();
  let waitingForSpace: (() => void) | null = null;

  function clearSpaceListeners() {
    buf1.waitSpace(null);
    buf2.waitSpace(null);
  }

  // Unified Multicast Writable Sink
  // Intercepts written chunks, paces the pipeline using strict headroom checks,
  // and closes/aborts queues reactively.
  const multicastSink = new WritableStream<Uint8Array>({
    async write(chunk) {
      const len = chunk.byteLength;

      if (len > cacheSize) {
        throw new Error(`[teeStream] Chunk size (${
            len} bytes) exceeds buffer capacity (${cacheSize} bytes).`);
      }

      // Dynamic backpressure check: block pipeline write if EITHER active queue
      // does not have enough headroom.
      while ((buf1.isActive && buf1.freeBytes < len) ||
             (buf2.isActive && buf2.freeBytes < len)) {
        await new Promise<void>(resolve => {
          waitingForSpace = resolve;

          const onSpaceFreed = () => {
            const b1Ready = !buf1.isActive || buf1.freeBytes >= len;
            const b2Ready = !buf2.isActive || buf2.freeBytes >= len;
            if (b1Ready && b2Ready) {
              waitingForSpace = null;
              clearSpaceListeners();
              resolve();
            } else {
              // Wait again on subsequent space reclaimed events
              if (buf1.isActive) buf1.waitSpace(onSpaceFreed);
              if (buf2.isActive) buf2.waitSpace(onSpaceFreed);
            }
          };

          if (buf1.isActive) buf1.waitSpace(onSpaceFreed);
          if (buf2.isActive) buf2.waitSpace(onSpaceFreed);
        });
      }

      // Write chunk to both active buffers
      if (buf1.isActive) buf1.write(chunk);
      if (buf2.isActive) buf2.write(chunk);
    },
    close() {
      buf1.close();
      buf2.close();
    },
    abort(err) {
      if (buf1.isActive) buf1.setError(err);
      if (buf2.isActive) buf2.setError(err);
    }
  });

  // Pipe the network stream to our multicast sink in the background.
  sourceStream.pipeTo(multicastSink, {signal: abortController.signal})
      .catch(err => {
        if (buf1.isActive) buf1.setError(err);
        if (buf2.isActive) buf2.setError(err);
      });

  // Create a teed consumer stream wrapping a private RingBuffer queue.
  function makeTeedStream(buf: RingBuffer, otherBuf: RingBuffer) {
    const generator = buf.readAll();

    return new ReadableStream<Uint8Array>(
        {
          pull(controller) {
            // Pull next chunk out of the RingBuffer's async generator.
            return (async () => {
              try {
                const {done, value} = await generator.next();
                if (done) {
                  controller.close();
                  return;
                }
                if (value) {
                  controller.enqueue(value);
                }
              } catch (err) {
                controller.error(err);
              }
            })();
          },
          cancel(reason) {
            buf.deactivate();
            buf.close();

            // Manually wake up the other stream since this stream is done and
            // will not trigger a space notification.
            if (waitingForSpace) {
              const cb = waitingForSpace;
              waitingForSpace = null;
              clearSpaceListeners();
              cb();
            }

            // Cancel/abort the parent network stream only if BOTH teed
            // consumers are cancelled
            if (!otherBuf.isActive) {
              abortController.abort(reason);
            }
          }
        },
        // highWaterMark: 0 to avoid automatically pulling the next chunk.
        // We can not automatically pull the next chunk from the source stream
        // because by pulling the next chunk, we mark the current chunk as
        // consumed in the RingBuffer, and its underlying ArrayBuffer may be
        // overwritten before the downstream consumer has read it.
        {highWaterMark: 0});
  }

  return [makeTeedStream(buf1, buf2), makeTeedStream(buf2, buf1)];
}
