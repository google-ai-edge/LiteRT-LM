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

/** Options for `createMonitoredStream()`. */
export declare interface MonitoredStreamOptions {
  /**
   * Fails the stream if it doesn't contain exactly this many bytes, so that a
   * truncated download is never stored.
   */
  expectedSize?: number;
  /** Called with the total number of bytes read so far, after each chunk. */
  onProgress?: (loadedBytes: number) => void;
}

/** A stream returned by `createMonitoredStream()`. */
export declare interface MonitoredStream {
  readonly stream: ReadableStream<Uint8Array>;
  /**
   * Fails `stream` with `reason` and cancels the source. Does nothing if
   * `stream` has already ended.
   *
   * Unlike `stream.cancel()`, this works after `stream` has been passed to a
   * consumer that locked it, e.g. a `Response` passed to `cache.put()`, and it
   * makes that consumer fail.
   */
  abort(reason?: unknown): void;
}

/**
 * Wraps a stream so that it can be aborted after it has been handed to a
 * consumer, and optionally checks its length and reports progress.
 *
 * Chunks are passed through without copying, and the next chunk is only read
 * from the source once the consumer has asked for it, so a chunk that's only
 * valid until the next read (see `teeStream()`) stays valid.
 */
export function createMonitoredStream(
    source: ReadableStream<Uint8Array>,
    options: MonitoredStreamOptions = {}): MonitoredStream {
  const reader = source.getReader();
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  let ended = false;
  let loadedBytes = 0;

  const fail = (reason: unknown) => {
    if (ended) return;
    ended = true;
    controller.error(reason);
    reader.cancel(reason).catch(() => {});
  };

  const stream = new ReadableStream<Uint8Array>(
      {
        start(c) {
          controller = c;
        },
        async pull(c) {
          let result: ReadableStreamReadResult<Uint8Array>;
          try {
            result = await reader.read();
          } catch (e) {
            if (!ended) {
              ended = true;
              c.error(e);
            }
            return;
          }
          if (ended) return;
          const {expectedSize} = options;
          if (result.done) {
            if (expectedSize !== undefined && loadedBytes !== expectedSize) {
              fail(new Error(`Stream ended after ${
                  loadedBytes} bytes, but ${expectedSize} were expected`));
              return;
            }
            ended = true;
            c.close();
            return;
          }
          loadedBytes += result.value.byteLength;
          if (expectedSize !== undefined && loadedBytes > expectedSize) {
            fail(new Error(`Stream is longer than the expected ${
                expectedSize} bytes`));
            return;
          }
          options.onProgress?.(loadedBytes);
          c.enqueue(result.value);
        },
        cancel(reason) {
          ended = true;
          return reader.cancel(reason);
        },
      },
      // Don't read ahead of the consumer. See the function comment.
      {highWaterMark: 0});

  return {stream, abort: fail};
}
