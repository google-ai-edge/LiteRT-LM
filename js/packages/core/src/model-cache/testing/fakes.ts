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

// Helpers for model cache tests.

/** A stream source whose chunks are pushed by the test. */
export class ControlledSource {
  readonly stream: ReadableStream<Uint8Array>;
  /** Number of chunks the stream has been asked for. */
  pulls = 0;
  cancelled = false;
  cancelReason: unknown;
  private controller!: ReadableStreamDefaultController<Uint8Array>;

  constructor() {
    this.stream = new ReadableStream<Uint8Array>(
        {
          start: controller => {
            this.controller = controller;
          },
          pull: () => {
            this.pulls++;
          },
          cancel: reason => {
            this.cancelled = true;
            this.cancelReason = reason;
          },
        },
        {highWaterMark: 0});
  }

  enqueue(chunk: Uint8Array): void {
    this.controller.enqueue(chunk);
  }

  close(): void {
    this.controller.close();
  }

  error(reason: unknown): void {
    this.controller.error(reason);
  }
}

/** Returns a stream of `data` in chunks of `chunkSize` bytes. */
export function streamOf(
    data: Uint8Array, chunkSize = 4): ReadableStream<Uint8Array> {
  let position = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (position >= data.length) {
        controller.close();
        return;
      }
      controller.enqueue(data.slice(position, position + chunkSize));
      position += chunkSize;
    },
  });
}

/** Reads a whole stream. */
export async function readAll(stream: ReadableStream<Uint8Array>):
    Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value.slice());
  }
  return concat(chunks);
}

/** Concatenates byte arrays. */
export function concat(chunks: Uint8Array[]): Uint8Array {
  const result =
      new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

/** Returns `size` bytes of test data. */
export function testData(size: number): Uint8Array<ArrayBuffer> {
  return Uint8Array.from({length: size}, (unused, i) => (i * 7 + 3) % 256);
}

/** Waits for pending promise callbacks and stream operations to run. */
export function flush(): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, 0);
  });
}
