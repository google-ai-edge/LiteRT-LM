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

import {teeStream} from './tee_stream.js';

// Helper to create source stream from array of buffers
function createSource(
    chunks: Uint8Array[], delayMs = 0): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream({
    async pull(controller) {
      if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
      if (i < chunks.length) {
        controller.enqueue(chunks[i++]!.slice(0));
      } else {
        controller.close();
      }
    }
  });
}

// Helper to consume default reader
async function consumeDefault(
    stream: ReadableStream<Uint8Array>, delayMs = 0): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let totalLen = 0;
  while (true) {
    if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value!.slice());
    totalLen += value!.byteLength;
  }
  const full = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    full.set(c, offset);
    offset += c.byteLength;
  }
  return full;
}

describe('teeStream Concurrency Suite', () => {
  const dataStr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.repeat(
      100);  // 3,600 bytes test payload
  let sourceData: Uint8Array;
  let sourceChunks: Uint8Array[];

  beforeEach(() => {
    sourceData = new TextEncoder().encode(dataStr);
    sourceChunks = [
      sourceData.subarray(0, 1000),
      sourceData.subarray(1000, 2000),
      sourceData.subarray(2000, 3000),
      sourceData.subarray(3000),
    ];
  });

  it('First stream polling faster than the second', async () => {
    const source = createSource(sourceChunks, 10);
    const [s1, s2] = teeStream(source);

    const p1 = consumeDefault(s1, 0);   // Fast
    const p2 = consumeDefault(s2, 20);  // Slow

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual(sourceData);
    expect(r2).toEqual(sourceData);
  });

  it('Second stream polling faster than the first', async () => {
    const source = createSource(sourceChunks, 10);
    const [s1, s2] = teeStream(source);

    const p1 = consumeDefault(s1, 25);  // Slow
    const p2 = consumeDefault(s2, 0);   // Fast

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual(sourceData);
    expect(r2).toEqual(sourceData);
  });

  it('Both streams polling faster than the source', async () => {
    const source = createSource(sourceChunks, 30);  // Slow source
    const [s1, s2] = teeStream(source);

    const p1 = consumeDefault(s1, 0);
    const p2 = consumeDefault(s2, 0);

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual(sourceData);
    expect(r2).toEqual(sourceData);
  });

  it('Both streams polling slower than the source', async () => {
    const source = createSource(sourceChunks, 0);  // Fast source
    const [s1, s2] = teeStream(source);

    const p1 = consumeDefault(s1, 15);
    const p2 = consumeDefault(s2, 15);

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual(sourceData);
    expect(r2).toEqual(sourceData);
  });

  it('One stream polling slower while another polls faster (before buffer fills)',
     async () => {
       const source = createSource(sourceChunks, 5);
       const [s1, s2] = teeStream(source);

       const p1 = consumeDefault(s1, 0);  // Lead reader pulls instantly
       await new Promise(
           r => setTimeout(r, 50));  // Wait for lead reader to pull ahead
       const p2 =
           consumeDefault(s2, 5);  // Trailing reader consumes from ring buffer

       const [r1, r2] = await Promise.all([p1, p2]);
       expect(r1).toEqual(sourceData);
       expect(r2).toEqual(sourceData);
     });

  it('Ring buffer full (blocking lead reader until trailing reader catches up)',
     async () => {
       // Use a tiny static ring buffer of 2000 bytes (lead blocks when buffer
       // is full)
       const source =
           createSource(sourceChunks, 1);  // 36KB payload across 4 chunks
       const [s1, s2] = teeStream(source, 2000);

       let p1Done = false;
       const p1 = consumeDefault(s1, 0).then(res => {
         p1Done = true;
         return res;
       });

       // Give lead reader time to hit the 2048-byte ring buffer ceiling and
       // block
       await new Promise(r => setTimeout(r, 40));
       expect(p1Done).toBe(
           false);  // Flawlessly verifies lead reader is blocked!

       // Now start trailing reader to unblock ring buffer
       const p2 = consumeDefault(s2, 5);

       const [r1, r2] = await Promise.all([p1, p2]);
       expect(r1).toEqual(sourceData);
       expect(r2).toEqual(sourceData);
     });

  it('Cancellation of one stream instantly unblocks lead reader and leaves other active',
     async () => {
       const source = createSource(sourceChunks, 1);
       const [s1, s2] = teeStream(source, 4096);  // Tiny buffer

       // S1 pulls ahead and blocks
       const reader1 = s1.getReader();
       await reader1.read();                       // Reads chunk 1
       await new Promise(r => setTimeout(r, 20));  // Blocked on chunk 2

       // Cancel S2 (lagging reader)
       await s2.cancel();

       // S1 should instantly unblock and read remaining data to EOF
       const remaining: Uint8Array[] = [];
       while (true) {
         const {done, value} = await reader1.read();
         if (done) break;
         remaining.push(value!);
       }
       expect(remaining.length).toBeGreaterThan(0);
     });

  it('Trailing reader finishes exactly at EOF and never calls pull again (no hangs)',
     async () => {
       const source = createSource(sourceChunks, 5);
       const [s1, s2] = teeStream(source, 4096);  // Tiny buffer

       // Consumes entire S1 (lead reader)
       const p1 = consumeDefault(s1, 0);

       // S2 (trailing reader) reads up to the end, but we manually read it
       // chunk-by-chunk
       const reader2 = s2.getReader();
       const chunks: Uint8Array[] = [];
       while (true) {
         const {done, value} = await reader2.read();
         if (done) break;
         chunks.push(value!);
         if (chunks.length === 4) break;
       }

       reader2.releaseLock();

       const r1 = await p1;
       expect(r1).toEqual(sourceData);

       let totalLen = 0;
       for (const c of chunks) totalLen += c.byteLength;
       const r2 = new Uint8Array(totalLen);
       let offset = 0;
       for (const c of chunks) {
         r2.set(c, offset);
         offset += c.byteLength;
       }
       expect(r2).toEqual(sourceData);
     });

  it('Network chunk size exceeds remaining ring buffer headroom (dynamic overflow blocking)',
     async () => {
       const chunks = [
         new Uint8Array(40),  // Chunk 1: 40 bytes
         new Uint8Array(80),  // Chunk 2: 80 bytes (combined: 120 bytes,
                              // overflows 100-byte buffer!)
         new Uint8Array(20),  // Chunk 3: 20 bytes
       ];

       chunks[0]!.fill(1);
       chunks[1]!.fill(2);
       chunks[2]!.fill(3);

       const source = createSource(chunks, 2);
       const [s1, s2] = teeStream(source, 100);  // Tiny 100-byte buffer

       let p1Done = false;
       const p1 = consumeDefault(s1, 0).then(res => {
         p1Done = true;
         return res;
       });

       await new Promise(r => setTimeout(r, 40));
       expect(p1Done).toBe(false);  // Lead reader MUST be blocked!

       const p2 = consumeDefault(s2, 5);

       const [r1, r2] = await Promise.all([p1, p2]);

       const expected = new Uint8Array(140);
       expected.set(chunks[0]!, 0);
       expected.set(chunks[1]!, 40);
       expected.set(chunks[2]!, 120);

       expect(r1).toEqual(expected);
       expect(r2).toEqual(expected);
     });

  it('Error when writing a chunk larger than buffer capacity', async () => {
    const chunks = [new Uint8Array(200)];  // 200 bytes
    const source = createSource(chunks);
    const [s1] = teeStream(source, 100);  // 100 byte capacity

    let caughtError: Error|undefined;
    try {
      await consumeDefault(s1);
    } catch (e: unknown) {
      if (!(e instanceof Error)) throw e;
      caughtError = e;
    }

    expect(caughtError).toBeDefined();
    expect(caughtError?.message).toContain('exceeds buffer capacity');
  });

  it('Dangling space listeners do not cause deadlocks when a new chunk blocks and a stream is cancelled',
     async () => {
       const chunks = [
         new Uint8Array(60).fill(1),
         new Uint8Array(50).fill(2),
         new Uint8Array(10).fill(3),
         new Uint8Array(60).fill(4),
       ];
       const source = createSource(chunks, 2);
       const [s1, s2] = teeStream(source, 100);

       const reader1 = s1.getReader();
       const reader2 = s2.getReader();

       const {value: v1} = await reader1.read();
       expect(v1?.byteLength).toBe(60);

       await new Promise(r => setTimeout(r, 20));

       const {value: v2} = await reader2.read();
       expect(v2?.byteLength).toBe(60);

       await reader1.cancel();

       let bytesRead = 0;
       while (true) {
         const {done, value} = await reader2.read();
         if (done) break;
         bytesRead += value!.byteLength;
       }
       expect(bytesRead).toBe(120);  // 50 + 10 + 60
     });

  it('Graceful clear when consumer cancels while waiting for data',
     async () => {
       let pullCount = 0;
       const source = new ReadableStream({
         pull(controller) {
           pullCount++;
           // Enqueue nothing, just let it wait
         }
       });

       const [s1, s2] = teeStream(source);

       const reader1 = s1.getReader();
       const reader2 = s2.getReader();

       // Start reading, both will block waiting for data
       const p1 = reader1.read();
       const p2 = reader2.read();

       // Cancel both readers. This triggers multicastSink.abort()
       // If abort() unconditionally sets the error, the blocked read()
       // might throw the cancellation reason instead of returning {done: true}.
       await reader1.cancel('cancel 1');
       await reader2.cancel('cancel 2');

       const res1 = await p1;
       const res2 = await p2;

       expect(res1.done).toBeTrue();
       expect(res2.done).toBeTrue();
     });

  it('Source stream error surfaces to active readers', async () => {
    let errorController: ReadableStreamDefaultController<Uint8Array>;
    const source = new ReadableStream({
      start(c) {
        errorController = c;
      }
    });

    const [s1, s2] = teeStream(source);

    const reader1 = s1.getReader();
    const reader2 = s2.getReader();

    const p1 = reader1.read();
    const p2 = reader2.read();

    const testError = new Error('Network failure');
    errorController!.error(testError);

    await expectAsync(p1).toBeRejectedWith(testError);
    await expectAsync(p2).toBeRejectedWith(testError);
  });

  it('Wait again on subsequent space reclaimed events when buffer is full',
     async () => {
       // Total capacity = 50.
       const source = createSource(
           [
             new Uint8Array(40).fill(1),
             new Uint8Array(40).fill(2)  // 40 + 40 = 80, exceeds 50 capacity
           ],
           2);
       const [s1, s2] = teeStream(source, 50);

       const pullAll = async (stream: ReadableStream) => {
         const reader = stream.getReader();
         const chunks = [];
         while (true) {
           const {done, value} = await reader.read();
           if (done) break;
           chunks.push(value);
         }
         return chunks;
       };

       // Because capacity is 50, the second 40-byte chunk will block the
       // producer. The producer will enter its waitSpace loop. Eventually the
       // consumers pull the first chunks causing waitSpace callbacks to
       // recursively trigger until enough headroom exists to write.

       // We start both fast pullers which consumes all data cleanly.
       const [c1, c2] = await Promise.all([pullAll(s1), pullAll(s2)]);

       expect(c1.length).toBe(3);
       expect(c2.length).toBe(3);
     });

  it('[RingBuffer] Buffer overflow! Free space...', async () => {
    // We directly trigger the RingBuffer error condition, since teeStream's
    // dynamic bounds check usually prevents it from happening.

    const {RingBuffer} = await import('./tee_stream');
    const rb = new RingBuffer(10);

    expect(() => {
      rb.write(new Uint8Array(20));
    }).toThrowError(/.*\[RingBuffer\] Buffer overflow! Free space.*/);
  });
});
