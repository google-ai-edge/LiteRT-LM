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

import {createMonitoredStream} from './stream_helpers.js';
import {ControlledSource, readAll, streamOf, testData} from './testing/fakes.js';

describe('createMonitoredStream', () => {
  it('passes the data through and reports progress', async () => {
    const data = testData(10);
    const progress: number[] = [];
    const {stream} = createMonitoredStream(streamOf(data, 4), {
      expectedSize: 10,
      onProgress: loadedBytes => {
        progress.push(loadedBytes);
      },
    });

    expect(await readAll(stream)).toEqual(data);
    expect(progress).toEqual([4, 8, 10]);
  });

  it('fails if the stream is shorter than expected', async () => {
    const {stream} =
        createMonitoredStream(streamOf(testData(10)), {expectedSize: 11});

    await expectAsync(readAll(stream))
        .toBeRejectedWithError(/ended after 10 bytes, but 11 were expected/);
  });

  it('fails if the stream is longer than expected', async () => {
    const {stream} =
        createMonitoredStream(streamOf(testData(10)), {expectedSize: 9});

    await expectAsync(readAll(stream))
        .toBeRejectedWithError(/longer than the expected 9 bytes/);
  });

  it('aborts a locked stream and cancels the source', async () => {
    const source = new ControlledSource();
    const monitored = createMonitoredStream(source.stream);
    const reader = monitored.stream.getReader();
    const read = reader.read();
    const reason = new Error('aborted');

    monitored.abort(reason);

    await expectAsync(read).toBeRejectedWith(reason);
    expect(source.cancelled).toBeTrue();
    expect(source.cancelReason).toBe(reason);
  });

  it('ignores abort after the stream has ended', async () => {
    const monitored = createMonitoredStream(streamOf(testData(4)));
    await readAll(monitored.stream);

    expect(() => {
      monitored.abort(new Error('too late'));
    }).not.toThrow();
  });

  it('only reads the source when the consumer asks', async () => {
    const source = new ControlledSource();
    const {stream} = createMonitoredStream(source.stream);
    const reader = stream.getReader();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(source.pulls).toBe(0);

    const read = reader.read();
    source.enqueue(new Uint8Array([1]));
    expect((await read).value).toEqual(new Uint8Array([1]));
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(source.pulls).toBe(1);
  });
});
