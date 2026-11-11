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

import './camera_capture';

import type {LitertCameraCapture} from './camera_capture.js';

describe('litert-camera-capture', () => {
  let element: LitertCameraCapture;
  let openStates: boolean[];

  beforeEach(() => {
    element = document.createElement('litert-camera-capture');
    openStates = [];
    element.addEventListener('open-change', (e: Event) => {
      openStates.push((e as CustomEvent<{open: boolean}>).detail.open);
    });
    document.body.append(element);
  });

  afterEach(() => {
    element.remove();
    Reflect.deleteProperty(navigator, 'mediaDevices');
  });

  /** Replaces `navigator.mediaDevices.getUserMedia` (absent on plain http). */
  function fakeGetUserMedia(result: () => Promise<MediaStream>) {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {getUserMedia: result},
    });
  }

  function query(selector: string): HTMLElement|null {
    return element.shadowRoot!.querySelector<HTMLElement>(selector);
  }

  it('opens with Take photo focused and closes on Escape', async () => {
    fakeGetUserMedia(async () => new MediaStream());

    expect(await element.open()).toBeTrue();
    expect(query('.modal-backdrop')).not.toBeNull();
    expect(element.shadowRoot!.activeElement?.id).toBe('btn-capture-photo');

    query('.modal-backdrop')!.dispatchEvent(new KeyboardEvent(
        'keydown', {key: 'Escape', bubbles: true, cancelable: true}));
    await element.updateComplete;

    expect(query('.modal-backdrop')).toBeNull();
    expect(openStates).toEqual([true, false]);
  });

  it('stays closed when the camera is unavailable', async () => {
    fakeGetUserMedia(async () => {
      throw new Error('Permission denied');
    });
    spyOn(console, 'warn');

    expect(await element.open()).toBeFalse();
    await element.updateComplete;

    expect(query('.modal-backdrop')).toBeNull();
    expect(openStates).toEqual([]);
  });

  /** Returns a stream with real frames, like a camera's. */
  function createFrameStream(): MediaStream {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 24;
    canvas.getContext('2d')!.fillRect(0, 0, 32, 24);
    return canvas.captureStream();
  }

  async function waitForFirstFrame() {
    const video = query('video') as HTMLVideoElement;
    if (video.videoWidth > 0) return;
    await new Promise(resolve => {
      video.addEventListener('loadeddata', resolve, {once: true});
    });
  }

  it('sends a JPEG photo and closes when Take photo is clicked', async () => {
    fakeGetUserMedia(async () => createFrameStream());
    await element.open();
    await waitForFirstFrame();
    const captured = new Promise<Blob>(resolve => {
      element.addEventListener('photo-captured', (e: Event) => {
        resolve((e as CustomEvent<{photo: Blob}>).detail.photo);
      });
    });

    query('#btn-capture-photo')!.click();

    expect((await captured).type).toBe('image/jpeg');
    expect(openStates).toEqual([true, false]);
  });

  it('ignores Take photo until the camera shows a frame', async () => {
    fakeGetUserMedia(async () => new MediaStream());
    await element.open();
    const captured = jasmine.createSpy('photo-captured');
    element.addEventListener('photo-captured', captured);

    query('#btn-capture-photo')!.click();

    expect(captured).not.toHaveBeenCalled();
    expect(query('.modal-backdrop')).not.toBeNull();
  });

  it('stops a stream that arrives after the modal was closed', async () => {
    const stream = createFrameStream();
    let grant!: () => void;
    fakeGetUserMedia(() => new Promise(resolve => {
      grant = () => {
        resolve(stream);
      };
    }));
    const opened = element.open();

    element.close();
    grant();

    expect(await opened).toBeTrue();
    expect(stream.getTracks().every(track => track.readyState === 'ended'))
        .toBeTrue();
    expect(query('.modal-backdrop')).toBeNull();
    expect(openStates).toEqual([]);
  });
});
