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

import {SUPPORTED_AUDIO_MIME_TYPES} from '@litert-lm/core';

/** Sample rate the model's audio encoder expects. */
const TARGET_SAMPLE_RATE = 16000;

/** Encodes mono float samples in [-1, 1] as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);              // Subchunk1Size (PCM)
  view.setUint16(20, 1, true);               // AudioFormat (PCM)
  view.setUint16(22, 1, true);               // NumChannels (mono)
  view.setUint32(24, sampleRate, true);      // SampleRate
  view.setUint32(28, sampleRate * 2, true);  // ByteRate
  view.setUint16(32, 2, true);               // BlockAlign
  view.setUint16(34, 16, true);              // BitsPerSample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]!));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], {type: 'audio/wav'});
}

/**
 * Decodes any audio format the browser supports (e.g. Ogg, WebM, or M4A) and
 * re-encodes it as a 16kHz mono WAV file.
 */
async function reencodeAsWav(blob: Blob): Promise<Blob> {
  const decodeContext = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeContext.decodeAudioData(await blob.arrayBuffer());
  } finally {
    void decodeContext.close();
  }

  // Downmix and resample in one pass.
  const length = Math.max(
      1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE));
  const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE);
}

/**
 * Returns `blob` if the model can decode it, or a WAV re-encoding of it
 * otherwise.
 */
export async function toSupportedAudio(blob: Blob): Promise<Blob> {
  if (SUPPORTED_AUDIO_MIME_TYPES.has(blob.type)) return blob;
  return reencodeAsWav(blob);
}

/**
 * Records audio from the microphone. The recording is returned in a format
 * the model supports.
 */
export class AudioRecorder {
  private mediaRecorder: MediaRecorder|null = null;
  private chunks: Blob[] = [];
  private startTime = 0;

  get isRecording(): boolean {
    return this.mediaRecorder !== null;
  }

  /** Seconds since recording started, or 0 when not recording. */
  get durationSeconds(): number {
    return this.isRecording ? (Date.now() - this.startTime) / 1000 : 0;
  }

  /** Starts recording. Rejects if microphone access is denied. */
  async start(): Promise<void> {
    if (this.mediaRecorder) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true},
    });
    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(stream);
    this.mediaRecorder.addEventListener('dataavailable', (e: BlobEvent) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    });
    this.mediaRecorder.start();
    this.startTime = Date.now();
  }

  /** Stops recording and returns the recording as a supported audio file. */
  async stop(): Promise<Blob> {
    const recorder = this.mediaRecorder;
    if (!recorder) throw new Error('Not recording.');
    const stopped = new Promise<void>(resolve => {
      recorder.addEventListener('stop', () => {
        resolve();
      }, {once: true});
    });
    recorder.stop();
    await stopped;
    this.release(recorder);
    const recording = new Blob(this.chunks, {type: recorder.mimeType});
    this.chunks = [];
    return toSupportedAudio(recording);
  }

  /** Stops recording and discards the recording. */
  cancel() {
    const recorder = this.mediaRecorder;
    if (!recorder) return;
    if (recorder.state !== 'inactive') recorder.stop();
    this.release(recorder);
    this.chunks = [];
  }

  private release(recorder: MediaRecorder) {
    for (const track of recorder.stream.getTracks()) {
      track.stop();
    }
    this.mediaRecorder = null;
  }
}
