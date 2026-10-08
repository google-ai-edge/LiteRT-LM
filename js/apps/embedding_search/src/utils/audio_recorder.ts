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

/** Sample rate the audio graph is asked for; the engine resamples anyway. */
const PREFERRED_SAMPLE_RATE = 16000;

/**
 * In-browser microphone recorder that captures mono audio and packages raw
 * 16-bit linear PCM into a standard WAV container.
 *
 * The AudioContext is requested at 16 kHz, but browsers are free to ignore
 * that hint, so the WAV header always carries the rate the context actually
 * ran at. The engine's audio preprocessor resamples to the model's rate.
 */
export class AudioRecorder {
  private mediaStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  // TODO: Move capture to an AudioWorkletNode. That needs a worklet module
  // served as a separate static file by both the dev server and the Vite
  // build (safevalues does not allow a Blob-URL script module), so the
  // deprecated but universally supported ScriptProcessorNode stays for now.
  private scriptProcessor: ScriptProcessorNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private audioChunks: Float32Array[] = [];
  private recording = false;
  private starting = false;
  private sessionGeneration = 0;
  /** Sample rate of the context that produced `audioChunks`. */
  private sampleRate = PREFERRED_SAMPLE_RATE;

  get isRecording(): boolean {
    return this.recording;
  }

  /**
   * Starts microphone recording.
   */
  async start(): Promise<void> {
    if (this.recording || this.starting) return;
    this.starting = true;
    const generation = ++this.sessionGeneration;

    this.audioChunks = [];
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: PREFERRED_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (e) {
      if (this.sessionGeneration === generation) {
        this.starting = false;
      }
      throw e;
    }

    // If stop() or cancel() was called while awaiting getUserMedia, release the
    // newly acquired tracks immediately and abort startup.
    if (this.sessionGeneration !== generation) {
      stopTracks(stream);
      return;
    }

    try {
      this.mediaStream = stream;

      const AudioContextClass =
          window.AudioContext ||
          (window as unknown as {webkitAudioContext: typeof AudioContext})
              .webkitAudioContext;
      this.audioContext =
          new AudioContextClass({sampleRate: PREFERRED_SAMPLE_RATE});
      this.sampleRate = this.audioContext.sampleRate;

      this.sourceNode = this.audioContext.createMediaStreamSource(stream);
      this.scriptProcessor =
          this.audioContext.createScriptProcessor(4096, 1, 1);

      this.scriptProcessor.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!this.recording) return;
        const channelData = event.inputBuffer.getChannelData(0);
        this.audioChunks.push(new Float32Array(channelData));
      };

      this.sourceNode.connect(this.scriptProcessor);
      const muteGain = this.audioContext.createGain();
      muteGain.gain.value = 0;
      this.scriptProcessor.connect(muteGain);
      muteGain.connect(this.audioContext.destination);
    } catch (e) {
      // Setting up the audio graph failed: release the microphone again and
      // leave the recorder ready for another attempt.
      await this.teardown();
      if (this.sessionGeneration === generation) {
        this.starting = false;
      }
      throw e;
    }

    this.starting = false;
    this.recording = true;
  }

  /**
   * Stops recording and returns the recorded audio as an audio/wav Blob.
   */
  async stop(): Promise<Blob> {
    this.sessionGeneration++;
    this.starting = false;
    this.recording = false;
    await this.teardown();

    const totalLength =
        this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.audioChunks = [];

    return this.encodeWav(merged, this.sampleRate);
  }

  /**
   * Cancels active recording without saving.
   */
  async cancel(): Promise<void> {
    this.sessionGeneration++;
    this.starting = false;
    this.recording = false;
    this.audioChunks = [];
    await this.teardown();
  }

  /** Releases the microphone and the audio graph. Safe to call repeatedly. */
  private async teardown(): Promise<void> {
    if (this.mediaStream) {
      stopTracks(this.mediaStream);
      this.mediaStream = null;
    }

    if (this.scriptProcessor) {
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.audioContext) {
      const context = this.audioContext;
      this.audioContext = null;
      try {
        await context.close();
      } catch (_) {}
    }
  }

  /**
   * Encodes float samples into a 16-bit linear PCM WAV container.
   */
  private encodeWav(samples: Float32Array, sampleRate: number): Blob {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // "RIFF" chunk descriptor
    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, 'WAVE');

    // "fmt " sub-chunk
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);             // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);              // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true);              // NumChannels (1 = mono)
    view.setUint32(24, sampleRate, true);     // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (rate * 1 ch * 2 B)
    view.setUint16(32, 2, true);              // BlockAlign (1 * 2)
    view.setUint16(34, 16, true);             // BitsPerSample (16)

    // "data" sub-chunk
    this.writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples with clamping
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const sample = Math.max(-1, Math.min(1, samples[i]!));
      const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset, int16, true);
      offset += 2;
    }

    return new Blob([buffer], {type: 'audio/wav'});
  }

  private writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }
}

function stopTracks(stream: MediaStream) {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
