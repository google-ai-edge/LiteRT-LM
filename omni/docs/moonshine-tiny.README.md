---
license: mit
library_name: litert
base_model:
- UsefulSensors/moonshine-tiny
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
tags:
- audio
- automatic-speech-recognition
- litert
- tflite
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Moonshine Tiny — LiteRT

[Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) is a
27M-parameter encoder-decoder speech-recognition model from Moonshine AI
(formerly Useful Sensors), introduced in
[Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608).
It transcribes English speech and is designed for fast on-device inference.

This repository packages the model for [LiteRT](https://ai.google.dev/edge/litert):
a float32 model, an int8 model (dynamic-range-quantized decoder with a
float32 encoder), and ahead-of-time compiled variants for a range of MediaTek
and Qualcomm SoCs so the model can run on the device NPU.

## Model description

Each `.tflite` file contains two signatures that together form the
transcription loop:

| Signature | Inputs | Output |
|---|---|---|
| `encode` | raw audio `[1, 80000]` float32 (5 s at 16 kHz, zero-padded) | encoder states `[1, 207, 288]` float32 |
| `decode` | states `[1, 207, 288]`, tokens `[1, 64]` int32, additive causal mask `[1, 1, 64, 64]` float32 | logits `[1, 64, 32768]` float32 |

- The audio frontend is inside the graph: the model takes a raw 16 kHz
  waveform in `[-1, 1]` — no mel-spectrogram extraction is needed.
- The window is fixed at 5 seconds. Longer audio is transcribed in
  consecutive 5 s windows; shorter audio is zero-padded.
- Decoding is greedy: start token `1`, EOS token `2`, at most 64 tokens per
  window. The decoder re-scores the full token buffer each step (no KV
  cache), so decode time grows with the number of emitted tokens.
- The tokenizer is not duplicated in this repository — load `tokenizer.json`
  from the base model repository (see the script below).

### Files

| File | Description |
|---|---|
| `moonshine_tiny_5s_f32.tflite` | float32 model (109 MB) |
| `moonshine_tiny_5s_i8.tflite` | int8 model (52 MB): float32 encoder + dynamic-range int8 decoder |
| `moonshine_tiny_5s_f32_MediaTek_*.tflite` | float32 AOT-compiled for MediaTek NPUs (per SoC) |
| `moonshine_tiny_5s_f32_Qualcomm_*.tflite` | float32 AOT-compiled for Qualcomm NPUs (per SoC) |

## How to use

**1. Install dependencies**

```bash
pip install ai-edge-litert numpy tokenizers huggingface_hub
```

**2. Save the script** below as `transcribe.py`:

```python
#!/usr/bin/env python3
"""Transcribe a wav file with litert-community/moonshine-tiny (LiteRT)."""
import argparse
import wave

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

WINDOW_SAMPLES = 80000  # 5 s at 16 kHz
MAX_TOKENS = 64
START_TOKEN = 1
EOS_TOKEN = 2


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Reads a wav file as float32 mono at 16 kHz."""
    with wave.open(path, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(round(len(audio) * 16000 / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


class MoonshineTiny:
    """5 s window encoder/decoder with greedy decoding."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = CompiledModel.from_file(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.encode_idx = self.model.get_signature_index("encode")
        self.decode_idx = self.model.get_signature_index("decode")
        # Additive causal mask: 0 on and below the diagonal, -1e9 above.
        causal = np.tril(np.ones((MAX_TOKENS, MAX_TOKENS), dtype=bool))
        self.mask = np.where(causal, 0.0, -1e9).astype(np.float32)[None, None]

    def _transcribe_window(self, audio: np.ndarray) -> str:
        """Transcribes up to 5 s of 16 kHz audio."""
        buf = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
        buf[0, : len(audio)] = audio

        enc_in = self.model.create_input_buffers(self.encode_idx)
        enc_out = self.model.create_output_buffers(self.encode_idx)
        enc_in[0].write(buf)
        self.model.run_by_index(self.encode_idx, enc_in, enc_out)
        states = enc_out[0].read((1, 207, 288), np.float32)

        tokens = np.full((1, MAX_TOKENS), EOS_TOKEN, dtype=np.int32)
        tokens[0, 0] = START_TOKEN
        dec_in = self.model.create_input_buffers(self.decode_idx)
        dec_out = self.model.create_output_buffers(self.decode_idx)
        dec_in[0].write(states)
        dec_in[2].write(self.mask)

        decoded = []
        for position in range(1, MAX_TOKENS):
            dec_in[1].write(tokens)
            self.model.run_by_index(self.decode_idx, dec_in, dec_out)
            logits = dec_out[0].read((1, MAX_TOKENS, 32768), np.float32)
            next_token = int(np.argmax(logits[0, position - 1]))
            if next_token == EOS_TOKEN:
                break
            tokens[0, position] = next_token
            decoded.append(next_token)
        return self.tokenizer.decode(decoded).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribes audio of any length in consecutive 5 s windows."""
        parts = [
            self._transcribe_window(audio[i : i + WINDOW_SAMPLES])
            for i in range(0, max(len(audio), 1), WINDOW_SAMPLES)
        ]
        return " ".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a wav file.")
    parser.add_argument(
        "--variant", default="f32", choices=["f32", "i8"], help="Model variant."
    )
    args = parser.parse_args()

    model_path = hf_hub_download(
        "litert-community/moonshine-tiny", f"moonshine_tiny_5s_{args.variant}.tflite"
    )
    tokenizer_path = hf_hub_download("UsefulSensors/moonshine-tiny", "tokenizer.json")

    asr = MoonshineTiny(model_path, tokenizer_path)
    audio = load_wav_16k_mono(args.wav)
    print(asr.transcribe(audio))


if __name__ == "__main__":
    main()
```

**3. Run it** on a 16 kHz mono wav file:

```bash
python transcribe.py --wav sample.wav
```

### Android sample app

For an on-device Android demo that runs Moonshine (and other ASR models) with
hardware acceleration, see the LiteRT
[speech recognition sample](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/speech_recognition).

## Performance

Measured on one 5 s window of continuous speech (11 output tokens), CPU
inference, median of 10 runs. The macOS and Raspberry Pi rows use the Python
Interpreter API as in the script above (XNNPack, 4 threads,
`ai-edge-litert` 2.1.6); the iPhone rows use the LiteRT CompiledModel C API
with the CPU accelerator at default threading:

| Device | Variant | Encode | Decode | Window total | RTF |
|---|---|---|---|---|---|
| iPhone 17 Pro | f32 | 10.9 ms | 70.2 ms | 81.2 ms | 0.016 |
| iPhone 17 Pro | i8 | 10.7 ms | 69.3 ms | 80.0 ms | 0.016 |
| Apple M4 Max (macOS) | f32 | 8.1 ms | 79.4 ms | 87.5 ms | 0.017 |
| Apple M4 Max (macOS) | i8 | 7.4 ms | 72.6 ms | 80.0 ms | 0.016 |
| Raspberry Pi 5 | f32 | 50.7 ms | 444.7 ms | 495.3 ms | 0.099 |
| Raspberry Pi 5 | i8 | 50.3 ms | 267.4 ms | 317.7 ms | 0.064 |

RTF = processing time / audio duration (lower is better; below 1.0 is faster
than real time). Decode dominates and scales with the number of emitted
tokens, so dense speech takes proportionally longer than sparse speech. The
i8 model's int8 decoder makes it about 1.6x faster than f32 on the Pi 5's
Cortex-A76; on Apple silicon (M4 Max, iPhone 17 Pro) the two are equally
fast. The greedy decode is deterministic across platforms: the same window
produces bit-identical f32 token sequences on all three devices, the i8
model reproduces the f32 token sequence exactly on the dense test window on
both Apple devices, and its transcripts are identical between the M4 Max
and the Pi 5 on all 12 test clips.

### Snapdragon NPU / GPU — Galaxy S26

Measured on a physical Samsung Galaxy S26 (SM-S942Q, Snapdragon 8 Elite Gen 5 / SM8850, Hexagon v81, Android 16) with LiteRT `CompiledModel` 2.2.0 — 5 warm-up runs then 50 timed runs, one accelerator per process, every row at device thermal status `NONE`, delegate placement confirmed from logcat per row. The figures are the **`encode` signature only** (the model's first signature, which the API's default `run()` executes — the encoder over the raw 80,000-sample / 5 s waveform input); not comparable to the end-to-end window totals in the CPU table above. NPU in HTP BURST mode, on-device (JIT) compilation.

| File | Compute unit | Encode (median / min) | Load |
|---|---|---|---|
| `moonshine_tiny_5s_f32.tflite` | **GPU (Adreno)** | **5.4 ms** / 5.4 ms | 1.7 s |
| `moonshine_tiny_5s_i8.tflite` | NPU (Hexagon, JIT) — first launch | 1294 ms / 1281 ms | 13.1 s |
| `moonshine_tiny_5s_i8.tflite` | NPU (Hexagon, JIT) — cached | 1264 ms / 1247 ms | 0.37 s |

What the table says:

- **On this device the GPU running the f32 file is the only fast accelerated path**: 5.4 ms for the 5 s window's encoder, encoder-only RTF 0.001.
- **The Hexagon runs the i8 encoder in ~1.26 s** — slower than the Pi 5's CPU in the table above. The two accelerators also refuse each other's file: f32 compiles for the NPU but fails at output-buffer creation, and i8 fails to compile on the GPU. So there is no NPU recommendation to make from this run.
- `moonshine_tiny_5s_f32_Qualcomm_SM8850.tflite` (the embedded precompiled context) failed to invoke on both accelerators in this harness and is not quoted.

### Accuracy note

In a 12-clip spot check (LibriSpeech dev-clean samples plus two
public-domain clips), the f32 model transcribes clips of up to 5 s at
near-reference quality, and the i8 model matches it: the same overall word
error rate on the 12-clip harness (within chunking noise), word-level
divergence from the f32 transcripts of 2.5%, and encoder output that is
bit-identical to f32 because the encoder is not quantized. The encoder is kept in float32
deliberately — the convolutional audio frontend on the raw waveform does not
survive dynamic-range quantization (an earlier fully-quantized i8 upload
degraded badly for exactly this reason) — while the decoder, which dominates
latency, carries the int8 weights.

For the source model's quality, the
[Moonshine paper](https://arxiv.org/abs/2410.15608) reports that Moonshine
Tiny matches Whisper tiny.en word error rates across standard evaluation
datasets at about 5x less compute.

## License and attribution

The original Moonshine Tiny model is released by Moonshine AI under the MIT
license; these converted artifacts inherit it. If you use this model, please
cite:

```bibtex
@misc{jeffries2024moonshinespeechrecognitionlive,
  title={Moonshine: Speech Recognition for Live Transcription and Voice Commands},
  author={Nat Jeffries and Evan King and Manjunath Kudlur and Guy Nicholson and James Wang and Pete Warden},
  year={2024},
  eprint={2410.15608},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2410.15608},
}
```
