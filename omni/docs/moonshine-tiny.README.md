---
license: mit
library_name: litert-lm
pipeline_tag: automatic-speech-recognition
base_model: UsefulSensors/moonshine-tiny
base_model_relation: quantized
language:
  - en
tags:
  - litert
  - litert-lm
  - tflite
  - asr
  - automatic-speech-recognition
  - speech-to-text
  - on-device
  - android
  - ios
  - moonshine
  - int8
  - npu
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Moonshine-Tiny for LiteRT-LM

Ultra-lightweight (~30 MB), fully on-device English speech-to-text in a single
file.

This is
[UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)
(27M-parameter encoder-decoder speech recognition transformer from Moonshine AI,
formerly Useful Sensors, introduced in
[Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608))
packaged as a self-contained `.litertlm` bundle for
[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni),
Google's open-source on-device inference runtime across Android, iOS, macOS,
Linux, and C++, alongside standalone `.tflite` and AOT-compiled NPU variants for
[LiteRT](https://github.com/google-ai-edge/LiteRT). Everything needed to
transcribe English speech with punctuation and capitalization is inside the one
`.litertlm` file: INT8 encoder-decoder transformer with an in-graph raw-waveform
convolutional front-end, 32,768-token Hugging Face BPE tokenizer, and
`Speech2TextMetadata` — no Python and no network required.

-   **Languages:** English (`en`) with automatic punctuation and capitalization
-   **Footprint:** **29.7 MB** `.litertlm` bundle (smallest speech-to-text model
    in LiteRT-LM Omni)
-   **Audio front-end:** Raw 16 kHz waveform input (`[1, 80000]` float32 per 5 s
    window) — the convolutional audio stem is built directly into the TFLite
    `encode` graph, so no external STFT or mel-spectrogram computation is needed
-   **Streaming:** 5-second sliding windows with configurable overlap
    (`overlap_ratio` of `0.4` to `0.75`) and cross-window seam alignment for
    arbitrary-length audio

## Which file should I use?

| File | Size | Backend | Recommended for |
|---|---:|---|---|
| **`moonshine_tiny_5s_i8.litertlm`** ⭐ | 29.7 MB | CPU / GPU (`--backend=cpu` recommended) | **[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni) (`OmniEngine` / `omni_runner`) across Android, iOS, macOS, Linux, and C++.** All-in-one `.litertlm` bundle with INT8 encoder-decoder model, bundled 32,768-token tokenizer, and `Speech2TextMetadata`. |
| `moonshine_tiny_5s_i8.tflite` | 51.9 MB | CPU | Standalone [LiteRT](https://ai.google.dev/edge/litert) `.tflite` model (FP32 encoder + dynamic-range INT8 decoder; see [Standalone `.tflite` usage](#standalone-tflite-usage-python-litert) below). |
| `moonshine_tiny_5s_f32.tflite` | 109 MB | CPU / GPU | Unquantized FP32 standalone `.tflite` model (`encode` and `decode` signatures). |
| `moonshine_tiny_5s_f32_<SoC>.tflite` | 77–102 MB | NPU + CPU | Standalone LiteRT NPU pipelines AOT-compiled for a specific mobile SoC (see below). |

How the variants differ:

-   **`.litertlm` vs. `.tflite`** — the `.litertlm` file bundles the `.tflite`
    graph together with the compressed Hugging Face `tokenizer.json` and
    `Speech2TextMetadata` protobuf in one self-contained archive for
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)'s
    `OmniEngine` and `omni_runner`. Standalone `.tflite` files contain only the
    neural network graph and require loading `tokenizer.json` from the upstream
    repository.
-   **AOT-compiled NPU `.tflite` files (`_<SoC>.tflite`)** — pre-compiled for
    on-device NPU accelerators via the LiteRT NeuroPilot / QNN compilers:

| SoC family | Available `<SoC>` suffixes |
|---|---|
| **Qualcomm Snapdragon** | `Qualcomm_SM8450` (8 Gen 1), `SM8550` (8 Gen 2), `SM8650` (8 Gen 3), `SM8750` (8 Elite), `SM8850` (8 Elite Gen 5), `SA8255`, `SA8295` |
| **MediaTek Dimensity** | `MediaTek_MT6877`, `MT6878`, `MT6879`, `MT6897`, `MT6983` (9000), `MT6985` (9200), `MT6989` (9300), `MT6991` (9400), `MT6993`, `MT8171`, `MT8188`, `MT8189` |

## Download

```sh
pip install -U huggingface_hub

# Recommended: INT8 .litertlm bundle (29.7 MB, English)
hf download litert-community/moonshine-tiny \
    moonshine_tiny_5s_i8.litertlm --local-dir .
```

---

## Usage with LiteRT-LM Omni (`omni_runner` & C++)

All commands below accept mono or multi-channel WAV files at any sample rate
(automatically resampled to 16 kHz mono) and stream confirmed and unconfirmed
transcript segments to stdout (or `--output_path`). Prerequisites (Bazel,
Android NDK) are in the LiteRT-LM
[build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md)
and the
[LiteRT-LM Omni documentation](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

### 1. Download a model

```sh
pip install -U huggingface_hub
hf download litert-community/moonshine-tiny \
    moonshine_tiny_5s_i8.litertlm --local-dir .
```

> [!TIP]
> If you omit `--model_path` and pass `--model_name=moonshine-tiny`,
> `omni_runner` automatically downloads `moonshine_tiny_5s_i8.litertlm` into
> `--cache_dir` on first run.

### 2. Build the runner

```sh
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
bazel build -c opt //omni:omni_runner                            # Linux / macOS
bazel build -c opt --config=android_arm64 //omni:omni_runner     # Android (needs ANDROID_NDK_HOME)
```

### 3. Transcribe speech

**On Linux or macOS (CPU):**

```sh
bazel-bin/omni/omni_runner --model_name=moonshine-tiny \
    --model_path=/path/to/moonshine_tiny_5s_i8.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav
```

**On an Android phone (CPU, recommended):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/moonshine_tiny_5s_i8.litertlm \
         runtime/components/testdata/audio_sample.wav $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=moonshine-tiny \
    --backend=cpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/moonshine_tiny_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

**On an Android phone (GPU):**

```sh
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=moonshine-tiny \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/moonshine_tiny_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

> `--model_name=moonshine-tiny` (or alias `moonshine`) tells the runner which
> model family the file is (it is inferred automatically when the file is named
> `moonshine-tiny.litertlm`). XNNPACK weight packs and compiled GPU kernels are
> cached in `--cache_dir` so subsequent runs start fast.

The full list of flags and the C++ `OmniEngine` / `OmniSession` streaming API
are documented in the
[LiteRT-LM Omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

### Use from C++ (`OmniEngine`)

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/proto/speech2text_session_config.pb.h"

using ::litert::omni::OmniEngine;
using ::litert::omni::OmniSession;
using ::litert::omni::PushInputSource;

OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kCpu,
    .cache_dir = "/data/local/tmp/omni/cache",
    .model_path = "/data/local/tmp/omni/moonshine_tiny_5s_i8.litertlm",
};
litert::lm::proto::Speech2TextSessionConfig asr_config;
asr_config.set_overlap_ratio(0.4f);

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("moonshine-tiny", settings, &asr_config));
auto input_source = std::make_unique<PushInputSource>();
PushInputSource* input = input_source.get();
ABSL_ASSIGN_OR_RETURN(auto session,
                      engine->CreateSession(std::move(input_source)));

// Push input audio metadata once, followed by one or more PCM float chunks.
ABSL_RETURN_IF_ERROR(input->PushInput(
    OmniSession::AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
ABSL_RETURN_IF_ERROR(input->PushInput(
    OmniSession::AudioInput{.pcm_samples = std::move(pcm_chunk)}));
input->Finish();
// Pull OmniSession::TextOutput updates (confirmed_text + unconfirmed_text)
// with session->ProcessNext() until OutOfRange or NotFound, then call session->Flush().
```

In your `BUILD` file, link only the Moonshine target
(`//omni/speech2text:moonshine`) to avoid pulling in other speech, TTS, or image
generation models:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:moonshine",
    ],
)
```

---

## Standalone `.tflite` Usage (Python LiteRT)

If you are using the standalone `.tflite` files (`moonshine_tiny_5s_f32.tflite`
or `moonshine_tiny_5s_i8.tflite`) directly with the Python
[LiteRT](https://ai.google.dev/edge/litert) `CompiledModel` API instead of
LiteRT-LM Omni:

### Model signatures

Each `.tflite` file contains two signatures that together form the transcription
loop:

| Signature | Inputs | Output |
|---|---|---|
| `encode` | raw audio `[1, 80000]` float32 (5 s at 16 kHz, zero-padded) | encoder states `[1, 207, 288]` float32 |
| `decode` | states `[1, 207, 288]`, tokens `[1, 64]` int32, additive causal mask `[1, 1, 64, 64]` float32 | logits `[1, 64, 32768]` float32 |

-   The audio frontend is inside the graph: the model takes a raw 16 kHz
    waveform in `[-1, 1]` — no mel-spectrogram extraction is needed.
-   The window is fixed at 5 seconds. Longer audio is transcribed in consecutive
    5 s windows; shorter audio is zero-padded.
-   Decoding is greedy: start token `1`, EOS token `2`, at most 64 tokens per
    window. The decoder re-scores the full token buffer each step (no KV cache),
    so decode time grows with the number of emitted tokens.
-   When using the standalone `.tflite` files directly, load `tokenizer.json`
    from the base model repository (`UsefulSensors/moonshine-tiny`), as shown
    below (the `.litertlm` bundle already embeds `tokenizer.json`).

### Run with Python `ai-edge-litert`

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

For an on-device Android demo that runs standalone `.tflite` Moonshine models
with hardware acceleration, see the LiteRT
[speech recognition sample](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/speech_recognition).

## Performance (Standalone `.tflite`)

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

## What's in the file

Inside `moonshine_tiny_5s_i8.litertlm`:

| Component | Details |
|---|---|
| **Audio front-end** | Self-contained C++ resampler + `DummyPreprocessor` (`Speech2TextMetadata`): resamples any input audio rate to 16 kHz mono and zero-pads short final chunks to 80,000 samples (5.0 s). The convolutional waveform front-end runs inside the TFLite `encode` graph. |
| **Encoder-decoder (`tf_lite_encoder_decoder`)** | 27M-parameter Moonshine Tiny transformer (`~29.1 MB`, INT8 weights) with two signatures: `encode` (raw waveform `[1, 80000]` float32 → encoder hidden states `[1, 207, 288]` float32) and `decode` (encoder states `[1, 207, 288]`, token IDs `[1, 64]` int32, additive causal mask `[1, 1, 64, 64]` float32 → logits `[1, 64, 32768]` float32). |
| **Decoder (`StatelessDecoder`)** | Greedy autoregressive decoding from start token `1` (`<s>`) until stop token `2` (`</s>`) or 64 tokens per 5 s window, with automatic trailing repetition truncation. |
| **Tokenizer (`HF_Tokenizer_Zlib`)** | Zlib-compressed Hugging Face `tokenizer.json` (32,768-token BPE vocabulary with punctuation and capitalization). |
| **Text merger** | Cross-window overlap seam alignment across consecutive 5 s windows (`overlap_ratio = 0.4` in `omni_runner`). |
| **Output** | Streaming `OmniSession::TextOutput` chunks (`confirmed_text` + `unconfirmed_text`). |

The models were exported from the upstream PyTorch checkpoint
(`UsefulSensors/moonshine-tiny`) with
[litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized
with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

-   **English only:** Moonshine Tiny is trained on English speech only. For
    multilingual speech recognition, use
    [`litert-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3)
    (25 European languages),
    [`litert-community/Qwen3-ASR-0.6B`](https://huggingface.co/litert-community/Qwen3-ASR-0.6B)
    (52 languages and dialects), or
    [`litert-community/whisper-tiny`](https://huggingface.co/litert-community/whisper-tiny)
    (99 languages).
-   **Fixed 5-second encoder window (max 64 tokens per window):** Longer audio
    is processed in overlapping 5-second windows. Because `decode` re-evaluates
    the 64-token buffer at each step without a stateful KV cache, `--backend=cpu`
    is recommended for lowest end-to-end latency.
-   **SoC-specific NPU files are standalone `.tflite` graphs:** The pre-compiled
    `moonshine_tiny_5s_f32_<SoC>.tflite` files target the standalone LiteRT NPU
    C++/Kotlin APIs rather than `.litertlm` bundles.

## License & acknowledgements

-   **Weights & tokenizer:** [MIT](https://opensource.org/licenses/MIT),
    inherited from
    [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)
    by Moonshine AI (formerly Useful Sensors).
-   **Conversion & runtime:**
    [litert-torch](https://github.com/google-ai-edge/litert-torch),
    [LiteRT](https://github.com/google-ai-edge/LiteRT), and
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)
    (Apache-2.0).

If you use this model, please cite:

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
