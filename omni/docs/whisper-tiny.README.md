---
license: apache-2.0
library_name: litert-lm
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-tiny
base_model_relation: quantized
language:
  - en
  - zh
  - de
  - es
  - ru
  - ko
  - fr
  - ja
  - pt
  - tr
  - pl
  - ca
  - nl
  - ar
  - sv
  - it
  - id
  - hi
  - fi
  - vi
  - he
  - uk
  - el
  - ms
  - cs
  - ro
  - da
  - hu
  - ta
  - no
  - th
  - ur
  - hr
  - bg
  - lt
  - la
  - mi
  - ml
  - cy
  - sk
  - te
  - fa
  - lv
  - bn
  - sr
  - az
  - sl
  - kn
  - et
  - mk
  - br
  - eu
  - is
  - hy
  - ne
  - mn
  - bs
  - kk
  - sq
  - sw
  - gl
  - mr
  - pa
  - si
  - km
  - sn
  - yo
  - so
  - af
  - oc
  - ka
  - be
  - tg
  - sd
  - gu
  - am
  - yi
  - lo
  - uz
  - fo
  - ht
  - ps
  - tk
  - nn
  - mt
  - sa
  - lb
  - my
  - bo
  - tl
  - mg
  - as
  - tt
  - haw
  - ln
  - ha
  - ba
  - jw
  - su
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
  - whisper
  - int8
  - npu
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Whisper-Tiny for LiteRT-LM

Compact (~42 MB), on-device speech-to-text in a single file.

This is
[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
(39M-parameter encoder-decoder Transformer introduced in
[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356),
Radford et al.) packaged as a self-contained `.litertlm` bundle for
[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni),
Google's open-source on-device inference runtime across Android, iOS, macOS,
Linux, and C++, alongside standalone `.tflite` and AOT-compiled NPU variants for
[LiteRT](https://github.com/google-ai-edge/LiteRT). Everything needed to
transcribe arbitrary-length audio with punctuation and capitalization is inside
the one `.litertlm` file: INT8 Whisper Tiny encoder-decoder transformer,
51,865-token multilingual BPE tokenizer, and `Speech2TextMetadata` (16 kHz
resampler and 30-second 80-bin Slaney log-mel front-end configuration) — no
Python and no network required.

-   **Footprint:** **41.8 MB** `.litertlm` bundle (`3.6×` smaller than the
    151 MB FP32 checkpoint)
-   **Languages:** Multilingual Whisper Tiny checkpoint (99 languages in the
    upstream vocabulary; defaults to English transcription `<|en|>` when no
    custom `decode_start_token_ids` are specified)
-   **Windowing:** 30-second audio windows (`input_milliseconds = 30000`, 3,000
    mel frames at 10 ms hop) with configurable sliding-window overlap and seam
    alignment for recordings longer than 30 seconds

## Which file should I use?

| File | Size | Backend | Recommended for |
|---|---:|---|---|
| **`whisper_tiny_30s_i8.litertlm`** ⭐ | 41.8 MB | CPU / GPU (`--backend=cpu` recommended) | **[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni) (`OmniEngine` / `omni_runner`) across Android, iOS, macOS, Linux, and C++.** All-in-one `.litertlm` bundle with INT8 30 s Whisper Tiny model, tokenizer, and `Speech2TextMetadata`. |
| `whisper_tiny_30s_i8.tflite` | 41.1 MB | CPU / GPU | Direct [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr) integration without LiteRT-LM (`encode` and `decode` signatures). |
| `whisper_tiny_30s_f32.tflite` | 151 MB | CPU / GPU | Unquantized FP32 reference baseline. |
| `whisper_tiny_30s_f32_<SoC>.tflite` | 119–158 MB | NPU + CPU | Standalone LiteRT NPU pipelines AOT-compiled for a specific mobile SoC (see below). |

How the variants differ:

-   **`.litertlm` vs. `.tflite`** — the `.litertlm` file bundles the `.tflite`
    graph together with the compressed Hugging Face `tokenizer.json` and
    `Speech2TextMetadata` protobuf in one self-contained archive for
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)'s
    `OmniEngine` and `omni_runner`. Standalone `.tflite` files contain only the
    neural network graph and expect the host app to supply 80-bin Whisper
    log-mel preprocessing and BPE detokenization.
-   **INT8 (`_i8`) vs. FP32 (`_f32`)** — INT8 dynamic range quantization shrinks
    the model from **151 MB to 41.8 MB** (`3.6×` smaller on disk) and
    accelerates mobile ARM64 CPU inference.
-   **AOT-compiled NPU `.tflite` files (`_<SoC>.tflite`)** — pre-compiled for
    on-device NPU accelerators via the LiteRT NeuroPilot / QNN compilers:

| SoC family | Available `<SoC>` suffixes |
|---|---|
| **Qualcomm Snapdragon** | `Qualcomm_SM8450` (8 Gen 1), `SM8550` (8 Gen 2), `SM8650` (8 Gen 3), `SM8750` (8 Elite), `SM8850` (8 Elite Gen 5), `SA8255`, `SA8295` |
| **MediaTek Dimensity** | `MediaTek_MT6877`, `MT6878`, `MT6879`, `MT6897`, `MT6983` (9000), `MT6985` (9200), `MT6989` (9300), `MT6991` (9400), `MT6993`, `MT8171`, `MT8188`, `MT8189` |

## Download

```sh
pip install -U huggingface_hub

# Recommended: INT8 .litertlm bundle (41.8 MB)
hf download litert-community/whisper-tiny \
    whisper_tiny_30s_i8.litertlm --local-dir .
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
hf download litert-community/whisper-tiny \
    whisper_tiny_30s_i8.litertlm --local-dir .
```

> [!TIP]
> If you omit `--model_path` and pass `--model_name=whisper-tiny`,
> `omni_runner` automatically downloads `whisper_tiny_30s_i8.litertlm` into
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
bazel-bin/omni/omni_runner --model_name=whisper-tiny \
    --model_path=/path/to/whisper_tiny_30s_i8.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav
```

**On an Android phone (CPU, recommended):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/whisper_tiny_30s_i8.litertlm \
         runtime/components/testdata/audio_sample.wav $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=whisper-tiny \
    --backend=cpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/whisper_tiny_30s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

**On an Android phone (GPU):**

```sh
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=whisper-tiny \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/whisper_tiny_30s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

> `--model_name=whisper-tiny` (or alias `whisper`) tells the runner which model
> family the file is (it is inferred automatically when the file is named
> `whisper-tiny.litertlm`). XNNPACK weight packs and compiled GPU kernels are
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
    .model_path = "/data/local/tmp/omni/whisper_tiny_30s_i8.litertlm",
};
litert::lm::proto::Speech2TextSessionConfig asr_config;
asr_config.set_overlap_ratio(0.4f);

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("whisper-tiny", settings, &asr_config));
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

In your `BUILD` file, link only the Whisper target
(`//omni/speech2text:whisper`) to avoid pulling in other speech, TTS, or image
generation models:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:whisper",
    ],
)
```

## What's in the file

Inside `whisper_tiny_30s_i8.litertlm`:

| Component | Details |
|---|---|
| **Audio front-end** | Self-contained C++ `LogMelSpectrogramProcessor` (`Speech2TextMetadata`): resamples any input audio rate to 16 kHz mono, computes 400-point Hann STFT (25 ms window, 10 ms hop, 3,000 frames per 30 s window), 80-bin Slaney power mel filterbank (`mel_floor = 1e-10`), and Whisper dynamic-range log normalization (`NORM_TYPE_WHISPER`, `[1, 80, 3000]`). |
| **Encoder-decoder (`tf_lite_encoder_decoder`)** | 39M-parameter Whisper Tiny transformer (`~41.1 MB`, INT8 weights) with two signatures: `encode` (`[1, 80, 3000]` log-mel spectrogram → encoder hidden states `[1, 1500, 384]`) and `decode` (encoder states, token IDs, and additive causal mask → vocabulary logits). |
| **Decoder (`StatelessDecoder`)** | Greedy autoregressive decoding seeded by `decode_start_token_ids = [50258, 50259, 50359, 50363]` (`<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`) until `<|endoftext|>` (`50257`), with automatic trailing repetition truncation. |
| **Tokenizer (`HF_Tokenizer_Zlib`)** | Zlib-compressed Whisper multilingual BPE `tokenizer.json` (51,865 tokens with punctuation, capitalization, and language/task control tokens). |
| **Text merger** | Cross-window overlap seam alignment across consecutive 30 s windows (`overlap_ratio = 0.4` in `omni_runner`). |
| **Output** | Streaming `OmniSession::TextOutput` chunks (`confirmed_text` + `unconfirmed_text`). |

The models were exported from the upstream PyTorch checkpoint
(`openai/whisper-tiny`) with
[litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized
with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

-   **Fixed 30-second encoder window:** Whisper Tiny's encoder always processes
    a 30-second (3,000-frame) mel-spectrogram window; clips shorter than 30
    seconds are zero-padded to 30 seconds before `encode`. For lower-latency 5 s
    streaming windows, consider
    [`litert-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3),
    [`litert-community/parakeet-ctc-0.6b`](https://huggingface.co/litert-community/parakeet-ctc-0.6b),
    [`litert-community/Qwen3-ASR-0.6B`](https://huggingface.co/litert-community/Qwen3-ASR-0.6B),
    or
    [`litert-community/moonshine-tiny`](https://huggingface.co/litert-community/moonshine-tiny).
-   **Default decoder prefix targets English (`<|en|>`):** By default,
    `WhisperSessionFactory` seeds `StatelessDecoder` with
    `[50258, 50259, 50359, 50363]` (`<|startoftranscript|>`, `<|en|>`,
    `<|transcribe|>`, `<|notimestamps|>`). Transcribing another language with
    Whisper Tiny requires overriding `decode_start_token_ids` in
    `Speech2TextMetadata` with that language's Whisper language token ID. For
    automatic zero-config multilingual speech recognition, use
    [`litert-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3)
    (25 European languages) or
    [`litert-community/Qwen3-ASR-0.6B`](https://huggingface.co/litert-community/Qwen3-ASR-0.6B)
    (52 languages and dialects).
-   **SoC-specific NPU files are standalone `.tflite` graphs:** The pre-compiled
    `whisper_tiny_30s_f32_<SoC>.tflite` files target the standalone LiteRT NPU
    C++/Kotlin APIs rather than `.litertlm` bundles.

## License & acknowledgements

-   **Weights & tokenizer:**
    [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from
    [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
    ([Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356),
    Radford et al., OpenAI).
-   **Conversion & runtime:**
    [litert-torch](https://github.com/google-ai-edge/litert-torch),
    [LiteRT](https://github.com/google-ai-edge/LiteRT), and
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)
    (Apache-2.0).
-   **Training data, as disclosed by the upstream model:** 680,000 hours of
    labeled multilingual and multitask speech data collected from the internet.
    This conversion adds no training data and no personal information.
