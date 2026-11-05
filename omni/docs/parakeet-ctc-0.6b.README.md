---
license: cc-by-4.0
library_name: litert-lm
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-ctc-0.6b
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
  - parakeet
  - ctc
  - fastconformer
  - int8
  - npu
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Parakeet-CTC-0.6B for LiteRT-LM

Fully on-device English speech-to-text in a single file.

This is
[nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
(600M-parameter FastConformer encoder + Connectionist Temporal Classification,
CTC, decoder) packaged as a self-contained `.litertlm` bundle for
[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni),
Google's open-source on-device inference runtime across Android, iOS, macOS,
Linux, and C++, alongside standalone `.tflite` and AOT-compiled NPU variants for
[LiteRT](https://github.com/google-ai-edge/LiteRT). Everything needed to
transcribe arbitrary-length English audio is inside the one `.litertlm` file:
INT8 FastConformer-CTC encoder, 1,024-token English SentencePiece tokenizer, and
`Speech2TextMetadata` (16 kHz resampler and 80-bin Slaney log-mel front-end
configuration) — no Python and no network required.

-   **Languages:** English (`en`, lowercase alphabet with spaces and
    apostrophes)
-   **Architecture:** Non-autoregressive single-pass FastConformer-CTC encoder
    (500 mel frames @ 10 ms → 63 frames @ 80 ms) with greedy CTC repeat collapse
    and blank removal (`CtcDecoder`)
-   **Streaming:** 5-second sliding windows with configurable overlap
    (`overlap_ratio` of `0.4` to `0.75`), warm-start early first-chunk emission,
    and frame-timestamp dynamic-programming seam alignment
    (`TimestampTextMerger`) for arbitrary-length audio

## Which file should I use?

| File | Size | Backend | Recommended for |
|---|---:|---|---|
| **`parakeet_ctc_0.6b_5s_i8.litertlm`** ⭐ | 597 MB | CPU / GPU (`--backend=cpu` or `--backend=gpu`) | **[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni) (`OmniEngine` / `omni_runner`) across Android, iOS, macOS, Linux, and C++.** All-in-one `.litertlm` bundle with INT8 FastConformer-CTC encoder, tokenizer, and `Speech2TextMetadata`. |
| `parakeet_ctc_0.6b_5s_i8.tflite` | 596 MB | CPU / GPU | Direct [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr) integration without LiteRT-LM. |
| `parakeet_ctc_0.6b_5s_f32.tflite` | 2.35 GB | CPU / GPU | Unquantized FP32 reference baseline. |
| `parakeet_ctc_0.6b_5s_f32_<SoC>.tflite` | 1.18–1.21 GB | NPU + CPU | Standalone LiteRT NPU pipelines AOT-compiled for a specific mobile SoC (see below). |

How the variants differ:

-   **`.litertlm` vs. `.tflite`** — the `.litertlm` file bundles the `.tflite`
    graph together with the compressed Hugging Face `tokenizer.json` and
    `Speech2TextMetadata` protobuf in one self-contained archive for
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)'s
    `OmniEngine` and `omni_runner`. Standalone `.tflite` files contain only the
    neural network graph and expect the host app to supply mel preprocessing,
    CTC decoding, and detokenization.
-   **INT8 (`_i8`) vs. FP32 (`_f32`)** — INT8 dynamic range quantization shrinks
    the model from **2.35 GB to 597 MB** (`4×` smaller on disk and in RAM) and
    accelerates mobile ARM64 CPU inference via `SDOT`/`I8MM` SIMD instructions,
    while running the entire 5-second window in a single non-autoregressive
    forward pass on either CPU or GPU.
-   **AOT-compiled NPU `.tflite` files (`_<SoC>.tflite`)** — pre-compiled for
    on-device NPU accelerators via the LiteRT NeuroPilot / QNN / Google Tensor
    compilers:

| SoC family | Available `<SoC>` suffixes |
|---|---|
| **Google Tensor** | `Google_Tensor_G3` |
| **Qualcomm Snapdragon** | `Qualcomm_SM8450` (8 Gen 1), `SM8550` (8 Gen 2), `SM8650` (8 Gen 3), `SM8750` (8 Elite), `SM8850` (8 Elite Gen 5), `SA8255`, `SA8295` |
| **MediaTek Dimensity** | `MediaTek_MT6877`, `MT6878`, `MT6879`, `MT6897`, `MT6983` (9000), `MT6985` (9200), `MT6989` (9300), `MT6991` (9400), `MT6993`, `MT8171`, `MT8188`, `MT8189` |

## Download

```sh
pip install -U huggingface_hub

# Recommended: INT8 .litertlm bundle (597 MB, English)
hf download litert-community/parakeet-ctc-0.6b \
    parakeet_ctc_0.6b_5s_i8.litertlm --local-dir .
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
hf download litert-community/parakeet-ctc-0.6b \
    parakeet_ctc_0.6b_5s_i8.litertlm --local-dir .
```

> [!TIP]
> If you omit `--model_path` and pass `--model_name=parakeet-ctc-0.6b`,
> `omni_runner` automatically downloads `parakeet_ctc_0.6b_5s_i8.litertlm` into
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
bazel-bin/omni/omni_runner --model_name=parakeet-ctc-0.6b \
    --model_path=/path/to/parakeet_ctc_0.6b_5s_i8.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav
```

**On an Android phone (CPU):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/parakeet_ctc_0.6b_5s_i8.litertlm \
         runtime/components/testdata/audio_sample.wav $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=parakeet-ctc-0.6b \
    --backend=cpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/parakeet_ctc_0.6b_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

**On an Android phone (GPU):**

```sh
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=parakeet-ctc-0.6b \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/parakeet_ctc_0.6b_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

> `--model_name=parakeet-ctc-0.6b` (or alias `parakeet-ctc`) tells the runner
> which model family the file is (it is inferred automatically when the file is
> named `parakeet-ctc-0.6b.litertlm`). XNNPACK weight packs and compiled GPU
> kernels are cached in `--cache_dir` so subsequent runs start fast.

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
    .model_path = "/data/local/tmp/omni/parakeet_ctc_0.6b_5s_i8.litertlm",
};
litert::lm::proto::Speech2TextSessionConfig asr_config;
asr_config.set_overlap_ratio(0.4f);
asr_config.set_text_merger_type(
    litert::lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("parakeet-ctc-0.6b", settings, &asr_config));
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

In your `BUILD` file, link only the Parakeet-CTC target
(`//omni/speech2text:parakeet_ctc`) to avoid pulling in other speech, TTS, or
image generation models:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:parakeet_ctc",
    ],
)
```

## What's in the file

| Component | Details |
|---|---|
| **Audio front-end** | Self-contained C++ preprocessor (`Speech2TextMetadata`): resamples any input rate to 16 kHz mono, applies `0.97` pre-emphasis, 512-point Hann STFT (25 ms window, 10 ms hop, 500 frames per 5 s window), 80-bin Slaney log-mel filterbank (0–8 kHz), and per-feature mean/std normalization (`[1, 80, 500]`). |
| **Encoder & CTC head** | 600M-parameter FastConformer-CTC (`tf_lite_audio_encoder`) with 8× depthwise-separable conv subsampling (500 mel frames @ 10 ms → 63 frames @ 80 ms, `1024`-dim) and linear CTC head outputting `[1, 63, 1025]` frame-level logits (`1,024` tokens + `<blank>`), quantized to channel-wise INT8 weights. |
| **CTC decoder** | C++ `CtcDecoder`: single-pass greedy argmax per frame with consecutive repeat deduplication, `<blank>` removal, window-boundary `DecodeRange` slicing, and per-token frame timestamp emission. |
| **Tokenizer** | Zlib-compressed Hugging Face `tokenizer.json` (`HF_Tokenizer_Zlib`, 1,024-token English SentencePiece vocabulary) + incremental word-timestamp detokenizer. |
| **Text merger** | `TimestampTextMerger`: monotone dynamic-programming seam alignment across overlapping 5 s windows (`overlap_ratio = 0.4` in `omni_runner`, `0.75` default in `AsrSession`). |
| **Output** | Streaming `OmniSession::TextOutput` chunks (`confirmed_text` + `unconfirmed_text`). |

The models were exported from the upstream NeMo checkpoint
(`parakeet-ctc-0.6b.nemo`) to LiteRT via PyTorch and
[litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized
with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

-   **English lowercase only (no punctuation or capitalization):** Like the
    upstream `nvidia/parakeet-ctc-0.6b` checkpoint, this model outputs lowercase
    English characters, spaces, and apostrophes without punctuation or
    capitalization. If you need automatic punctuation, capitalization, or
    multilingual European speech recognition, use
    [`litert-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3)
    (`--model_name=parakeet-tdt-0.6b-v3`) or
    [`litert-community/Qwen3-ASR-0.6B`](https://huggingface.co/litert-community/Qwen3-ASR-0.6B)
    (`--model_name=qwen3-asr-0.6b`).
-   **Fixed 5-second encoder window:** Longer audio is processed in overlapping
    5-second windows (2.5 to 3.0 s stride). Words spoken right across a window
    boundary are emitted immediately in `unconfirmed_text` and promoted to
    `confirmed_text` once the next overlapping window resolves the boundary seam.
-   **SoC-specific NPU files are standalone `.tflite` graphs:** The pre-compiled
    `parakeet_ctc_0.6b_5s_f32_<SoC>.tflite` files target the standalone LiteRT
    NPU C++/Kotlin APIs (such as the
    [LiteRT ASR Android sample app](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr))
    rather than `.litertlm` bundles.

## License & acknowledgements

-   **Weights & tokenizer:**
    [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en),
    inherited from
    [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
    (trained by NVIDIA NeMo on 64,000 hours of English speech;
    [FastConformer](https://arxiv.org/abs/2305.05084), Rekesh et al.).
-   **Conversion & runtime:**
    [litert-torch](https://github.com/google-ai-edge/litert-torch),
    [LiteRT](https://github.com/google-ai-edge/LiteRT), and
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)
    (Apache-2.0).
-   **Training data, as disclosed by the upstream model:** 64,000 hours of
    English speech from the NeMo ASR Set 3.0 corpus (LibriSpeech, Fisher,
    Switchboard-1, WSJ-0/1, National Speech Corpus, VCTK, VoxPopuli, Europarl,
    Multilingual LibriSpeech, Mozilla Common Voice, People's Speech, and
    NVIDIA-internal datasets). This conversion adds no training data and no
    personal information.
