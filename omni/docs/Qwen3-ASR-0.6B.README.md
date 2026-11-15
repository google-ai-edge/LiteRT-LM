---
license: apache-2.0
library_name: litert-lm
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ASR-0.6B
base_model_relation: quantized
language:
  - zh
  - en
  - yue
  - ar
  - de
  - fr
  - es
  - pt
  - id
  - it
  - ko
  - ru
  - th
  - vi
  - ja
  - tr
  - hi
  - ms
  - nl
  - sv
  - da
  - fi
  - pl
  - cs
  - fil
  - fa
  - el
  - hu
  - mk
  - ro
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
  - qwen3-asr
  - qwen3
  - int8
  - npu
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Qwen3-ASR-0.6B for LiteRT-LM

Fully on-device, 52-language/dialect speech-to-text in a single file.

This is
[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
(600M-parameter audio-language model built on Qwen3-Omni) packaged as a
self-contained `.litertlm` bundle for
[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni),
Google's open-source on-device inference runtime across Android, iOS, macOS,
Linux, and C++, alongside standalone `.tflite` and AOT-compiled NPU variants for
[LiteRT](https://github.com/google-ai-edge/LiteRT). Everything needed to
transcribe arbitrary-length speech across 30 languages and 22 Chinese dialects
with automatic language identification, punctuation, and capitalization is
inside the one `.litertlm` file: INT8 audio encoder + projector, 28-layer Qwen3
autoregressive LM decoder with stateful KV cache, token embedding table, Qwen3
BPE tokenizer, and `LlmMetadata` / `ExecutorMetadata` — no Python and no network
required.

-   **Languages:** 30 languages and 22 Chinese dialects with automatic language
    identification, punctuation, and capitalization (see
    [Supported languages](#supported-languages))
-   **Architecture:** Whisper-style 128-bin Slaney log-mel front-end + audio
    encoder (`tf_lite_audio_encoder_hw`) + 28-layer Qwen3 autoregressive
    language model decoder (`tf_lite_prefill_decode` + `tf_lite_embedder`) with
    stateful KV cache (`LmDecoder`)
-   **Streaming:** 5-second sliding windows with configurable overlap
    (`overlap_ratio` of `0.4` to `0.75`), warm-start early first-chunk emission,
    and cross-window overlap seam alignment for arbitrary-length audio

## Which file should I use?

| File | Size | Backend | Recommended for |
|---|---:|---|---|
| **`qwen3_asr_0.6b_5s_i8.litertlm`** ⭐ | 960 MB | CPU / GPU (`--backend=cpu` or `--backend=gpu`) | **[LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni) (`OmniEngine` / `omni_runner`) across Android, iOS, macOS, Linux, and C++.** All-in-one `.litertlm` bundle with INT8 audio encoder, 28-layer Qwen3 LM decoder (`prefill`/`decode`), embedder, tokenizer, and `LlmMetadata`. |
| `qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | CPU / GPU | Direct [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr) integration without LiteRT-LM. |
| `qwen3_asr_0.6b_5s_f32.tflite` | 3.13 GB | CPU / GPU | Unquantized FP32 reference baseline. |
| `qwen3_asr_0.6b_5s_f32_<SoC>.tflite` | 1.88–2.19 GB | NPU + CPU | Standalone LiteRT NPU pipelines AOT-compiled for a specific mobile SoC (see below). |

How the variants differ:

-   **`.litertlm` vs. `.tflite`** — the `.litertlm` file bundles the audio
    encoder (`tf_lite_audio_encoder_hw`), the 28-layer Qwen3 LM with stateful KV
    cache (`tf_lite_prefill_decode`), the token embedder (`tf_lite_embedder`),
    the compressed Hugging Face `tokenizer.json`, and `LlmMetadata` /
    `ExecutorMetadata` in one self-contained archive for
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)'s
    `OmniEngine` and `omni_runner`. Standalone `.tflite` files contain only the
    neural network graph and expect the host app to supply mel preprocessing and
    detokenization.
-   **INT8 (`_i8`) vs. FP32 (`_f32`)** — INT8 dynamic range quantization shrinks
    the model from **3.13 GB to 960 MB** (`3.3×` smaller on disk) and
    accelerates mobile ARM64 CPU and GPU inference.
-   **AOT-compiled NPU `.tflite` files (`_<SoC>.tflite`)** — pre-compiled for
    on-device NPU accelerators via the LiteRT NeuroPilot / QNN compilers:

| SoC family | Available `<SoC>` suffixes |
|---|---|
| **Qualcomm Snapdragon** | `Qualcomm_SM8450` (8 Gen 1), `SM8550` (8 Gen 2), `SM8650` (8 Gen 3), `SM8750` (8 Elite), `SM8850` (8 Elite Gen 5), `SA8255`, `SA8295` |
| **MediaTek Dimensity** | `MediaTek_MT6877`, `MT6878`, `MT6879`, `MT6897`, `MT6983` (9000), `MT6985` (9200), `MT6989` (9300), `MT6991` (9400), `MT6993`, `MT8171`, `MT8188`, `MT8189` |

## Download

```sh
pip install -U huggingface_hub

# Recommended: INT8 .litertlm bundle (960 MB, 52 languages & dialects)
hf download litert-community/Qwen3-ASR-0.6B \
    qwen3_asr_0.6b_5s_i8.litertlm --local-dir .
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
hf download litert-community/Qwen3-ASR-0.6B \
    qwen3_asr_0.6b_5s_i8.litertlm --local-dir .
```

> [!TIP]
> If you omit `--model_path` and pass `--model_name=qwen3-asr-0.6b`,
> `omni_runner` automatically downloads `qwen3_asr_0.6b_5s_i8.litertlm` into
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
bazel-bin/omni/omni_runner --model_name=qwen3-asr-0.6b \
    --model_path=/path/to/qwen3_asr_0.6b_5s_i8.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav
```

**On an Android phone (CPU):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/qwen3_asr_0.6b_5s_i8.litertlm \
         runtime/components/testdata/audio_sample.wav $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=qwen3-asr-0.6b \
    --backend=cpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/qwen3_asr_0.6b_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

**On an Android phone (GPU):**

```sh
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=qwen3-asr-0.6b \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/qwen3_asr_0.6b_5s_i8.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

> `--model_name=qwen3-asr-0.6b` (or alias `qwen3-asr`) tells the runner which
> model family the file is (it is inferred automatically when the file is named
> `qwen3-asr-0.6b.litertlm`). XNNPACK weight packs and compiled GPU kernels are
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
    .model_path = "/data/local/tmp/omni/qwen3_asr_0.6b_5s_i8.litertlm",
};
litert::lm::proto::Speech2TextSessionConfig asr_config;
asr_config.set_overlap_ratio(0.4f);

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("qwen3-asr-0.6b", settings, &asr_config));
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

In your `BUILD` file, link only the Qwen3-ASR target
(`//omni/speech2text:qwen3_asr`) to avoid pulling in other speech, TTS, or image
generation models:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:qwen3_asr",
    ],
)
```

## Supported languages

`qwen3-asr-0.6b` supports automatic language identification and speech
recognition across **30 languages** and **22 Chinese dialects** — no language
flag is required. Internally, the Qwen3 decoder emits `language <Lang><asr_text>`
followed by the transcript, and `LmDecoder` automatically strips tokens up to
and including `<asr_text>` (`token_id = 151704`).

| Category | Supported languages / dialects |
|---|---|
| **30 Languages** | Chinese (`zh`), English (`en`), Cantonese (`yue`), Arabic (`ar`), German (`de`), French (`fr`), Spanish (`es`), Portuguese (`pt`), Indonesian (`id`), Italian (`it`), Korean (`ko`), Russian (`ru`), Thai (`th`), Vietnamese (`vi`), Japanese (`ja`), Turkish (`tr`), Hindi (`hi`), Malay (`ms`), Dutch (`nl`), Swedish (`sv`), Danish (`da`), Finnish (`fi`), Polish (`pl`), Czech (`cs`), Filipino (`fil`), Persian (`fa`), Greek (`el`), Hungarian (`hu`), Macedonian (`mk`), Romanian (`ro`) |
| **22 Chinese Dialects** | Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent), Wu, Minnan |

## What's in the file

Inside `qwen3_asr_0.6b_5s_i8.litertlm`:

| Section | Details |
|---|---|
| **Audio front-end (C++)** | Self-contained C++ `LogMelSpectrogramProcessor`: resamples any input audio rate to 16 kHz mono, computes 400-point Hann STFT (25 ms window, 10 ms hop, 500 frames per 5 s window), 128-bin Slaney power mel filterbank (`mel_floor = 1e-10`), and Whisper dynamic-range log normalization (`NORM_TYPE_WHISPER`, `[1, 128, 500]`). |
| `tf_lite_audio_encoder_hw` | Audio encoder + multimodal projector (`~181 MB`, INT8 weights) mapping the 5-second `[1, 128, 500]` log-mel spectrogram into audio token embeddings for the Qwen3 language model. |
| `tf_lite_prefill_decode` | 28-layer Qwen3 autoregressive language model (`~582 MB`, INT8 weights) with stateful KV cache (`kv_cache_k_0..27` and `kv_cache_v_0..27`, max sequence length 512). |
| `tf_lite_embedder` | Token embedding table (`~150 MB`) used by `LiteRtLmRunner` during prompt prefill and autoregressive token decoding. |
| `LlmMetadataProto` & `ExecutorMetadataProto` | ChatML audio prompt template (`<|audio_start|><|audio_pad|><|audio_end|>`), stop tokens (`151643` `<|endoftext|>`, `151645` `<|im_end|>`), and 28-layer KV cache state buffer descriptors. |
| `HF_Tokenizer_Zlib` | Zlib-compressed Qwen3 Hugging Face `tokenizer.json` (~2.1 MB compressed; 151,936 vocabulary + audio/ASR control tokens including `<asr_text>` = `151704`). |
| **Decoder & Text merger (C++)** | `LmDecoder` (prefills audio embeddings via `LiteRtLmRunner`, decodes greedily, skips the predicted `language <Lang><asr_text>` prefix up to token `151704`, and truncates trailing repetitions) + cross-window seam alignment across overlapping 5 s windows. |

The models were exported from the upstream PyTorch checkpoint
(`Qwen/Qwen3-ASR-0.6B`) with
[litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized
with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

-   **Fixed 5-second audio encoder window:** Longer audio is processed in
    overlapping 5-second windows (2.5 to 3.0 s stride). Words spoken right
    across a window boundary are emitted immediately in `unconfirmed_text` and
    promoted to `confirmed_text` once the next overlapping window resolves the
    boundary seam.
-   **Memory footprint:** Because the bundle includes both the audio encoder and
    the 28-layer Qwen3 language model (`960 MB` on disk), devices with at least
    3–4 GB of available RAM are recommended. For a ~30–42 MB footprint, see
    [`litert-community/moonshine-tiny`](https://huggingface.co/litert-community/moonshine-tiny)
    or
    [`litert-community/whisper-tiny`](https://huggingface.co/litert-community/whisper-tiny).
-   **SoC-specific NPU files are standalone `.tflite` graphs:** The pre-compiled
    `qwen3_asr_0.6b_5s_f32_<SoC>.tflite` files target the standalone LiteRT NPU
    C++/Kotlin APIs rather than `.litertlm` bundles.

## License & acknowledgements

-   **Weights & tokenizer:**
    [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from
    [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) by the
    Qwen team, Alibaba Group.
-   **Conversion & runtime:**
    [litert-torch](https://github.com/google-ai-edge/litert-torch),
    [LiteRT](https://github.com/google-ai-edge/LiteRT), and
    [LiteRT-LM Omni](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni)
    (Apache-2.0).
