---
license: apache-2.0
library_name: litert-lm
pipeline_tag: text-to-speech
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
base_model_relation: quantized
language:
  - en
  - zh
  - ja
  - ko
  - es
  - fr
  - de
  - it
  - pt
  - ru
tags:
  - litert
  - litert-lm
  - tts
  - text-to-speech
  - on-device
  - android
  - qwen3-tts
  - int8
  - int4
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Qwen3-TTS-12Hz-0.6B-Base for LiteRT-LM

Fully on-device, 10-language neural speech language model packaged in a single `.litertlm` file.

This is [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) packaged as a self-contained `.litertlm` bundle for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni), Google's open-source on-device inference runtime. Everything needed to synthesize 24 kHz speech is inside a single file: the 28-layer Qwen3 Talker language model, the 5-layer Multi-Token Prediction (MTP) residual codebook drafter, the RVQ + causal ConvNet neural audio codec decoder, embedding tables, the Qwen2 BPE tokenizer, and the enrolled speaker embedding (`demo_speaker`).

- **Languages (10):** English, Mandarin Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian (plus automatic language detection with `--language=auto`)
- **Single-file deployment:** No external `.npy` tables, tokenizer files, or Python host loops required — runs directly through the C++ `OmniEngine` / `OmniSession` API or `omni_runner` CLI
- **Hardware support:** CPU (XNNPACK) and GPU (LiteRT OpenCL delegate for the 28-layer Talker LM, reducing host RAM usage by **~450–500 MB**)

## Which file should I use?

| File | Size | Talker Quantization | Runs on | Recommended for |
|---|---:|---|---|---|
| **`qwen3_tts_int8.litertlm`** ⭐ | 2.55 GB | Dynamic INT8 (`wi8_afp32`, 441 MB) | **GPU** (`--backend=gpu`) or **CPU** (`--backend=cpu`) | **Recommended default for all devices.** Faster synthesis, faster engine load (~2.1 s on GPU, ~0.5 s on CPU), and ~470 MB lower host memory on GPU (1.98 GB Peak RSS on Galaxy S26 GPU vs 2.45 GB on CPU). |
| `qwen3_tts_int4.litertlm` | 2.34 GB | Blockwise-32 OCTAV INT4 (244 MB) | GPU or CPU | Storage-constrained deployments where saving ~200 MB of disk space is prioritized over load speed. |

Both files share the same FP16-weights MTP drafter (`220 MB`), FP32 neural codec decoder (`457 MB`), embedding lookup tables, tokenizer, and `demo_speaker` voice embedding.

> [!NOTE]
> The earlier standalone `.tflite` and `.npy`/`.npz` preview files that used to live at the root of this repository have been moved to [`legacy/`](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/tree/main/legacy) and are **deprecated**. New integrations should use the self-contained `.litertlm` files above with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

## Quick start

Both commands below produce a 24 kHz mono WAV file. Prerequisites (Bazel, Android NDK) are in the LiteRT-LM [build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

### 1. Download a model

```sh
pip install -U huggingface_hub
# Recommended: INT8 bundle
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base qwen3_tts_int8.litertlm --local-dir .
# Optional: INT4 bundle (smaller file size)
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base qwen3_tts_int4.litertlm --local-dir .
```

### 2. Build the runner

```sh
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
bazel build -c opt //omni:omni_runner                            # Linux / macOS
bazel build -c opt --config=android_arm64 //omni:omni_runner     # Android (needs ANDROID_NDK_HOME)
```

### 3. Synthesize speech

**On desktop / server (CPU):**

```sh
bazel-bin/omni/omni_runner --model_name=qwen3-tts \
    --backend=cpu --num_threads=4 \
    --model_path=/path/to/qwen3_tts_int8.litertlm \
    --language=en-US \
    --text="Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM." \
    --output_path=/tmp/hello.wav
```

**On an Android phone (Snapdragon GPU or CPU):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR $DEVICE_DIR/cache
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/qwen3_tts_int8.litertlm $DEVICE_DIR

# Run on GPU (lower host RAM usage: ~1.77 GB Peak RSS) or CPU (--backend=cpu)
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner --model_name=qwen3-tts \
    --backend=gpu --num_threads=4 --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/qwen3_tts_int8.litertlm \
    --language=en-US \
    --text="Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM." \
    --output_path=$DEVICE_DIR/hello.wav

adb pull $DEVICE_DIR/hello.wav .
```

> `--model_name=qwen3-tts` tells the runner which model family the file is (it is only inferred automatically when the file is literally named `qwen3_tts.litertlm`). On GPU, the first run compiles OpenCL kernels and caches them in `--cache_dir` so subsequent runs start in ~3.5 s.

The full list of flags and the C++ `OmniEngine` / `OmniSession` streaming API are documented in the [LiteRT-LM omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

### Use from C++

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"

using ::litert::omni::OmniEngine;
using ::litert::omni::OmniSession;
using ::litert::omni::PushInputSource;

OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kGpu,
    .num_threads = 4,
    .cache_dir = "/data/local/tmp/omni/cache",
    .model_path = "/data/local/tmp/omni/qwen3_tts_int8.litertlm",
};
litert::lm::proto::TtsSessionConfig tts_config;
tts_config.set_language("en-US");

ABSL_ASSIGN_OR_RETURN(auto engine, OmniEngine::Create("qwen3-tts", settings, &tts_config));
auto input_source = std::make_unique<PushInputSource>();
PushInputSource* input = input_source.get();
ABSL_ASSIGN_OR_RETURN(auto session, engine->CreateSession(std::move(input_source)));

ABSL_RETURN_IF_ERROR(input->PushInput(OmniSession::TextInput{
    .text = "Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM.",
}));
input->Finish();
// Pull OmniSession::AudioOutput chunks with session->ProcessNext() until OutOfRange, then session->Flush().
```

## Supported languages

Pass any of the language names or BCP-47 tags below to `--language` (or `TtsSessionConfig::set_language`):

| Language | Accepted `--language` values |
|---|---|
| Automatic detection | `auto` |
| English | `en`, `en-US`, `en-GB`, `english` |
| Mandarin Chinese | `zh`, `zh-CN`, `cmn`, `chinese` |
| Japanese | `ja`, `ja-JP`, `japanese` |
| Korean | `ko`, `ko-KR`, `korean` |
| Spanish | `es`, `es-ES`, `spanish` |
| French | `fr`, `fr-FR`, `french` |
| German | `de`, `de-DE`, `german` |
| Italian | `it`, `it-IT`, `italian` |
| Portuguese | `pt`, `pt-BR`, `portuguese` |
| Russian | `ru`, `ru-RU`, `russian` |

## Samples

All 10 clips below were synthesized on a Samsung Galaxy S26 with `qwen3_tts_int8.litertlm` (`--backend=gpu`) using the bundled `demo_speaker` voice and verified by transcribing them back with an ASR model.

| Language | `--language` | Text | Audio |
|---|---|---|---|
| English | `en` | Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/en_demo_speaker.wav"></audio> |
| Mandarin Chinese | `zh` | 你好，这是在移动设备上直接运行的语音合成模型测试。 | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/zh_demo_speaker.wav"></audio> |
| Japanese | `ja` | こんにちは、これはデバイス上で動作する音声合成モデルのテストです。 | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/ja_demo_speaker.wav"></audio> |
| Korean | `ko` | 안녕하세요, 기기에서 직접 실행되는 음성 합성 모델 테스트입니다. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/ko_demo_speaker.wav"></audio> |
| Spanish | `es` | Hola, hoy estamos probando la síntesis de voz directamente en el dispositivo. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/es_demo_speaker.wav"></audio> |
| French | `fr` | Bonjour, nous testons aujourd'hui la synthèse vocale directement sur l'appareil. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/fr_demo_speaker.wav"></audio> |
| German | `de` | Hallo, heute testen wir die Sprachsynthese direkt auf dem Gerät. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/de_demo_speaker.wav"></audio> |
| Italian | `it` | Buongiorno, oggi proviamo la sintesi vocale direttamente sul telefono. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/it_demo_speaker.wav"></audio> |
| Portuguese | `pt` | Olá, hoje estamos testando a síntese de voz diretamente no dispositivo. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/pt_demo_speaker.wav"></audio> |
| Russian | `ru` | Привет, сегодня мы тестируем синтез речи непосредственно на устройстве. | <audio controls src="https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/samples/ru_demo_speaker.wav"></audio> |

## Quality

**How we measured.** We synthesized 1,000 challenging evaluation sentences across all 10 supported languages (100 sentences per language covering heteronyms, homographs, liaison, gemination, numbers, dates, currencies, abbreviations, loanwords, and tone sandhi) with `qwen3_tts_int8.litertlm`, transcribed the generated audio with Whisper large-v3, and computed the word error rate (WER) or character error rate (CER, for Chinese, Japanese, and Korean) after text normalization. We report both the overall mean error rate across all 100 sentences per language and the phonetics & prosody error rate (`PHON`, natural-word sentences without unexpanded digits or symbolic abbreviations). Lower is better.

| Language | Sentences | Metric | Overall error rate | Phonetics & prosody (`PHON`) |
|---|---:|---|---:|---:|
| Korean (`ko`) | 100 | CER | **1.27 %** | **0.43 %** |
| Mandarin Chinese (`zh`) | 100 | CER | **3.26 %** | **1.65 %** |
| French (`fr`) | 100 | WER | **3.27 %** | **2.28 %** |
| Portuguese (`pt`) | 100 | WER | **5.21 %** | **1.53 %** |
| Italian (`it`) | 100 | WER | **6.23 %** | **1.09 %** |
| Japanese (`ja`) | 100 | CER | **6.98 %** | **7.71 %** |
| Spanish (`es`) | 100 | WER | **7.18 %** | **1.39 %** |
| English (`en`) | 100 | WER | **7.27 %** | **3.98 %** |
| German (`de`) | 100 | WER | **8.99 %** | **0.57 %** |
| Russian (`ru`) | 100 | WER | **10.90 %** | **3.44 %** |
| **All 10 languages** | **1,000** | **Mean** | **6.06 %** | **2.47 %** |

## Performance

Measured with `omni_runner` (`android_arm64`, 4 CPU threads) on retail-equivalent Android phones after one warm-up run (cached GPU shaders in `--cache_dir`), synthesizing a ~6.5 s English sentence (`"The quick brown fox jumps over the lazy dog. Today we are testing speech synthesis on a phone."`) and a ~7.4 s Mandarin sentence (`"你好，在移动设备上直接运行语音合成模型，不仅速度非常快，而且完全不需要联网。"`). **RTF** = synthesis time ÷ audio duration (lower is faster). **Peak memory** is the process's peak resident set (`VmHWM`), including model load.

| Device | File | Backend | Engine load (warm) | English RTF | Mandarin RTF | Peak memory (`VmHWM`) |
|---|---|---|---:|---:|---:|---:|
| Galaxy S26 (Snapdragon 8 Elite Gen 5, Adreno) | **`qwen3_tts_int8.litertlm`** ⭐ | **GPU** | 2.1 s | **2.97** (18.3 s / 6.16 s) | **3.05** (21.2 s / 6.96 s) | **1,980 MB (~1.98 GB)** |
| Galaxy S26 (Snapdragon 8 Elite Gen 5) | **`qwen3_tts_int8.litertlm`** ⭐ | **CPU** | **0.5 s** | **2.59** (16.8 s / 6.48 s) | **2.57** (19.6 s / 7.60 s) | 2,449 MB (~2.45 GB) |
| Galaxy S26 | `qwen3_tts_int4.litertlm` | GPU | 3.3 s | 2.93 (17.6 s / 6.00 s) | 2.92 (19.4 s / 6.64 s) | 2,030 MB (~2.03 GB) |
| Galaxy S26 | `qwen3_tts_int4.litertlm` | CPU | 1.3 s | 2.84 (17.7 s / 6.24 s) | 2.81 (19.6 s / 6.96 s) | 2,282 MB (~2.28 GB) |
| Pixel 10 Pro (Tensor G5, PowerVR) | **`qwen3_tts_int8.litertlm`** ⭐ | **CPU** | **1.2 s** | **5.65** (36.6 s / 6.48 s) | **7.40** (56.2 s / 7.60 s) | 2,322 MB (~2.32 GB) |
| Pixel 10 Pro (Tensor G5, PowerVR) | **`qwen3_tts_int8.litertlm`** ⭐ | GPU | 3.1 s | 8.48 (56.3 s / 6.64 s) | 8.29 (61.7 s / 7.44 s) | **1,925 MB (~1.93 GB)** |
| Pixel 10 Pro | `qwen3_tts_int4.litertlm` | CPU | 2.0 s | 7.52 (46.9 s / 6.24 s) | 7.73 (53.8 s / 6.96 s) | 2,150 MB (~2.15 GB) |

### Backend notes

- **GPU vs CPU memory:** When `--backend=gpu` is selected, the 28-layer Talker LM (`441 MB` INT8 weights + 28-layer KV cache) runs on the GPU while the 16-step inner MTP drafter loop, text projection, embedding gather tables, and neural codec vocoder execute on CPU (avoiding 16 synchronous CPU↔GPU round-trips per 80 ms frame). This reduces host Peak RSS (`VmHWM`) by **~470 MB on Galaxy S26** (`1.98 GB` vs `2.45 GB`) and **~400 MB on Pixel 10 Pro** (`1.93 GB` vs `2.32 GB`).
- **Galaxy S26:** Both `--backend=gpu` (RTF `2.97`, `1.98 GB` RAM, `2.1 s` warm load) and `--backend=cpu` (RTF `2.59`, `2.45 GB` RAM, `0.5 s` warm load) perform well; choose GPU when minimizing host memory footprint is important, or CPU for the fastest load and synthesis time.
- **Pixel 10 Pro:** `--backend=cpu` is fastest (`qwen3_tts_int8.litertlm`, RTF `5.65`, load `1.2 s`), while `--backend=gpu` lowers host memory to `1.93 GB` (RTF `8.48`, load `3.1 s`).

## What's in the file

| Section (`TtsMetadata` role) | Details | Execution backend (`--backend=gpu`) |
|---|---|---|
| `tf_lite_talker` | 28-layer Qwen3 autoregressive speech LM (`prefill_32`, `prefill_128`, `decode`, KV cache 1024) predicting codebook 0 at 12.5 Hz. Dynamic INT8 (`441 MB` in `qwen3_tts_int8.litertlm`) or Blockwise-32 OCTAV INT4 (`244 MB` in `qwen3_tts_int4.litertlm`). | **GPU** |
| `tf_lite_mtp_drafter` | 5-layer inner Multi-Token Prediction transformer (`220 MB`, FP16 weights) invoked 16× per 12.5 Hz frame to predict residual codebooks 1–15. | **CPU** (keeps all 16 inner steps in CPU cache with zero CPU↔GPU sync overhead) |
| `tf_lite_text_projection` | 2-layer MLP (`25 MB`, FP32) projecting text embeddings into the Talker hidden dimension. | **CPU** |
| `tf_lite_codec_decoder` | RVQ + 8-layer transformer + causal ConvNet upsampler (`457 MB`, FP32) rendering 64-frame chunks into 24 kHz PCM audio. | **CPU** (1D conv lengths up to `122,886` exceed mobile OpenCL 2D texture width limits of `16,384`) |
| `tf_lite_text_embedding` | `[151936, 2048]` gather lookup table (`1.24 GB`). | **CPU** |
| `tf_lite_mtp_embedding` | 15 residual codebook gather lookup tables (`126 MB`). | **CPU** |
| `tf_lite_codec_embedding` | Codebook 0 gather lookup table (`12.6 MB`). | **CPU** |
| `HF_Tokenizer_Zlib` | Compressed Qwen2 `tokenizer.json`. | **CPU** |
| `demo_speaker` | Enrolled 1024-dim float32 speaker x-vector (`4 KB`). | **CPU** |

## Limitations

- Bundled with a single default voice (`demo_speaker`, x-vector speaker embedding). Custom voice enrollment requires extracting a 1024-d x-vector off-device with the speaker encoder.
- Prompt prefill uses `prefill_32` / `prefill_128` chunks; the Talker KV cache length is 1024 tokens and audio generation per chunk is capped at 512 frames (~41 s of audio at 12.5 Hz). Long inputs are automatically split at sentence boundaries by `StreamTextSource`.

## License & acknowledgements

- Base model & weights: Apache-2.0, inherited from [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by the Qwen team, Alibaba Group.
- Conversion & runtime: [litert-torch](https://github.com/google-ai-edge/litert-torch) and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0).
