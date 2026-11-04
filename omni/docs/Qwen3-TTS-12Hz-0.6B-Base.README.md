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
  - ios
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

Fully on-device, 10-language neural speech language model in a single file.

This is [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) packaged as a self-contained `.litertlm` bundle for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni), Google's open-source on-device inference runtime. Everything needed to turn text into 24 kHz speech is inside the one file: the 28-layer Qwen3 Talker language model, the 5-layer Multi-Token Prediction (MTP) residual codebook drafter, the RVQ + causal ConvNet neural audio codec decoder, embedding tables, the Qwen2 BPE tokenizer, and the enrolled speaker embedding (`demo_speaker`) — no external `.npy` tables, no phonemizer libraries, and no Python host loops required.

- **Languages:** English, Mandarin Chinese, Japanese, Korean, Spanish, French, German, Italian, Portuguese, Russian (plus automatic language detection with `--language=auto`; see [Supported languages](#supported-languages))
- **Voice:** Bundled multilingual speaker embedding (`demo_speaker`, 24 kHz mono output)
- **Speed:** Synthesizes speech near real time on a Galaxy S26 GPU (RTF 1.24, ~1.44 GB peak memory) or CPU (RTF 1.29, ~1.90 GB peak memory)
- **Quality:** 5.40 % mean WER/CER across all 10 languages (2.14 % on the phonetics & prosody subset) on a 1,000-sentence intelligibility benchmark (see [Quality](#quality))

## Which file should I use?

| File | Size | Talker Quantization | Runs on | Recommended for |
|---|---:|---|---|---|
| **`qwen3_tts_int8_v2.litertlm`** ⭐ | 2.44 GB | Dynamic INT8 (`wi8_afp32`, 441 MB) | Talker LM on **GPU** (`--backend=gpu`) or **CPU** (`--backend=cpu`) | **Recommended default for all devices.** Faster synthesis (near real time on Galaxy S26: RTF 1.24 on GPU, 1.29 on CPU), faster engine load (~2.0 s on GPU, ~0.5 s on CPU), and ~235–460 MB lower host memory on GPUs supported by the LiteRT GPU delegate (OpenCL, Metal, or WebGPU). |
| `qwen3_tts_int4_v2.litertlm` | 2.24 GB | Blockwise-32 OCTAV INT4 (244 MB) | GPU or CPU | Storage-constrained deployments where saving ~200 MB of disk space is prioritized over load and synthesis speed. |

Both files share the same dynamic-INT8 MTP drafter (`111 MB`, with per-codebook `head_<i>` signatures), FP32 neural codec decoder (`460 MB`, with 64/32/16-frame window signatures), embedding lookup tables, Qwen2 BPE tokenizer, and `demo_speaker` voice embedding:

- **GPU execution (`--backend=gpu`)** — delegates the 28-layer Talker LM (`441 MB` INT8 weights + KV cache) to any GPU supported by the LiteRT GPU delegate (OpenCL, Metal, or WebGPU), while keeping the 16-step inner MTP drafter loop, gather embeddings, and neural codec vocoder on CPU (XNNPACK) to avoid CPU↔GPU round-trip overhead per 80 ms frame. This lowers host peak RSS by ~460 MB on Galaxy S26 (`1.44 GB` vs `1.90 GB`).
- **CPU execution (`--backend=cpu`)** — runs all stages on CPU via XNNPACK with fast cold/warm engine initialization (~0.5–0.8 s) when no GPU delegate is used.

> [!NOTE]
> The v1 bundles (`qwen3_tts_int8.litertlm` and `qwen3_tts_int4.litertlm`) remain in the repository for compatibility with earlier runtimes built before 2026-10-29. The earlier standalone `.tflite` and `.npy`/`.npz` preview files have been moved to [`legacy/`](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/tree/main/legacy) and are **deprecated**. New integrations should use `qwen3_tts_int8_v2.litertlm` or `qwen3_tts_int4_v2.litertlm` above with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

## Quick start

Both commands below produce a 24 kHz mono WAV file. Prerequisites (Bazel, Android NDK) are in the LiteRT-LM [build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

### 1. Download a model

```sh
pip install -U huggingface_hub
# Recommended: INT8 bundle (2.44 GB, 10 languages)
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base qwen3_tts_int8_v2.litertlm --local-dir .
# Or: INT4 bundle (2.24 GB, smaller file size)
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base qwen3_tts_int4_v2.litertlm --local-dir .
```

### 2. Build the runner

```sh
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
# Pick ONE of the following; `bazel-bin/` always points at the last build.
bazel build -c opt //omni:omni_runner                            # Linux / macOS
bazel build -c opt --config=android_arm64 //omni:omni_runner     # Android (needs ANDROID_NDK_HOME; tested with NDK r28c)
```

> If you build for both, copy the first binary out of `bazel-bin/` before running the second build.

### 3. Synthesize speech

**On desktop / server (CPU):**

```sh
bazel-bin/omni/omni_runner --model_name=qwen3-tts \
    --backend=cpu --num_threads=4 \
    --model_path=/path/to/qwen3_tts_int8_v2.litertlm \
    --language=en-US \
    --text="Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM." \
    --output_path=/tmp/hello.wav
```

**On an Android phone (GPU or CPU):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR $DEVICE_DIR/cache
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/qwen3_tts_int8_v2.litertlm $DEVICE_DIR

# Run on GPU (lower host RAM usage: ~1.44 GB peak RSS on Galaxy S26) or CPU (--backend=cpu)
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner --model_name=qwen3-tts \
    --backend=gpu --num_threads=4 --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/qwen3_tts_int8_v2.litertlm \
    --language=en-US \
    --text="Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM." \
    --output_path=$DEVICE_DIR/hello.wav

adb pull $DEVICE_DIR/hello.wav .
```

> `--model_name=qwen3-tts` tells the runner which model family the file is (it is only inferred automatically when the file is literally named `qwen3-tts.litertlm`). On GPU, the first run compiles GPU kernels and caches them in `--cache_dir` so later runs start in ~2 s.

Change `--language` to switch language (e.g. `--language=zh-CN`, `--language=ja`, or `--language=auto`). The full list of flags and the C++ `OmniEngine` / `OmniSession` streaming API are documented in the [LiteRT-LM omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

### Use from C++ (`OmniEngine`)

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/proto/text2speech_session_config.pb.h"  // //runtime/proto:text2speech_session_config_cc_proto

using ::litert::omni::OmniEngine;
using ::litert::omni::OmniSession;
using ::litert::omni::PushInputSource;

OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kGpu,
    .num_threads = 4,
    .cache_dir = "/data/local/tmp/omni/cache",
    .model_path = "/data/local/tmp/omni/qwen3_tts_int8_v2.litertlm",
};
litert::lm::proto::Text2SpeechSessionConfig tts_config;
tts_config.set_language("en-US");

ABSL_ASSIGN_OR_RETURN(auto engine, OmniEngine::Create("qwen3-tts", settings, &tts_config));
auto input_source = std::make_unique<PushInputSource>();
PushInputSource* input = input_source.get();
ABSL_ASSIGN_OR_RETURN(auto session, engine->CreateSession(std::move(input_source)));

ABSL_RETURN_IF_ERROR(input->PushInput(OmniSession::TextInput{
    .text = "Hello! Today we are testing on-device speech synthesis powered by LiteRT-LM.",
}));
input->Finish();
// Pull AudioOutput chunks with session->ProcessNext() until OutOfRange, then session->Flush().
```

## Supported languages

Pass any of the language names or BCP 47 tags below to `--language` (or `Text2SpeechSessionConfig::set_language`):

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

All 10 clips below were synthesized on a Samsung Galaxy S26 with `qwen3_tts_int8_v2.litertlm` (`--backend=gpu`) using the bundled `demo_speaker` voice and verified by transcribing them back with an ASR model.

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

**How we measured.** We synthesized 1,000 challenging evaluation sentences across all 10 supported languages (100 sentences per language covering heteronyms, homographs, liaison, gemination, numbers, dates, currencies, abbreviations, loanwords, and tone sandhi) with `qwen3_tts_int8_v2.litertlm`, transcribed the generated audio with Whisper large-v3, and computed the word error rate (WER) or character error rate (CER, for Chinese, Japanese, and Korean) after text normalization. Below we report the mean error rate across the 973 / 1,000 sentences that terminated normally before the 512-frame (`40.96 s`) cap, alongside the phonetics & prosody subset (`PHON`, natural-word sentences without unexpanded digits or symbolic abbreviations). Lower is better.

| Language | Sentences | Metric | Overall error rate | Phonetics & prosody (`PHON`) |
|---|---:|---|---:|---:|
| Korean (`ko`) | 100 | CER | **1.45 %** | **0.62 %** |
| French (`fr`) | 97 | WER | **2.53 %** | **1.91 %** |
| Mandarin Chinese (`zh`) | 100 | CER | **3.03 %** | **1.58 %** |
| Italian (`it`) | 97 | WER | **4.77 %** | **0.69 %** |
| Portuguese (`pt`) | 100 | WER | **4.93 %** | **0.64 %** |
| English (`en`) | 99 | WER | **5.16 %** | **3.16 %** |
| German (`de`) | 85 | WER | **6.06 %** | **0.89 %** |
| Japanese (`ja`) | 96 | CER | **6.93 %** | **6.49 %** |
| Spanish (`es`) | 99 | WER | **8.05 %** | **1.40 %** |
| Russian (`ru`) | 100 | WER | **11.14 %** | **3.83 %** |
| **All 10 languages** | **973 / 1,000** | **Mean** | **5.40 %** | **2.14 %** |

> Under default stochastic codebook-0 sampling (`temperature = 0.9`, `top_k = 50`), 27 / 1,000 sentences (2.7 %) missed the codec end-of-speech token and ran to the 512-frame (`40.96 s`) cap, producing trailing noise or repetition after the spoken text (including those 27 cap-hit utterances raises the unfiltered 1,000-sentence mean to 11.06 % overall / 3.19 % `PHON`).

## Performance

Measured with `omni_runner` (`android_arm64`, 4 CPU threads) built from LiteRT-LM at head on retail-equivalent Android phones, averaging two runs after one warm-up run (cached GPU shaders in `--cache_dir`), synthesizing a ~6.5 s English sentence (`"The quick brown fox jumps over the lazy dog. Today we are testing speech synthesis on a phone."`) and a ~7.5 s Mandarin sentence (`"你好，在移动设备上直接运行语音合成模型，不仅速度非常快，而且完全不需要联网。"`). **RTF** = synthesis time ÷ audio duration (lower is faster; 1.0 = real time). **Peak memory** is the process's peak resident set (`Max RSS`), including model load.

| Device | File | Backend | Engine load (warm) | English RTF | Mandarin RTF | Peak memory |
|---|---|---|---:|---:|---:|---:|
| Galaxy S26 (Snapdragon 8 Elite Gen 5, Adreno) | **`qwen3_tts_int8_v2.litertlm`** ⭐ | **GPU** | 2.0 s | **1.24** (7.5 s / 6.08 s) | **1.26** (9.6 s / 7.68 s) | **1,441 MB (~1.44 GB)** |
| Galaxy S26 (Snapdragon 8 Elite Gen 5) | **`qwen3_tts_int8_v2.litertlm`** ⭐ | **CPU** | **0.5 s** | **1.29** (8.6 s / 6.64 s) | **1.22** (9.7 s / 8.00 s) | 1,898 MB (~1.90 GB) |
| Galaxy S26 | `qwen3_tts_int4_v2.litertlm` | GPU | 3.3 s | 1.36 (8.4 s / 6.16 s) | 1.31 (8.1 s / 6.16 s) | 1,477 MB (~1.48 GB) |
| Galaxy S26 | `qwen3_tts_int4_v2.litertlm` | CPU | 1.4 s | 1.48 (9.1 s / 6.16 s) | 1.41 (9.4 s / 6.64 s) | 1,720 MB (~1.72 GB) |
| Pixel 10 Pro (Tensor G5, PowerVR) | **`qwen3_tts_int8_v2.litertlm`** ⭐ | **CPU** | **0.8 s** | **3.21** (21.3 s / 6.64 s) | **3.62** (29.0 s / 8.00 s) | 1,844 MB (~1.84 GB) |
| Pixel 10 Pro (Tensor G5, PowerVR) | **`qwen3_tts_int8_v2.litertlm`** ⭐ | GPU | 1.9 s | 5.43 (32.4 s / 5.96 s) | 5.52 (38.9 s / 7.04 s) | **1,609 MB (~1.61 GB)** |
| Pixel 10 Pro | `qwen3_tts_int4_v2.litertlm` | CPU | 1.6 s | 3.98 (24.5 s / 6.16 s) | 4.11 (27.3 s / 6.64 s) | 1,663 MB (~1.66 GB) |
| Pixel 10 Pro | `qwen3_tts_int4_v2.litertlm` | GPU | 3.4 s | 6.00 (36.7 s / 6.12 s) | 5.25 (35.7 s / 6.80 s) | 1,578 MB (~1.58 GB) |

Compared with the previous release of these files, the re-exported MTP drafter (dynamic INT8 weights and per-head LM signatures) cuts English RTF on Galaxy S26 from 2.97 to 1.24 on GPU and from 2.59 to 1.29 on CPU, and lowers peak memory by about 540 MB.

### Backend notes

- **GPU vs CPU memory:** When `--backend=gpu` is selected, the 28-layer Talker LM (`441 MB` INT8 weights + 28-layer KV cache) runs on the GPU while the 16-step inner MTP drafter loop, text projection, embedding gather tables, and neural codec vocoder execute on CPU (avoiding 16 synchronous CPU↔GPU round-trips per 80 ms frame). This reduces host peak RSS by **~460 MB on Galaxy S26** (`1.44 GB` vs `1.90 GB`) and **~235 MB on Pixel 10 Pro** (`1.61 GB` vs `1.84 GB`).
- **Galaxy S26:** Both `--backend=gpu` (RTF `1.24`, `1.44 GB` RAM, `2.0 s` warm load) and `--backend=cpu` (RTF `1.29`, `1.90 GB` RAM, `0.5 s` warm load) run close to real time; choose GPU when minimizing host memory footprint is important, or CPU for the fastest load.
- **Pixel 10 Pro:** `--backend=cpu` is fastest (`qwen3_tts_int8_v2.litertlm`, RTF `3.21`, load `0.8 s`), while `--backend=gpu` lowers host memory to `1.61 GB` (RTF `5.43`, load `1.9 s`).

## What's in the file

| Section (`Text2SpeechMetadata` role) | Details | Execution backend (`--backend=gpu`) |
|---|---|---|
| `tf_lite_talker` | 28-layer Qwen3 autoregressive speech LM (`prefill_32`, `prefill_128`, `decode`, KV cache 1024) predicting codebook 0 at 12.5 Hz. Dynamic INT8 (`441 MB` in `qwen3_tts_int8_v2.litertlm`) or Blockwise-32 OCTAV INT4 (`244 MB` in `qwen3_tts_int4_v2.litertlm`). | **GPU** |
| `tf_lite_mtp_drafter` | 5-layer inner Multi-Token Prediction transformer (`111 MB`, dynamic INT8 weights) invoked 16× per 12.5 Hz frame to predict residual codebooks 1–15. The decode step emits the final hidden state; the 15 codebook LM heads are separate `head_<i>` signatures so each step only evaluates the head it needs. | **CPU** (keeps all 16 inner steps in CPU cache with zero CPU↔GPU sync overhead) |
| `tf_lite_text_projection` | 2-layer MLP (`25 MB`, FP32) projecting text embeddings into the Talker hidden dimension. | **CPU** |
| `tf_lite_codec_decoder` | RVQ + 8-layer transformer + causal ConvNet upsampler (`460 MB`, FP32) rendering 64-frame chunks into 24 kHz PCM audio. Also exports `decode_32` / `decode_16` window signatures (sharing the same weights) so streaming runtimes can render the first audio chunk and the tail through smaller windows. | **CPU** (1D conv lengths up to `122,886` exceed mobile GPU 2D texture width limits of `16,384`) |
| `tf_lite_text_embedding` | `[151936, 2048]` gather lookup table (`1.24 GB`). | **CPU** |
| `tf_lite_mtp_embedding` | 15 residual codebook gather lookup tables (`126 MB`). | **CPU** |
| `tf_lite_codec_embedding` | Codebook 0 gather lookup table (`12.6 MB`). | **CPU** |
| `HF_Tokenizer_Zlib` | Compressed Qwen2 `tokenizer.json`. | **CPU** |
| `demo_speaker` | Enrolled 1024-dim float32 speaker x-vector (`4 KB`). | **CPU** |

The models were exported from the original PyTorch checkpoint with [litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

- Bundled with a single default multilingual voice (`demo_speaker`, x-vector speaker embedding). Custom voice enrollment requires extracting a 1024-d x-vector off-device with the speaker encoder.
- Prompt prefill uses `prefill_32` / `prefill_128` chunks; the Talker KV cache length is 1024 tokens and audio generation per chunk is capped at 512 frames (~41 s of audio at 12.5 Hz). Long inputs are automatically split at sentence boundaries by `StreamTextSource`.
- Because the bundle is ~2.24–2.44 GB, mobile and desktop deployments with sufficient RAM (≥ 4 GB device memory recommended) are the primary targets.

## Changelog

- **2026-10-29 (`qwen3_tts_int8_v2.litertlm`, `qwen3_tts_int4_v2.litertlm`):** Re-exported the MTP drafter with dynamic INT8 weights (`220 MB` → `111 MB`); its decode step now returns the final hidden state and the 15 codebook LM heads are separate `head_<i>` signatures, so each of the 16 inner steps evaluates only one head. The codec decoder adds `decode_32` / `decode_16` window signatures. Synthesis is ~2.4× faster on Galaxy S26 and files are ~105 MB smaller. Requires LiteRT-LM built with [the per-head MTP runtime](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni); the v1 files (`qwen3_tts_int8.litertlm`, `qwen3_tts_int4.litertlm`) are retained for older runtimes.
- **2026-10-17 (`qwen3_tts_int8.litertlm`, `qwen3_tts_int4.litertlm`):** Initial `.litertlm` release.

## License & acknowledgements

- Weights: Apache-2.0, inherited from [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by the Qwen team, Alibaba Group.
- Conversion & runtime: [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [litert-torch](https://github.com/google-ai-edge/litert-torch) (Apache-2.0).
