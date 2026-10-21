---
license: apache-2.0
library_name: litert-lm
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
base_model_relation: quantized
language:
  - en
  - es
  - fr
  - it
  - pt
  - hi
  - zh
  - ja
tags:
  - litert
  - litert-lm
  - tts
  - text-to-speech
  - on-device
  - android
  - kokoro
  - int8
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Kokoro-82M for LiteRT-LM

Fully on-device, multilingual text-to-speech in a single file.

This is [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82M-parameter StyleTTS 2 + ISTFTNet) packaged as a `.litertlm` bundle for
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), Google's open-source on-device inference runtime. Everything needed to turn text into
24 kHz speech is inside the one file: INT8 acoustic + vocoder models, all 54 voices, and self-contained text
front-ends (text normalization and grapheme-to-phoneme) for **9 languages / dialects** — no Python, no network, and
no external GPL phoneme libraries, so the whole stack is Apache-2.0.

- **Languages:** American & British English, Spanish, French, Italian, Brazilian Portuguese, Hindi, Mandarin Chinese, Japanese
- **Voices:** all 54 Kokoro v1.0 voices (see [Languages & voices](#languages--voices))
- **Speed:** synthesizes speech 4–5× faster than real time on a Galaxy S26 GPU, with ~0.5 GB peak memory
- **Quality:** on an 800-sentence, 8-language intelligibility benchmark it matches or beats the original Kokoro running on a server (see [Quality](#quality))

## Which file should I use?

| File | Size | Runs on | Recommended for |
|---|---:|---|---|
| **`kokoro_int8_gpu.litertlm`** ⭐ | 351 MB | Acoustic model on CPU, vocoder on **GPU** (`--backend=gpu`) | **Any device with a GPU supported by the LiteRT GPU delegate (OpenCL, WebGPU or Metal)** — phones, tablets, laptops. Fastest and about **half the memory** (~0.5 GB peak). |
| `kokoro_int8.litertlm` | 351 MB | CPU or GPU | Devices without a usable GPU (e.g. servers, CI, older hardware). Uses ~1 GB or more peak memory. |

Both files contain the same acoustic model, voices and text front-ends. They differ only in how the vocoder is
quantized:

- **GPU version** — vocoder convolutions use INT8 weights with floating-point activations, which the LiteRT GPU
  delegate runs natively on OpenCL, WebGPU and Metal GPUs. Run it with `--backend=gpu` and the LiteRT GPU
  accelerator libraries next to the runner (on Android: `prebuilt/android_arm64/*.so`, see below). With
  `--backend=cpu`, or when no GPU accelerator is available, the vocoder silently runs on the CPU instead: speech is
  still intelligible but may be noisier, so use `kokoro_int8.litertlm` for CPU-only devices.
- **CPU/GPU version** — the vocoder convolutions carry explicit dequantize ops so XNNPACK (CPU) also produces clean
  audio. This costs extra memory, so only pick it if you cannot use the GPU.

> [!NOTE]
> The earlier FP32 `.tflite` preview that used to live at the root of this repo (three bucketed graphs + a fixed-length
> demo build, English only, CPU only) has moved to [`legacy/`](https://huggingface.co/litert-community/Kokoro-82M/tree/main/legacy)
> and is **deprecated**. New integrations should use the `.litertlm` files above.

## Quick start

Both commands below produce a 24 kHz mono WAV file. Prerequisites (Bazel, Android NDK) are in the LiteRT-LM
[build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

### 1. Download a model

```sh
pip install -U huggingface_hub
# Recommended: GPU version (any device with a LiteRT-supported GPU)
hf download litert-community/Kokoro-82M kokoro_int8_gpu.litertlm --local-dir .
# Or: CPU/GPU version (no GPU available)
hf download litert-community/Kokoro-82M kokoro_int8.litertlm --local-dir .
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

**On a machine without a usable GPU (CPU):**

```sh
bazel-bin/omni/omni_runner --model_name=kokoro \
    --model_path=/path/to/kokoro_int8.litertlm \
    --language=en-US --voice=af_heart \
    --text="Hello from Kokoro, running fully on device with LiteRT-LM." \
    --output_path=/tmp/hello.wav
```

**On an Android phone (GPU, recommended):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/kokoro_int8_gpu.litertlm $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner --model_name=kokoro \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/kokoro_int8_gpu.litertlm \
    --language=en-US --voice=af_heart \
    --text="Hello from Kokoro, running fully on device with LiteRT-LM." \
    --output_path=$DEVICE_DIR/hello.wav

adb pull $DEVICE_DIR/hello.wav .
```

> `--model_name=kokoro` tells the runner which model family the file is. Keep it: builds before 2026-10-17 only infer
> it when the file is literally named `kokoro.litertlm`. The first GPU run compiles kernels and is slower; they are cached in
> `--cache_dir` so later runs start fast.

Change `--language` and `--voice` to switch language (e.g. `--language=ja --voice=jf_alpha`). The full list of flags
and the C++ `OmniEngine` / `OmniSession` streaming API are documented in the
[LiteRT-LM omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

### Use from C++

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/proto/text2speech_session_config.pb.h"  // //runtime/proto:text2speech_session_config_cc_proto

using ::litert::omni::OmniEngine;
using ::litert::omni::OmniSession;
using ::litert::omni::PushInputSource;

OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kGpu,
    .cache_dir = "/data/local/tmp/omni/cache",
    .model_path = "/data/local/tmp/omni/kokoro_int8_gpu.litertlm",
};
litert::lm::proto::Text2SpeechSessionConfig tts_config;
tts_config.set_language("en-US");
tts_config.set_voice("af_heart");

ABSL_ASSIGN_OR_RETURN(auto engine, OmniEngine::Create("kokoro", settings, &tts_config));
auto input_source = std::make_unique<PushInputSource>();
PushInputSource* input = input_source.get();
ABSL_ASSIGN_OR_RETURN(auto session, engine->CreateSession(std::move(input_source)));

ABSL_RETURN_IF_ERROR(input->PushInput(OmniSession::TextInput{.text = "Hello from Kokoro."}));
input->Finish();
// Pull OmniSession::AudioOutput chunks with session->ProcessNext() until OutOfRange, then session->Flush().
```

## Languages & voices

Pass any of these codes to `--language` (aliases such as `en`, `fr-FR`, `pt`, `zh-CN`, `cmn` also work) and a
matching voice to `--voice`. Voice names follow Kokoro's convention: first letter = language, second letter =
`f`emale / `m`ale.

| Language | `--language` | Voices |
|---|---|---|
| American English | `en-US` | `af_heart` ⭐, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| British English | `en-GB` | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` |
| Spanish | `es` | `ef_dora`, `em_alex`, `em_santa` |
| French | `fr` | `ff_siwis` |
| Italian | `it` | `if_sara`, `im_nicola` |
| Brazilian Portuguese | `pt-BR` | `pf_dora`, `pm_alex`, `pm_santa` |
| Hindi | `hi` | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |
| Mandarin Chinese | `zh` | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` |
| Japanese | `ja` | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo` |

Voice descriptions and quality grades are in the upstream [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).

## Samples

All clips were synthesized on a Samsung Galaxy S26 with `kokoro_int8_gpu.litertlm` (`--backend=gpu`) and
verified by transcribing them back with an ASR model.

| Language | Voice | Text | Audio |
|---|---|---|---|
| English (US) | `af_heart` | The quick brown fox jumps over the lazy dog. Artificial intelligence on edge devices enables private, low-latency speech synthesis. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/en-US_af_heart.wav"></audio> |
| English (UK) | `bf_emma` | Good morning! Running text to speech directly on a mobile processor without external phonetic libraries keeps the footprint small. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/en-GB_bf_emma.wav"></audio> |
| Spanish | `ef_dora` | La síntesis de voz en el dispositivo permite una comunicación rápida, natural y completamente privada sin conexión a internet. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/es_ef_dora.wav"></audio> |
| French | `ff_siwis` | Bonjour à tous, la génération vocale sur mobile fonctionne désormais sans bibliothèque externe avec une excellente qualité sonore. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/fr_ff_siwis.wav"></audio> |
| Italian | `if_sara` | Buongiorno! La sintesi vocale direttamente sul telefono offre prestazioni veloci e una pronuncia naturale in italiano. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/it_if_sara.wav"></audio> |
| Portuguese (BR) | `pf_dora` | Olá! A conversão de texto em fala no próprio dispositivo garante baixa latência e alta qualidade de áudio em português. | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/pt-BR_pf_dora.wav"></audio> |
| Hindi | `hf_alpha` | नमस्ते, मोबाइल डिवाइस पर सीधे काम करने वाली यह नई तकनीक बिना किसी बाहरी लाइब्रेरी के स्पष्ट आवाज़ उत्पन्न करती है। | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/hi_hf_alpha.wav"></audio> |
| Mandarin | `zf_xiaobei` | 你好，在移动设备上直接运行语音合成模型，不仅速度非常快，而且完全不需要依赖外部开源音素库。 | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/zh_zf_xiaobei.wav"></audio> |
| Japanese | `jf_alpha` | こんにちは。スマートフォン上で直接動作する音声合成エンジンにより、自然でクリアな日本語の音声を高速に生成できます。 | <audio controls src="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/samples/ja_jf_alpha.wav"></audio> |

## Quality

**How we measured.** We synthesized 800 deliberately difficult sentences (100 per language × 8 languages:
heteronyms, numbers, dates, currencies, abbreviations, liaison, tone sandhi, pitch accent, …) on a Samsung Galaxy
S26 Ultra with `kokoro_int8_gpu.litertlm`, transcribed the audio with Whisper large-v3, and computed the
word error rate (WER; character error rate, CER, for Chinese and Japanese) against the input text. Lower is better.

| Language | Metric | Error rate |
|---|---|---:|
| English | WER | 3.96 % |
| Spanish | WER | 2.21 % |
| French | WER | 3.73 % |
| Italian | WER | 1.52 % |
| Portuguese | WER | 0.80 % |
| Hindi | WER | 11.70 % |
| Mandarin Chinese | CER | 4.26 % |
| Japanese | CER | 4.05 % |
| **5 European languages (500 sentences)** | WER | **2.44 %** |
| **Chinese + Japanese (200 sentences)** | CER | **4.15 %** |

For context, the self-contained front-ends in this release match or beat the reference host phonemizer
sentence-for-sentence on 96 % of the 800 sentences (100/100 in Spanish and French, 98/100
Italian, 97/100 Portuguese, 99/100 Japanese). Most residual errors are text-normalization edge cases listed under
[Limitations](#limitations) or ASR spelling choices (e.g. *e-mail* vs *email*).

The text front-ends were also checked directly against Kokoro's reference host phonemizer (`misaki`): on a
100-word evaluation set per language they produce **identical phonemes** for English (US and UK), Spanish, French,
Italian, Portuguese and Hindi.

## Performance

Measured with `omni_runner` on retail-equivalent phones, synthesizing an 18.3 s English paragraph (three sentences)
with `af_heart`, 4 CPU threads, after one warm-up run. **RTF** = synthesis time ÷ audio duration (lower is faster;
1.0 = real time). **Peak memory** is the process's peak resident set (`VmHWM`), including model load.

| Device | File | Backend | Synthesis time | RTF | Peak memory |
|---|---|---|---:|---:|---:|
| Galaxy S26 (Snapdragon 8 Elite Gen 5, Adreno) | `kokoro_int8_gpu.litertlm` ⭐ | GPU | 4.0 s | **0.22** | **485 MB** |
| Galaxy S26 | `kokoro_int8.litertlm` | GPU | 3.9 s | 0.21 | 1,030–1,225 MB |
| Galaxy S26 | `kokoro_int8.litertlm` | CPU | 4.7 s | 0.26 | 975 MB |
| Pixel 10 Pro (Tensor G5, PowerVR) | `kokoro_int8_gpu.litertlm` ⭐ | GPU | 17.0 s | 0.93 | **415–550 MB** |
| Pixel 10 Pro | `kokoro_int8.litertlm` | GPU | 17.0 s | 0.93 | 1,070–1,275 MB |
| Pixel 10 Pro | `kokoro_int8.litertlm` | CPU | 17.1 s | 0.94 | 975 MB |

- Engine load time (model parse + GPU kernel cache load): 1.5–3 s on GPU, ~0.5 s on CPU. The very first GPU run on a
  device additionally compiles kernels (a few extra seconds); they are cached in `--cache_dir`.
- Across the 800-sentence benchmark in [Quality](#quality) (Galaxy S26 Ultra, GPU file), aggregate RTF was **0.30**
  (mean 1.19 s to synthesize a 3.9 s sentence), with Mandarin fastest (0.24) and French slowest (0.36).
- The GPU file keeps the vocoder's INT8 weights packed on the GPU, which is why it needs roughly **half the memory** of
  the CPU/GPU file (whose explicit dequantize ops materialize FP32 vocoder weights in host memory).
- On the Pixel 10 Pro the vocoder runs fully on the GPU too, but the acoustic model (CPU in both files) accounts
  for almost all of the 17 s on Tensor G5, so the backend choice barely changes the total. Speeding up the acoustic
  model on Tensor CPUs is a known follow-up.
- Long inputs are split into sentences / clauses and synthesized as a stream, so time-to-first-audio is roughly the
  synthesis time of the first sentence regardless of total text length.

## What's in the file

| Component | Details |
|---|---|
| Acoustic model | Kokoro text encoder, duration/prosody predictors and decoder, INT8 weights (dynamic quantization) with the six bidirectional LSTMs fused into a single custom op. Runs on CPU. |
| Vocoder | ISTFTNet generator, INT8 convolution weights. Runs on GPU (`kokoro_int8_gpu.litertlm`) or CPU/GPU (`kokoro_int8.litertlm`). The final inverse STFT runs in C++. |
| Voices | 54 style embeddings (`af_heart`, `bf_emma`, `zf_xiaobei`, …) |
| Text front-end | Rule-based text normalization for every language (numbers, dates, currency, abbreviations), a 5 MB English lexicon with heteronym/part-of-speech handling, rule-based G2P for Spanish, French, Italian, Portuguese and Hindi, a Japanese morphological dictionary and a Chinese word segmenter + pinyin lexicon with tone sandhi, and a Traditional→Simplified Chinese table. |
| Output | 24 kHz, 16-bit mono PCM |

The models were exported from the original PyTorch checkpoint with [litert-torch](https://github.com/google-ai-edge/litert-torch)
and quantized with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

- The acoustic model takes about 124 phonemes (~10 s of speech) per run. Longer sentences are split automatically, at punctuation when possible and otherwise between words, which can add a short pause mid-sentence.
- Mixed-script input works (e.g. English words inside Chinese text), but all-caps acronyms inside Chinese text are read as words rather than spelled out.
- Very large numbers (≥ 1,000,000) in Portuguese and Hindi and some 4-digit currency amounts in English may be read as years rather than cardinals.
- The GPU version needs a GPU supported by the LiteRT GPU delegate (OpenCL, WebGPU or Metal); it was tested on Qualcomm Adreno and Pixel (PowerVR). Use `kokoro_int8.litertlm` elsewhere.

## License & acknowledgements

- Weights: Apache-2.0, inherited from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (trained by `@rzvzn`; architecture by [StyleTTS 2](https://arxiv.org/abs/2306.07691) / [ISTFTNet](https://arxiv.org/abs/2203.02395), Li et al.).
- Bundled lexicons: English from [misaki](https://github.com/hexgrad/misaki) (Apache-2.0); Japanese from [unidic-lite](https://github.com/polm/unidic-lite) (BSD); Chinese from [jieba](https://github.com/fxsjy/jieba) / [pypinyin](https://github.com/mozillazg/python-pinyin) (MIT).
- Runtime: [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0).
- Training data, as disclosed by the upstream model: a few hundred hours of permissive / non-copyrighted audio with IPA labels. This conversion adds no training data and no personal information.
