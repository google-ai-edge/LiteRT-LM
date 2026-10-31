---
license: cc-by-4.0
library_name: litert-lm
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt-0.6b-v3
base_model_relation: quantized
language:
  - bg
  - hr
  - cs
  - da
  - nl
  - en
  - et
  - fi
  - fr
  - de
  - el
  - hu
  - it
  - lv
  - lt
  - mt
  - pl
  - pt
  - ro
  - sk
  - sl
  - es
  - sv
  - ru
  - uk
tags:
  - litert
  - litert-lm
  - mediapipe
  - tflite
  - asr
  - automatic-speech-recognition
  - speech-to-text
  - on-device
  - android
  - ios
  - web
  - parakeet
  - tdt
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

# Parakeet-TDT-0.6B-v3 for MediaPipe & LiteRT-LM

Fully on-device, 25-language speech-to-text in a single file.

This is
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(600M-parameter FastConformer encoder + Token-and-Duration Transducer, TDT,
decoder) packaged as a `.litertlm` bundle for the
**[MediaPipe Audio Transcriber (`AudioTranscriber`) Task](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_transcriber)**
and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), Google's
open-source on-device inference runtime, across Android, iOS, macOS, Python,
Web, and C++, alongside standalone `.tflite` and AOT-compiled NPU variants for
[LiteRT](https://github.com/google-ai-edge/LiteRT). Everything
needed to transcribe arbitrary-length audio with automatic language detection,
punctuation, and capitalization is inside the one `.litertlm` file: INT8
encoder, hybrid stateful/stateless TDT decoder, unified 8,192-token multilingual
SentencePiece tokenizer, and `Speech2TextMetadata` (16 kHz resampler and 128-bin
Slaney log-mel front-end configuration) — no Python and no network required.

-   **Languages:** 25 European languages with automatic language detection,
    punctuation, and capitalization (see
    [Supported languages](#supported-languages))
-   **Streaming:** 5-second sliding windows with configurable overlap
    (`overlap_ratio` of `0.4` to `0.5`) and word-timestamp dynamic-programming
    seam alignment (`TimestampTextMerger`) for arbitrary-length audio
-   **Speed:** transcribes speech **9.5–15.6× faster than real time** (`RTF` of
    `0.06` to `0.11`, **12.1× mean RTFx**) on a Galaxy S24 Ultra CPU, with
    **765–779 MB** peak memory (**3.2× less memory** and **3.0× faster** than
    FP32)
-   **Quality:** **5.07 %–5.15 % average WER** across a 5-dataset English
    streaming benchmark (280 utterances, 35.6 min), matching the unquantized
    FP32 checkpoint (`5.24 %`) within `0.09–0.17 pp` (see [Quality](#quality))

## Which file should I use?

| File | Size | Backend | Recommended for |
|---|---:|---|---|
| **`parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm`** ⭐ | 615 MB | CPU / GPU (`Delegate.CPU` / `--backend=cpu` recommended) | **MediaPipe (`AudioTranscriber`) and LiteRT-LM (`OmniEngine` / `omni_runner`) across Android, iOS, macOS, Linux, Python, and Web.** All-in-one `.litertlm` bundle with INT8 stateful model, tokenizer, and `Speech2TextMetadata`. |
| `parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite` | 614 MB | CPU / GPU | Direct [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr) integration without LiteRT-LM (`encode`, stateless `decode`, and stateful `decode_1`). |
| `parakeet_tdt_0.6b_v3_5s_i8.tflite` | 614 MB | CPU / GPU | Standalone INT8 `.tflite` with stateless `encode` and `decode` signatures only. |
| `parakeet_tdt_0.6b_v3_5s_f32_stateful.tflite`<br>`parakeet_tdt_0.6b_v3_5s_f32.tflite` | 2.42 GB | CPU / GPU | Unquantized FP32 reference baselines (stateful and stateless; ~2.5 GB peak RAM). |
| `*_f32_stateful_<SoC>.tflite`<br>`*_f32_<SoC>.tflite` | 1.23–1.30 GB | NPU + CPU | Standalone LiteRT NPU pipelines AOT-compiled for a specific mobile SoC (see below). |

How the variants differ:

-   **`.litertlm` vs. `.tflite`** — the `.litertlm` file bundles the `.tflite`
    graph together with the compressed Hugging Face `tokenizer.json` and
    `Speech2TextMetadata` protobuf in one self-contained archive for MediaPipe's
    `AudioTranscriber` and LiteRT-LM's `OmniEngine`. Standalone `.tflite` files
    contain only the neural network graph and expect the host app to supply mel
    preprocessing and detokenization.
-   **Stateful (`_stateful`) vs. stateless** — stateful models expose three
    signatures: `encode`, stateless multi-token `decode`, and single-token
    stateful `decode_1` (which carries explicit 2-layer LSTM hidden and cell
    states `[2, 1, 640]`). During decoding, `OmniEngine` runs stateless `decode`
    for the first 4 tokens (`<sos>` onset context) and then zero-copy
    transitions the LSTM state into single-token `decode_1`, avoiding
    full-prefix recomputation at every step.
-   **INT8 (`_i8`) vs. FP32 (`_f32`)** — INT8 dynamic range quantization shrinks
    the model from **2.42 GB to 615 MB** (`4×` smaller on disk, `3.2×` lower
    peak RAM at `765–779 MB` vs. `2,458–2,478 MB` on mobile CPU) and accelerates
    mobile ARM64 CPU inference by **3.0–3.4×** via `SDOT`/`I8MM` SIMD
    instructions with no loss in transcription accuracy (`5.07 %`–`5.15 %` vs.
    `5.24 %` mean WER).
-   **AOT-compiled NPU `.tflite` files (`_<SoC>.tflite`)** — pre-compiled for
    on-device NPU accelerators via the LiteRT NeuroPilot / QNN / Google Tensor
    compilers:

| SoC family | Available `<SoC>` suffixes |
|---|---|
| **Google Tensor** | `Google_Tensor_G5` (Pixel 10), `Google_Tensor_G6` |
| **Qualcomm Snapdragon** | `Qualcomm_SM8450` (8 Gen 1), `SM8550` (8 Gen 2), `SM8650` (8 Gen 3), `SM8750` (8 Elite), `SM8850` (8 Elite Gen 5), `SA8255`, `SA8295` |
| **MediaTek Dimensity** | `MediaTek_MT6877`, `MT6878`, `MT6879`, `MT6897`, `MT6983` (9000), `MT6985` (9200), `MT6989` (9300), `MT6991` (9400), `MT6993`, `MT8171`, `MT8188`, `MT8189` |

## Download

```sh
pip install -U huggingface_hub

# Recommended: INT8 stateful .litertlm bundle (615 MB, 25 European languages)
hf download litert-community/parakeet-tdt-0.6b-v3 \
    parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm --local-dir .
```

---

## MediaPipe Audio Transcriber Task Capabilities

When loaded with the [MediaPipe `AudioTranscriber` Task](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_transcriber), Parakeet-TDT-0.6B-v3 supports:

- **Single-file `.litertlm` deployment**: Accepts mono floating-point (`float32`) PCM audio in `[-1.0, 1.0]` at any sample rate (automatically resampled to 16 kHz when `sampleRateHz` / `sample_rate_hz` is set) with built-in 128-bin Slaney log-mel preprocessing, hybrid stateful/stateless TDT decoding, and 25-language SentencePiece detokenization—zero external Python or audio library dependencies.
- **Low-latency streaming transcription (`transcribe` / `transcribeStream` + `finishStream` / `finish_stream`)**: Feed live microphone audio in small chunks (`endOfStream = false` / `end_of_stream=False`) and receive recognized `TranscriptionSegment` / `Segment` updates continuously while the user speaks, then call `finishStream` / `finish_stream` to flush any remaining buffered audio.
- **Single-call audio clip transcription**: Pass a complete audio recording with `endOfStream = true` (`end_of_stream=True`) or followed by `finishStream()` to transcribe an entire clip in one pass.
- **Real-time interim & finalized segments (`enableInterimResults` / `enable_interim_results`)**: Optionally emit low-latency non-final hypotheses (`isFinal = false` / `is_final=False`) while a 5-second sliding window is still in progress, and automatically promote cross-window seam-aligned words to finalized segments (`isFinal = true` / `is_final=True`) via `TimestampTextMerger`.
- **Segment-level timestamps (`fromTimestampMs` / `toTimestampMs`)**: Every emitted segment includes start and end timestamps in milliseconds relative to the start of the stream (`fromTimestampMs` / `toTimestampMs` on Android & Web, `fromTimestampInMilliseconds` / `toTimestampInMilliseconds` on iOS & macOS, `from_timestamp_ms` / `to_timestamp_ms` in Python & C++).
- **Engine prewarming (`prewarm`)**: Run a warm-up pass ahead of time on a background thread so the first spoken utterance is transcribed without cold-start initialization latency.

---

## Usage with MediaPipe Audio Transcriber Task

### 1. Android (Kotlin / Java)

See the [MediaPipe Audio Transcriber Android Guide](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_transcriber/android) and [Android Sample App](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/audio_transcriber/android) for a complete project setup.

#### Add the Gradle dependency

Add the MediaPipe Audio Tasks dependency to your module's `build.gradle.kts` (`minSdk = 26`, `arm64-v8a`):

```kotlin
dependencies {
    implementation("com.google.mediapipe:tasks-audio:latest.release")
}
```

#### Initialize and prewarm the task

```kotlin
import com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber
import com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriber.AudioTranscriberOptions
import com.google.mediapipe.tasks.audio.audiotranscriber.AudioTranscriberResult
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate

val options = AudioTranscriberOptions.builder()
    .setBaseOptions(
        BaseOptions.builder()
            .setModelAssetPath("/data/user/0/com.example.app/files/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm")
            .setDelegate(Delegate.CPU) // CPU recommended for Parakeet-TDT (or Delegate.GPU)
            .build()
    )
    .setSampleRateHz(16000)
    .setEnableInterimResults(true)
    .setCacheDir(context.cacheDir.absolutePath)
    .build()

val transcriber = AudioTranscriber.createFromOptions(context, options)

// Optional: warm up the inference engine on a background thread before recording starts:
transcriber.prewarm()
```

#### Streaming microphone transcription & single-clip transcription

```kotlin
// 1. Streaming microphone chunks on a background worker thread (endOfStream = false):
while (isRecording) {
    val read = audioRecord.read(chunk, 0, chunk.size, AudioRecord.READ_BLOCKING)
    if (read > 0) {
        val result: AudioTranscriberResult =
            transcriber.transcribe(chunk.copyOf(read), /* endOfStream= */ false)
        for (segment in result.segments()) {
            println(
                "[${segment.fromTimestampMs()}..${segment.toTimestampMs()} ms] " +
                    "(final=${segment.isFinal()}): ${segment.transcript()}"
            )
        }
    }
}
// Flush remaining buffered audio when recording stops:
val finalResult: AudioTranscriberResult = transcriber.finishStream()
println("Final tail: ${finalResult.text}")

// 2. Single-call audio clip transcription (endOfStream = true automatically flushes and finalizes):
val clipResult = transcriber.transcribe(audioClipSamples, /* endOfStream= */ true)
println("Clip transcript: ${clipResult.text}")

// Release native resources when done:
transcriber.close()
```

---

### 2. iOS & macOS (Swift)

See the [MediaPipe Audio Transcriber iOS Guide](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_transcriber/ios) and [iOS Sample App](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/audio_transcriber/ios).

#### Add the Swift Package

In Xcode, select **File > Add Package Dependencies...**, enter `https://github.com/google-ai-edge/mediapipe`, and add the **`MediaPipeTasksAudio`** library to your target.

#### Initialize and transcribe speech

```swift
import Foundation
import MediaPipeTasksAudio

let modelPath = Bundle.main.path(
    forResource: "parakeet_tdt_0.6b_v3_5s_i8_stateful",
    ofType: "litertlm"
)!
let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("mediapipe_asr_cache").path

let options = AudioTranscriberOptions()
options.baseOptions.modelAssetPath = modelPath
options.baseOptions.delegate = .CPU // CPU recommended (or .GPU)
options.sampleRateHz = 16000 // Or 48000 to match AVAudioEngine inputNode sampleRate
options.enableInterimResults = true
options.cacheDirectory = cacheDir

let transcriber = try AudioTranscriber(options: options)
try transcriber.prewarm()

// 1. Streaming microphone buffers on a serial background queue:
func processAudioChunk(_ samples: [Float]) throws {
    guard !samples.isEmpty else { return }
    let result = try samples.withUnsafeBufferPointer { buffer in
        try transcriber.transcribe(
            samples: buffer.baseAddress!,
            count: UInt(buffer.count),
            endOfStream: false
        )
    }
    for segment in result.segments {
        print(
            "[\(segment.fromTimestampInMilliseconds)..\(segment.toTimestampInMilliseconds) ms] " +
            "(final=\(segment.isFinal)): \(segment.transcript)"
        )
    }
}

// Flush remaining buffered audio when the user stops speaking:
let finalResult = try transcriber.finishStream()
print("Finalized tail: \(finalResult.text)")

// Release native resources when done:
try transcriber.close()
```

---

### 3. Python

```python
import numpy as np
import soundfile as sf
from mediapipe.tasks.python.audio import audio_transcriber
from mediapipe.tasks.python.core import base_options

audio_samples, sample_rate = sf.read("audio_sample.wav", dtype="float32")
if audio_samples.ndim > 1:
    audio_samples = np.mean(audio_samples, axis=1)

options = audio_transcriber.AudioTranscriberOptions(
    base_options=base_options.BaseOptions(
        model_asset_path="parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
        delegate=base_options.BaseOptions.Delegate.CPU,
    ),
    sample_rate_hz=sample_rate,
    enable_interim_results=True,
    cache_dir="/tmp/mediapipe_asr_cache",
)

with audio_transcriber.AudioTranscriber.create_from_options(options) as transcriber:
    transcriber.prewarm()

    # 1. Unary audio clip transcription (end_of_stream=True automatically flushes and finalizes)
    result = transcriber.transcribe(audio_samples, end_of_stream=True)
    for seg in result.segments:
        print(f"[{seg.from_timestamp_ms}..{seg.to_timestamp_ms} ms] (final={seg.is_final}): {seg.transcript}")

    # 2. Streaming chunk-by-chunk transcription (e.g. 500 ms chunks)
    chunk_size = sample_rate // 2
    for offset in range(0, len(audio_samples), chunk_size):
        chunk = audio_samples[offset : offset + chunk_size]
        partial = transcriber.transcribe(chunk, end_of_stream=False)
        for seg in partial.segments:
            print(f"{'FINAL' if seg.is_final else 'INTERIM'}: {seg.transcript}")
    final_tail = transcriber.finish_stream()
    if final_tail.text:
        print(f"FINAL: {final_tail.text}")
```

---

### 4. Web (JavaScript / TypeScript)

```typescript
import { AudioTranscriber, FilesetResolver } from "@mediapipe/tasks-audio";

const wasmFileset = await FilesetResolver.forAudioTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@latest/wasm"
);

const transcriber = await AudioTranscriber.createFromOptions(wasmFileset, {
  baseOptions: {
    modelAssetPath: "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
    delegate: "CPU", // Or "GPU" (WebGPU)
  },
  sampleRateHz: 16000,
  enableInterimResults: true,
});

await transcriber.prewarm();

// 1. Unary audio clip transcription (endOfStream defaults to true, which flushes and finalizes)
const clipResult = await transcriber.transcribe(audioClipFloat32Array);
console.log("Transcript:", clipResult.text);

// 2. Streaming microphone chunks (endOfStream = false)
const streamResult = await transcriber.transcribeStream(micChunkFloat32Array);
for (const seg of streamResult.segments) {
  console.log(
    `[${seg.fromTimestampMs}..${seg.toTimestampMs} ms] (final=${seg.isFinal}): ${seg.transcript}`
  );
}
const finalResult = await transcriber.finishStream();

transcriber.close();
```

---

### 5. C++

```cpp
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"

using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriberOptions;
using ::mediapipe::tasks::core::BaseOptions;

auto options = std::make_unique<AudioTranscriberOptions>();
options->base_options.model_asset_path =
    "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm";
options->base_options.delegate = BaseOptions::Delegate::CPU;
options->sample_rate_hz = 16000;
options->enable_interim_results = true;
options->cache_dir = "/tmp/mediapipe_asr_cache";

ABSL_ASSIGN_OR_RETURN(auto transcriber,
                      AudioTranscriber::Create(std::move(options)));
ABSL_RETURN_IF_ERROR(transcriber->Prewarm());

// Streaming audio chunks (end_of_stream = false):
ABSL_ASSIGN_OR_RETURN(
    auto partial,
    transcriber->Transcribe(pcm_chunk, /*end_of_stream=*/false));
for (const auto& seg : partial.segments) {
  // seg.transcript, seg.is_final, seg.from_timestamp_ms, seg.to_timestamp_ms
}

// Flush remaining buffered audio at the end of the stream:
ABSL_ASSIGN_OR_RETURN(auto final_result, transcriber->FinishStream());
ABSL_RETURN_IF_ERROR(transcriber->Close());
```

---

## Configuration Options Reference

### `AudioTranscriberOptions`

| Option (Android / iOS / Web) | Option (Python / C++) | Description | Default |
|---|---|---|---|
| `baseOptions.modelAssetPath` | `base_options.model_asset_path` | Path (or URL on Web) to the `.litertlm` model bundle (`parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm`). | *Required* |
| `baseOptions.delegate` | `base_options.delegate` | Execution delegate (`CPU` recommended on Android, iOS, and desktop for lowest autoregressive TDT decode latency; `GPU` also supported). | `CPU` (`GPU` on Web) |
| `sampleRateHz` | `sample_rate_hz` | Input mono PCM float32 sample rate in Hz. Automatically resampled to 16 kHz when different from `16000`. | `16000` |
| `enableInterimResults` | `enable_interim_results` | Whether to emit non-final interim hypotheses (`isFinal = false` / `is_final = false`) while audio is streaming. | `false` |
| `modelArchitecture` | `model_architecture` | ASR architecture family (`AUTO` / `.auto` / `kAuto` or `OMNI` / `.omni` / `kOmni`). Automatically inferred from the `.litertlm` bundle when set to `AUTO`. | `AUTO` |
| `cacheDir` (`cacheDirectory` on iOS) | `cache_dir` | Writable directory path to persist compiled XNNPACK weight packs and GPU kernel caches across runs. | `""` (`None` in Python) |

---

## Usage with LiteRT-LM Omni (`omni_runner` & C++)

All commands below accept mono or multi-channel WAV files at any sample rate
(automatically resampled to 16 kHz mono) and stream confirmed and unconfirmed
transcript segments to stdout (or `--output_path`). Prerequisites (Bazel,
Android NDK) are in the LiteRT-LM
[build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

### 1. Download a model

```sh
pip install -U huggingface_hub
hf download litert-community/parakeet-tdt-0.6b-v3 \
    parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm --local-dir .
```

> [!TIP]
> If you omit `--model_path` and pass `--model_name=parakeet-tdt-0.6b-v3`,
> `omni_runner` automatically downloads
> `parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm` into `--cache_dir` on first
> run.

### 2. Build the runner

```sh
git clone https://github.com/google-ai-edge/LiteRT-LM.git && cd LiteRT-LM
bazel build -c opt //omni:omni_runner                            # Linux / macOS
bazel build -c opt --config=android_arm64 //omni:omni_runner     # Android (needs ANDROID_NDK_HOME)
```

### 3. Transcribe speech

**On Linux or macOS (CPU):**

```sh
bazel-bin/omni/omni_runner --model_name=parakeet-tdt-0.6b-v3 \
    --model_path=/path/to/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav
```

**On an Android phone (CPU, fastest):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm \
         runtime/components/testdata/audio_sample.wav $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=parakeet-tdt-0.6b-v3 \
    --backend=cpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

**On an Android phone (GPU):**

```sh
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --model_name=parakeet-tdt-0.6b-v3 \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav
```

> `--model_name=parakeet-tdt-0.6b-v3` (or alias `parakeet-tdt`) tells the runner
> which model family the file is (it is inferred automatically when the file is
> named `parakeet-tdt-0.6b-v3.litertlm`). XNNPACK weight packs and compiled GPU
> kernels are cached in `--cache_dir` so subsequent runs start fast.

The full list of flags and the C++ `OmniEngine` / `OmniSession` streaming API
are documented in the
[LiteRT-LM omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

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
    .model_path =
        "/data/local/tmp/omni/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
};
litert::lm::proto::Speech2TextSessionConfig asr_config;
asr_config.set_overlap_ratio(0.4f);
asr_config.set_text_merger_type(
    litert::lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("parakeet-tdt-0.6b-v3", settings, &asr_config));
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

In your `BUILD` file, link only the Parakeet-TDT target
(`//omni/speech2text:parakeet_tdt`) to avoid pulling in TTS or image generation
models:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:parakeet_tdt",
    ],
)
```

## Supported languages

`parakeet-tdt-0.6b-v3` transcribes **25 European languages** with a single
unified 8,192-token SentencePiece vocabulary. Language identification,
punctuation, and capitalization are performed **automatically** by the model —
no language code flag or prompt token is required.

| Language | Code | FLEURS WER | Language | Code | FLEURS WER |
|---|---|---:|---|---|---:|
| Bulgarian | `bg` | 6.64 % | Lithuanian | `lt` | 12.16 % |
| Croatian | `hr` | 7.69 % | Maltese | `mt` | 8.33 % |
| Czech | `cs` | 6.80 % | Polish | `pl` | 7.31 % |
| Danish | `da` | 11.46 % | Portuguese | `pt` | 6.45 % |
| Dutch | `nl` | 6.48 % | Romanian | `ro` | 6.53 % |
| English | `en` | 4.85 % | Russian | `ru` | 5.51 % |
| Estonian | `et` | 7.72 % | Slovak | `sk` | 5.66 % |
| Finnish | `fi` | 6.13 % | Slovenian | `sl` | 11.53 % |
| French | `fr` | 5.15 % | Spanish | `es` | 3.45 % |
| German | `de` | 5.04 % | Swedish | `sv` | 9.35 % |
| Greek | `el` | 11.09 % | Ukrainian | `uk` | 6.72 % |
| Hungarian | `hu` | 8.85 % | **25-lang mean** | — | **7.73 %** |
| Italian | `it` | 3.00 % | | | |
| Latvian | `lv` | 9.93 % | | | |

Full per-language benchmarks across FLEURS, Multilingual LibriSpeech (MLS), and
CoVoST 2 are available in the upstream
[nvidia/parakeet-tdt-0.6b-v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Quality

**How we measured.** We evaluated
`parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm` (INT8) against the unquantized
FP32 baseline (`parakeet_tdt_0.6b_v3_5s_f32_stateful.tflite`) across **5 English
streaming benchmarks** totaling **280 utterances (~35.6 minutes of audio)**:
`mini` (5 short voice commands), `ytlego10` (10 YouTube clips), `ytlego30` (30
conversational YouTube clips), `regression45` (44 multi-domain long-form clips),
and `greco3_short` (191 short utterances at 16 & 24 kHz). All runs used 5.0 s
sliding windows (`overlap_ratio = 0.5`), `TimestampTextMerger` overlap
alignment, and standard Whisper / Hugging Face Open ASR Leaderboard English text
normalization (`ASR_HUGGINGFACE`). Lower WER is better.

| Benchmark | Clips (Audio) | INT8 Host CPU | INT8 S24U CPU | INT8 S24U GPU | FP32 Baseline |
|---|---:|---:|---:|---:|---:|
| **`mini`** | 5 (15.2 s) | **0.00 %** | **0.00 %** | **0.00 %** | 0.00 % |
| **`ytlego10`** | 10 (3.1 min) | 5.16 % | 5.16 % | **5.00 %** | 5.65 % |
| **`ytlego30`** | 30 (9.2 min) | 6.47 % | 6.58 % | **6.42 %** | 6.58 % |
| **`regression45`** | 44 (11.1 min) | **6.48 %** | 6.94 % | 6.53 % | 6.89 % |
| **`greco3_short`** | 191 (12.0 min) | 7.23 % | **7.09 %** | 7.79 % | **7.09 %** |
| **5-dataset mean** | **280 (35.6 min)** | **5.07 %** | **5.15 %** | **5.15 %** | **5.24 %** |

Key takeaways:

-   **Zero quantization degradation:** INT8 dynamic range quantization matches
    the 2.42 GB FP32 baseline within `0.09–0.17 pp` on the 5-dataset mean and
    slightly improves overall accuracy (`5.07 %` to `5.15 %` INT8 vs. `5.24 %`
    FP32).
-   **Consistent across CPU and GPU:** Host x86_64 CPU, mobile ARM64 CPU
    (Snapdragon 8 Gen 3), and mobile OpenCL GPU (Adreno 750) all converge to
    `5.07 %` to `5.15 %` mean WER.
-   **Comparison with other LiteRT-LM Omni ASR models:** On the same 5-dataset
    streaming suite (Host CPU, INT8), `parakeet-tdt-0.6b-v3` (FastConformer-TDT
    0.6B) achieves the lowest WER among all supported LiteRT-LM speech-to-text
    models:

| Model (`--model_name`) | Size | `mini` | `yt10` | `yt30` | `reg45` | `greco3` | **Mean WER** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`parakeet-tdt-0.6b-v3`** ⭐ | 615 MB | **0.00 %** | 5.16 % | **6.47 %** | **6.48 %** | **7.23 %** | **5.07 %** |
| `qwen3-asr-0.6b` | 960 MB | 7.14 % | **4.84 %** | 8.19 % | 8.78 % | 7.37 % | 7.26 % |
| `parakeet-ctc-0.6b` | 597 MB | 7.14 % | 5.81 % | 7.54 % | 7.09 % | 10.43 % | 7.60 % |
| `whisper-tiny` | 42 MB | 17.86 % | 8.55 % | 14.66 % | 8.88 % | 18.22 % | 13.63 % |

-   **Upstream English benchmark reference:** On the Hugging Face Open ASR
    Leaderboard (full-utterance offline evaluation across 8 standard English
    benchmarks), `nvidia/parakeet-tdt-0.6b-v3` achieves **6.32 % average WER**
    (`1.93 %` LibriSpeech `test-clean`, `3.59 %` `test-other`, `2.75 %`
    TEDLIUM-v3, `3.97 %` SPGISpeech, `6.14 %` VoxPopuli, `9.59 %` GigaSpeech,
    `11.16 %` AMI, `11.39 %` Earnings22).

## Performance

Measured with `omni_runner` on a Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3,
Adreno 750) and a Linux x86_64 workstation across the 30-clip `ytlego30`
benchmark (552.7 s / 9.2 min of multi-window conversational speech) and the
191-clip `greco3_short` benchmark (718.4 s / 12.0 min of short utterances),
comparing `parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm` (**INT8 `.litertlm`**)
and `parakeet_tdt_0.6b_v3_5s_f32_stateful.tflite` (**FP32 `.tflite`**) with 4
CPU threads and a warm cache. **RTFx** = audio duration ÷ transcription time
(higher is faster; `10×` means 10 s of audio transcribed in 1 s). **RTF** =
transcription time ÷ audio duration (lower is faster; `1.0` = real time). **Peak
RAM** is the process's peak resident set (`VmHWM`), including model load.

| Device | Variant | Backend | `ytlego30` RTFx (RTF) | `greco3` RTFx (RTF) | Peak RAM |
|---|---|---|---|---|---:|
| **Galaxy S24 Ultra** | **INT8 `.litertlm`** ⭐ | **CPU** | **9.59×** (`0.104`) | **15.55×** (`0.064`) | **765–779 MB** |
| Galaxy S24 Ultra | INT8 `.litertlm` | GPU | 5.22× (`0.192`) | 9.05× (`0.111`) | 796–800 MB |
| Galaxy S24 Ultra | FP32 `.tflite` | CPU | 3.15× (`0.318`) | 4.62× (`0.216`) | 2,458–2,478 MB |
| Galaxy S24 Ultra | FP32 `.tflite` | GPU | 4.30× (`0.233`) | 7.85× (`0.127`) | 2,569–2,575 MB |
| **Linux x86_64** | **INT8 `.litertlm`** ⭐ | **CPU** | **5.40×** (`0.185`) | **9.03×** (`0.111`) | 875–893 MB |
| Linux x86_64 | FP32 `.tflite` | CPU | 5.09× (`0.197`) | 8.79× (`0.114`) | 2,583–2,601 MB |

-   **Why CPU is faster than GPU for Parakeet-TDT:** While the FastConformer
    `encode` pass runs once per 5-second window, the autoregressive TDT decoder
    (`decode_1`) executes ~60–90 small single-token LSTM + joint-network
    invocations per window. On mobile OpenCL GPUs, per-step GPU queue dispatch
    latency dominates those small sequential invocations, whereas modern
    ARMv8.2+/ARMv9 CPUs execute INT8 matrix-vector products inline via
    `SDOT`/`I8MM` SIMD instructions with microsecond latency.
-   **Why short utterances have higher RTFx (9.0–15.6×) than long overlapping
    streams (4.6–10.9×):** Clips shorter than 5 seconds require a single 5 s
    encoder window (`0.06` to `0.11` RTF), whereas continuous multi-window audio
    with `overlap_ratio = 0.5` steps the 5 s window forward every 2.5 seconds
    (`2×` encoder passes per unit of audio time) to guarantee seamless
    cross-boundary word alignment. Reducing `overlap_ratio` to `0.4` (the
    `omni_runner` CLI default) increases long-form streaming RTFx by ~20 %.
-   **3.2× lower peak memory with INT8:** Dynamic range quantization reduces
    peak resident memory (`VmHWM`) from `2,458`–`2,575 MB` on FP32 to
    **765–800 MB** on INT8 on Galaxy S24 Ultra (and from `2,583`–`2,601 MB` to
    **875–893 MB** on Linux x86_64) across CPU and GPU backends.

## What's in the file

| Component | Details |
|---|---|
| **Audio front-end** | Self-contained C++ preprocessor (`Speech2TextMetadata`): resamples any input rate to 16 kHz mono, applies `0.97` pre-emphasis, 512-point Hann STFT (25 ms window, 10 ms hop, 500 frames per 5 s window), 128-bin Slaney log-mel filterbank (0–8 kHz), and per-feature mean/std normalization. |
| **Encoder** | 600M-parameter FastConformer (`encode`) with 8× depthwise-separable conv subsampling (500 mel frames @ 10 ms → 63 frames @ 80 ms, `1024`-dim), quantized to channel-wise INT8 weights. |
| **TDT decoder** | 2-layer LSTM prediction network (`decode` & stateful `decode_1`, state `[2, 1, 640]`) + Token-and-Duration Transducer joint network outputting `8,198` logits (`8,192` vocab + `<blank>` + duration steps `0..4`). |
| **Tokenizer** | Zlib-compressed Hugging Face `tokenizer.json` (8,192 multilingual SentencePiece tokens with punctuation & capitalization) + incremental word-timestamp detokenizer. |
| **Text merger** | `TimestampTextMerger`: monotone dynamic-programming seam alignment across overlapping 5 s windows (`overlap_ratio = 0.4` in `omni_runner`, `0.75` in `AsrSession` / `AudioTranscriber`). |
| **Output** | Streaming `OmniSession::TextOutput` chunks (`confirmed_text` + `unconfirmed_text`). |

The models were exported from the upstream NeMo checkpoint
(`parakeet-tdt-0.6b-v3.nemo`) to LiteRT via PyTorch and
[litert-torch](https://github.com/google-ai-edge/litert-torch) and quantized
with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

-   **Fixed 5-second encoder window:** Longer audio is processed in overlapping
    5-second windows (2.5 to 3.0 s stride). Words spoken right across a window
    boundary are emitted immediately in `unconfirmed_text` and promoted to
    `confirmed_text` once the next overlapping window resolves the boundary seam
    (~1–2 s confirmation lag).
-   **European languages only:** The model was trained on the 25 European
    languages listed in [Supported languages](#supported-languages). For CJK,
    Arabic, Hindi, or other non-European languages, use
    [`litert-community/Qwen3-ASR-0.6B`](https://huggingface.co/litert-community/Qwen3-ASR-0.6B)
    (`--model_name=qwen3-asr-0.6b`) or
    [`litert-community/whisper-tiny`](https://huggingface.co/litert-community/whisper-tiny)
    (`--model_name=whisper-tiny`).
-   **Mobile GPU dispatch overhead on autoregressive steps:** Because TDT
    decoding runs ~60–90 sequential single-token `decode_1` invocations per 5 s
    window, `--backend=cpu` is faster than `--backend=gpu` on current Android
    phones (9.5–15.6× vs. 4.6–9.1× on Galaxy S24 Ultra).
-   **SoC-specific NPU files are standalone `.tflite` graphs:** The pre-compiled
    `parakeet_tdt_0.6b_v3_5s_f32_*_<SoC>.tflite` files target the standalone
    LiteRT NPU C++/Kotlin APIs (such as the
    [LiteRT ASR Android sample app](https://github.com/google-ai-edge/LiteRT/tree/main/litert/samples/asr))
    rather than `.litertlm` bundles.

## License & acknowledgements

-   **Weights & tokenizer:**
    [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en),
    inherited from
    [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
    (trained by the NVIDIA NeMo team on the
    [Granary](https://arxiv.org/abs/2505.13404) and NeMo ASR Set 3.0 corpora;
    [FastConformer](https://arxiv.org/abs/2305.05084), Rekesh et al.;
    [Token-and-Duration Transducer](https://arxiv.org/abs/2304.06795), Xu et
    al.;
    [Parakeet-TDT-0.6B-v3 Technical Report](https://arxiv.org/abs/2509.14128),
    Sekoyan et al.).
-   **Conversion & runtime:**
    [litert-torch](https://github.com/google-ai-edge/litert-torch),
    [LiteRT](https://github.com/google-ai-edge/LiteRT),
    [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_transcriber),
    and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0).
-   **Training data, as disclosed by the upstream model:** 660,000 hours of
    multilingual speech across 25 European languages from the public
    [Granary](https://huggingface.co/datasets/nvidia/Granary) corpus
    (YouTube-Commons, Yodas2, Mozilla Common Voice, Multilingual LibriSpeech,
    VoxPopuli) and 10,000 hours of human-transcribed NeMo ASR Set 3.0 data. This
    conversion adds no training data and no personal information.
