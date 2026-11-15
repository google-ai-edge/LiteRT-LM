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
  - mediapipe
  - tts
  - text-to-speech
  - on-device
  - android
  - ios
  - web
  - kokoro
  - int8
  - fp16
---

<!-- mdformat off(Hugging Face model card GFM) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->
<!-- disableFinding(WHITESPACE_LINES) -->

# Kokoro-82M for MediaPipe & LiteRT-LM

Fully on-device, multilingual text-to-speech in a single file.

This is [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82M-parameter StyleTTS 2 + ISTFTNet) packaged as a `.litertlm` bundle for the
**[MediaPipe Text-to-Speech (`TextToSpeech`) Task](https://ai.google.dev/edge/mediapipe/solutions/audio/text_to_speech)** and
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), Google's open-source on-device inference runtime, across Android, iOS, macOS, Python, Web, and C++. Everything needed to turn text into
24 kHz speech is inside the one file: INT8 acoustic + vocoder models, all 54 voices, and self-contained text
front-ends (text normalization and grapheme-to-phoneme) for **9 languages / dialects** — no Python, no network, and
no external GPL phoneme libraries, so the whole stack is Apache-2.0.

- **Languages:** American & British English, Spanish, French, Italian, Brazilian Portuguese, Hindi, Mandarin Chinese, Japanese
- **Voices:** all 54 Kokoro v1.0 voices (see [Languages & voices](#languages--voices))
- **Speed:** synthesizes speech 6–10× faster than real time on a Galaxy S26 GPU with FP16 precision, with ~0.35 GB peak memory
- **Quality:** on an 800-sentence, 8-language intelligibility benchmark it matches or beats the original Kokoro running on a server (see [Quality](#quality))

## Which file should I use?

| File | Size | Runs on | Recommended for |
|---|---:|---|---|
| **`kokoro_int8_gpu_fp16.litertlm`** ⭐ | 295 MB | Acoustic model on CPU, vocoder on **GPU** (`Delegate.GPU` / `--backend=gpu`), FP16 or FP32 GPU precision | **Any device with a GPU supported by the LiteRT GPU delegate (OpenCL, WebGPU or Metal)** — phones, tablets, laptops. Fastest and about **half the memory** (~0.35–0.5 GB peak). |
| `kokoro_int8_fp16.litertlm` | 295 MB | CPU or GPU (FP16 or FP32) | Devices without a usable GPU (e.g. servers, CI, older hardware, WASM CPU). Uses ~0.7–0.95 GB peak memory. |

Both files contain the same acoustic model weights, voices and text front-ends, and their vocoder graph is exported so
that intermediate values stay inside the FP16 range (enabling native FP16 GPU execution, which roughly halves vocoder
latency compared to FP32). The two files differ only in how the vocoder is quantized:

- **GPU version** — vocoder convolutions use INT8 weights with floating-point activations, which the LiteRT GPU
  delegate runs natively on OpenCL, WebGPU and Metal GPUs. Run it with `Delegate.GPU` / `--backend=gpu` and the LiteRT GPU
  accelerator libraries next to the runner (on Android: `prebuilt/android_arm64/*.so`, see below). With
  `Delegate.CPU` / `--backend=cpu`, or when no GPU accelerator is available, the vocoder silently runs on the CPU instead: speech is
  still intelligible but may be noisier, so use `kokoro_int8_fp16.litertlm` for CPU-only devices.
- **CPU/GPU version** — the vocoder convolutions carry explicit dequantize ops so XNNPACK (CPU) also produces clean
  audio. This costs extra memory, so only pick it if you cannot use the GPU.

> [!NOTE]
> The earlier FP32 `.tflite` preview that used to live at the root of this repo (three bucketed graphs + a fixed-length
> demo build, English only, CPU only) has moved to [`legacy/`](https://huggingface.co/litert-community/Kokoro-82M/tree/main/legacy)
> and is **deprecated**. New integrations should use the `.litertlm` files above.

## Download

```sh
pip install -U huggingface_hub

# Recommended: GPU version (295 MB, 54 voices across 9 languages / dialects)
hf download litert-community/Kokoro-82M kokoro_int8_gpu_fp16.litertlm --local-dir .

# Or: CPU/GPU version (when no GPU is available)
hf download litert-community/Kokoro-82M kokoro_int8_fp16.litertlm --local-dir .
```

---

## MediaPipe Text-to-Speech Task Capabilities

When loaded with the [MediaPipe `TextToSpeech` Task](https://ai.google.dev/edge/mediapipe/solutions/audio/text_to_speech), Kokoro-82M supports:

- **Single-file `.litertlm` deployment**: Produces normalized 24 kHz mono 32-bit floating-point (`float32`) PCM audio in `[-1.0, 1.0]` with all 54 voices and 9-language G2P / text normalization built in—zero external GPL phoneme libraries or Python dependencies.
- **Low-latency streaming speech synthesis (`synthesizeStream` / `synthesize_stream`)**: Automatically segments multi-sentence inputs at sentence and clause boundaries and streams `AudioChunk` buffers as each slice finishes synthesis.
- **Real-time LLM-to-TTS token streaming (`openStream` / `start_stream`)**: Push incremental text deltas (`pushText` / `push_text`) directly from an on-device LLM stream and call `finishInput` / `finish_input` when done. Speech synthesis begins immediately once the first sentence boundary arrives.
- **Word-level timestamp alignment (`wordTimestamps` / `word_timestamps`)**: Both unary `AudioResult` and streaming `AudioChunk` results include pause-aware `WordTimestamp` entries (`word`, `startChar`, `endChar`, `startSec`, `endSec`) mapped directly to character offsets in the input string for real-time read-along text highlighting.
- **Fast voice & language switching**: Switch across all 54 bundled voices and 9 languages per utterance using `SynthesisOptions` (`voiceId`, `locale`) without reloading model weights. When `voiceId` is set (e.g. `"af_heart"`, `"ef_dora"`, `"zf_xiaobei"`, `"jf_alpha"`), the matching language is inferred automatically if `locale` is omitted.
- **Speaking rate control & lock-free barge-in cancellation**: Adjust `speakingRate` (`(0.0, 4.0]`, default `1.0`) and abort in-flight synthesis within milliseconds via `cancel()` or native Kotlin Coroutine / Swift `Task` cancellation.

---

## Usage with MediaPipe Text-to-Speech Task

### 1. Android (Kotlin)

See the [MediaPipe Text-to-Speech Android Guide](https://ai.google.dev/edge/mediapipe/solutions/audio/text_to_speech/android) and [Android Sample App](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_to_speech/android) for a complete project setup.

#### Add the Gradle dependency

Add the MediaPipe Audio Tasks dependency to your module's `build.gradle.kts` (`minSdk = 26`, `arm64-v8a`):

```kotlin
dependencies {
    implementation("com.google.mediapipe:tasks-audio:latest.release")
}
```

#### Initialize the task and list voices

```kotlin
import com.google.mediapipe.tasks.audio.texttospeech.SynthesisOptions
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeech
import com.google.mediapipe.tasks.audio.texttospeech.TextToSpeechOptions
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate

val options = TextToSpeechOptions(
    baseOptions = BaseOptions.builder()
        .setModelAssetPath("/data/user/0/com.example.app/files/kokoro_int8_gpu_fp16.litertlm")
        .setDelegate(Delegate.GPU) // Or Delegate.CPU with kokoro_int8_fp16.litertlm
        .build(),
    defaultVoiceId = "af_heart",
    speakingRate = 1.0f,
    cacheDirectory = context.cacheDir.absolutePath,
)

val tts = TextToSpeech.create(options)

// Enumerate all 54 bundled voices at runtime:
for (voice in tts.listVoices()) {
    println("${voice.voiceId} (${voice.displayName}) - locale=${voice.locale}, ${voice.sampleRateHz} Hz")
}
```

#### Unary synthesis & word-level timestamps

```kotlin
// Non-blocking coroutine synthesis (or use tts.synthesize(...) on a background thread):
val result = tts.synthesizeAsync(
    "Hello from Kokoro running on-device with MediaPipe Text to Speech!",
    SynthesisOptions(voiceId = "af_heart"),
)
// result.samples: FloatArray (24 kHz mono float32 PCM in [-1.0, 1.0])
// result.sampleRateHz: 24000
// result.durationSeconds: Float

// Word-level timestamps for synchronized read-along UI highlighting:
for (wt in result.wordTimestamps) {
    println("Word '${wt.word}' chars=[${wt.startChar}, ${wt.endChar}) time=${wt.startSec}s..${wt.endSec}s")
}
```

#### Streaming synthesis & incremental LLM-to-TTS streaming

```kotlin
// 1. Streaming a multi-sentence prompt sentence-by-sentence:
tts.synthesizeStream(
    "Streaming synthesis plays the first sentence right away. Subsequent sentences arrive while audio plays.",
    SynthesisOptions(voiceId = "af_heart"),
).collect { chunk ->
    streamingAudioTrack.write(chunk.samples, 0, chunk.samples.size, AudioTrack.WRITE_BLOCKING)
    // chunk.wordTimestamps provides word timing offsets for the current chunk
}

// 2. Incremental token-by-token streaming from an on-device LLM:
tts.openStream(options = SynthesisOptions(voiceId = "af_heart")).use { stream ->
    stream.pushText("Hello from an ")
    stream.pushText("on-device LLM! ")
    stream.pushText("Audio starts streaming before generation finishes.")
    stream.finishInput()

    while (true) {
        val chunk = stream.readNext() ?: break
        streamingAudioTrack.write(chunk.samples, 0, chunk.samples.size, AudioTrack.WRITE_BLOCKING)
    }
}

// Release native resources when done:
tts.close()
```

---

### 2. iOS & macOS (Swift)

See the [MediaPipe Text-to-Speech iOS Guide](https://ai.google.dev/edge/mediapipe/solutions/audio/text_to_speech/ios) and [iOS Sample App](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_to_speech/ios).

#### Add the Swift Package

In Xcode, select **File > Add Package Dependencies...**, enter `https://github.com/google-ai-edge/mediapipe`, and add the **`MediaPipeTasksAudio`** library to your target.

#### Initialize and synthesize speech

```swift
import Foundation
import MediaPipeTasksAudio

let modelPath = Bundle.main.path(forResource: "kokoro_int8_gpu_fp16", ofType: "litertlm")!
let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("mediapipe_tts_cache").path

let options = TextToSpeechOptions(
    baseOptions: TTSBaseOptions(
        modelAssetPath: modelPath,
        delegate: .gpu // Or .cpu with kokoro_int8_fp16.litertlm
    ),
    defaultVoiceId: "af_heart",
    speakingRate: 1.0,
    cacheDirectory: cacheDir
)

let tts = try createTextToSpeech(options: options)

// Enumerate bundled voices:
for voice in try tts.listVoices() {
    print("\(voice.voiceId) - locale=\(voice.locale), \(voice.sampleRateHz) Hz")
}

// 1a. Async unary synthesis:
let asyncResult = try await tts.synthesizeAsync(
    "Hello from Kokoro on Apple platforms!",
    options: SynthesisOptions(voiceId: "af_heart")
)
// asyncResult.samples: [Float] (24 kHz mono PCM), asyncResult.sampleRateHz: 24000

// 1b. Synchronous unary synthesis with word-level timestamps (run on a background thread):
let result = try tts.synthesize(
    "Hello from Kokoro on Apple platforms!",
    options: SynthesisOptions(voiceId: "af_heart")
)
for wt in result.wordTimestamps {
    print("\(wt.word) [\(wt.startChar)..<\(wt.endChar)]: \(wt.startSec)s - \(wt.endSec)s")
}

// 2. Low-latency streaming synthesis:
for try await chunk in tts.synthesizeStream(
    "Streaming synthesis emits 24 kHz PCM chunks as soon as each sentence completes.",
    options: SynthesisOptions(voiceId: "bf_emma")
) {
    // Schedule chunk.samples (24 kHz mono Float) onto AVAudioPlayerNode
}

// 3. Incremental LLM-to-TTS streaming:
let stream = try tts.openStream(options: SynthesisOptions(voiceId: "af_heart"))
defer { stream.close() }

try stream.pushText("Streaming tokens from ")
try stream.pushText("an on-device language model.")
try stream.finishInput()

while let chunk = try stream.readNext() {
    // Play chunk.samples & highlight chunk.wordTimestamps
}
```

---

### 3. Python

```python
import soundfile as sf
from mediapipe.tasks.python.audio import text_to_speech
from mediapipe.tasks.python.core import base_options

options = text_to_speech.TextToSpeechOptions(
    base_options=base_options.BaseOptions(
        model_asset_path="kokoro_int8_fp16.litertlm"
    ),
    default_voice_id="af_heart",
    speaking_rate=1.0,
)

with text_to_speech.TextToSpeech.create_from_options(options) as tts:
    # List bundled voices
    for voice in tts.list_voices():
        print(f"{voice.voice_id}: locale={voice.locale}, {voice.sample_rate_hz} Hz")

    # 1. Unary synthesis + word-level timestamps
    result = tts.synthesize(
        "Hello from Kokoro running on-device with MediaPipe in Python!",
        voice_id="af_heart",
    )
    sf.write("output.wav", result.samples, result.sample_rate_hz)
    for wt in result.word_timestamps:
        print(f"{wt.word} [{wt.start_char}:{wt.end_char}] {wt.start_sec:.2f}s - {wt.end_sec:.2f}s")

    # 2. Streaming synthesis
    for chunk in tts.synthesize_stream(
        "First sentence streams right away. Second sentence follows seamlessly.",
        voice_id="af_heart",
    ):
        print(f"Chunk {chunk.chunk_index}: {len(chunk.samples)} samples (final={chunk.is_final})")

    # 3. Incremental LLM token streaming
    with tts.start_stream(voice_id="af_heart") as stream:
        stream.push_text("Streaming tokens ")
        stream.push_text("directly into Kokoro.")
        stream.finish_input()
        for chunk in stream:
            pass  # Play chunk.samples in real time
```

---

### 4. Web (JavaScript / TypeScript)

```typescript
import { FilesetResolver, TextToSpeech } from "@mediapipe/tasks-audio";

const wasmFileset = await FilesetResolver.forAudioTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@latest/wasm"
);

const tts = await TextToSpeech.createFromOptions(wasmFileset, {
  baseOptions: {
    modelAssetPath: "kokoro_int8_gpu_fp16.litertlm",
    delegate: "GPU", // Or "CPU" with kokoro_int8_fp16.litertlm
  },
  defaultVoiceId: "af_heart",
  speakingRate: 1.0,
});

// List bundled voices
const voices = await tts.listVoices();
console.log("Available voices:", voices);

// 1. Unary synthesis
const result = await tts.synthesize(
  "Hello from Kokoro in the browser via MediaPipe WebAssembly!",
  { voiceId: "af_heart" }
);
// result.samples: Float32Array, result.sampleRateHz: 24000, result.wordTimestamps

// 2. Streaming synthesis
const stream = await tts.synthesizeStream(
  "Streaming speech directly to the Web Audio API.",
  { voiceId: "bf_emma" }
);
for await (const chunk of stream) {
  // Schedule chunk.samples (Float32Array at chunk.sampleRateHz) on AudioContext
}

// 3. Incremental LLM token streaming (finishInput = false)
const llmStream = await tts.synthesizeStream("", { voiceId: "af_heart" }, /* finishInput= */ false);
await llmStream.pushText("Streaming tokens ");
await llmStream.pushText("from WebGPU LLM into Kokoro.");
await llmStream.finishInput();
for await (const chunk of llmStream) {
  // Play chunk.samples and highlight chunk.wordTimestamps
}

tts.close();
```

---

### 5. C++

```cpp
#include "mediapipe/tasks/cc/audio/text_to_speech/text_to_speech.h"

using ::mediapipe::tasks::audio::text_to_speech::SynthesisOptions;
using ::mediapipe::tasks::audio::text_to_speech::TextToSpeech;
using ::mediapipe::tasks::audio::text_to_speech::TextToSpeechOptions;
using ::mediapipe::tasks::core::BaseOptions;

TextToSpeechOptions options;
options.base_options.model_asset_path = "kokoro_int8_gpu_fp16.litertlm";
options.base_options.delegate = BaseOptions::Delegate::GPU;
options.default_voice_id = "af_heart";
options.speaking_rate = 1.0f;
options.cache_directory = "/tmp/mediapipe_tts_cache";

ABSL_ASSIGN_OR_RETURN(auto tts, TextToSpeech::Create(options));

// Unary synthesis:
SynthesisOptions synth_opts{.voice_id = "af_heart"};
ABSL_ASSIGN_OR_RETURN(auto result, tts->Synthesize("Hello from C++!", synth_opts));
// result.samples (std::vector<float>), result.sample_rate_hz (24000), result.word_timestamps

// Streaming synthesis:
ABSL_ASSIGN_OR_RETURN(auto stream, tts->SynthesizeStream("Streaming from C++.", synth_opts));
while (true) {
  auto chunk = stream->ReadNext();
  if (absl::IsOutOfRange(chunk.status())) break;
  ABSL_RETURN_IF_ERROR(chunk.status());
  // Consume chunk->samples and chunk->word_timestamps
}
```

---

## Configuration Options Reference

### `TextToSpeechOptions`

| Option (Kotlin / Swift / JS) | Option (Python / C++) | Description | Default |
|---|---|---|---|
| `baseOptions.modelAssetPath` | `base_options.model_asset_path` | Path (or URL on Web) to the `.litertlm` model bundle. | *Required* |
| `baseOptions.delegate` | `base_options.delegate` | Execution delegate (`GPU` recommended with `kokoro_int8_gpu_fp16.litertlm`; `CPU` with `kokoro_int8_fp16.litertlm`). | `CPU` |
| `defaultVoiceId` | `default_voice_id` | Default voice identifier (e.g. `"af_heart"`, `"bf_emma"`, `"ef_dora"`, `"zf_xiaobei"`, `"jf_alpha"`). | `"af_heart"` |
| `defaultLocale` | `default_locale` | Default BCP 47 language tag (e.g. `"en-US"`, `"es"`, `"zh"`, `"ja"`). Automatically inferred from voice ID when omitted. | Inferred from voice (`"en-US"` in Python) |
| `speakingRate` | `speaking_rate` | Speaking speed multiplier in `(0.0, 4.0]`. | `1.0` |
| `cacheDirectory` | `cache_directory` | Writable directory path to persist compiled GPU kernel and XNNPACK weight caches across runs. | `""` |

### `SynthesisOptions` (per-utterance override)

| Option (Kotlin / Swift / JS) | Option (Python / C++) | Description | Default |
|---|---|---|---|
| `voiceId` | `voice_id` | Overrides the voice profile for a single synthesis or streaming call. | Task default |
| `locale` | `locale` | Optional BCP 47 locale override (e.g. `"en-GB"`, `"es"`, `"fr"`, `"zh"`, `"ja"`). Inferred from `voiceId` if unset. | Inferred from `voiceId` |

---

## Usage with LiteRT-LM Omni (`omni_runner` & C++)

Both commands below produce a 24 kHz mono WAV file. Prerequisites (Bazel, Android NDK) are in the LiteRT-LM
[build guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

### 1. Download a model

```sh
pip install -U huggingface_hub
# Recommended: GPU version (any device with a LiteRT-supported GPU)
hf download litert-community/Kokoro-82M kokoro_int8_gpu_fp16.litertlm --local-dir .
# Or: CPU/GPU version (no GPU available)
hf download litert-community/Kokoro-82M kokoro_int8_fp16.litertlm --local-dir .
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
    --model_path=/path/to/kokoro_int8_fp16.litertlm \
    --language=en-US --voice=af_heart \
    --text="Hello from Kokoro, running fully on device with LiteRT-LM." \
    --output_path=/tmp/hello.wav
```

**On an Android phone (GPU, recommended):**

```sh
DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner prebuilt/android_arm64/*.so \
         /path/to/kokoro_int8_gpu_fp16.litertlm $DEVICE_DIR

adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner --model_name=kokoro \
    --backend=gpu --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/kokoro_int8_gpu_fp16.litertlm \
    --language=en-US --voice=af_heart \
    --text="Hello from Kokoro, running fully on device with LiteRT-LM." \
    --output_path=$DEVICE_DIR/hello.wav

adb pull $DEVICE_DIR/hello.wav .
```

> `--model_name=kokoro` tells the runner which model family the file is. Keep it: builds before 2026-10-17 only infer
> it when the file is literally named `kokoro.litertlm`. The first GPU run compiles kernels and is slower; they are cached in
> `--cache_dir` so later runs start fast. GPU runs use FP16 precision by default; add `--force_fp32_on_gpu` for FP32
> (about 2× slower).

Change `--language` and `--voice` to switch language (e.g. `--language=ja --voice=jf_alpha`). The full list of flags
and the C++ `OmniEngine` / `OmniSession` streaming API are documented in the
[LiteRT-LM omni README](https://github.com/google-ai-edge/LiteRT-LM/tree/main/omni).

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
    .cache_dir = "/data/local/tmp/omni/cache",
    .model_path = "/data/local/tmp/omni/kokoro_int8_gpu_fp16.litertlm",
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
// Pull OmniSession::AudioOutput chunks with session->ProcessNext() until
// OutOfRange or NotFound, then call session->Flush() (link //omni/text2speech:kokoro).
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

All clips were synthesized on a Samsung Galaxy S26 (`--backend=gpu`) and verified by transcribing them back with an ASR
model.

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
S26 Ultra, transcribed the audio with Whisper large-v3, and computed the word error rate (WER; character error rate,
CER, for Chinese and Japanese) against the input text. Lower is better.

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

Measured with head `omni_runner` on retail-equivalent phones, synthesizing an 18.3 s English paragraph (three sentences)
with `af_heart`, 4 CPU threads, mean of 3 runs after one warm-up run. GPU runs use the default FP16 precision.
**RTF** = synthesis time ÷ audio duration (lower is faster; 1.0 = real time). **Peak memory** is the process's peak
resident set (`VmHWM`), including model load.

| Device | File | Backend | Synthesis time | RTF | Peak memory |
|---|---|---|---:|---:|---:|
| Galaxy S26 (Snapdragon 8 Elite Gen 5, Adreno) | `kokoro_int8_gpu_fp16.litertlm` ⭐ | GPU | **2.2 s** | **0.12** | **340–360 MB** |
| Galaxy S26 | `kokoro_int8_fp16.litertlm` | GPU | 2.2 s | 0.12 | 690–800 MB |
| Galaxy S26 | `kokoro_int8_fp16.litertlm` | CPU | 4.8 s | 0.26 | 915–930 MB |
| Pixel 10 Pro (Tensor G5, PowerVR) | `kokoro_int8_gpu_fp16.litertlm` ⭐ | GPU | **9.1 s** | **0.50** | **400–505 MB** |
| Pixel 10 Pro | `kokoro_int8_fp16.litertlm` | GPU | 9.0 s | 0.49 | 830–940 MB |
| Pixel 10 Pro | `kokoro_int8_fp16.litertlm` | CPU | 17.0 s | 0.93 | 925–945 MB |

- Engine load time (model parse + GPU kernel cache load): 1.2–3 s on GPU, ~0.5 s on CPU. The very first GPU run on a
  device additionally compiles kernels (a few extra seconds); they are cached in `--cache_dir`.
- The GPU file keeps the vocoder's INT8 weights packed on the GPU, which is why it needs roughly **half the memory** of
  the CPU/GPU file (whose explicit dequantize ops materialize FP32 vocoder weights in host memory).
- CPU timings rise as the phone heats up (up to ~6 s on the S26 after several back-to-back runs).
- Long inputs are split into sentences / clauses and synthesized as a stream, so time-to-first-audio is roughly the
  synthesis time of the first sentence regardless of total text length.

### GPU FP16 vs FP32

Galaxy S26 (SM-S942U1), head `omni_runner`, 4 CPU threads, one short paragraph per language, mean of 5 warm runs.
FP16 is the default; FP32 is `--force_fp32_on_gpu`.

| Language (audio length) | `kokoro_int8_gpu_fp16.litertlm`, GPU FP16 | Same file, GPU FP32 | `kokoro_int8_fp16.litertlm`, CPU |
|---|---:|---:|---:|
| English (8.3 s) | **1.09 s** (RTF 0.13) | 1.99 s (RTF 0.24) | 2.39 s (RTF 0.29) |
| Spanish (4.3 s) | **0.55 s** (RTF 0.13) | 0.99 s (RTF 0.23) | 1.24 s (RTF 0.29) |
| Japanese (5.4 s) | **0.55 s** (RTF 0.10) | 1.00 s (RTF 0.18) | 1.24 s (RTF 0.23) |
| Mandarin (6.8 s) | **1.08 s** (RTF 0.16) | 2.00 s (RTF 0.29) | 2.42 s (RTF 0.35) |

## What's in the file

| Component | Details |
|---|---|
| Acoustic model | Kokoro text encoder, duration/prosody predictors and decoder, INT8 weights (dynamic quantization) with the six bidirectional LSTMs fused into a single custom op. Runs on CPU. |
| Vocoder | ISTFTNet generator, INT8 convolution weights. Runs on GPU (`kokoro_int8_gpu_fp16.litertlm`) or CPU/GPU (`kokoro_int8_fp16.litertlm`); the vocoder graph is FP16-safe. The final inverse STFT runs in C++. |
| Voices | 54 style embeddings (`af_heart`, `bf_emma`, `zf_xiaobei`, …) |
| Text front-end | Rule-based text normalization for every language (numbers, dates, currency, abbreviations), a 5 MB English lexicon with heteronym/part-of-speech handling, rule-based G2P for Spanish, French, Italian, Portuguese and Hindi, a Japanese morphological dictionary and a Chinese word segmenter + pinyin lexicon with tone sandhi, and a Traditional→Simplified Chinese table. |
| Output | 24 kHz, 16-bit mono PCM |

The models were exported from the original PyTorch checkpoint with [litert-torch](https://github.com/google-ai-edge/litert-torch)
and quantized with the AI Edge Quantizer. No retraining or fine-tuning was done.

## Limitations

- The acoustic model takes about 124 phonemes (~10 s of speech) per run. Longer sentences are split automatically, at punctuation when possible and otherwise between words, which can add a short pause mid-sentence.
- Mixed-script input works (e.g. English words inside Chinese text), but all-caps acronyms inside Chinese text are read as words rather than spelled out.
- Very large numbers (≥ 1,000,000) in Portuguese and Hindi and some 4-digit currency amounts in English may be read as years rather than cardinals.
- The GPU version needs a GPU supported by the LiteRT GPU delegate (OpenCL, WebGPU or Metal); it was tested on Qualcomm Adreno and Pixel (PowerVR). Use `kokoro_int8_fp16.litertlm` elsewhere.
- GPU FP16 output was verified on Qualcomm Adreno (Galaxy S26). If a GPU produces noisy audio in FP16, pass `--force_fp32_on_gpu`.

## License & acknowledgements

- Weights: Apache-2.0, inherited from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (trained by `@rzvzn`; architecture by [StyleTTS 2](https://arxiv.org/abs/2306.07691) / [ISTFTNet](https://arxiv.org/abs/2203.02395), Li et al.).
- Bundled lexicons: English from [misaki](https://github.com/hexgrad/misaki) (Apache-2.0); Japanese from [unidic-lite](https://github.com/polm/unidic-lite) (BSD); Chinese from [jieba](https://github.com/fxsjy/jieba) / [pypinyin](https://github.com/mozillazg/python-pinyin) (MIT).
- Runtime: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/audio/text_to_speech) and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0).
- Training data, as disclosed by the upstream model: a few hundred hours of permissive / non-copyrighted audio with IPA labels. This conversion adds no training data and no personal information.
