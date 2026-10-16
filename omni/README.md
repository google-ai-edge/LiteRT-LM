# LiteRT-LM Streaming Multimodality Model Examples

This folder contains an **experimental** effort (aka **"LiteRT Omni"**) inside
LiteRT-LM to provide **rich model examples**, enabling LiteRT to run complex
**streaming multimodality models** from a single `.litertlm` model file. It
expands LiteRT-LM to run diverse models—such as **ASR**, **TTS**, **Diffusion**,
and more.

It exposes a single, modality-agnostic C++ interface (`OmniEngine` and
`OmniSession`) and a single command-line runner (`//omni:omni_runner`) that runs
every supported model across **CPU**, **GPU**, and **NPU** backends. It can
accept **Any input** (audio, text, and vision) and produce **Any output**.

Today it supports **Automatic Speech Recognition (ASR)**, **Text-to-Speech
(TTS)**, and **Text-to-Image** generation, and more models are coming.
Pre-converted on-device models are available in the Hugging Face
[`litert-community`](https://huggingface.co/litert-community).

## Supported Models

| Modality | Model Name (`--model_name`) | Hugging Face (`litert-community`) | Bazel Target | Input | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text** | `parakeet-tdt-0.6b-v3` | [parakeet-tdt-0.6b-v3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3) | `//omni/speech2text:parakeet_tdt` | `AudioInputMetadata` + `AudioInput` | `TextOutput` |
| **Speech-to-Text** | `parakeet-ctc-0.6b` | [parakeet-ctc-0.6b](https://huggingface.co/litert-community/parakeet-ctc-0.6b) | `//omni/speech2text:parakeet_ctc` | `AudioInputMetadata` + `AudioInput` | `TextOutput` |
| **Speech-to-Text** | `whisper-tiny` | [whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | `//omni/speech2text:whisper` | `AudioInputMetadata` + `AudioInput` | `TextOutput` |
| **Speech-to-Text** | `qwen3-asr-0.6b` | [Qwen3-ASR-0.6B](https://huggingface.co/litert-community/Qwen3-ASR-0.6B) | `//omni/speech2text:qwen3_asr` | `AudioInputMetadata` + `AudioInput` | `TextOutput` |
| **Speech-to-Text** | `moonshine-tiny` | [moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny) | `//omni/speech2text:moonshine` | `AudioInputMetadata` + `AudioInput` | `TextOutput` |
| **Text-to-Speech** | `kokoro` (`kokoro-82m`) | [Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M) | `//omni/text2speech:kokoro` | `TextInput` | `AudioOutput` |
| **Text-to-Speech** | `qwen3-tts` | [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) | `//omni/text2speech:qwen3_tts` | `TextInput` | `AudioOutput` |
| **Text-to-Image** | `bonsai-flux2` | [Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) | `//omni/text2image:bonsai_flux2` | `ImageGenInputMetadata` *(optional)* + `TextInput` | `ImageOutput` |
| **Text-to-Image** | `flux2-klein` (`flux.2-klein-4b`) | [FLUX.2-klein-4B-LiteRT](https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT) | `//omni/text2image:flux2_klein` | `ImageGenInputMetadata` *(optional)* + `TextInput` | `ImageOutput` |

--------------------------------------------------------------------------------

## 1. Quick Start: Build and Run `omni_runner`

Run the commands below from the root of the LiteRT-LM GitHub repository (see
[Build and Run](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md)
for prerequisites: Bazel from `.bazelversion`, and Android NDK r28b+ for Android
builds).

### Build once (Linux / macOS)

```sh
bazel build //omni:omni_runner
```

### Run ASR, TTS, or Text-to-Image with the same binary

`omni_runner` selects the task automatically from your input flags
(`--audio_path` for ASR, `--text` for TTS or Text-to-Image) and infers
`--model_name` from `--model_path` when omitted:

```sh
# 1. Automatic Speech Recognition (ASR): WAV -> text transcript
bazel-bin/omni/omni_runner \
    --model_path=/path/to/parakeet-tdt-0.6b-v3.litertlm \
    --audio_path=runtime/components/testdata/audio_sample.wav

# 2. Text-to-Speech (TTS): text -> 24 kHz WAV file
bazel-bin/omni/omni_runner \
    --model_path=/path/to/kokoro.litertlm \
    --language=en-US \
    --text="Hello from LiteRT-Omni, running with a unified multimodal API." \
    --output_path=/tmp/hello.wav

# 3. Text-to-Image: text prompt -> PNG image
bazel-bin/omni/omni_runner \
    --model_path=/path/to/bonsai-flux2.litertlm \
    --text="A red fox sitting in fresh snow at sunrise" \
    --img_size=256 --steps=4 \
    --output_path=/tmp/fox.png
```

### Run on Android (CPU & GPU)

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
bazel build --config=android_arm64 //omni:omni_runner

DEVICE_DIR=/data/local/tmp/omni
adb shell mkdir -p $DEVICE_DIR
adb push bazel-bin/omni/omni_runner \
    prebuilt/android_arm64/*.so \
    /path/to/parakeet-tdt-0.6b-v3.litertlm \
    /path/to/kokoro.litertlm \
    runtime/components/testdata/audio_sample.wav \
    $DEVICE_DIR

# ASR on Android GPU (compiled GPU programs & weights are cached in --cache_dir)
adb shell LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/omni_runner \
    --backend=gpu \
    --cache_dir=$DEVICE_DIR/cache \
    --model_path=$DEVICE_DIR/parakeet-tdt-0.6b-v3.litertlm \
    --audio_path=$DEVICE_DIR/audio_sample.wav

# TTS on Android CPU
adb shell $DEVICE_DIR/omni_runner \
    --model_path=$DEVICE_DIR/kokoro.litertlm \
    --language=en-US \
    --text="Hello from LiteRT-Omni on Android." \
    --output_path=$DEVICE_DIR/hello.wav
```

### Common `omni_runner` Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model_path` | `""` | Path to a `.litertlm` package or model directory. |
| `--model_name` | *(inferred)* | Model name (e.g., `parakeet-tdt-0.6b-v3`, `kokoro`, `bonsai-flux2`). |
| `--backend` | `"cpu"` | Hardware accelerator: `cpu`, `gpu`, or `npu`. |
| `--cache_dir` | `"/tmp/omni_cache"` | Directory for downloaded models and serialized GPU/XNNPack caches. |
| `--audio_path` | `""` | Input WAV file for ASR transcription. |
| `--text` | `""` | Input text for TTS synthesis or Text-to-Image generation. |
| `--output_path` | `""` | Output file path (`.wav` for TTS, `.png` for Text-to-Image, or optional `.txt` for ASR). |
| `--language` / `--voice` | `"en-US"` / `""` | TTS language code (BCP-47) and optional voice ID (e.g., `af_heart`). |
| `--img_size` / `--steps` / `--seed` | `0` / `4` / `42` | Text-to-Image resolution, denoising steps, and random seed. |

--------------------------------------------------------------------------------

## 2. C++ API Quick Start

Every modality in LiteRT-Omni shares the exact same workflow:

1.  Create an `OmniEngine` once with your model name/path and `Settings`.
2.  Create a lightweight `OmniSession` connected to a `PushInputSource`.
3.  Push `OmniSession::Input` items (`AudioInput` or `TextInput`) and pull
    `OmniSession::Output` items (`TextOutput`, `AudioOutput`, or `ImageOutput`)
    via `ProcessNext()` and `Flush()`.

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"

using ::litert::omni::OmniEngine;
using ::litert::omni::OmniSession;
using ::litert::omni::PushInputSource;

// 1. Load the engine once (model_name can be "parakeet-tdt-0.6b-v3",
//    "kokoro", "bonsai-flux2", etc.).
OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kGpu,
    .cache_dir = "/tmp/omni_cache",
    .model_path = "/path/to/model.litertlm",
};
ABSL_ASSIGN_OR_RETURN(auto engine, OmniEngine::Create("kokoro", settings));

// 2. Create a streaming session with a PushInputSource.
auto input_source = std::make_unique<PushInputSource>();
PushInputSource* raw_input = input_source.get();
ABSL_ASSIGN_OR_RETURN(auto session,
                      engine->CreateSession(std::move(input_source)));

// 3. Push inputs (TextInput for TTS / Text2Image, or AudioInput for ASR).
ABSL_RETURN_IF_ERROR(raw_input->PushInput(
    OmniSession::TextInput{.text = "Hello from LiteRT-Omni."}));
raw_input->Finish();

// 4. Pull outputs until the stream ends, then flush any trailing output.
while (true) {
  auto next = session->ProcessNext();
  if (!next.ok()) {
    if (absl::IsOutOfRange(next.status()) || absl::IsNotFound(next.status())) {
      ABSL_ASSIGN_OR_RETURN(auto tail, session->Flush());
      // Handle `tail` (TextOutput, AudioOutput, or ImageOutput)...
      break;
    }
    return next.status();
  }
  // Handle `*next` (TextOutput, AudioOutput, or ImageOutput)...
}
```

### Selective Linking to Minimize Binary Size

`//omni:omni_runner` links all supported ASR, TTS, and Text-to-Image model
targets so a single binary can run any model. In your own application, you can
depend on `//omni:omni_engine` and link **only** the specific model target(s)
you need from the [Supported Models](#supported-models) table above (e.g., only
`//omni/speech2text:parakeet_tdt`), avoiding pulling unused model
implementations or dependencies into your final binary or shared library:

```python
cc_binary(
    name = "my_asr_app",
    srcs = ["my_asr_app.cc"],
    deps = [
        "//omni:omni_engine",
        "//omni:omni_session",
        "//omni/speech2text:parakeet_tdt",  # Link only Parakeet-TDT
    ],
)
```

--------------------------------------------------------------------------------

## 3. Core Architecture

*   **`OmniEngine`** (`omni_engine.h`): Heavyweight, thread-safe engine that
    owns compiled LiteRT models, tokenizers, hardware accelerator environments
    (CPU, GPU, NPU), and worker thread pools. Creating an `OmniEngine` performs
    one-time model resolution and accelerator compilation.
*   **`OmniSession`** (`omni_session.h`): Lightweight, stateful streaming
    session created by `OmniEngine::CreateSession`. Multiple sessions can be
    created from a single `OmniEngine` with per-session configuration overrides:
    *   **Synchronous pulling**: Call `ProcessNext()` in a loop until
        `absl::OutOfRangeError` / `absl::NotFoundError`, then call `Flush()`.
    *   **Asynchronous streaming**: Call `ProcessAsync(callback)` to run stages
        concurrently on the engine's thread pool and receive `Output` chunks via
        callback.
*   **`MultiStagedSession` & `Stage<T>`** (`multi_staged_session.h`,
    `base/stage.h`): Composable pipeline engine that connects typed stages:
    *   **ASR**: `AudioSource` → `AudioPreprocessor` → `SpeechRecognizer` →
        `Detokenizer` → `TextMerger`
    *   **TTS**: `StreamTextSource` → `TextFrontend` → `AcousticModel` →
        `Vocoder`
    *   **Text-to-Image**: `PromptSource` → `TextEncoderStage` → `Denoiser` →
        `ImageDecoder`

--------------------------------------------------------------------------------

## 4. Three-Layer Configuration & Metadata

LiteRT-Omni separates runtime settings, model-intrinsic metadata, and per-stream
session configuration into three layers:

1.  **Layer 1 — Engine Settings & Factory Defaults** (`OmniEngine::Settings`,
    `OmniSessionFactory`):
    *   Hardware backend (`kCpu`, `kGpu`, `kNpu`), thread count, `cache_dir`,
        `model_path`, and fallback model/session defaults provided by the
        model's `SessionFactory`.
2.  **Layer 2 — `.litertlm` Package Metadata** (embedded in `.litertlm` files):
    *   `Speech2TextMetadata`, `Text2SpeechMetadata`, `Text2ImageMetadata`:
        Model-intrinsic parameters packaged with the weights (e.g., sample rate,
        mel spectrogram config, special token IDs).
3.  **Layer 3 — Session Config** (passed to `OmniEngine::Create`):
    *   `Speech2TextSessionConfig` (overlap ratio, text merger type),
        `Text2SpeechSessionConfig` (language, voice), `Text2ImageSessionConfig`
        (width, height, inference steps, seed).

```cpp
// Configure TTS voice and language when creating the engine:
litert::lm::proto::Text2SpeechSessionConfig tts_config;
tts_config.set_language("en-US");
tts_config.set_voice("af_heart");

ABSL_ASSIGN_OR_RETURN(
    auto engine,
    OmniEngine::Create("kokoro", settings, &tts_config));
```

--------------------------------------------------------------------------------

## 5. Extending LiteRT-Omni: Registering Custom Models

`OmniRegistry` (`omni_registry.h`) decouples `OmniEngine` from concrete model
implementations. To add a new model, implement an `OmniSessionFactory` and
register a `RegistrySpec` with `OmniRegistry::Get().RegisterSessionFactory`:

*   **`model_names`**: Canonical model name followed by any aliases.
*   **`get_session_factory`**: Callback that compiles shared model resources and
    returns an `OmniSessionFactory`.
*   **`session_config_type_id`**: Short key (`"Speech2Text"`, `"Text2Speech"`,
    `"Text2Image"`, which expands to `"litert.lm.proto.<Key>SessionConfig"` and
    `"litert.lm.proto.<Key>Metadata"`) or full proto type name.
*   **`model_remote_url`** *(optional)*: Default URL for automatic download when
    no local `model_path` is supplied.

```cpp
namespace litert::omni::asr {
namespace {

constexpr absl::string_view kModelNames[] = {
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt",
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetParakeetTdtSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& options,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  // Compile shared models/tokenizers once and return an OmniSessionFactory.
}

[[maybe_unused]] const bool kParakeetTdtRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetParakeetTdtSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/"
          "resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::asr
```

> **Note on Static Registration**: Set `alwayslink = 1` on the `cc_library`
> target in `BUILD` so the linker retains the self-registration translation unit
> when linking binaries like `//omni:omni_runner`.

### Running Unit Tests

```sh
bazel test //omni/...
```
