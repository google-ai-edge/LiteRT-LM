# LiteRT-Omni

**LiteRT-Omni** is the unified multimodal streaming runtime in LiteRT-LM for
on-device generative and perception pipelines across audio, text, and vision
modalities—including **Automatic Speech Recognition (ASR)**, **Text-to-Speech
(TTS)**, and **Text-to-Image generation**.

LiteRT-Omni provides a single, modality-agnostic C++ and Kotlin/JNI surface
(`OmniEngine` and `OmniSession`) backed by an asynchronous, multi-staged
streaming execution pipeline (`MultiStagedSession`).

--------------------------------------------------------------------------------

## Core Concepts

*   **`OmniEngine`** (`omni_engine.h`): The heavyweight, thread-safe engine that
    owns compiled LiteRT models, tokenizers, hardware accelerator environments
    (CPU, GPU, NPU), and worker thread pools for a loaded model. Creating an
    `OmniEngine` performs one-time model resolution, optional downloading, and
    accelerator compilation.
*   **`OmniSession`** (`omni_session.h`): A lightweight, stateful streaming
    session created by `OmniEngine::CreateSession`. Each session owns its own
    per-stream pipeline stages and recurrent/decoder states while sharing the
    underlying `OmniEngine` weights.
    *   **Inputs (`OmniSession::Input`)**: Streamed into the session via an
        `OmniSession::InputSource` (such as `PushInputSource`), supporting
        `TextInput`, `AudioInput`, `AudioInputMetadata`,
        `ImageGenInputMetadata`, and `EndOfInput`.
    *   **Outputs (`OmniSession::Output`)**: Pulled synchronously via
        `ProcessNext()` / `Flush()` or streamed asynchronously via
        `ProcessAsync(callback)`, emitting `TextOutput`, `AudioOutput`,
        `ImageOutput`, or `EndOfOutput`.
*   **`MultiStagedSession` & `Stage<T>`** (`multi_staged_session.h`,
    `base/stage.h`): Composable pipeline execution engine that connects stages
    (e.g., `AudioSource` → `AudioPreprocessor` → `SpeechRecognizer` →
    `Detokenizer` → `TextMerger` for ASR; `StreamTextSource` → `TextFrontend` →
    `AcousticModel` → `Vocoder` for TTS; `PromptSource` → `TextEncoderStage` →
    `Denoiser` → `ImageDecoder` for Text-to-Image).

--------------------------------------------------------------------------------

## Three-Layer Configuration & Metadata Architecture

LiteRT-Omni separates model metadata, session configuration, and runtime
environment settings into three distinct layers:

1.  **Layer 1 — Registration Defaults & Engine Settings** (per-model
    registration & per-`OmniEngine`):
    *   **Types**: `OmniEngine::ModelSpec`, `OmniEngine::EngineSettings`
    *   **Purpose**: Default download URLs, fallback model metadata, default
        session config, hardware backend (`kCpu`, `kGpu`, `kNpu`), cache
        directory, thread count, and local file paths.
2.  **Layer 2 — `.litertlm` Package Metadata** (embedded in the `.litertlm`
    model file):
    *   **Types**:
        *   `litert.lm.proto.AsrMetadata` (`asr_metadata.proto`)
        *   `litert.lm.proto.TtsMetadata` (`tts_metadata.proto`)
        *   `litert.lm.proto.Text2ImageMetadata` (`text2image_metadata.proto`)
    *   **Purpose**: Purely model-intrinsic architecture parameters packaged
        with the model weights (e.g., audio sample rate, input window duration,
        mel spectrogram parameters, special token IDs, supported languages).
        Contains **no** session-specific or runtime-specific fields.
3.  **Layer 3 — Session Config** (per-`OmniSession` via `CreateSession`):
    *   **Types**:
        *   `litert.lm.proto.AsrSessionConfig` (`asr_session_config.proto`)
        *   `litert.lm.proto.TtsSessionConfig` (`tts_session_config.proto`)
        *   `litert.lm.proto.Text2ImageSessionConfig`
            (`text2image_session_config.proto`)
    *   **Purpose**: Per-stream session parameters that vary across sessions
        without reloading the engine (e.g., chunk overlap ratio, text merger
        strategy, synthesis language/voice, text chunking, image
        dimensions/steps/seed).

### Override Precedence

1.  **Engine Creation (`OmniEngine::Create(model_id, settings)`)**:
    *   Starts from the `ModelSpec` registered for `model_id`
        (`default_engine_settings`, `default_session_config`, and
        `default_metadata`).
    *   Any non-empty / positive field in the caller's `EngineSettings`
        overrides `ModelSpec::default_engine_settings` (for example, overriding
        `model_remote_url`, `model_path`, `cache_dir`, `num_threads`, or
        `backend`).
    *   When loading a `.litertlm` package, embedded `.litertlm` metadata
        (**Layer 2**) is merged on top of `ModelSpec::default_metadata` (**Layer
        1**).
2.  **Session Creation (`engine->CreateSession(input_source,
    session_config)`)**:
    *   Starts from `ModelSpec::default_session_config` (**Layer 1**).
    *   If a non-null `session_config` proto (**Layer 3**) is provided, its
        populated fields are merged onto the default session config via Protobuf
        `MergeFrom`.

--------------------------------------------------------------------------------

## Session Factory Registration (`OmniEngine::RegisterSessionFactory`)

`OmniEngine` has zero compile-time dependencies on concrete modality engines.
Instead, each modality registers its `SessionFactoryCreator` and supported
models with `OmniEngine::RegisterSessionFactory`.

### Two-Hop Registry Lookup

Internally, `OmniEngine` maintains a two-hop registry:

```text
model_id
  --> (session_config_proto_type_id, metadata_proto_type_id) + ModelSpec defaults
  --> SessionFactoryCreator
```

This allows multiple models to share a single `SessionFactoryCreator` while each
`ModelSpec` supplies its own per-model `default_engine_settings`,
`default_session_config`, and `default_metadata`. Conversely, two factories can
share the same `.litertlm` metadata proto type while using different session
config proto types (or vice versa).

### Protobuf Type ID Derivation Rules

When calling `OmniEngine::RegisterSessionFactory(factory,
session_config_proto_type_id, metadata_proto_type_id, models)`:

1.  **`session_config_proto_type_id` (Primary Key)**:
    *   **Short key `"Xxx"`** (no `.`): Resolves to
        `"litert.lm.proto.XxxSessionConfig"`.
        *   `"Asr"` → `"litert.lm.proto.AsrSessionConfig"`
        *   `"Tts"` → `"litert.lm.proto.TtsSessionConfig"`
        *   `"Text2Image"` → `"litert.lm.proto.Text2ImageSessionConfig"`
    *   **Full proto type name `"d.e.f"`** (contains `.`): Used as-is
        (`"d.e.f"`).
2.  **`metadata_proto_type_id` (Optional for Short Keys)**:
    *   **Empty `""` (when `session_config_proto_type_id` is a short key
        `"Xxx"`)**: Derives `"litert.lm.proto.XxxMetadata"`.
        *   `("Asr", "")` → `("litert.lm.proto.AsrSessionConfig",
            "litert.lm.proto.AsrMetadata")`
        *   `("Text2Image", "")` → `("litert.lm.proto.Text2ImageSessionConfig",
            "litert.lm.proto.Text2ImageMetadata")`
    *   **Short key `"Yyy"`** (no `.`): Resolves to
        `"litert.lm.proto.YyyMetadata"`.
    *   **Full proto type name `"a.b.c"`** (contains `.`): Used as-is
        (`"a.b.c"`).
    *   **Empty `""` when `session_config_proto_type_id` is a full type name
        `"d.e.f"`**: Rejected with `absl::InvalidArgumentError` because a
        metadata type ID cannot be automatically derived from an arbitrary full
        proto type name.

### Type Validation

*   At **registration time**, if `ModelSpec::default_session_config` is
    non-null, `RegisterSessionFactory` verifies that
    `default_session_config->GetTypeName()` matches the resolved
    `session_config_proto_type_id`. Similarly, if `ModelSpec::default_metadata`
    is non-null, its `GetTypeName()` must match the resolved
    `metadata_proto_type_id`.
*   At **session creation time**, if `session_config` is passed to
    `OmniEngine::CreateSession(input_source, session_config)`, `OmniEngine`
    verifies that `session_config->GetTypeName() ==
    engine->session_config_proto_type_id()` before forwarding it to the
    underlying `OmniSessionFactory`, returning `absl::InvalidArgumentError` on
    mismatch.

--------------------------------------------------------------------------------

## Examples

### 1. Registering a Model Session Factory

To comply with the Google C++ Style Guide (which forbids non-POD static/global
variables with dynamic destructors), construct `OmniEngine::ModelSpec` entries
as local variables inside a registration function using helper functions that
return `std::unique_ptr<SessionConfig>` and `std::unique_ptr<Metadata>`:

```cpp
#include <array>
#include <memory>
#include <string>
#include "third_party/absl/log/absl_check.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "omni/asr/asr_session.h"
#include "omni/omni_engine.h"
#include "runtime/proto/asr_metadata.proto.h"
#include "runtime/proto/asr_session_config.proto.h"

namespace litert::omni::asr {
namespace {

std::unique_ptr<lm::proto::AsrSessionConfig> DefaultParakeetTdtSessionConfig() {
  auto config = std::make_unique<lm::proto::AsrSessionConfig>();
  config->set_overlap_ratio(0.4f);
  config->set_text_merger_type(
      lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);
  return config;
}

std::unique_ptr<lm::proto::AsrMetadata> DefaultParakeetTdtMetadata() {
  auto meta = std::make_unique<lm::proto::AsrMetadata>();
  meta->mutable_asr_model_type()->mutable_parakeet();
  meta->set_sample_rate_hz(16000);
  meta->set_input_milliseconds(5000);
  meta->set_decode_start_token_id(8192);
  auto* mel = meta->mutable_log_mel_spectrogram_config();
  mel->set_n_fft(512);
  mel->set_n_mels(128);
  mel->set_n_frames(500);
  mel->set_preemphasis(0.97f);
  constexpr absl::string_view kStateBufferPatterns[] = {
      "encode_output_0_output",
      "decode_args_0",
      "decode_1_args_0",
      "decode_1_args_2",
      "decode_1_args_3",
      "decode_1_output_1_output",
      "decode_1_output_2_output",
      "decode_output_1_output",
      "decode_output_2_output",
  };
  for (absl::string_view pattern : kStateBufferPatterns) {
    meta->add_state_buffer_name_patterns(std::string(pattern));
  }
  return meta;
}

void RegisterParakeetModels() {
  OmniEngine::ModelSpec models[] = {
      {
          .model_id = "parakeet-tdt-0.6b-v3",
          .default_engine_settings =
              {
                  .model_remote_url =
                      "https://huggingface.co/litert-community/"
                      "parakeet-tdt-0.6b-v3/resolve/main/"
                      "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
                  .npu_model_remote_url_pattern =
                      "https://huggingface.co/litert-community/"
                      "parakeet-tdt-0.6b-v3/resolve/main/"
                      "parakeet_tdt_0.6b_v3_5s_f32_stateful_<target>.litertlm",
              },
          .default_session_config = DefaultParakeetTdtSessionConfig(),
          .default_metadata = DefaultParakeetTdtMetadata(),
      },
  };
  ABSL_CHECK_OK(OmniEngine::RegisterSessionFactory(
      &AsrSessionFactory::CreateFactory,
      /*session_config_proto_type_id=*/"Asr",
      /*metadata_proto_type_id=*/"", absl::MakeSpan(models)));
}

[[maybe_unused]] const bool kParakeetRegistered =
    (RegisterParakeetModels(), true);

}  // namespace
}  // namespace litert::omni::asr
```

> **Toolchain & Linker Portability (GCC, Clang, Xcode, MSVC)**:
>
> *   **C++20 Designated Initializers**: Designated initializers require C++20
>     (`-std=c++20` on GCC/Clang/Xcode, `/std:c++20` on MSVC) and must appear in
>     exact struct declaration order (`model_id`, `default_engine_settings`,
>     `default_session_config`, `default_metadata`).
> *   **Static Archive Dead-Stripping**: If a registration translation unit has
>     no symbols referenced by the main binary (or if `kParakeetRegistered` is
>     the only symbol in its `.cc` file inside a `.a` / `.lib` static archive),
>     linkers on all four toolchains (GNU `ld`/`lld` on GCC/Clang, Apple
>     `ld64`/`ld_prime` on Xcode, and `link.exe` on MSVC) will skip extracting
>     that object file from the archive. To ensure static self-registration runs
>     across all platforms:
>     *   In **Bazel**, set `alwayslink = 1` on the `cc_library` target
>         (translates to `-Wl,--whole-archive` on GCC/Clang, `-Wl,-force_load`
>         on Xcode, and `/WHOLEARCHIVE` on MSVC), **or** place the registration
>         in a `.cc` file whose symbols (such as `AsrSessionFactory`) are
>         already referenced by the binary.
>     *   In **CMake** (or when whole-archive linking is not used), link with
>         `$<LINK_LIBRARY:WHOLE_ARCHIVE,target>` or expose and call
>         `RegisterParakeetModels()` explicitly during initialization.

### 2. Creating an `OmniEngine` and `OmniSession` with Per-Session Overrides

```cpp
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/proto/asr_session_config.proto.h"

// 1. Create the engine (overrides default backend and local model path, while
//    inheriting any unset EngineSettings fields from the registered ModelSpec).
OmniEngine::EngineSettings engine_settings{
    .backend = OmniEngine::EngineSettings::Backend::kGpu,
    .cache_dir = "/data/local/tmp/omni_cache",
    .model_path = "/data/local/tmp/parakeet_tdt_0.6b_v3.litertlm",
};
ABSL_ASSIGN_OR_RETURN(
    auto engine, OmniEngine::Create("parakeet-tdt-0.6b-v3", engine_settings));

// 2. Optionally customize per-session parameters via AsrSessionConfig.
litert::lm::proto::AsrSessionConfig session_config;
session_config.set_overlap_ratio(0.25f);
session_config.set_text_merger_type(
    litert::lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN);

auto input_source = std::make_unique<litert::omni::PushInputSource>();
auto* raw_input_source = input_source.get();
ABSL_ASSIGN_OR_RETURN(
    auto session,
    engine->CreateSession(std::move(input_source), &session_config));

// 3. Push audio samples and stream transcription outputs.
ABSL_RETURN_IF_ERROR(raw_input_source->PushInput(
    litert::omni::OmniSession::AudioInput{.pcm_samples = pcm_chunk}));
raw_input_source->Finish();

ABSL_ASSIGN_OR_RETURN(auto output, session->Flush());
```
