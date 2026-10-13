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

1.  **Layer 1 — SessionFactory Defaults & Engine Settings** (per-model
    `OmniSessionFactory` & per-`OmniEngine`):
    *   **Types**: `OmniRegistry::RegistrySpec`, `OmniEngine::Settings`,
        `OmniSessionFactory`
    *   **Purpose**: Model registration (`model_names`, `get_session_factory`,
        `session_config_type_id`, `metadata_type_id`, `model_remote_url`),
        fallback model metadata and session config defaults filled by the
        model's `SessionFactory` if not set, hardware backend (`kCpu`, `kGpu`,
        `kNpu`), cache directory, thread count, and local file paths.
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

1.  **Session Factory Creation (`OmniRegistry::CreateSessionFactory(model_name,
    settings, session_config, metadata)`)**:
    *   When loading a `.litertlm` package, embedded `.litertlm` metadata
        (**Layer 2**) is merged with any caller-provided `metadata` override.
    *   The model's `SessionFactory` fills in any unset model metadata or
        session config fields with its built-in defaults (**Layer 1**).
2.  **Session Creation (`engine->CreateSession(input_source,
    session_config)`)**:
    *   Starts from the resolved session config in the `SessionFactory` (**Layer
        1**).
    *   If a non-null `session_config` proto (**Layer 3**) is provided, its
        populated fields override the default session config.

--------------------------------------------------------------------------------

## Session Factory Registration (`OmniRegistry::RegisterSessionFactory`)

`OmniRegistry` (`omni_registry.h`) decouples `OmniEngine` from concrete modality
model implementations. Each model registers a `RegistrySpec` containing its
`model_names` (first element is the canonical model name; remaining elements are
aliases), `get_session_factory` callback, `session_config_type_id`, optional
`metadata_type_id`, and optional `model_remote_url` with
`OmniRegistry::Get().RegisterSessionFactory`.

### Protobuf Type ID Derivation Rules

When calling `OmniRegistry::Get().RegisterSessionFactory(spec)`:

1.  **`spec.session_config_type_id` (Primary Key)**:
    *   **Short key `"Xxx"`** (no `.`): Resolves to
        `"litert.lm.proto.XxxSessionConfig"`.
        *   `"Asr"` → `"litert.lm.proto.AsrSessionConfig"`
        *   `"Tts"` → `"litert.lm.proto.TtsSessionConfig"`
        *   `"Text2Image"` → `"litert.lm.proto.Text2ImageSessionConfig"`
    *   **Full proto type name `"d.e.f"`** (contains `.`): Used as-is
        (`"d.e.f"`).
2.  **`spec.metadata_type_id` (Optional for Short Keys)**:
    *   **Omitted / empty `""` (when `session_config_type_id` is a short key
        `"Xxx"`)**: Derives `"litert.lm.proto.XxxMetadata"`.
        *   `("Asr", std::nullopt)` → `("litert.lm.proto.AsrSessionConfig",
            "litert.lm.proto.AsrMetadata")`
        *   `("Text2Image", std::nullopt)` →
            `("litert.lm.proto.Text2ImageSessionConfig",
            "litert.lm.proto.Text2ImageMetadata")`
    *   **Short key `"Yyy"`** (no `.`): Resolves to
        `"litert.lm.proto.YyyMetadata"`.
    *   **Full proto type name `"a.b.c"`** (contains `.`): Used as-is
        (`"a.b.c"`).
    *   **Omitted / empty `""` when `session_config_type_id` is a full type name
        `"d.e.f"`**: Rejected with `absl::InvalidArgumentError` because a
        metadata type ID cannot be automatically derived from an arbitrary full
        proto type name.

### Type Validation

*   At **session factory creation time**, if `session_config` or `metadata` is
    passed to `OmniRegistry::CreateSessionFactory`, `OmniRegistry` verifies that
    `session_config->GetTypeName()` and `metadata->GetTypeName()` match the
    registered proto type IDs, returning `absl::InvalidArgumentError` on
    mismatch.
*   At **session creation time**, if `session_config` is passed to
    `OmniEngine::CreateSession(input_source, session_config)`, `OmniEngine`
    forwards it to the underlying `OmniSessionFactory`.

--------------------------------------------------------------------------------

## Examples

### 1. Registering a Model Session Factory

To comply with the Google C++ Style Guide (which forbids non-POD static/global
variables with dynamic destructors), register each model using an immediately
invoked lambda that passes a `RegistrySpec` to
`OmniRegistry::Get().RegisterSessionFactory`:

```cpp
namespace litert::omni::asr {
namespace {

constexpr absl::string_view kModelNames[] = {
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v3-int8",
    "parakeet-tdt",
    "parakeet-tdt-int8",
};

// Resolves configs and instantiates `OmniSession` on demand.
class ParakeetTdtSessionFactory : public OmniSessionFactory {
 public:
  ParakeetTdtSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata,
      ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable lm_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model)
      : session_config_(ResolveSessionConfig(session_config)),
        metadata_(ResolveMetadata(metadata)),
        environment_(std::move(env)),
        lm_resources_(std::move(lm_resources)),
        tokenizer_(std::move(tokenizer)),
        compiled_model_(std::move(compiled_model)) {}

 protected:
  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    // Instantiate pipeline stages and return an OmniSession.
  }

 private:
  static lm::proto::AsrMetadata ResolveMetadata(
      const MessageLite* absl_nullable metadata) {
    lm::proto::AsrMetadata resolved;
    if (metadata != nullptr) {
      resolved.CheckTypeAndMergeFrom(*metadata);
    }
    // Set default values if not set by user.
    return resolved;
  }

  static lm::proto::AsrSessionConfig ResolveSessionConfig(
      const MessageLite* absl_nullable session_config) {
    lm::proto::AsrSessionConfig resolved;
    if (session_config != nullptr) {
      resolved.CheckTypeAndMergeFrom(*session_config);
    }
    // Set default values if not set by user.
    return resolved;
  }

  lm::proto::AsrSessionConfig session_config_;
  lm::proto::AsrMetadata metadata_;
  ::litert::Environment environment_;
  std::shared_ptr<lm::ModelResources> absl_nullable lm_resources_;
  std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer_;
  std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model_;
};

// Instantiates common objects shared among sessions (returning on error), then
// returns the session factory.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetParakeetTdtSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& options,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "lm_resources must not be null (model_path: ", settings.model_path,
        ")."));
  }
  ABSL_ASSIGN_OR_RETURN(
      auto compiled_model,
      CreateCompiledModel(settings, env, options, lm_resources.get(),
                          lm::proto::AsrMetadata::TF_LITE_ENCODER_DECODER));
  ABSL_ASSIGN_OR_RETURN(
      auto tokenizer,
      CreateTokenizer(lm_resources.get(), settings.tokenizer_path));
  return std::make_unique<ParakeetTdtSessionFactory>(
      settings, session_config, metadata, std::move(env),
      std::move(lm_resources), std::move(tokenizer), std::move(compiled_model));
}

[[maybe_unused]] const bool kParakeetTdtRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetParakeetTdtSessionFactory,
      .session_config_type_id = "Asr",
      .model_remote_url =
          "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/"
          "resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::asr
```

> **Toolchain & Linker Portability (GCC, Clang, Xcode, MSVC)**:
>
> *   **C++20 Designated Initializers**: Designated initializers require C++20
>     (`-std=c++20` on GCC/Clang/Xcode, `/std:c++20` on MSVC) and must appear in
>     exact struct declaration order (`model_names`, `get_session_factory`,
>     `session_config_type_id`, `metadata_type_id`, `model_remote_url`).
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
//    inheriting any unset Settings fields from the registered ModelSpec).
OmniEngine::Settings settings{
    .backend = OmniEngine::Settings::Backend::kGpu,
    .cache_dir = "/data/local/tmp/omni_cache",
    .model_path = "/data/local/tmp/parakeet_tdt_0.6b_v3.litertlm",
};
ABSL_ASSIGN_OR_RETURN(
    auto engine, OmniEngine::Create("parakeet-tdt-0.6b-v3", settings));

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
