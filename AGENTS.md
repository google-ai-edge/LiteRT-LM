# LiteRT-LM Repository AI Rules

This is a C++ based repository for the LiteRT-LM engine, supporting multiple
platform bindings (such as Kotlin/JNI).

## Domain-Specific Context Routing

Agents MUST NOT guess build configurations or API usages. Before executing any
task, you MUST read the requirement file for the specific platform binding you
are working on:

*   **Kotlin / JNI Bindings (Android)**: Read `agents/kotlin/requirements.md`

## Global Repository Constraints

*   **Deterministic Execution**: Always use specific toolchain versions defined
    in the platform requirement files. Do not hallucinate deprecated build
    tools.
*   **Performance Optimization**: Multi-Token Prediction (MTP) via speculative
    decoding is universally recommended for all tasks on GPU backends.

## Core Concepts

```
Engine          - loads model weights; create once, reuse across sessions
  └── Conversation   - high-level stateful chat (recommended for most apps)
  └── Session        - low-level prefill/decode; use for scoring, checkpoints, cloning
```

Messages use an OpenAI-compatible format: `role` is `user`, `model`, `tool`, or
`system`; `content` is a string or an array of typed parts (`text`, `image`,
`audio`).

## Model Acquisition

LiteRT-LM models are distributed as `.litertlm` bundle files via HuggingFace.
You can download them using the `litert-lm` command-line tool or
programmatically using the `huggingface_hub` library.

**Example CLI Usage:** `bash uv tool install litert-lm litert-lm run
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm
gemma-4-E2B-it.litertlm --prompt="Hello"`

**Example Python Usage:** `python from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
repo_id="litert-community/gemma-4-E2B-it-litert-lm",
filename="gemma-4-E2B-it.litertlm", local_dir="./models" )`

## Quick Starts

### Python

```python
import litert_lm
with litert_lm.Engine("model.litertlm", backend=litert_lm.Backend.GPU) as engine:
    with engine.create_conversation() as conversation:
        response = conversation.send_message("Hello")
        print(response['content'][0]['text'])
```

**Full guide:**
[Python API on ai.google.dev](https://ai.google.dev/edge/litert-lm/python)

### Kotlin

```kotlin
val engineConfig = EngineConfig(modelPath = "model.litertlm", backend = Backend.GPU())
Engine(engineConfig).use { engine ->
    engine.initialize()
    engine.createConversation().use { conversation ->
        conversation.sendMessageAsync("Hello").collect { print(it.toString()) }
    }
}
```

**Full guide:**
[Android API on ai.google.dev](https://ai.google.dev/edge/litert-lm/android)

### C++

```cpp
// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create("model.litertlm");
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    *model_assets,
    /*backend=*/litert::lm::Backend::CPU);
CHECK_OK(engine_settings);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(*engine_settings);
CHECK_OK(engine);

// 3. Create a Conversation
auto conversation_config = ConversationConfig::CreateDefault(**engine);
CHECK_OK(conversation_config);
absl::StatusOr<std::unique_ptr<Conversation>> conversation = Conversation::Create(**engine, *conversation_config);
CHECK_OK(conversation);

// 4. Send message to the LLM with blocking call.
absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
    JsonMessage{
        {"role", "user"},
        {"content", "What is the tallest building in the world?"}
    });
CHECK_OK(model_message);

// 5. Print the model message.
std::cout << *model_message << std::endl;
```

**Full guide:**
[C++ API on ai.google.dev](https://ai.google.dev/edge/litert-lm/cpp)

### Swift

```swift
let engineConfig = try EngineConfig(
    modelPath: "model.litertlm",
    backend: .gpu)
let engine = Engine(engineConfig: engineConfig)
try await engine.initialize()
let conversation = try await engine.createConversation()
let response = try await conversation.sendMessage(Message("Hello"))
print(response.toString)
```

**Full guide:**
[iOS/macOS API on ai.google.dev](https://ai.google.dev/edge/litert-lm/swift)

## Topic Index

Topic                                   | Languages | Document
--------------------------------------- | --------- | --------
Overview — benchmarks, supported models | All       | [Overview on ai.google.dev](https://ai.google.dev/edge/litert-lm/overview)
Command Line Interface (CLI)            | All       | [CLI Guide on ai.google.dev](https://ai.google.dev/edge/litert-lm/cli)
Python API                              | Python    | [Python API on ai.google.dev](https://ai.google.dev/edge/litert-lm/python)
Android API                             | Kotlin    | [Android API on ai.google.dev](https://ai.google.dev/edge/litert-lm/android)
C++ API                                 | C++       | [C++ API on ai.google.dev](https://ai.google.dev/edge/litert-lm/cpp)
Swift API                               | Swift     | [iOS/macOS API on ai.google.dev](https://ai.google.dev/edge/litert-lm/swift)
Flutter API                             | Dart      | [Flutter API on ai.google.dev](https://ai.google.dev/edge/litert-lm/flutter)
Build & Inspect Models                  | CLI       | [litert-lm-builder on ai.google.dev](https://ai.google.dev/edge/litert-lm/file_builder)

## Known Constraints

-   **Engine init is slow** (2–10 s). Always initialize on a background thread.
-   **Not thread-safe.** Serialize calls to Engine/Conversation, or use separate
    Sessions per thread.
-   **KV cache grows unboundedly.** For long sessions use
    `SaveCheckpoint`/`RewindToCheckpoint` (C++) or periodically reset the
    conversation.
-   **Tool calling requires a compatible model.** Gemma 4, Qwen3, and
    FunctionGemma variants have explicit tool calling support. Check with `uvx
    litert-lm-peek --litertlm_file model.litertlm`.
-   **GPU backend may be unavailable.** Always provide a CPU fallback; the
    runtime throws if the requested backend is absent.
