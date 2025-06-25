# LiteRT-LM

A C++ library to efficiently run language models across edge platforms.

## Description

Language models are no longer a single model but really a pipeline of models and
components working together. LiteRT-LM builds on top of
[LiteRT](https://github.com/google-ai-edge/LiteRT) to enable these pipelines
including:

*   **C++ api** to efficiently run language models
*   **Cross-Platform** support via portable C++ for broad deployment scenarios
*   **Flexible** so you can customize for your specific feature
*   **Hardware Acceleration** to unlock the full potential of your device's
    hardware

### Status: Early Preview

Expect our first full release of LiteRT-LM late summer / early fall. We heard
the community feedback regarding Google AI Edge's Gemma 3n LiteRT preview. You
want access on more platforms, more visibility into the underlying stack, and
more flexibility. LiteRT-LM can help with all three.

### 🚀 What's New

*   ***June 24, 2025*** **: Run Gemma models with NPU Support (`v0.7.0`)**
    Unlock significant performance gains! Our latest release leverages the power
    of Neural Processing Units (NPUs) on devices with Qualcomm and MediaTek
    chipsets to run the Gemma3 1B model with incredible efficiency.

    **Note:** LiteRT-LM NPU acceleration is only available through an Early
    Access Program. Please check out
    [this page](https://ai.google.dev/edge/litert/next/npu) for more information
    about how to sign it up.
*   ***June 10, 2025*** **: The Debut of LiteRT-LM: A New Framework for
    On-Device LLMs** We're proud to release an early preview (`v0.6.1`) of the
    LiteRT-LM codebase! This foundational release enables you to run the latest
    Gemma series models across a wide range of devices with initial support for
    CPU execution and powerful GPU acceleration on Android.

### Supported Backends & Platforms

Platform     | CPU Support | GPU Support | NPU Support |
:----------- | :---------: | :-----------: | :-----------:
**Android**  | ✅           | ✅            | ✅ |
**macOS**    | ✅           | *Coming Soon* | - |
**Windows**  | ✅           | *Coming Soon* | - |
**Linux**    | ✅           | *Coming Soon* | - |
**Embedded** | ✅           | *Coming Soon* | - |

### Supported Models and Performance

Currently supported models during our Preview (as `.litertlm` format).

Model       | Quantization      | Context size | Model Size (Mb) | Download link
:---------- | :---------------: | :----------: | :-------------: | :-----------:
Gemma3-1B   | 4-bit per-channel | 4096         | 557             | [download](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm)
Gemma3n-E2B | 4-bit per-channel | 4096         | 2965            | [download](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm-preview)
Gemma3n-E4B | 4-bit per-channel | 4096         | 4235            | [download](https://huggingface.co/google/gemma-3n-E4B-it-litert-lm-preview)

Below are the performance numbers of running each model on various devices. Note
that the benchmark is measured with 1024 tokens prefill and 256 tokens decode (
with performance lock on Android devices).

| Model | Device | Backend | Prefill (tokens/sec) | Decode (tokens/sec) | Context size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3-1B | MacBook Pro<br>(2023 M3) | CPU | 422.98 | 66.89 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | CPU | 243.24 | 43.56 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | GPU | 1876.5 | 44.57 | 4096 |
| Gemma3-1B | Samsung S25<br>(Ultra) | NPU | 5836.6 | 84.8 | 1280 |
| Gemma3n-E2B | MacBook Pro<br>(2023 M3) | CPU | 232.5 | 27.6 | 4096 |
| Gemma3n-E2B | Samsung S24<br>(Ultra) | CPU | 110.5 | 16.1 | 4096 |
| Gemma3n-E2B | Samsung S24<br>(Ultra) | GPU | 816.4 | 15.6 | 4096 |
| Gemma3n-E4B | MacBook Pro<br>(2023 M3) | CPU | 170.1 | 20.1 | 4096 |
| Gemma3n-E4B | Samsung S24<br>(Ultra) | CPU | 73.5 | 9.2 | 4096 |
| Gemma3n-E4B | Samsung S24<br>(Ultra) | GPU | 548.0 | 9.4 | 4096 |

## Quick Start

This guide provides the necessary steps to build and execute a Large Language
Model (LLM) on your device. Note that the LiteRT-LM runtime is designed to work
with models in the `.litertlm` format. You can find and download compatible
models in the
[Supported Models and Performance](#supported-models-and-performance) section.

**Want to try it out first?** Before proceeding with the full setup, you can use
the pre-built binary below to run the LiteRT-LM immediately:

-   [Android Arm64](https://github.com/google-ai-edge/LiteRT-LM/releases/latest/download/litert_lm_main.android_arm64)
-   [MacOS](https://github.com/google-ai-edge/LiteRT-LM/releases/latest/download/litert_lm_main.macos_arm64)
-   [Linux x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/latest/download/litert_lm_main.linux_x86_64)
-   [Windows x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/latest/download/litert_lm_main.windows_x86_64.exe)
-   [iOS Arm64](https://github.com/google-ai-edge/LiteRT-LM/releases/latest/download/litert_lm_main.ios_sim_arm64)

*Tip: you may have to explicitly approve the usage of pre-built binaries. For
example, in MacOS, you should go to **System Settings > Privacy & Security >
Security** to approve the binary. *

### Prerequisites

Before you begin, please ensure you have the following installed:

-   **Git**: To clone the repository and manage versions.
-   **Bazel (version 7.6.1)**: This project uses `bazel` as its build system.

#### Get the Source Code

Current stable branch tag: [![Latest Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)

First, clone the repository to your local machine. We strongly recommend
checking out the latest stable release tag to ensure you are working with a
stable version of the code.

**Clone the repository:**

```
git clone git@github.com:google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
```

**Fetch the latest tags from the remote repository:**

```
git fetch --tags
```

**Checkout the latest stable release ([![Latest Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)):**

To start working, create a new branch from the stable tag. This is the
recommended approach for development.

```
git checkout -b <my-feature-branch> <release-tag, e.g. "v0.6.1">
```

You are now on a local branch created from the tag and ready to work.

#### Install Bazel

This project requires Bazel version **7.6.1**. You can skip this if you already
have it set up.

The easiest way to manage Bazel versions is to install it via
[Bazelisk](https://github.com/bazelbuild/bazelisk). Bazelisk will automatically
download and use the correct Bazel version specified in the project's
.bazelversion file.

Alternatively, you can install Bazel manually by following the official
installation [instructions](https://bazel.build/install) for your platform.

### Build and Run the Command Line Demo

**LiteRT-LM** allows you to deploy and run LLMs on various platforms, including
Android, Linux, MacOS, and Windows. `runtime/engine/litert_lm_main.cc` is a
[command line demo](#litert_lm_main) that shows how to initialize and interact
with the model.

Please check the corresponding section below depending on your target deployment
device and your development platform.

<details>
<summary><strong>Deploy to Windows</strong></summary>

Building on Windows requires several prerequisites to be installed first.

#### Prerequisites

1.  **Visual Studio 2022** - Install from Microsoft Store to get the MSVC
    toolchain.
2.  **Git for Windows** - Install from https://git-scm.com/download/win
    (includes Git Bash needed for flatbuffer generation scripts).
3.  **Python 3.11** - Install from Microsoft Store for Python dependencies.
4.  **Bazel** - Install using Windows Package Manager (winget): `powershell
    winget install --id=Bazel.Bazelisk -e`.
5.  Download the `.litertlm` model from the
    [Supported Models and Performance](#supported-models-and-performance)
    section.

#### Building and Running

Once you've downloaded the `.litertlm` file, set the path for convenience:

```powershell
$Env:MODEL_PATH = "C:\path\to\your_model.litertlm"
```

Build the binary:

```powershell
# Build litert_lm_main for Windows.
bazelisk build //runtime/engine:litert_lm_main --config=windows
```

Run the binary (make sure you run the following command in **powershell**):

```powershell
# Run litert_lm_main.exe with a model .litertlm file.
bazel-bin\runtime\engine\litert_lm_main.exe `
    --backend=cpu `
    --model_path=$Env:MODEL_PATH
```

</details>

<details>
<summary><strong>Deploy to Linux / Embedded</strong></summary>

`clang` is used to build LiteRT-LM on linux. Build `litert_lm_main`, a CLI
executable and run models on CPU. Note that you should download the `.litertlm`
model from the
[Supported Models and Performance](#supported-models-and-performance) section.
Note that one can also deploy the model to Raspberry Pi using the same setup and
command in this section.

Once you've downloaded the `.litertlm` file, set the path for convenience:

```
export MODEL_PATH=<path to your .litertlm file>
```

Build the binary:

```
bazel build //runtime/engine:litert_lm_main
```

Run the binary:

```
bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=$MODEL_PATH
```

</details>

<details>
<summary><strong>Deploy to MacOS</strong></summary>

Xcode command line tools include clang. Run `xcode-select --install` if not
installed before. Note that you should download the `.litertlm` model from the
[Supported Models and Performance](#supported-models-and-performance) section.

Once you've downloaded the `.litertlm` file, set the path for convenience:

```
export MODEL_PATH=<path to your .litertlm file>
```

Build the binary:

```
bazel build //runtime/engine:litert_lm_main
```

Run the binary:

```
bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=$MODEL_PATH
```

</details>

<details>
<summary><strong>Deploy to Android</strong></summary>

To be able to interact with your Android device, please make sure you've
properly installed
[Android Debug Bridge](https://developer.android.com/tools/adb) and have a
connected device that can be accessed via `adb`.

**Note:** If you are interested in trying out LiteRT-LM with NPU acceleration,
please check out [this page](https://ai.google.dev/edge/litert/next/npu) for
more information about how to sign it up for an Early Access Program.

<details>
<summary><strong>Develop in Linux</strong></summary>

To be able to build the binary for Android, one needs to install NDK r28b or
newer from https://developer.android.com/ndk/downloads#stable-downloads.
Specific steps are:

-   Download the `.zip` file from
    https://developer.android.com/ndk/downloads#stable-downloads.
-   Unzip the `.zip` file to your preferred location (say
    `/path/to/AndroidNDK/`)
-   Make `ANDROID_NDK_HOME` to point to the NDK directory. It should be
    something like:

```
export ANDROID_NDK_HOME=/path/to/AndroidNDK/
```

*Tips: make sure your `ANDROID_NDK_HOME` points to the directory that has
`README.md` in it.*

With the above set up, let's try to build the `litert_lm_main` binary:

```
bazel build --config=android_arm64 //runtime/engine:litert_lm_main
```

</details>

<details>
<summary><strong>Develop in MacOS</strong></summary>

Xcode command line tools include clang. Run `xcode-select --install` if not
installed before.

To be able to build the binary for Android, one needs to install NDK r28b or
newer from https://developer.android.com/ndk/downloads#stable-downloads.
Specific steps are:

-   Download the `.dmg` file from
    https://developer.android.com/ndk/downloads#stable-downloads.
-   Open the `.dmg` file and move the `AndroidNDK*` file to your preferred
    location (say `/path/to/AndroidNDK/`)
-   Make `ANDROID_NDK_HOME` to point to the NDK directory. It should be
    something like:

```
export ANDROID_NDK_HOME=/path/to/AndroidNDK/AndroidNDK*.app/Contents/NDK/
```

*Tips: make sure your `ANDROID_NDK_HOME` points to the directory that has
`README.md` in it.*

With the above set up, let's try to build the `litert_lm_main` binary:

```
bazel build --config=android_arm64 //runtime/engine:litert_lm_main
```

</details>

After the binary is successfully built, we can now try to run the model on
device. Make sure you have the write access to the `DEVICE_FOLDER`:

In order to run the binary on your Android device, we have to push a few assets
/ binaries. First set your `DEVICE_FOLDER`, please make sure you have the write
access to it (typically you can put things under `/data/local/tmp/`):

```
export DEVICE_FOLDER=/data/local/tmp/
adb shell mkdir -p $DEVICE_FOLDER
```

To run with **CPU** backend, simply push the main binary and the `.litertlm`
model to device and run.

```
# Skip model push if it is already there
adb push $MODEL_PATH $DEVICE_FOLDER/model.litertlm

adb push bazel-bin/runtime/engine/litert_lm_main $DEVICE_FOLDER

adb shell $DEVICE_FOLDER/litert_lm_main \
    --backend=cpu \
    --model_path=$DEVICE_FOLDER/model.litertlm
```

To run with **GPU** backend, we need additional `.so` files. They are located in
the `prebuilt/` subfolder in the repo (we currently only support `arm64`).

```
# Skip model push if it is already there
adb push $MODEL_PATH $DEVICE_FOLDER/model.litertlm

adb push prebuilt/android_arm64/*.so $DEVICE_FOLDER
adb push bazel-bin/runtime/engine/litert_lm_main $DEVICE_FOLDER

adb shell LD_LIBRARY_PATH=$DEVICE_FOLDER \
    $DEVICE_FOLDER/litert_lm_main \
    --backend=gpu \
    --model_path=$DEVICE_FOLDER/model.litertlm
```

Note that the first time a given model is loaded on a given device, it will take
longer to load. This is because the model weights are being arranged to run
optimally on your particular device's GPU. Subsequent loads will be faster
because the optimized weights are cached on your device.

</details>

### Command Line Demo Usage <span id="litert_lm_main"></span>

`litert_lm_main` is a command line demo for running and evaluating large
language models (LLMs) using our LiteRT [Engine/Session interface](#engine). It
provides basic functionalities as the following:

-   generating text based on a user-provided prompt.
-   executing the inference on various hardware backends, e.g. CPU / GPU.
-   includes options for performance analysis, allowing users to benchmark
    prefill and decoding speeds, as well as monitor peak memory consumption
    during the run.
-   supports both synchronous and asynchronous execution modes.

Below are a few example commands (please update accordingly when using `adb`):

**Run the model with default prompt**

```
<path to binary directory>/litert_lm_main \
    --backend=cpu \
    --model_path=$MODEL_PATH
```

**Benchmark the model performance**

```
<path to binary directory>/litert_lm_main \
    --backend=cpu \
    --model_path=$MODEL_PATH \
    --benchmark \
    --benchmark_prefill_tokens=1024 \
    --benchmark_decode_tokens=256 \
    --async=false
```

*Tip: when benchmarking on Android devices, remember to use `taskset` to pin the
executable to the main core for getting the consistent numbers, e.g. `taskset
f0`.*

**Run the model with your prompt**

```
<path to binary directory>/litert_lm_main \
    --backend=cpu \
    --input_prompt=\"Write me a song\"
    --model_path=$MODEL_PATH
```

More detailed description about each of the flags are in the following table:

| Flag Name | Description | Default Value |
| :--- | :--- | :--- |
| `backend` | Executor backend to use for LLM execution (e.g., cpu, gpu). | `"gpu"` |
| `model_path` | Path to the `.litertlm` file for LLM execution. | `""` |
| `input_prompt` | Input prompt to use for testing LLM execution. | `"What is the tallest building in the world?"` |
| `benchmark` | Benchmark the LLM execution. | `false` |
| `benchmark_prefill_tokens` | If benchmark is true and this value is > 0, the benchmark will use this number to set the prefill tokens, regardless of the input prompt. If this is non-zero, `async` must be `false`. | `0` |
| `benchmark_decode_tokens` | If benchmark is true and this value is > 0, the benchmark will use this number to set the number of decode steps, regardless of the input prompt. | `0` |
| `async` | Run the LLM execution asynchronously. | `true` |
| `report_peak_memory_footprint` | Report peak memory footprint. | `false` |

## LiteRT-LM API <span id="engine"></span>

The LiteRT-LM provides a C++ API for executing Language Models. It is designed
around two primary classes: `Engine` and `Session`.

-   The **`Engine`** is the main entry point. It's responsible for loading the
    model and its associated resources (like the tokenizer) from storage and
    preparing them for execution. It acts as a factory for creating `Session`
    objects.
-   The **`Session`** represents a single, stateful conversation or interaction
    with the LLM. It holds the context (like conversation history) and provides
    the methods to actually generate text. Each `Session` is an independent
    instance, allowing for multiple interactions.

### Basic Workflow for Text-in-Text-out Inference

The typical lifecycle for using the runtime is:

1.  **Create an `Engine`**: Initialize a single `Engine` with the model path and
    configuration. This is a heavyweight object that holds the model weights.
2.  **Create a `Session`**: Use the `Engine` to create one or more lightweight
    `Session` objects.
3.  **Generate Content**: Use a `Session` object to run inference, either
    through a simple one-shot API or through more granular prefill/decode steps.

Below is the simplest way to generate text and is recommended for most use
cases. It mirrors
[Gemini text generation APIs](https://ai.google.dev/gemini-api/docs).

-   `GenerateContent`: A blocking call that takes user input and returns the
    complete model response.
-   `GenerateContentStream`: A non-blocking call that streams the model's
    response back token-by-token through an observer.

Example code snippet:

```cpp
#include "third_party/odml/litert_lm/runtime/engine/engine.h"

// ...

// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create(model_path);
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::CPU);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
CHECK_OK(engine);

// 3. Create a Session for a new conversation.
auto session_config = SessionConfig::CreateDefault();
absl::StatusOr<std::unique_ptr<Engine::Session>> session = (*engine)->CreateSession(session_config);
CHECK_OK(session);

// 4. Generate content using the high-level API.
absl::StatusOr<Responses> responses = (*session)->GenerateContent(
    {InputText("What is the tallest building in the world?")});
CHECK_OK(responses);

// 5. Print the response.
std::cout << *responses << std::endl;
```

### Inference with GPU Backend

On Android, the runtime can pick GPU as the backend for inference instead of
CPU, by passing `litert::lm::Backend::GPU` in `EngineSettings::CreateDefault()`.

```cpp
// ...

// Set GPU as backend instead of CPU.
auto engine_settings = EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::GPU);

// ...
```

When the engine is created, it looks for `libLiteRtGpuAccelerator.so` and
`libLiteRtTopKSampler.so` from the directories specified in `LD_LIBRARY_PATH`,
rpath in the app binary or default location by system dynamic linker. For
example, if an app binary and .so files are packaged in an APK by Android SDK,
.so files are unpacked by Android Package Manager where the app binary can find
them, i.e. under app's `/lib` directory.

### Advanced Control Over Prefill/Decode

This API provides fine-grained control over the two phases of transformer
inference: prefill and decode. This can be useful for advanced scenarios or
performance optimization.

-   **Prefill**: The `RunPrefill` or `RunPrefillAsync` methods process the input
    prompt and populate the model's internal state (KV cache).
-   **Decode**: The `RunDecode` or `RunDecodeAsync` methods generate new tokens
    one at a time based on the prefilled state.

Example code snippet:

```cpp
#include "third_party/odml/litert_lm/runtime/engine/engine.h"

// ...

// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create(model_path);
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::CPU);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
CHECK_OK(engine);

// 3. Create a Session for a new conversation.
auto session_config = SessionConfig::CreateDefault();
absl::StatusOr<std::unique_ptr<Engine::Session>> session = (*engine)->CreateSession(session_config);
CHECK_OK(session);

// 4. Prefill some prompts.
CHECK_OK((*session)->RunPrefill({InputText("What's the tallest building in the world?")}));
CHECK_OK((*session)->RunPrefill({InputText(" and what's the tallest building in the United States?")}));

// 5. Start decoding.
auto responses = (*session)->RunDecode();

// 6. Print the response.
std::cout << *responses << std::endl;
```

## FAQ

### LiteRT vs LiteRT-LM vs MediaPipe GenAI Tasks

LiteRT, LiteRT-LM, and MediaPipe GenAI Tasks are three libraries within the
Google AI Edge stack that build on each other. By exposing functionality at
different abstraction layers, we hope to enable developers to balance their
respective needs between flexibility and complexity.

[LiteRT](https://ai.google.dev/edge/litert) is Google AI Edge's underlying
on-device runtime. Developer can convert individual PyTorch, TensorFlow, and JAX
models to LiteRT and run them on-device.

**LiteRT-LM** gives developers the pipeline framework to stitch together
multiple LiteRT models with pre and post processing components (e.g. tokenizer,
vision encoder, text decoder).

[MediaPipe GenAI Tasks](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)
are out-of-the-box native APIs (Kotlin, Swift, JS) to run langauge models by
just setting a few parameters such as temperature and topK.

### .litertlm vs .task

MediaPipe GenAI Tasks currently use `.task` files to represent language models.
Task files are a zip of multiple LiteRT files, components, and metadata.
`.litertlm` is an evolution of the `.task` file format to include additional
metadata and enable better compression.

During our LiteRT-LM preview, we will release a small number of `.litertlm
files`. MediaPipe APIs will continue to use `.task` files. Once we have the
first full release of LiteRT-LM, we will migrate MediaPipe APIs to use the new
`.litertlm` files and release a wider collection of .litertlm files on the
[LiteRT Hugging Face Community](https://huggingface.co/litert-community)

## Reporting Issues

If you encounter a bug or have a feature request, we encourage you to use the
[GitHub Issues](https://github.com/google-ai-edge/LiteRT-LM/issues/new) page to
report it.

Before creating a new issue, please search the existing issues to avoid
duplicates. When filing a new issue, please provide a clear title and a detailed
description of the problem, including steps to reproduce it. The more
information you provide, the easier it will be for us to help you.
