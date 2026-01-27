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
Full release is coming soon.
We heard the community feedback regarding Google AI Edge's Gemma 3n LiteRT
preview. You want access on more platforms, more visibility into the underlying
stack, and more flexibility. LiteRT-LM can help with all three.

### 🚀 What's New

*   ***Nov 2025*** **: Desktop GPU support and more (`v0.8.0`)**
    - Desktop GPU support.
    - Simple CLI for Desktop: [Link to Quick Start section](#quick_start)
    - Multi-Modality support: Vision and Audio input are supported when models
      support it. [See more details here](#multimodal)
    - Kotlin API for Android and JVM (Linux, MacOS, Windows): [Link to LiteRT-LM
      Kotlin API](./kotlin/README.md)
    - Conversation API: [Link to Conversation API](./docs/conversation.md)
    - Function calling support: [Link to Tool Use](./docs/tool_use.md)

*   ***June 24, 2025*** **: Run Gemma models with NPU Support (`v0.7.0`)**
    Unlock significant performance gains! Our latest release leverages the power
    of Neural Processing Units (NPUs) on devices with Qualcomm and MediaTek
    chipsets to run the Gemma3 1B model with incredible efficiency.

    **Note:** LiteRT-LM NPU acceleration is only available through an Early
    Access Program. Please check out [this
    page](https://ai.google.dev/edge/litert/next/npu) for more information about
    how to sign it up.
*   ***June 10, 2025*** **: The Debut of LiteRT-LM: A New Framework for
    On-Device LLMs** We're proud to release an early preview (`v0.6.1`) of the
    LiteRT-LM codebase! This foundational release enables you to run the latest
    Gemma series models across a wide range of devices with initial support for
    CPU execution and powerful GPU acceleration on Android.

### Supported Backends & Platforms

Platform     | CPU Support | GPU Support | NPU Support |
:----------- | :---------: | :-----------: | :-----------:
**Android**  | ✅           | ✅            | ✅ |
**macOS**    | ✅           | ✅            | - |
**Windows**  | ✅           | ✅            | - |
**Linux**    | ✅           | ✅            | - |
**Embedded** | ✅           | -             | - |

### Supported Models and Performance

Currently supported models during our Preview (as `.litertlm` format).

| Model               | Usage Type                     | Quantization      | Context size | Model Size (Mb) | Give it a try                                                                                                                                                             |
| :------------------ | :----------------------------- | :---------------- | :----------- | :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gemma3-1B           | Chat Ready                     | 4-bit per-channel | 4096         | 557             | [Download](https://huggingface.co/litert-community/Gemma3-1B-IT/blob/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm)                                             |
| Gemma-3n-E2B        | Chat Ready                     | 4-bit per-channel | 4096         | 2965            | [Download](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm-preview)                                                                                               |
| Gemma-3n-E4B        | Chat Ready                     | 4-bit per-channel | 4096         | 4235            | [Download](https://huggingface.co/google/gemma-3n-E4B-it-litert-lm-preview)                                                                                               |
| phi-4-mini          | Chat Ready                     | 8-bit per-channel | 4096         | 3728            | [Download](https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm)                            |
| qwen2.5-1.5b        | Chat Ready                     | 8-bit per-channel | 4096         | 1524            | [Download](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm)                        |
| FunctionGemma-270M  | Base (Fine-tuning required)    | 8-bit per-channel | 1024         | 288             | [Fine-tuning Guide](https://ai.google.dev/gemma/docs/mobile-actions)                                                                                                      |
| ↪ TinyGarden-270M   | Demo                           | 8-bit per-channel | 1024         | 288             | [Download](https://huggingface.co/google/functiongemma-270m-it/blob/main/tiny_garden.litertlm) / [Try App](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&hl=en_US) |

Below are the performance numbers of running each model on various devices. Note
that the benchmark is measured with 1024 tokens prefill and 256 tokens decode (
with performance lock on Android devices).

| Model | Device | Backend | Prefill (tokens/sec) | Decode (tokens/sec) | Context size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3-1B | MacBook Pro<br>(2023 M3) | CPU | 422.98 | 66.89 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | CPU | 243.24 | 43.56 | 4096 |
| Gemma3-1B | Samsung S24<br>(Ultra) | GPU | 1876.5 | 44.57 | 4096 |
| Gemma3-1B | Samsung S25<br>(Ultra) | NPU | 5836.6 | 84.8 | 1280 |
| Gemma-3n-E2B | MacBook Pro<br>(2023 M3) | CPU | 232.5 | 27.6 | 4096 |
| Gemma-3n-E2B | Samsung S24<br>(Ultra) | CPU | 110.5 | 16.1 | 4096 |
| Gemma-3n-E2B | Samsung S24<br>(Ultra) | GPU | 816.4 | 15.6 | 4096 |
| Gemma-3n-E4B | MacBook Pro<br>(2023 M3) | CPU | 170.1 | 20.1 | 4096 |
| Gemma-3n-E4B | Samsung S24<br>(Ultra) | CPU | 73.5 | 9.2 | 4096 |
| Gemma-3n-E4B | Samsung S24<br>(Ultra) | GPU | 548.0 | 9.4 | 4096 |
| FunctionGemma | Samsung S25<br>(Ultra) | CPU | 1718.4 | 125.9 | 1024 |

## Quick Start <span id="quick_start"></span>

**Want to try it out first?** Before proceeding with the full setup, you can use
the pre-built binary below to run the LiteRT-LM immediately.
### Desktop CLI (LIT)

-   [MacOS ARM64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.macos_arm64)
-   [Linux x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.linux_x86_64)
-   [Linux ARM64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.linux_arm64)
-   [Windows x86_64](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.8.0/lit.windows_x86_64.exe)

After the download the `lit` binary, just run `lit` to see the options.
Simple use case is like:

```shell
# Set the HuggingFace token in the HUGGING_FACE_HUB_TOKEN environment variable
# so that lit can pull the model from HuggingFace.

# On Linux or MacOS
export HUGGING_FACE_HUB_TOKEN="**your huggingface token**"

# On Windows Command Prompt
set HUGGING_FACE_HUB_TOKEN=your_huggingface_token

# On Windows Powershell
$env:HUGGING_FACE_HUB_TOKEN = "your_huggingface_token"
```

```shell
lit list --show_all
lit pull gemma3-1b
lit run gemma3-1b [--backend=<cpu|gpu>]
```

Note: **Running GPU on Windows needs DirectXShaderCompiler**
 <span id="windows_gpu"></span>
 Download the dxc_2025_07_14.zip or the latest zip file from
 https://github.com/microsoft/DirectXShaderCompiler/releases, unzip the file and
 locate the right architecture directory under `bin`, copy the `dxil.dll` and
 `dxcompiler.dll` into the same directory as the executable like `lit` or
 `litert_lm_main`.

Tip: For more functionality, use `lit --help` or `lit <command> --help`

Tip: Follow this [link](https://huggingface.co/docs/hub/en/security-tokens) to
get your own hugging face token

Tip: you may have to `chmod +x lit` and explicitly approve the usage of
pre-built binaries. For example, in MacOS, you should go to **System Settings >
Privacy & Security > Security** to approve the binary.

### Mobile Apps

-   [Android AI Edge Gallery App](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery&hl=en_US&pli=1)
-   iOS (Coming soon)

Note that the LiteRT-LM runtime is designed to work
with models in the `.litertlm` format. You can find and download compatible
models in the [Supported Models and
Performance](#supported-models-and-performance) section.

Note: that the first time a given model is loaded on a given device, it will
take longer to load. This is because the model weights are being arranged to run
optimally on your particular device. Subsequent loads will be faster
because the optimized weights are cached on your device.

## Build and Run
This guide provides the necessary steps to build and execute a Large Language
Model (LLM) on your device.
Follow the instructions below to build and run the sample code.

### Prerequisites

-   **Git**: To clone the repository and manage versions.
-   **Bazel (version 7.6.1)**: This project uses `bazel` as its build system.

#### Get the Source Code

Current stable branch tag: [![Latest
Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)

First, clone the repository to your local machine. We strongly recommend
checking out the latest stable release tag to ensure you are working with a
stable version of the code.

**Clone the repository:**

```
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
```

**Fetch the latest tags from the remote repository:**

```
git fetch --tags
```

**Checkout the latest stable release ([![Latest
Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)):**

To start working, create a new branch from the stable tag. This is the
recommended approach for development.

```
git checkout -b <my-feature-branch> <release-tag, e.g. "v0.8.0">
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

### Build and Run the Demo

**LiteRT-LM** allows you to deploy and run LLMs on various platforms, including
Android, Linux, MacOS, and Windows. `runtime/engine/litert_lm_main.cc` is a
[demo](#litert_lm_main) that shows how to initialize and interact
with the model.

Please check the corresponding section below depending on your target deployment
device and your development platform.

> Note: In order to run on GPU on all platforms, we need to take extra steps:
>
> 1. Install [Git LFS](https://git-lfs.com), then run `git lfs checkout` to fetch
  prebuilt binaries. These steps are required only once if not done before.
> 1. Add `--define=litert_link_capi_so=true`
  `--define=resolve_symbols_in_exec=false` in the build command.
> 1. `mkdir -p <test_dir>; cp <your litert_lm_main> <test_dir>; cp ./prebuilt/<your OS>/<shared libaries> <test_dir>/`
 and make sure the prebuilt .so/.dll/.dylib files are in the same directory as
  `litert_lm_main` binary
> 1. Running GPU on Windows needs DirectXShaderCompiler. See
 [this Note](#windows_gpu) for more details.

<details>
<summary><strong>Deploy to Windows</strong></summary>

Building on Windows requires several prerequisites to be installed first.

#### Prerequisites

1.  **Visual Studio 2022** - Download from
https://visualstudio.microsoft.com/downloads/ and install. Make sure it install
the MSVC toolchain for all users, usually under this directory C:\Program Files.
2.  **Git for Windows** - Install from https://git-scm.com/download/win
    (includes Git Bash needed for flatbuffer generation scripts).
3.  **Python 3.13** - Download from https://www.python.org/downloads/ and
install for all users.
4.  **Bazel** - Install using Windows Package Manager (winget): `powershell
    winget install --id=Bazel.Bazelisk -e`.
5.  **Java** - Install from https://www.oracle.com/java/technologies/downloads/
    and set JAVA_HOME to point at the jdk directory.
6.  **Enable long path** Make sure the LongPathsEnabled is true in the Registry.
    If needed, use `bazelisk --output_base=C:\bzl` to shorten the output path
    further. Otherwise, compilation errors related to file permission could
    happen.
7.  Download the `.litertlm` model from the
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

</details>

### Demo Usage <span id="litert_lm_main"></span>

`litert_lm_main` is a demo for running and evaluating large
language models (LLMs) using our LiteRT [Engine/Session interface](#engine). It
provides basic functionalities as the following:

-   generating text based on a user-provided prompt.
-   executing the inference on various hardware backends, e.g. CPU / GPU.
-   includes options for performance analysis, allowing users to benchmark
    prefill and decoding speeds, as well as monitor peak memory consumption
    during the run.
-   supports both synchronous and asynchronous execution modes.

<details>
<summary><strong>Example commands</strong></summary>

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

</details>

## LiteRT-LM API <span id="engine"></span>

The LiteRT-LM provides a C++ API for executing Language Models. It is designed
around two primary classes: `Engine` and `Session`.

-   The **`Engine`** is the main entry point. It's responsible for loading the
    model and its associated resources (like the tokenizer) from storage and
    preparing them for execution. It acts as a factory for creating
    `Conversation` or `Session` objects.
-   The **`Conversation`**: This class represents a single, stateful
    conversation with the LLM and is the recommended entry point for most users.
    It internally manages a `Session` and handles complex data processing tasks.
    These tasks include maintaining the initial context, managing tool
    definitions, preprocessing multimodal data, and applying Jinja prompt
    templates with role-based message formatting. Each `Conversation` instance
    is independent, allowing for multiple concurrent interactions.
-   The **`Session`**: This class also represents a single, stateful interaction
    with the LLM, holding the context and providing methods for text generation.
    `Session` is intended for advanced users who need fine-grained control over
    the prefill and decode phases, such as implementing chunked prefill or
    custom multimodal preprocessing. Like `Conversation`, each `Session`
    instance is independent.

### Basic Workflow for Text-in-Text-out Inference

The typical lifecycle for using the runtime is:

1.  **Create an `Engine`**: Initialize a single `Engine` with the model path and
    configuration. This is a heavyweight object that holds the model weights.
2.  **Create a `Conversation`**: Use the `Engine` to create one or more
    lightweight `Conversation` objects.
3.  **Send Message**: Utilize the `Conversation` object's methods to send
    messages to the LLM and receive responses, effectively enabling a chat-like
    interaction.

Below is the simplest way to send message and get model response. It is
recommended for most use cases. It mirrors [Gemini Chat
APIs](https://ai.google.dev/gemini-api/docs/text-generation#multi-turn-conversations).

-   `SendMessage`: A blocking call that takes user input and returns the
    complete model response.
-   `SendMessageAsync`: A non-blocking call that streams the model's response
    back token-by-token through callbacks.

Example code snippet:

```cpp
#include "runtime/engine/engine.h"

// ...

// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create(model_path);
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::CPU);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
CHECK_OK(engine);

// 3. Create a Conversation
auto conversation_config = ConversationConfig::CreateDefault(**engine);
CHECK_OK(conversation_config)
absl::StatusOr<std::unique_ptr<Conversation>> conversation = Conversation::Create(**engine, *conversation_config);
CHECK_OK(conversation);

// 4. Send message to the LLM.
absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
    JsonMessage{
        {"role", "user"},
        {"content", "What is the tallest building in the world?"}
    });
CHECK_OK(model_message);

// 5. Print the model message.
std::cout << *model_message << std::endl;

```

### Inference with GPU Backend

The runtime can pick GPU as the backend for inference instead of
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

### Inference with Multimodal data <span id="multimodal"></span>

To use multimodality, the engine must be created with vision and audio backend
depending on the multimodality to be used.

```cpp
// Create engine with proper multimodality backend, depending on which backend
// the model support. Note for Gemma-3N models, vision_backend must be GPU and
// audio_backend must be CPU.
auto engine_settings = EngineSettings::CreateDefault(
    model_assets,
    /*backend=*/litert::lm::Backend::CPU,
    /*vision_backend*/litert::lm::Backend::GPU,
    /*audio_backend*/litert::lm::Backend::CPU,
);

// The same steps to create Engine and Conversation as above...

// Send message to the LLM with image data.
absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
    JsonMessage{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "Describe the following image: "}},
          {{"type", "image"}, {"path", "/file/path/to/image.jpg"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

// Send message to the LLM with audio data.
model_message = (*conversation)->SendMessage(
    JsonMessage{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "Transcribe the audio: "}},
          {{"type", "audio"}, {"path", "/file/path/to/audio.wav"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

// The content can include multiple image or audio data.
model_message = (*conversation)->SendMessage(
    JsonMessage{
        {"role", "user"},
        {"content", { // Now content must be an array.
          {{"type", "text"}, {"text", "First briefly describe the two images "}},
          {{"type", "image"}, {"path", "/file/path/to/image1.jpg"}},
          {{"type", "text"}, {"text", "and "}},
          {{"type", "image"}, {"path", "/file/path/to/image2.jpg"}},
          {{"type", "text"}, {"text", " then transcribe the content in the audio"}},
          {{"type", "audio"}, {"path", "/file/path/to/audio.wav"}}
        }},
    });
CHECK_OK(model_message);

// Print the model message.
std::cout << *model_message << std::endl;

```

For multimodal data input, `content` is a list of `part`. Where `part` is a
Json, and currently expect following structs:

```json
{
  "type": "text",
  "text": "this is a text"
}

{
  "type": "image",
  "path": "/path/to/image.jpg"
}

{
  "type": "image",
  "blob": "base64 encoded image bytes as string",
}

{
  "type": "audio",
  "path": "/path/to/audio.wav"
}

{
  "type": "audio",
  "blob": "base64 encoded audio bytes as string",
}
```

### Advanced Control Over Prefill/Decode

**`Session`** API provides fine-grained control over the two phases of
transformer inference: prefill and decode. This can be useful for advanced
scenarios or performance optimization.

-   **Prefill**: The `RunPrefill` or `RunPrefillAsync` methods process the input
    prompt and populate the model's internal state (KV cache).
-   **Decode**: The `RunDecode` or `RunDecodeAsync` methods generate new tokens
    one at a time based on the prefilled state.

Example code snippet:

```cpp
#include "runtime/engine/engine.h"

// ...

// 1. Define model assets and engine settings.
auto model_assets = ModelAssets::Create(model_path);
CHECK_OK(model_assets);

auto engine_settings = EngineSettings::CreateDefault(
    model_assets, litert::lm::Backend::CPU);

// 2. Create the main Engine object.
absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
CHECK_OK(engine);

// 3. Create a Session for a new context.
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
are out-of-the-box native APIs (Kotlin, Swift, JS) to run language models by
just setting a few parameters such as temperature and topK.

### .litertlm vs .task

MediaPipe GenAI Tasks currently use `.task` files to represent language models.
Task files are a zip of multiple LiteRT files, components, and metadata.
`.litertlm` is an evolution of the `.task` file format to include additional
metadata and enable better compression.

During our LiteRT-LM preview, we will release a small number of `.litertlm`
files. MediaPipe APIs will continue to use `.task` files. Once we have the first
full release of LiteRT-LM, we will migrate MediaPipe APIs to use the new
`.litertlm` files and release a wider collection of `.litertlm` files on the
[LiteRT Hugging Face Community](https://huggingface.co/litert-community)

## Reporting Issues

If you encounter a bug or have a feature request, we encourage you to use the
[GitHub Issues](https://github.com/google-ai-edge/LiteRT-LM/issues/new) page to
report it.

Before creating a new issue, please search the existing issues to avoid
duplicates. When filing a new issue, please provide a clear title and a detailed
description of the problem, including steps to reproduce it. The more
information you provide, the easier it will be for us to help you.
