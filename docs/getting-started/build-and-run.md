# Building from Source (Advanced)

🛑 **Note for App Developers:** You do **not** need to build this project from
 source to use it in your apps. If you are using Kotlin, Swift, or Python,
 please use our pre-built SDKs. More details in [technical overview](https://ai.google.dev/edge/litert-lm/overview).

This section provides instructions for compiling the core LiteRT-LM C++
framework from scratch. You should only follow these steps if you are:

* **A core contributor** fixing bugs or adding features to the LiteRT-LM engine.
* **A native C++ developer** who requires custom compilation flags for an
embedded system.

  - [Deploy to Windows](#deploy_to_windows)
  - [Deploy to Linux](#deploy_to_linux)
  - [Deploy to MacOS](#deploy_to_macos)
  - [Deploy to Android](#deploy_to_android)



## Build and Run

This guide provides the necessary steps to build and execute a Large Language
Model (LLM) on your device. Follow the instructions below to build and run the
sample code.

### Prerequisites

-   **Git**: To clone the repository and manage versions.
-   **Bazel (version 7.6.1)**: This project uses `bazel` as its build system.

#### Get the Source Code

Current stable branch tag:
[![Latest Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)

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

**Checkout the latest stable release
([![Latest Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)):**

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
[demo](#demo-usage) that shows how to initialize and interact with the model.

Please check the corresponding section below depending on your target deployment
device and your development platform.

Make sure [Git LFS](https://git-lfs.com) is installed, and run `git lfs pull` to
fetch the latest prebuilt binaries.

> Note: In order to run on GPU on all platforms, we need to take extra steps:
>
> 1.  Add `--define=litert_runtime_link_mode=dynamic`
>     `--define=resolve_symbols_in_exec=false` in the build command.
> 1.  `mkdir -p <test_dir>; cp <your litert_lm_main> <test_dir>; cp
>     ./prebuilt/<your OS>/<shared libaries> <test_dir>/` and make sure the
>     prebuilt .so/.dll/.dylib files are in the same directory as
>     `litert_lm_main` binary
> 1.  Running GPU on Windows needs DirectXShaderCompiler. See
>     [this Note](../../README.md#windows_gpu) for more details.

<details> <span id="deploy_to_windows"></span>
<summary><strong>Deploy to Windows</strong></summary>

Building on Windows requires several prerequisites to be installed first.

#### Prerequisites

1.  **Visual Studio 2022** - Download from
    https://visualstudio.microsoft.com/downloads/ and install. Make sure it
    install the MSVC toolchain for all users, usually under this directory
    C:\Program Files.
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
    [Supported Models and Performance](../../README.md#supported-models-and-performance)
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

<details> <span id="deploy_to_linux"></span>
<summary><strong>Deploy to Linux / Embedded</strong></summary>

`clang` is used to build LiteRT-LM on linux. Build `litert_lm_main`, a CLI
executable and run models on CPU. Note that you should download the `.litertlm`
model from the
[Supported Models and Performance](../../README.md#supported-models-and-performance) section.
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

<details> <span id="deploy_to_macos"></span>
<summary><strong>Deploy to MacOS</strong></summary>

Xcode command line tools include clang. Run `xcode-select --install` if not
installed before. Note that you should download the `.litertlm` model from the
[Supported Models and Performance](../../README.md#supported-models-and-performance) section.

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

<details> <span id="deploy_to_android"></span>
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

### Demo Usage <span id="demo-usage"></span>

`litert_lm_main` is a demo for running and evaluating large language models
(LLMs) using our LiteRT [Engine/Conversation interface](../api/cpp/conversation.md).
It provides basic functionalities as the following:

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
    --enable_profiling \
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

| Flag Name                      | Description          | Default Value       |
| :----------------------------- | :------------------- | :------------------ |
| `backend`                      | Executor backend to  | `"gpu"`             |
:                                : use for LLM          :                     :
:                                : execution (e.g.,     :                     :
:                                : cpu, gpu).           :                     :
| `model_path`                   | Path to the          | `""`                |
:                                : `.litertlm` file for :                     :
:                                : LLM execution.       :                     :
| `input_prompt`                 | Input prompt to use  | `"What is the       |
:                                : for testing LLM      : tallest building in :
:                                : execution.           : the world?"`        :
| `benchmark`                    | Benchmark the LLM    | `false`             |
:                                : execution.           :                     :
| `enable_profiling`             | Enable per-op        | `false`             |
:                                : profiling.           :                     :
| `benchmark_prefill_tokens`     | If benchmark is true | `0`                 |
:                                : and this value is >  :                     :
:                                : 0, the benchmark     :                     :
:                                : will use this number :                     :
:                                : to set the prefill   :                     :
:                                : tokens, regardless   :                     :
:                                : of the input prompt. :                     :
:                                : If this is non-zero, :                     :
:                                : `async` must be      :                     :
:                                : `false`.             :                     :
| `benchmark_decode_tokens`      | If benchmark is true | `0`                 |
:                                : and this value is >  :                     :
:                                : 0, the benchmark     :                     :
:                                : will use this number :                     :
:                                : to set the number of :                     :
:                                : decode steps,        :                     :
:                                : regardless of the    :                     :
:                                : input prompt.        :                     :
| `async`                        | Run the LLM          | `true`              |
:                                : execution            :                     :
:                                : asynchronously.      :                     :
| `report_peak_memory_footprint` | Report peak memory   | `false`             |
:                                : footprint.           :                     :

</details>
