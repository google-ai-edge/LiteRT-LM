# LiteRT LM: High-Performance LLM Inference on the Edge

## Description

**LiteRT LM** is a cutting-edge, high-performance inference engine designed to
deploy and run Large Language Models (LLMs) efficiently on a wide range of
portable devices, from laptops to mobile phones. Our mission is to empower
developers to seamlessly integrate state-of-the-art LLMs into their
applications, achieving unparalleled performance with minimal effort.

Built upon the robust foundations of the LiteRT framework, LiteRT LM provides a
unified, flexible, and easy-to-use solution for on-device AI. Whether you are
building an application for Android, Linux, macOS, or Windows, LiteRT LM is the
tool you need to bring the power of LLMs directly into the hands of your users.

## Key Features

*   **Cross-Platform:** Build once and deploy anywhere. LiteRT LM supports a
    wide array of operating systems.
*   **Hardware Accelerated:** Leverage the full potential of your device's
    hardware. Our engine supports multiple execution backends.
*   **Best-in-Class Performance:** Meticulously optimized for low latency and
    high throughput, ensuring a smooth and responsive user experience.
*   **Developer-Friendly API:** A simple and intuitive API designed to make the
    integration of LLMs into your projects as straightforward as possible.

### Supported Backends & Platforms

LiteRT LM is engineered to be versatile. Below is a summary of our current
platform and backend support.

Platform    | CPU Support | GPU Support
:---------- | :---------: | :-----------:
**Android** | ✅           | ✅
**Linux**   | ✅           | *Coming Soon*
**macOS**   | ✅           | *Coming Soon*
**Windows** | ✅           | *Coming Soon*

### Supported Models

Below are a list of currently supported models.

Model       | Quantization | Context size | Download link
:---------- | :----------: | :----------: | :-----------------:
Gemma3-1B   | 4-bit QAT    | 4096         | TODO: add link here
Gemma3n-E2B | 4-bit QAT    | 4096         | TODO: add link here
Gemma3n-E4B | 4-bit QAT    | 4096         | TODO: add link here

### Performance

We are committed to delivering top-tier performance. The table below shows the
benchmark numbers of running
[Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it-litert-preview) on
a Samsung S25 Ultra with 4096 KV cache size, 1024 tokens prefill, 256 tokens
decode.

TODO: update the following numbers

Weight Quantization | Backend | Prefill (tokens/sec) | Decode (tokens/sec) | Time to first token (sec) | Model size (MB) | Peak RSS Memory (MB) | GPU Memory (MB)
:------------------ | :------ | :------------------- | :------------------ | :------------------------ | :-------------- | :------------------- | :--------------
dynamic_int4        | CPU     | 163                  | 17.6                | 6.7                       | 2991            | 2704                 | 193
dynamic_int4        | GPU     | 620                  | 23.3                | 12.7                      | 2991            | 3408                 | 3408

## Quick Start

Getting started with LiteRT LM is easy. Follow the steps below to build and run
an LLM on your target device.

`bazel` is used as the build system. Install `bazel` via
[Bazelisk](https://github.com/bazelbuild/bazelisk) or follow the
[instructions](https://bazel.build/install) per platform.

### Linux

`clang` is used to build LiteRT LM on linux. Build `litert_lm_main`, a CLI
executable and run models on CPU.

```
bazel build //runtime/engine:litert_lm_main

bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

### Android

Install NDK r28b or newer from https://developer.android.com/ndk/downloads. To
run on CPU, only `litert_lm_main` is required.

```
export ANDROID_NDK_HOME=<ndk dir>

bazel build --config=android_arm64 //runtime/engine:litert_lm_main

adb push bazel-bin/runtime/engine/litert_lm_main /data/local/tmp

adb shell /data/local/tmp/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

To run on GPU, `libLiteRtGpuAccelerator.so` is required.

```
adb push prebuilt/android_arm64/*.so /data/local/tmp

adb shell LD_LIBRARY_PATH=/data/local/tmp \
    /data/local/tmp/litert_lm_main \
    --backend=gpu \
    --model_path=<model .litertlm or .task file>
```

To run on Qualcomm NPU, `libLiteRtDispatch_Qualcomm.so` in the maven package
from [TBD]() and Qualcomm's runtime shared libraries downloaded from [TBD]() are
required. Model file should be compiled with LiteRt compiler with Qualcomm's
plugin.

```
adb push libLiteRtDispatch_Qualcomm.so /data/local/tmp
adb push <qualcomm runtime shlibs> /data/local/tmp

adb shell LD_LIBRARY_PATH=/data/local/tmp \
    /data/local/tmp/litert_lm_main \
    --backend=qnn \
    --model_path=<model .tflite file>
```

### MacOS

Xcode command line tools include clang. Run `xcode-select --install` if not
installed before.

```
bazel build //runtime/engine:litert_lm_main

bazel-bin/runtime/engine/litert_lm_main \
    --backend=cpu \
    --model_path=<model .litertlm or .task file>
```

If bazel can't figure out the right version of MacOS SDK, append
`--macos_sdk_version=<sdk version>` in `bazel build`.
