# LiteRT-LM

LiteRT-LM is Google's production-ready, high-performance, open-source inference
framework for deploying Large Language Models on edge devices.

🔗 [Product Website](https://ai.google.dev/edge/litert-lm)

## 🔥 What's New: `v0.12.0`

-   **Swift APIs**: Natively integrate LiteRT-LM into iOS applications
    with Metal GPU acceleration. See the
    [Swift Guide](https://ai.google.dev/edge/litert-lm/swift).
-   **Web JavaScript APIs**: Run models inside web browsers with high
    performance via web GPU/CPU. See the
    [JavaScript Guide](https://ai.google.dev/edge/litert-lm/js) and try out our
    [in-browser chat demo in Chrome](https://google-ai-edge.github.io/LiteRT-LM/web_demos/chat/index.html).
-   **LiteRT-LM CLI Update**: The command-line interface now supports NPU,
    besides CPU and GPU backends across Linux, macOS, and Windows. See the
    [CLI Guide](https://ai.google.dev/edge/litert-lm/cli).
-   **Community-Maintained Flutter APIs**: Build cross-platform Flutter
    applications using the community
    [flutter_gemma](https://github.com/DenisovAV/flutter_gemma) package. See the
    [Flutter Guide](https://ai.google.dev/edge/litert-lm/flutter).

👉 Try Gemma4-E4B with MTP on Linux, macOS, Windows or Raspberry Pi with the
[LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/cli):

```bash
litert-lm run  \
   --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
   gemma-4-E4B-it.litertlm \
   --backend=gpu \
   --enable-speculative-decoding=true \
   --prompt="What is the capital of France?"
```

## 🌟 Key Features

-   📱 **Cross-Platform Support**: Android, iOS, Web, Desktop, and IoT (e.g.
    Raspberry Pi).
-   🚀 **Hardware Acceleration**: Peak performance via GPU and NPU accelerators.
-   👁️ **Multi-Modality**: Support for vision and audio inputs.
-   🔧 **Tool Use**: Function calling support for agentic workflows.
-   📚 **Broad Model Support**: Gemma, Llama, Phi-4, Qwen, and more.

![](./docs/api/kotlin/demo.gif)

--------------------------------------------------------------------------------

## 🚀 Production-Ready for Google's Products

LiteRT-LM powers on-device GenAI experiences in **Chrome**, **Chromebook Plus**,
**Pixel Watch**, and more.

You can also try the
[Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app to run
models immediately on your device.

**Install the app today from Google Play**                                                                                                                                                                                           | **Install the app today from App Store**
:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------:
<a href='https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery'><img alt='Get it on Google Play' height="120" src='https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png'/></a> | <a href="https://apps.apple.com/us/app/google-ai-edge-gallery/id6749645337?itscg=30200&itsct=apps_box_badge&mttnsubad=6749645337" style="display: inline-block;"> <img src="https://toolbox.marketingtools.apple.com/api/v2/badges/download-on-the-app-store/black/en-us?releaseDate=1771977600" alt="Download on the App Store" style="width: 246px; height: 90px; vertical-align: middle; object-fit: contain;" /></a>

### 📰 Blogs & Announcements

Link                                                                                                                                                                                                            | Description
:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------
[Blazing-fast on-device GenAI with LiteRT-LM](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/)                                                                                   | Unlock Gemma 4's full potential with blazing speed and incredible efficiency using newly added Swift, JavaScript, and Flutter APIs.
[Accelerating Gemma 4: faster inference with multi-token prediction drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)                                | An overview of how Multi-Token Prediction (MTP) drafters are making Gemma 4 models up to 3x faster at inference.
[Bring state-of-the-art agentic skills to the edge with Gemma 4](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)                                             | Deploy Gemma 4 in-app and across a broader range of devices with stellar performance and broad reach using LiteRT-LM.
[On-device GenAI in Chrome, Chromebook Plus and Pixel Watch](https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/)                                       | Deploy language models on wearables and browser-based platforms using LiteRT-LM at scale.
[On-device Function Calling in Google AI Edge Gallery](https://developers.googleblog.com/on-device-function-calling-in-google-ai-edge-gallery/)                                                                 | Explore how to fine-tune FunctionGemma and enable function calling capabilities powered by LiteRT-LM Tool Use APIs.
[Google AI Edge small language models, multimodality, and function calling](https://developers.googleblog.com/google-ai-edge-small-language-models-multimodality-rag-function-calling/)                         | Latest insights on RAG, multimodality, and function calling for edge language models.

--------------------------------------------------------------------------------

## 🏃 Quick Start

### 🔗 Key Links

-   👉 [Technical Overview](https://ai.google.dev/edge/litert-lm/overview)
    including performance benchmarks, model support, and more.
-   👉 [LiteRT-LM CLI Guide](https://ai.google.dev/edge/litert-lm/cli) including
    installation, getting started, and advanced usage.

### ⚡ Quick Try (No Code)

Try LiteRT-LM immediately from your terminal without writing a single line of
code using [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install litert-lm

litert-lm run \
  --from-huggingface-repo=google/gemma-3n-E2B-it-litert-lm \
  gemma-3n-E2B-it-int4 \
  --prompt="What is the capital of France?"
```

--------------------------------------------------------------------------------

### 📚 Supported Language APIs

Ready to get started? Explore our language-specific guides and setup
instructions.

Language             | Status          | Best For...             | Documentation
:------------------- | :-------------- | :---------------------- | :------------
**Python**           | ✅ Stable        | Prototyping & Scripting | [Python Guide](https://ai.google.dev/edge/litert-lm/python)
**Kotlin**           | ✅ Stable        | Android apps & JVM      | [Kotlin Guide](https://ai.google.dev/edge/litert-lm/android)
**Swift**            | 🚀 Early Preview | Native iOS & macOS      | [Swift Guide](https://ai.google.dev/edge/litert-lm/swift)
**JavaScript (web)** | 🚀 Early Preview | Browser environments    | [JavaScript Guide](https://ai.google.dev/edge/litert-lm/js)
**Flutter**          | 🚀 Community     | Cross-platform mobile   | [Flutter Guide](https://ai.google.dev/edge/litert-lm/flutter)
**C++**              | ✅ Stable        | High-performance native | [C++ Guide](https://ai.google.dev/edge/litert-lm/cpp)

#### 🏗️ Build From Source

This [guide](./docs/getting-started/build-and-run.md) shows how you can compile
LiteRT-LM from source. If you want to build the program from source, you should
checkout the stable
[![Latest Release](https://img.shields.io/github/v/release/google-ai-edge/LiteRT-LM)](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)
tag.

--------------------------------------------------------------------------------

## 📦 Releases

-   **v0.12.0**: Added early preview of Swift and Web JavaScript APIs, and
    community Flutter support. Updated LiteRT-LM CLI to have full CPU and GPU
    backend support across Linux, macOS, and Windows.
-   **v0.11.0**: Support Single Position Multi-token Prediction (MTP) for
    Gemma 4. Expand LiteRT-LM CLI to run natively on Windows with CPU and GPU
    backends.
-   **v0.10.1**: Deploy
    [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
    with stellar performance
    ([blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/))
    and introduce [LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/cli).
-   **v0.9.0**: Improvements to function calling capabilities, better app
    performance stability.
-   **v0.8.0**: Desktop GPU support and Multi-Modality.
-   **v0.7.0**: NPU acceleration for Gemma models.

For a full list of releases, see
[GitHub Releases](https://github.com/google-ai-edge/LiteRT-LM/releases).

--------------------------------------------------------------------------------
