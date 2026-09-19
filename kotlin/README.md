This document provides an overview of the test coverage and benchmark coverage
for the **LiteRT-LM Kotlin API**.

> [!NOTE] For the Kotlin API developer guide and getting started documentation,
> see [Getting Started Guide](../docs/api/kotlin/getting_started.md). \
> For detailed instructions on running device tests and performance benchmarks,
> refer to: - [Android Device Testing Guide](g3doc/device_test.md) -
> [Android Benchmarking Guide](g3doc/benchmark.md)

--------------------------------------------------------------------------------

## 1. Host JVM Unit Test Coverage

Host JVM unit tests execute locally or in hermetic build environments without
needing an Android emulator or physical device. They cover core API bindings,
serialization, data classes, conversation management, tool use schema
generation, and prompt template processing.

Target Name                        | Test Source File        | Component / Feature Tested                                               | Test Data / Models            | Bazel / Blaze Command
:--------------------------------- | :---------------------- | :----------------------------------------------------------------------- | :---------------------------- | :--------------------
`:CapabilitiesTest`                | `CapabilitiesTest.kt`   | Model capabilities extraction & config flags                             | `runtime/testdata`            | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:CapabilitiesTest`
`:JsonConvertersTest`              | `JsonConvertersTest.kt` | JSON serialization/deserialization for tools & messages                  | Unit mock JSON data           | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:JsonConvertersTest`
`:MessageTest`                     | `MessageTest.kt`        | Message content types (Text, ImageBytes, AudioBytes, Tool Call/Response) | Mock content data             | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:MessageTest`
`:ToolTest`                        | `ToolTest.kt`           | Tool declaration, `@Tool` / `@ToolParam` reflection & OpenAPI conversion | OpenAPI test schemas          | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:ToolTest`
`:ConversationTest`                | `ConversationTest.kt`   | Conversation turn management, system instructions & streaming Flow       | `runtime/testdata`            | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:ConversationTest`
`:SessionTest`                     | `SessionTest.kt`        | Low-level C JNI session wrapper lifecycle                                | `runtime/testdata`            | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:SessionTest`
`:BenchmarkTest`                   | `BenchmarkTest.kt`      | Metric logging & threshold bisection calculations                        | `runtime/testdata`            | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:BenchmarkTest`
`:Gemma4TemplateTest`              | `Gemma4TemplateTest.kt` | Gemma 4 prompt template rendering & Jinja context variables              | `runtime/components/testdata` | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:Gemma4TemplateTest`
`:IntegrationTest`                 | `IntegrationTest.kt`    | Full JVM E2E integration with real model                                 | Gemma 3 1B Int4               | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:IntegrationTest`
`:IntegrationNvidiaGpuArtisanTest` | `IntegrationTest.kt`    | E2E JVM GPU acceleration on Linux                                        | Gemma 3 1B HW                 | `bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:IntegrationNvidiaGpuArtisanTest`
`:IntegrationMacosGpuArtisanTest`  | `IntegrationTest.kt`    | E2E JVM GPU acceleration on macOS (Metal)                                | Gemma 3 1B HW                 | `bazel test -c opt --config=darwin_arm64 //kotlin/javatests/com/google/ai/edge/litertlm:IntegrationMacosGpuArtisanTest`

--------------------------------------------------------------------------------

## 2. Android Device Integration Test Coverage

Device integration tests validate LiteRT-LM Kotlin API functionality on real
Android devices (or emulators) across CPU, GPU, NPU, and TPU backends.

Model Family                        | Test Target Name(s)                                                                                                                                                                                                                                                                                | Backends Supported | Target Devices / Chipsets                                                | Modalities & Features Tested
:---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------- | :----------------------------------------------------------------------- | :---------------------------
**Gemma 3 (1B)**                    | `:Gemma3DeviceTest`<br>`:Gemma3_1bDeviceTest_QUALCOMM_SM8750_V79`<br>`:Gemma3DeviceTest_QUALCOMM_SM8650_V75`                                                                                                                                                                                       | CPU, GPU, NPU      | Samsung S25 Ultra, Pixel 9, S24 Exynos, Samsung S24                      | Text generation, conversation turns
**Gemma 3n (E2B)**                  | `:Gemma3nE2bDeviceTest`<br>`:Gemma3nE2bDeviceTest_GpuAudio`<br>`:Gemma3nDeviceTest_MEDIATEK_MT6993`                                                                                                                                                                                                | CPU, GPU, NPU      | S25 Ultra, Pixel 8, Pixel 9, S24 Exynos, Xiaomi 17T Pro                  | Multi-modality (Text, Vision, Audio processing)
**Gemma 4 (2.3B / E2B)**            | `:Gemma42bDeviceTest`<br>`:E2bExtWeightsDeviceTest`<br>`:Gemma42bLoraDeviceTest`<br>`:Gemma4DeviceTest_QUALCOMM_SM8850_V81_model`<br>`:Gemma4DeviceTest_QUALCOMM_SM8850_V81`<br>`:Gemma4DeviceTest_with_Mtp_On/Off_QUALCOMM_SM8850_V81`<br>`:Gemma4TPUDeviceTest_with_Mtp_On/Off_GOOGLE_TENSOR_G5` | CPU, GPU, NPU, TPU | S25 Ultra, Pixel 8, Pixel 9, S24 Exynos, Samsung S26 Ultra, Pixel 10 Pro | Multi-modality, External per-layer weights, LoRA adapters, Speculative Decoding (MTP)
**Function Gemma (1B)**             | `:FunctionGemmaDeviceTest`                                                                                                                                                                                                                                                                         | GPU                | Samsung S25 Ultra, Pixel 8                                               | Function calling, OpenAPI tool execution
**Kanana (1.07B)**                  | `:Kanana107Int4DeviceTest`                                                                                                                                                                                                                                                                         | GPU                | Samsung S25 Ultra, Samsung S24 Exynos                                    | Int4 text generation, Kakao architecture
**Qwen 2.5 (1.5B)**                 | `:Qwen2.5-1.5bDeviceTest`                                                                                                                                                                                                                                                                          | CPU                | Default CPU Devices                                                      | Instruct text generation
**Qwen 3 (4B)**                     | `:Qwen3_4bDeviceTest`                                                                                                                                                                                                                                                                              | GPU                | Samsung S25 Ultra                                                        | 4B param text generation
**MiniCPM5 (1B)**                   | `:MiniCpm5_1bDeviceTest`                                                                                                                                                                                                                                                                           | GPU                | Samsung S25 Ultra                                                        | Multilingual text generation
**Qed-Nano**                        | `:QedNanoDeviceTest`                                                                                                                                                                                                                                                                               | GPU                | Samsung S25 Ultra                                                        | Compact text generation
**Kumru 2B**                        | `:Kumru2bDeviceTest`                                                                                                                                                                                                                                                                               | GPU                | Samsung S25 Ultra                                                        | 2B text generation
**Prem 1B SQL**                     | `:Prem1bSqlDeviceTest`                                                                                                                                                                                                                                                                             | GPU                | Samsung S25 Ultra                                                        | Text-to-SQL generation
**Liquid AI LFM 2.5 (1.2B / 450M)** | `:Lfm2512bDeviceTest`<br>`:LfmDeviceTest_SAMSUNG_S24_EXYNOS`                                                                                                                                                                                                                                       | GPU, NPU           | Samsung S25 Ultra, Samsung S24 Exynos                                    | Text & Vision multi-modality on Samsung NPU
**FastVLM**                         | `:FastVlmDeviceTest_QUALCOMM_SM8750_V79`<br>`:FastVlmDeviceTest_QUALCOMM_SM8850_V81`                                                                                                                                                                                                               | NPU                | Samsung S25 Ultra, Samsung S26 Ultra                                     | Ultra-fast Vision-Language inference on Qualcomm NPU
**TinyGemma (270M)**                | `:TinyGemmaDeviceTest_QUALCOMM_SM8750_V79`                                                                                                                                                                                                                                                         | NPU                | Samsung S25 Ultra                                                        | Low-footprint text generation
**Constrained Decoding**            | `:ConstrainedDecoding_QUALCOMM_SM8750_V79`                                                                                                                                                                                                                                                         | NPU + CPU          | Samsung S25 Ultra                                                        | FST grammar constraints & logit masking on NPU forward pass
**Environment Verification**        | `:EnvironmentDeviceTest`<br>`:EnvironmentDeviceTest_QUALCOMM_SM8750_V79`                                                                                                                                                                                                                           | CPU, NPU           | Generic Devices, Samsung S25 Ultra                                       | Device environment & system library verification

--------------------------------------------------------------------------------

## 3. Android Benchmark Coverage

Performance benchmark suites measure latency, throughput, and memory consumption
across model families and hardware accelerators. Results are uploaded
continuously to
[go/litert-lm-android-benchmark](http://go/litert-lm-android-benchmark).

Model Family             | Benchmark Target Suite                                                                                                                                                                       | Backends Supported | Target Devices & Hardware                            | Benchmark Workloads & Metrics Tracked
:----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------- | :--------------------------------------------------- | :------------------------------------
**Gemma 3 (1B)**         | `:Gemma3BenchmarkTest`<br>`:Gemma3NPUBenchmarkTest`<br>`:Gemma3NPUBenchmarkV75Test`<br>`:Gemma3NPUBenchmarkWhhoneTest`                                                                       | CPU, GPU, NPU      | Samsung S25 Ultra, Pixel 9, S24 Exynos, Samsung S24  | TTFT, Prefill (tok/s), Decode (tok/s), Peak RSS/PSS, Init Time
**Gemma 3n (E2B)**       | `:Gemma3nBenchmarkTest`                                                                                                                                                                      | CPU, GPU           | Samsung S25 Ultra, Pixel 9, Samsung S24 Exynos       | Multi-modal Prefill & Decode performance
**Gemma 4 (2.3B / E2B)** | `:Gemma4BenchmarkTest`<br>`:Gemma4NPUBenchmarkTest`<br>`:Gemma4ConstrainedNPUBenchmarkTest`<br>`:Gemma4MtpNPUBenchmarkTest`<br>`:Gemma4TPUBenchmarkTest`<br>`:Gemma4TPUDrafterBenchmarkTest` | CPU, GPU, NPU, TPU | Samsung S25 Ultra, Pixel 9, S24 Exynos, Pixel 10 Pro | Standard execution, Grammar-constrained decoding overhead, Speculative decoding (MTP/Drafter) speedups
**Kanana (1.07B)**       | `:Kakao107Int4BenchmarkTest`                                                                                                                                                                 | CPU, GPU           | Samsung S25 Ultra, Pixel 9, S24 Exynos, Samsung S22  | Fixed prefill (512 tokens) and decode (128 tokens) latency
**TinyGemma (270M)**     | `:TinyGemmaNPUBenchmarkTest`                                                                                                                                                                 | NPU                | Samsung S25 Ultra                                    | Micro-model NPU latency & throughput

--------------------------------------------------------------------------------

## 4. Hardware Accelerator & Device Support Matrix

The table below summarizes the hardware platforms, vendor runtimes, target
devices, and execution backends validated by the test suite:

| Chipset /    | Vendor       | Representative | Supported     | Tested Models |
: SoC Family   : Runtime      : Target Device  : Backends      : & Features    :
:              : Library      :                :               :               :
| :----------- | :----------- | :------------- | :------------ | :------------ |
| **Snapdragon | Qualcomm NPU | Samsung S26    | NPU, GPU, CPU | Gemma 4 2B,   |
: 8 Elite Gen  : Runtime v81  : Ultra          :               : FastVLM,      :
: 5 (SM8850)** :              : Satellite      :               : Speculative   :
:              :              :                :               : Decoding      :
:              :              :                :               : (MTP)         :
| **Snapdragon | Qualcomm NPU | Samsung S25    | NPU, GPU, CPU | Gemma 3 1B,   |
: 8 Elite      : Runtime v79  : Ultra          :               : Gemma 4 2B,   :
: (SM8750)**   :              :                :               : FastVLM,      :
:              :              :                :               : TinyGemma,    :
:              :              :                :               : Constrained   :
:              :              :                :               : Decoding      :
| **Snapdragon | Qualcomm NPU | Samsung S24    | NPU, GPU, CPU | Gemma 3 1B    |
: 8 Gen 3      : Runtime v75  :                :               : NPU           :
: (SM8650)**   :              :                :               :               :
| **Snapdragon | OpenCL GPU   | Samsung S22    | GPU, CPU      | Kanana 107    |
: 8 Gen 1      :              :                :               : Int4 GPU      :
: (SM8450)**   :              :                :               :               :
| **Google     | Google       | Pixel 10 Pro   | TPU           | Gemma 4 E2B   |
: Tensor G5**  : Tensor TPU   :                :               : TPU,          :
:              : Runtime      :                :               : Speculative   :
:              :              :                :               : Drafter       :
| **Google     | OpenCL GPU / | Pixel 9 /      | GPU, CPU      | Gemma 3,      |
: Tensor G4 /  : ARM CPU      : Pixel 8        :               : Gemma 3n,     :
: G3**         :              :                :               : Gemma 4,      :
:              :              :                :               : Function      :
:              :              :                :               : Gemma         :
| **Exynos     | Samsung NPU  | Samsung S24    | NPU, GPU, CPU | Gemma 3,      |
: 2400**       : / Xclipse    : Exynos         :               : Gemma 3n,     :
:              : 940 GPU      :                :               : Gemma 4,      :
:              :              :                :               : Kanana, LFM   :
:              :              :                :               : 2.5           :
| **MediaTek   | MediaTek NPU | Xiaomi 17T Pro | NPU           | Gemma 3n Full |
: MT6993**     : Runtime v8   : Satellite      :               : Modality      :
:              :              :                :               : (Audio +      :
:              :              :                :               : Vision NPU)   :

--------------------------------------------------------------------------------

## 5. Continuous Integration Suites

LiteRT-LM uses TAP and Guitar for continuous testing and benchmark regression
alerting:

-   **Integration Test Suite Target**: `:all_device_tests` \
    Executes device integration tests continuously via `guitar_workflow_test`.
-   **Benchmark Suite Target**: `:all_benchmark_tests` \
    Executes performance benchmarks via `guitar_workflow_benchmark_test` and
    uploads metrics to Perfgate.

--------------------------------------------------------------------------------

## 6. How to Add a New Test (High-Level Guide)

This section outlines the high-level workflow for adding new tests or benchmarks
to the LiteRT-LM Kotlin codebase.

### A. Adding a Host JVM Unit Test

Host unit tests validate Kotlin API logic, JNI bindings, and prompt templates
without requiring Android hardware.

1.  **Create/Edit Kotlin Test File**:

    -   Place your test file in
        `kotlin/javatests/com/google/ai/edge/litertlm/`
        (e.g. `MyFeatureTest.kt`).
    -   Write standard JUnit 4 tests using Google Truth assertions
        (`assertThat(...)`).

2.  **Add Target in BUILD**:

    -   Add a `kt_jvm_test` rule in
        `kotlin/javatests/com/google/ai/edge/litertlm/BUILD`:

        ```bzl
        kt_jvm_test(
            name = "MyFeatureTest",
            srcs = ["MyFeatureTest.kt"],
            data = ["//runtime/testdata"],
            runtime_deps = ["//kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni"],
            deps = [
                "//third_party/java/junit",
                "//third_party/java/truth",
                "//kotlin/java/com/google/ai/edge/litertlm:litertlm-jvm",
            ],
        )
        ```

3.  **Verify Locally**:

    ```shell
    bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:MyFeatureTest
    ```

--------------------------------------------------------------------------------

### B. Adding an Android Device Integration Test

Device tests run instrumentation test cases on physical devices or emulators
using CPU, GPU, NPU, or TPU backends.

1.  **Lookup Model Path & Confirm Permissions**:

    -   Inspect model configurations in
        `runtime/engine/models.sh`.
    -   Verify the Guitar LOAS user (`mobileiq-gemini-guitar-jobs`) has read
        access to the CNS model path:

        ```shell
        /google/bin/releases/lhoss-security-team/tools/explainacls.par \
            --loas_user=mobileiq-gemini-guitar-jobs --path=<MODEL_PATH>
        ```

2.  **Register Device Definition (if needed)**:

    -   Ensure the target device is defined in
        `kotlin/javatests/com/google/ai/edge/litertlm/BUILD`
        using `shared_lab_android_device` or `satellite_lab_device`.

3.  **Declare Device Test Rule**:

    -   In
        `kotlin/javatests/com/google/ai/edge/litertlm/BUILD`,
        declare the macro:
        -   **For CPU/GPU**: Use `litertlm_device_test`.
        -   **For NPU/TPU**: Use `litertlm_npu_device_test` with required vendor
            runtime dependencies (e.g. `qualcomm_npu_runtime_v79` or
            `google_tensor_runtime`).
    -   Example:

        ```bzl
        litertlm_device_test(
            name = "MyModelDeviceTest",
            srcs = ["DeviceTest.kt"],
            model_path = "/cns/path/to/model.litertlm",
        )
        ```

4.  **Add to Continuous Integration Suite**:

    -   Include your new target label (e.g. `:MyModelDeviceTest`) in the `tests`
        list of `test_suite(name = "all_device_tests", ...)` in the `BUILD`
        file.

5.  **Run & Verify**:

    ```shell
    bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm:MyModelDeviceTest \
      --notest_loasd \
      --android_platforms=//buildenv/platforms/android:arm64-v8a \
      --android_ndk_min_sdk_version=26
    ```

--------------------------------------------------------------------------------

### C. Adding an Android Performance Benchmark

Performance benchmarks evaluate prefill/decode speeds, time-to-first-token, and
memory usage across device hardware.

1.  **Update `benchmark.bzl`**:

    -   In
        `kotlin/javatests/com/google/ai/edge/litertlm/benchmark/benchmark.bzl`,
        define the model struct:

        ```python
        MY_MODEL = _model(
            id = "my_model",
            path = "/cns/path/to/model.litertlm",
            extra_args = ",prefillTokens=512,decodeTokens=128",  # Optional
        )
        ```

    -   Add target device labels if not already listed.

2.  **Declare Benchmark Target in `benchmark/BUILD`**:

    -   In
        `kotlin/javatests/com/google/ai/edge/litertlm/benchmark/BUILD`:
        -   **For CPU/GPU**: Use `litert_lm_benchmark(name =
            "MyModelBenchmarkTest", model = MY_MODEL)`.
        -   **For NPU/TPU**: Use `litert_lm_npu_benchmark(name =
            "MyModelNPUBenchmarkTest", devices = [SAMSUNG_S25_ULTRA], model =
            MY_MODEL, npu_deps = [...])`.

3.  **Register in Continuous Benchmark Suite**:

    -   Add the target label (e.g., `:MyModelBenchmarkTest`) to
        `all_benchmark_tests` `test_suite` in `benchmark/BUILD` so metrics are
        automatically collected and uploaded to Perfgate on Guitar CI runs.

4.  **Run Benchmark Locally / Remotely**:

    ```shell
    bazel test -c opt //kotlin/javatests/com/google/ai/edge/litertlm/benchmark:MyModelBenchmarkTest \
      --notest_loasd \
      --android_platforms=//buildenv/platforms/android:arm64-v8a \
      --android_ndk_min_sdk_version=26
    ```
