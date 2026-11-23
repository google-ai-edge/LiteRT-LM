# LiteRT-LM Python API

Python bindings for LiteRT-LM, the **production-ready** orchestration layer to
run LLMs with LiteRT, engineered for **high-performance**, **cross-platform**
execution.

- **Documentation**: [https://ai.google.dev/edge/litert-lm/python](https://ai.google.dev/edge/litert-lm/python)
- **GitHub**: [https://github.com/google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)

## Supported platforms

-   Linux (x86_64, ARM64)
-   macOS (Apple Silicon)
-   Windows (x86_64, ARM64)
-   Android (ARM64, x86_64)

Python 3.10 or later is required. On Windows ARM64 (e.g. Snapdragon), use a
native ARM64 Python, which is available for Python 3.11 or later. An x64 Python
running under emulation installs the x86_64 wheel instead.

## NPU

`litert_lm.Backend.NPU()` is supported on Windows. The LiteRT dispatch library
is bundled in the wheel, but the vendor runtime has to be installed separately:

-   **Windows x86_64 (Intel)**: install OpenVINO with `pip install openvino`.
-   **Windows ARM64 (Qualcomm Snapdragon)**: install the Qualcomm AI Runtime
    (QNN) libraries with `pip install ai-edge-litert-sdk-qualcomm`.
    Alternatively, add the `lib\aarch64-windows-msvc` directory of a Qualcomm AI
    Runtime SDK to `PATH` and its `lib\hexagon-v*\unsigned` directories to
    `ADSP_LIBRARY_PATH`.
