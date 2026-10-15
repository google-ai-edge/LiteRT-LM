# Building on Windows ARM64

This guide covers building LiteRT-LM on **Windows on ARM** (for example
Snapdragon X Elite / X Plus laptops), including running models on the Qualcomm
Hexagon NPU.

The general [Deploy to Windows](./build-and-run.md#deploy_to_windows)
instructions target x86-64. On ARM64 they do not work as written, for one
reason: MSVC's `cl.exe` cannot compile this project for ARM64. Its
`arm64_neon.h` has no bf16 intrinsics (`vcvt_bf16_f32`, `vcombine_bf16`,
`vbfmmlaq_f32`, ...), and XNNPACK's ARM kernels need them. The ARM64 build
therefore uses **`clang-cl`**. It still uses the MSVC ABI, headers, libraries
and command-line flags; only the compiler binary changes.

Tested with Visual Studio (MSVC ARM64 tools), LLVM/clang-cl (18–20), Bazel 7.6.1
and QAIRT 2.49 on a Snapdragon X Elite (SC8380XP).

-   [Prerequisites](#prerequisites)
-   [One-time setup](#one-time-setup)
-   [Build](#build)
-   [Run on CPU](#run-on-cpu)
-   [Run on the NPU](#run-on-the-npu)
-   [Troubleshooting](#troubleshooting)
-   [Not supported](#not-supported)

## Prerequisites

Everything from the
[Windows prerequisites](./build-and-run.md#deploy_to_windows) (Git, Python,
Bazelisk, Java, long paths), with these differences:

1.  **Visual Studio** (2022 or newer) with:
    *   *MSVC ... ARM64/ARM64EC build tools*
2.  **LLVM/Clang (version 18, 19, or 20)**: Standalone LLVM for Windows on ARM64
    via `winget install LLVM.LLVM --version 20.1.8` (or from LLVM GitHub releases).
    *Note: A plain `winget install LLVM.LLVM` installs LLVM 23+, which is untested;
    pinning to `--version 20.1.8` is recommended. Visual Studio's bundled Clang
    adds Git commit URLs to its version string, which causes Bazel's toolchain
    generator to misparse Clang's built-in include directory. Standalone LLVM
    provides clean versioning that Bazel parses correctly.*
3.  **Git for Windows**, the full install or `PortableGit`. Bazel needs its
    `bash.exe` to run genrules. *MinGit* has no bash and will not work.
4.  **Developer Mode**, under Settings → System → For developers. The TensorFlow
    source archive contains symlinks, and without Developer Mode creating them
    requires administrator rights, so the fetch fails.
5.  **Bazel for windows-arm64.** Bazelisk downloads the native ARM64 build of
    the version in `.bazelversion`.
6.  **Short paths.** If you cannot enable `LongPathsEnabled`, keep the checkout
    path short (for example `C:\src\LiteRT-LM`) and pass a short `--output_base`
    such as `C:\bz`, as in all commands below.
7.  For the NPU only: the **Qualcomm AI Runtime (QAIRT) SDK 2.49**. Running on
    the NPU needs its DLLs (see [Run on the NPU](#run-on-the-npu)). The build
    downloads its own copy, unless you set `LITERT_QAIRT_SDK` so that it uses
    yours.

## Environment setup

Run these in PowerShell before building. Replace `<VS>` with your Visual Studio
install path (for example `C:\Program Files\Microsoft Visual
Studio\2022\Community`) and set `BAZEL_LLVM` to your standalone LLVM
installation:

```powershell
# Imports the ARM64 MSVC environment (INCLUDE, LIB, PATH) into PowerShell.
cmd /c "`"<VS>\VC\Auxiliary\Build\vcvarsarm64.bat`" >nul 2>&1 && set" |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:$($matches[1])" $matches[2] } }

$env:BAZEL_VC   = "<VS>\VC"
$env:BAZEL_LLVM = "C:\Program Files\LLVM"
$env:BAZEL_SH   = "C:\Program Files\Git\bin\bash.exe"

# Optional: use a QAIRT SDK already on disk instead of downloading it.
$env:LITERT_QAIRT_SDK = "C:\path\to\qairt\2.49.0.260730"
```

> **Tip:** If you prefer not passing `--config=windows_arm64` on every command,
> you can create a `.bazelrc.user` in the repository root containing:
> ```
> build --config=windows_arm64
> ```

## Build

```powershell
bazelisk --output_base=C:\bz build --config=windows_arm64 //tools/windows_arm64:npu
```

This builds:

| Output                                         | Path under `bazel-bin`  |
| ---------------------------------------------- | ----------------------- |
| `litert_lm_main.exe`                           | `runtime\engine\ `      |
| `litert_lm_advanced_main.exe`                  | `runtime\engine\ `      |
| `libLiteRtDispatch_Qualcomm.dll` (NPU dispatch | `tools\windows_arm64\ ` |
: library)                                       :                         :

Look up the output directory with:

```powershell
$bin = bazelisk --output_base=C:\bz info bazel-bin
```

If you only need CPU, build `//runtime/engine:litert_lm_main` instead:

```powershell
bazelisk --output_base=C:\bz build --config=windows_arm64 //runtime/engine:litert_lm_main
```
It does not need the QAIRT SDK.

## Run on CPU

```powershell
& "$bin\runtime\engine\litert_lm_main.exe" `
    --backend=cpu `
    --model_path="C:\models\<model_name>.litertlm"
```

## Run on the NPU

This needs a model whose text decoder was compiled for your SoC's Hexagon NPU,
for example `*_npu_text_SC8380XP.litertlm` for the Snapdragon X Elite.

Besides the Bazel outputs, the NPU needs Qualcomm's QNN runtime at run time. It
is a set of prebuilt DLLs, `QnnHtp.dll` and others, plus the Hexagon "skel"
libraries, all shipped only as binaries in the QAIRT SDK.

```powershell
$qairt = "C:\path\to\qairt\2.49.0.260730"

# QNN runtime DLLs, and the Hexagon skels for your SoC (v73 = Snapdragon X Elite).
$env:PATH              = "$qairt\lib\aarch64-windows-msvc;$env:PATH"
$env:ADSP_LIBRARY_PATH = "$qairt\lib\hexagon-v73\unsigned"

# The dispatch library must be in the same directory as the model.
Copy-Item "$bin\tools\windows_arm64\libLiteRtDispatch_Qualcomm.dll" C:\models\

& "$bin\runtime\engine\litert_lm_advanced_main.exe" `
    --model_path="C:\models\<model_name>_npu_text_SC8380XP.litertlm" `
    --backend=npu --vision_backend=cpu --audio_backend=cpu `
    --input_prompt="Describe this image. [image:C:\path\to\apple.png]"
```

Notes:

*   **The dispatch library is looked up in the model file's directory.**
    `litert_lm_main` and `litert_lm_advanced_main` have no flag to change this.
    If you don't want to copy files next to a large model, you can instead
    create a hardlink to the model in the dispatch library's directory.
*   **`ADSP_LIBRARY_PATH` is required**, and it must point at the skels that
    match your SoC's Hexagon version.
*   **Images and audio are given inline in the prompt**, as `[image:<path>]` and
    `[audio:<path>]`. There are no separate flags for them. `--vision_backend`
    and `--audio_backend` choose where each encoder runs.

## Troubleshooting

| Symptom                              | Cause and fix                         |
| ------------------------------------ | ------------------------------------- |
| The fetch fails while extracting     | Developer Mode is off. Enable it,     |
: `org_tensorflow` with a symlink or   : then fetch again.                     :
: permission error.                    :                                       :
| Genrules fail with `bash` not found, | `--shell_executable` or `BAZEL_SH` is |
: or `/bin/bash` errors.               : not set, or points at MinGit. Point   :
:                                      : both at a full Git for Windows        :
:                                      : `bash.exe`.                           :
| Compile errors with MSVC codes such  | Bazel is compiling with `cl.exe`, not |
: as `C2440` or `C3861` in XNNPACK     : clang-cl. Check that                  :
: `bf16` sources, or                   : `--config=windows_arm64` is passed or :
: `__builtin_expect` not found.        : set in `.bazelrc.user`.               :
| `absolute path inclusion(s) found in | Bazel's toolchain generator misparsed |
: rule ...`                            : the Clang version string (common with :
:                                      : Visual Studio's bundled Clang or LLVM :
:                                      : 23+). Install standalone LLVM 20.1.8  :
:                                      : via `winget install LLVM.LLVM --version 20.1.8` :
:                                      : and point `$env\:BAZEL_LLVM` to it    :
:                                      : (e.g. `C\:\Program Files\LLVM`).      :
| `No dispatch library found in <dir>` | `libLiteRtDispatch_Qualcomm.dll` is   |
:                                      : not in the model's directory.         :
| `Can't read future blob ... 3.3.x vs | Misleading. QNN could not load the    |
: 4.0.x`                               : Hexagon skel because                  :
:                                      : `ADSP_LIBRARY_PATH` is missing or     :
:                                      : wrong, and the fallback path it uses  :
:                                      : cannot read newer models. Point it at :
:                                      : `<qairt>\lib\hexagon-v<N>\unsigned`.  :
| `FastRPC buffer is not supported`    | The dispatch DLL was built without    |
:                                      : host-memory (raw tensor) support on   :
:                                      : Windows. Ensure LiteRT includes raw   :
:                                      : tensor memory support for Windows and :
:                                      : use the dispatch library built by     :
:                                      : `//tools/windows_arm64\:npu`.         :

## Not supported

**GPU.** LiteRT loads GPU support from a prebuilt accelerator library, and
`prebuilt/` has no `windows_arm64` version of it. The GPU backend cannot be
built from source either, as its source is not distributed with LiteRT. For
multimodal models, running vision on the CPU with `--vision_backend=cpu` works
well.
