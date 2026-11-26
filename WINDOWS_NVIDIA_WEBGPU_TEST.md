# Windows NVIDIA WebGPU / D3D12 validation

This branch is an experimental Windows stability patch for NVIDIA GPUs such as the RTX 3080. It targets a crash pattern seen after Dawn/WebGPU successfully selects an NVIDIA D3D12 adapter and LiteRT begins concurrent model-weight uploads.

## What the patch changes

`PATCH.litert_windows_nvidia_webgpu_upload` patches the LiteRT revision pinned by this LiteRT-LM workspace. On `_WIN32` only, it changes `num_threads_to_upload` to `0` before the WebGPU upload executors are created.

That means:

- D3D12/WebGPU GPU inference remains enabled.
- NVIDIA adapter selection is not disabled.
- Only the one-time model weight upload is serialized on Windows.
- Linux, Android, macOS, iOS, and Web behavior is unchanged.

This is intentionally a narrow A/B mitigation. If it fixes the Windows NVIDIA crash, it gives us a small upstreamable change. If the crash remains, the next experiment should separately investigate Windows host-mapped-pointer use rather than combining unrelated mitigations.

## Prerequisites

Use a native Windows 10/11 x64 development machine with:

1. A current NVIDIA display driver.
2. Visual Studio 2022 / Build Tools with the MSVC C++ workload.
3. Git for Windows and Git LFS.
4. Python 3.13 as required by LiteRT-LM's Windows build guide.
5. Java with `JAVA_HOME` configured.
6. Bazelisk (recommended):

```powershell
winget install --id=Bazel.Bazelisk -e
```

7. Windows long paths enabled. If needed, use a short Bazel output base such as `C:\bzl`.

## Clone the test branch

```powershell
git clone -b agent/fix-windows-nvidia-webgpu-upload https://github.com/ornab74/LiteRT-LM.git
cd LiteRT-LM
git lfs install
git lfs pull
```

## Build the patched Windows GPU bundle

From PowerShell:

```powershell
.\tools\windows\build_nvidia_webgpu_test.ps1
```

The helper:

1. Materializes the Windows Git-LFS runtime DLLs.
2. Checks out the exact LiteRT revision pinned by this LiteRT-LM workspace into a helper-owned temporary directory.
3. Applies `PATCH.litert_windows_nvidia_webgpu_upload` there.
4. Builds `//runtime/engine:litert_lm_main` with Windows GPU dynamic-link options.
5. Builds `@litert//litert/runtime/accelerators/gpu:libLiteRtWebGpuAccelerator` from the patched LiteRT source.
6. Copies the normal Windows runtime DLLs into the output bundle.
7. Replaces the bundled `libLiteRtWebGpuAccelerator.dll` with the freshly built patched DLL.

The default output is:

```text
out/windows-nvidia-webgpu/
```

## Build and immediately test a model

```powershell
.\tools\windows\build_nvidia_webgpu_test.ps1 `
  -ModelPath "C:\models\gemma-4-E2B-it.litertlm" `
  -Prompt "Reply with exactly: RTX GPU OK"
```

Or run the resulting executable manually:

```powershell
cd .\out\windows-nvidia-webgpu
.\litert_lm_main.exe `
  --backend=gpu `
  --model_path="C:\models\gemma-4-E2B-it.litertlm" `
  --input_prompt="Reply with exactly: RTX GPU OK"
```

## What counts as a successful RTX 3080 result

The important result is not merely that the executable starts. Capture the full console output and verify all of these:

1. Dawn/WebGPU discovers and selects the NVIDIA / D3D12 adapter.
2. The log contains:

```text
Windows WebGPU: disabling asynchronous weight uploads
```

3. The log then contains:

```text
# of threads to upload weights = 0
```

4. Model initialization finishes without the previous access violation / segmentation fault.
5. Prompt generation completes while `--backend=gpu` is still selected.

For additional confirmation, Windows Task Manager's GPU page can be used to observe the process using the NVIDIA GPU during inference. The console/backend result is the primary test; Task Manager is supplemental.

## Failure interpretation

### NVIDIA adapter is not discovered

That is a different failure from the upload crash. Capture the Dawn adapter enumeration logs, Windows version, NVIDIA driver version, and `nvidia-smi` output. Do not treat this patch as an adapter-discovery fix.

### The warning and `threads = 0` appear, but it still crashes

The asynchronous upload race has been removed and the failure is further downstream. Keep this patch isolated and test the next Windows-only candidate separately (notably native host-mapped-pointer behavior) so an upstream change remains attributable.

### It still says upload threads are greater than zero

The runtime is almost certainly loading an old/prebuilt `libLiteRtWebGpuAccelerator.dll`. Confirm the executable is being run from `out/windows-nvidia-webgpu` and that the freshly built DLL was copied there by the helper.

## Upstream plan

Do not open the Google upstream PR based only on compilation. First reproduce the original RTX 3080 failure, run this patched bundle on the same host/model, and save both logs. If the patched run succeeds, the upstream PR can be kept small and include the before/after Windows NVIDIA evidence plus references to the existing Windows WebGPU crash reports.
