[CmdletBinding()]
param(
  [string]$OutputDir = "out/windows-nvidia-webgpu",
  [string]$ModelPath = "",
  [string]$Prompt = "What is the capital of France?"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LiteRtRef = "b0f6c12088df229f6342f1af164caa66ffa7b010"
$LiteRtPatch = Join-Path $RepoRoot "PATCH.litert_windows_nvidia_webgpu_upload"
$OutputRoot = Join-Path $RepoRoot $OutputDir
$LiteRtRoot = Join-Path ([IO.Path]::GetTempPath()) ("litert-lm-nvidia-" + $LiteRtRef.Substring(0, 12))

function Require-Command([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $cmd) {
    throw "Required command '$Name' was not found in PATH."
  }
  return $cmd.Source
}

function Invoke-Checked([string]$Exe, [string[]]$Args, [string]$WorkingDirectory = $RepoRoot) {
  Write-Host "> $Exe $($Args -join ' ')" -ForegroundColor Cyan
  Push-Location $WorkingDirectory
  try {
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
      throw "Command failed with exit code $LASTEXITCODE: $Exe $($Args -join ' ')"
    }
  } finally {
    Pop-Location
  }
}

function Resolve-BazelOutput([string]$Bazel, [string[]]$CommonArgs, [string]$Target, [string]$Extension) {
  Push-Location $RepoRoot
  try {
    $lines = & $Bazel cquery @CommonArgs $Target "--output=files"
    if ($LASTEXITCODE -ne 0) {
      throw "bazel cquery failed for $Target"
    }
  } finally {
    Pop-Location
  }

  $candidate = $lines |
    Where-Object { $_ -and $_.Trim().ToLowerInvariant().EndsWith($Extension.ToLowerInvariant()) } |
    Select-Object -First 1
  if (-not $candidate) {
    throw "Could not resolve a $Extension output for Bazel target $Target. Output: $($lines -join '; ')"
  }

  if ([IO.Path]::IsPathRooted($candidate)) {
    return (Resolve-Path $candidate).Path
  }
  return (Resolve-Path (Join-Path $RepoRoot $candidate)).Path
}

function Assert-NotLfsPointer([string]$Path) {
  if (-not (Test-Path $Path)) {
    throw "Required runtime file is missing: $Path"
  }
  $firstLine = Get-Content $Path -TotalCount 1 -ErrorAction Stop
  if ($firstLine -eq "version https://git-lfs.github.com/spec/v1") {
    throw "Git LFS file is still a pointer: $Path. Run 'git lfs install' and 'git lfs pull'."
  }
}

$Git = Require-Command "git"
$BazelCommand = Get-Command "bazelisk" -ErrorAction SilentlyContinue
if (-not $BazelCommand) {
  $BazelCommand = Get-Command "bazel" -ErrorAction SilentlyContinue
}
if (-not $BazelCommand) {
  throw "Bazelisk/Bazel was not found. Recommended: winget install --id=Bazel.Bazelisk -e"
}
$Bazel = $BazelCommand.Source

if (-not (Test-Path $LiteRtPatch)) {
  throw "Patch not found: $LiteRtPatch"
}

Write-Host "=== LiteRT-LM Windows NVIDIA WebGPU test build ===" -ForegroundColor Green
Write-Host "LiteRT-LM repo : $RepoRoot"
Write-Host "LiteRT commit   : $LiteRtRef"
Write-Host "Patched checkout: $LiteRtRoot"
Write-Host "Output bundle   : $OutputRoot"

# Materialize the Windows runtime assets before packaging the test bundle.
Invoke-Checked $Git @("lfs", "install", "--local")
Invoke-Checked $Git @("lfs", "pull", "--include=prebuilt/windows_x86_64/*")

$PrebuiltRoot = Join-Path $RepoRoot "prebuilt\windows_x86_64"
if (-not (Test-Path $PrebuiltRoot)) {
  throw "Windows prebuilt directory was not found: $PrebuiltRoot"
}
Get-ChildItem $PrebuiltRoot -Filter "*.dll" | ForEach-Object { Assert-NotLfsPointer $_.FullName }

# Prepare an isolated checkout of the exact LiteRT revision pinned by this LiteRT-LM workspace.
# Destructive clean/reset commands are intentionally limited to this helper-owned temp directory.
if (-not (Test-Path (Join-Path $LiteRtRoot ".git"))) {
  if (Test-Path $LiteRtRoot) {
    Remove-Item -Recurse -Force $LiteRtRoot
  }
  Invoke-Checked $Git @("clone", "--filter=blob:none", "https://github.com/google-ai-edge/LiteRT.git", $LiteRtRoot) ([IO.Path]::GetTempPath())
}
Invoke-Checked $Git @("fetch", "--depth=1", "origin", $LiteRtRef) $LiteRtRoot
Invoke-Checked $Git @("checkout", "--detach", $LiteRtRef) $LiteRtRoot
Invoke-Checked $Git @("reset", "--hard", $LiteRtRef) $LiteRtRoot
Invoke-Checked $Git @("clean", "-fdx") $LiteRtRoot
Invoke-Checked $Git @("apply", "--check", $LiteRtPatch) $LiteRtRoot
Invoke-Checked $Git @("apply", $LiteRtPatch) $LiteRtRoot

$Override = $LiteRtRoot.Replace("\", "/")
$CommonArgs = @(
  "--config=windows",
  "--define=litert_runtime_link_mode=dynamic",
  "--define=resolve_symbols_in_exec=false",
  "--override_repository=litert=$Override"
)
$MainTarget = "//runtime/engine:litert_lm_main"
$AcceleratorTarget = "@litert//litert/runtime/accelerators/gpu:libLiteRtWebGpuAccelerator"

Invoke-Checked $Bazel (@("build") + $CommonArgs + @($MainTarget, $AcceleratorTarget))

$MainExe = Resolve-BazelOutput $Bazel $CommonArgs $MainTarget ".exe"
$AcceleratorDll = Resolve-BazelOutput $Bazel $CommonArgs $AcceleratorTarget ".dll"

if (Test-Path $OutputRoot) {
  Remove-Item -Recurse -Force $OutputRoot
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Copy-Item $MainExe (Join-Path $OutputRoot "litert_lm_main.exe") -Force
Copy-Item (Join-Path $PrebuiltRoot "*.dll") $OutputRoot -Force
# Always replace Google's bundled accelerator with the one built from the patched LiteRT source.
Copy-Item $AcceleratorDll (Join-Path $OutputRoot "libLiteRtWebGpuAccelerator.dll") -Force

$PatchedDll = Join-Path $OutputRoot "libLiteRtWebGpuAccelerator.dll"
Assert-NotLfsPointer $PatchedDll

Write-Host ""
Write-Host "Patched Windows GPU bundle created:" -ForegroundColor Green
Write-Host "  $OutputRoot"
Write-Host "Patched accelerator:" -ForegroundColor Green
Write-Host "  $PatchedDll"
Write-Host ""
Write-Host "Expected NVIDIA/D3D12 test log:" -ForegroundColor Yellow
Write-Host "  1. Dawn/WebGPU selects your NVIDIA adapter (RTX 3080 on the target machine)."
Write-Host "  2. 'Windows WebGPU: disabling asynchronous weight uploads ...'"
Write-Host "  3. '# of threads to upload weights = 0'"
Write-Host "  4. Model initialization and generation complete without the prior access violation/segfault."
Write-Host ""
Write-Host "This mitigation serializes weight upload only; GPU inference remains enabled." -ForegroundColor Yellow

if ($ModelPath) {
  $ResolvedModel = (Resolve-Path $ModelPath).Path
  Write-Host ""
  Write-Host "Launching RTX GPU smoke test..." -ForegroundColor Green
  $TestExe = Join-Path $OutputRoot "litert_lm_main.exe"
  Invoke-Checked $TestExe @(
    "--backend=gpu",
    "--model_path=$ResolvedModel",
    "--input_prompt=$Prompt"
  ) $OutputRoot
} else {
  Write-Host ""
  Write-Host "To run immediately with a model:" -ForegroundColor Cyan
  Write-Host ".\tools\windows\build_nvidia_webgpu_test.ps1 -ModelPath 'C:\path\to\model.litertlm'"
  Write-Host ""
  Write-Host "Or run the built bundle manually:" -ForegroundColor Cyan
  Write-Host "cd '$OutputRoot'"
  Write-Host ".\litert_lm_main.exe --backend=gpu --model_path='C:\path\to\model.litertlm' --input_prompt='Hello from the RTX 3080'"
}
