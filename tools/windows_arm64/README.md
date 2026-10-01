# Windows ARM64 build support

Build instructions:
[docs/getting-started/build-windows-arm64.md](../../docs/getting-started/build-windows-arm64.md).

What each file here is for:

| File          | Purpose                                            |
| ------------- | -------------------------------------------------- |
| `BUILD.bazel` | `:windows_arm64_clang_cl`, the platform that       |
:               : selects the clang-cl toolchain (used via          :
:               : `--config=windows_arm64`). `:npu` builds the      :
:               : runtime binaries and                              :
:               : `libLiteRtDispatch_Qualcomm.dll`.                 :

The dispatch DLL is renamed by `copy_file` because LiteRT outputs
`LiteRtDispatch_Qualcomm.dll`, but its runtime only loads dispatch libraries
whose names start with `libLiteRtDispatch` (`litert/core/dynamic_loading.h`).
