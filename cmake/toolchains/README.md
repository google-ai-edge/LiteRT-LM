# LiteRT-LM Toolchain Infrastructure

This directory contains cross-compilation toolchain wrappers and orchestration
scripts. They bridge CMake, the Android NDK, and the Cargo/Rust ecosystem to
ensure ABI stability and strict C++20 compliance across cross-compiled builds.

--------------------------------------------------------------------------------

## Architecture Overview

Cross-compiling LiteRT-LM for Android requires coordinating two distinct
ecosystems:

1.  **The C/C++ Toolchain:** Managed via CMake and the Android NDK.
2.  **The Rust/Cargo Toolchain:** Managed via `corrosion` / `cxx` bridges.

To prevent configuration mismatches and runtime failures, the toolchain
architecture is split into two specialized components:

```
cmake/toolchains/
├── litertlm_android.toolchain.cmake   # Toolchain wrapper (injected into target builds)
└── litertlm_android.script.cmake      # Mid-flight orchestrator hook (executed between phases)

```

--------------------------------------------------------------------------------

## 1. Toolchain Wrapper (`litertlm_android.toolchain.cmake`)

This script acts as a proxy wrapper around the official Android NDK toolchain
(`$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake`). Rather than replacing
the official toolchain, it configures prerequisites, resolves Rust linker paths,
and normalizes compiler flags.

### Key Responsibilities

*   **NDK Initialization & Defaults:** Enforces consistent defaults if not
    provided by the preset:
*   `ANDROID_ABI`: Default `arm64-v8a`
*   `ANDROID_PLATFORM`: Default `android-28`
*   `ANDROID_STL`: Default `c++_shared`

*   **ABI to Rust Target Translation:** Translates Android ABI targets into
    their corresponding Rust target triples and identifies the proper NDK Clang
    executable to serve as Cargo’s cross-linker:

*   `arm64-v8a` $\rightarrow$ `aarch64-linux-android`

*   `x86_64` $\rightarrow$ `x86_64-linux-android`

*   **Rust Linker Routing:** Constructs `RUST_LINKER_PATH` pointing to the
    API-specific NDK Clang binary (e.g., `aarch64-linux-android28-clang`) and
    sets `LITERTLM_RUST_LINKER_OVERRIDE` and `CARGO_TARGET_<TARGET>_LINKER`.

*   **Optimization Capping (`-O3` $\rightarrow$ `-O2`):** Post-processes
    `CMAKE_C_FLAGS_*` and `CMAKE_CXX_FLAGS_*` to downgrade `-O3` optimization
    levels to `-O2`. This mitigates aggressive compiler vectorization bugs and
    unexpected symbol stripping common to NDK Clang when compiling complex
    static payloads.

*   **Official Toolchain Ingestion:** Includes the real NDK toolchain file
    directly via `include("${_REAL_NDK_TOOLCHAIN}")`.

--------------------------------------------------------------------------------

## 2. Orchestration Script (`litertlm_android.script.cmake`)

This script executes at the orchestrator root level (`LANGUAGES NONE`) **between
Phase 1 (`prebuild`) and Phase 2 (`litert_lm`)**. It prepares the host and
target environments before the secondary CMake build begins.

### Key Responsibilities

*   **Cargo Cross-Flag Generation:** Constructs target-specific environment
    variables for the C/C++ compilation steps triggered by Cargo during
    build-script execution (`cc-rs`):
*   Sets `LITERTLM_CCRS_CXXFLAGS_KEY` / `LITERTLM_CCRS_CXXFLAGS_VAL` (e.g.,
    `--target=aarch64-linux-android28 -std=c++20`).
*   Sets `LITERTLM_CCRS_CFLAGS_KEY` / `LITERTLM_CCRS_CFLAGS_VAL`.

*   **Cargo Dependency Pre-fetching:** Runs `cargo fetch` inside the repository
    to populate the local or user Cargo cache prior to build isolation.

*   **In-Place `cxx` Crate Hotpatching:**

*   **The Issue:** Strict C++20 builds fail when consuming certain versions of
    the standard Rust `cxx` bridge headers due to missing aliases (`using
    element_type = T;`) required by standard pointer traits.

*   **The Fix:** The script scans
    `CARGO_HOME/registry/src/*/cxx-*/include/cxx.h`, checks if the patch has
    been applied, and in-place injects the required alias directly:

```cpp
using reference = typename std::add_lvalue_reference<T>::type;
using element_type = T;

```

--------------------------------------------------------------------------------

## Orchestration Flow

```
[Host Configuration (Root CMakeLists.txt)]
       │
       ▼
[Phase 1: Prebuild ExternalProject]
       │  (Builds protoc, flatc, absl with Host Toolchain)
       ▼
[Execute: litertlm_android.script.cmake]
       │  ├── cargo fetch
       │  ├── Hotpatch ~/.cargo/registry/.../cxx.h for C++20
       │  └── Set cc-rs CFLAGS / CXXFLAGS overrides
       ▼
[Phase 2: Target ExternalProject (litert_lm)]
       │  (Injected with litertlm_android.toolchain.cmake)
       │  ├── Wraps official NDK toolchain
       │  ├── Routes Android Clang to Cargo linker
       │  └── Caps optimization at -O2
       ▼
[Final Artifact: litert_lm_main]

```
