# LiteRT-LM: CMake Overview

The LiteRT-LM CMake build system provides a unified infrastructure for building
all required third-party dependencies, internal libraries, and the primary
`litert_lm_main` executable. It uses a Super-Build architecture to isolate the
build environment from the host machine and manage the integration of the
necessary C++, Rust, and Python toolchains. While the infrastructure contains
foundational support for macOS and Windows, current development and validation
are focused exclusively on native Linux builds and Android cross-compilation.
This strict environmental control enforces One Definition Rule (ODR) adherence
to resolve ABI conflicts across a complex dependency tree.

## Dependency Management & Project Structure

This project implements a Super-Build pattern to ensure One Definition Rule
(ODR) adherence. This approach is necessary to manage a converging dependency
tree where multiple components rely on different versions of the same core
libraries.

### Dependency Strategy

The build system leverages a hybrid approach to dependency management:

-   **FetchContent**: Used for conventional third-party libraries where
    integration is sufficient
-   **ExternalProject**: Reserved for dependencies requiring heavy modification.
    These are orchestrated to redirect include and library paths to a unified
    "source of truth" within the build directory. This prevents the symbol
    collisions that occur when multiple dependencies introduce conflicting
    versions of the same provider (e.g., Abseil or Protobuf).

### Package Infrastructure

Orchestration logic for external dependencies is modularized within
`cmake/packages/<name>`. To ensure a hermetic build environment and strict ODR
adherence, these modules implement a Source Transformation and Target Mapping
framework.

This framework does not simply wrap dependencies; it actively transforms them by
"de-nesting" internal third-party code and normalizing source-level paths to
align with the LiteRT-LM unified build structure.

```bash
cmake/packages/sentencepiece
├── sentencepiece.cmake                      # Primary orchestration module (ExternalProject_Add)
├── sentencepiece_config.cmake               # Defines package paths, versions, and configuration variables
├── sentencepiece_patcher.cmake              # Source-level transformation (Path normalization and dependency de-nesting)
├── sentencepiece_root_shim.cmake            # Injected logic for the package-level configuration
├── sentencepiece_src_shim.cmake             # Injected logic for the source-level build definitions
├── sentencepiece_aggregate.cmake            # Logic to consolidate build artifacts into a unified interface
├── sentencepiece_litert_lm_target_map.cmake # Phase-specific dictionary mapping internal targets to local archives
└── sentencepiece_target_map.cmake           # (Deprecated) Legacy un-phased target map
```

#### Transformation Highlights

-   **Hermeticity**: By removing nested third_party directories within
    dependencies, we force all components to resolve a single, verified version
    of core libraries (e.g., Abseil, Protobuf).

-   **Path Normalization**: Source files are patched in-place to canonicalize
    include paths, ensuring compatibility with the standalone CMake layout.

-   **Phase-Specific Target Redirection**: Using custom mapping logic generated
    on-the-fly, internal targets are transparently redirected to local static
    archives. These maps are generated per orchestration phase (e.g., `prebuild`
    vs. `litert_lm`), maintaining consistency across the dual-stage build
    without requiring a full monorepo environment.

### Project Layout

To maintain parity with the internal codebase and facilitate automated
maintenance, LiteRT-LM modeled its CMake target definitions to mirror the source
tree.

-   **Source-Locality**: Target definitions for LiteRT-LM components generally
    reside in the CMakeLists.txt file located in the same directory as their
    respective source files.

-   **Exceptions**: Shared resources (such as proto_lib) are consolidated into
    centralized configuration files to manage global visibility and reuse.

## Build Architecture & Compilation Pipeline

### Two-Stage Super-Build Architecture

The root CMakeLists.txt does not compile C++ code directly. Instead, it acts
purely as an orchestrator (LANGUAGES NONE), dividing the build into two strictly
isolated phases using ExternalProject_Add:

-   **Phase 1: Prebuild (Host Tools)**: Compiles build-time dependencies (e.g.,
    Abseil, GTest) and essential code generators (Protobuf's protoc,
    FlatBuffers' flatc) using the host machine's native compiler.

-   **Phase 2: Target Build (litert_lm)**: The primary C++ build phase. It
    receives the paths to the Phase 1 host tools, applies the target toolchain
    (e.g., Android NDK or local host), and handles the final compilation of the
    runtime and engine.

### Strict Hermeticity & Environment Isolation

To prevent the classic "works on my machine" failure and protect against
host-environment drift, the build enforces strict boundaries:

-   **Isolated Python Environments**: The orchestrator automatically bootstraps
    a hermetic Python 3.13 virtual environment using uv. CMake is forced to use
    this sandbox for both the interpreter (code generation) and C-API
    development headers, ensuring native Python bindings compile consistently.

-   **Cross-Compilation Safety**: The system dynamically toggles Python
    components based on the target. When cross-compiling (e.g., for Android), it
    strips the host C-API headers to prevent architecture mismatches while
    retaining the interpreter for build scripts.

### Advanced Linker & ABI Management

Linking massive, statically compiled frameworks like TensorFlow Lite alongside
third-party libraries often results in cyclical dependencies and symbol
stripping. This system utilizes a highly robust linker strategy:

-   **Global Target Registries**: Libraries are aggregated into flattened
    payloads via global CMake properties, bypassing fragile, manually sorted
    dependency graphs.

-   **Graph Flattening**: The final binary (litert_lm_main) wraps the entire
    dependency payload in --start-group / --end-group (and platform
    equivalents), forcing the linker to recursively resolve symbols and
    neutralizing static library ordering issues.

-   **Static Registration Enforcement**: Core ODML components and custom
    operators rely heavily on static initialization. The build uses
    --whole-archive to prevent the linker from aggressively stripping
    unreferenced ops, preventing runtime crashes.

-   **Platform-Aware ABI**: Linker flags are dynamically mapped across Apple
    (Mach-O), Android (Bionic), Linux (ELF), and Windows (MSVC / GNU ABI).
    Multiple-definition flags (e.g., -Wl,--allow-multiple-definition) are
    applied to safely merge converging symbols from shared dependencies like
    Abseil and Protobuf.

### Multi-Language & Code Generation Integration

LiteRT-LM integrates C++, Python, and Rust, requiring precise build-order
synchronization:

-   **Schema Generation Gates**: The build automates flatc and protoc execution
    to generate C++ headers. A strict generator_complete target acts as a
    compiler gate, guaranteeing no C++ translation unit is compiled before its
    generated headers exist on disk.

-   **Rust Integration**: Rust components are orchestrated via a CXX bridge
    (litertlm_cxx_bridge), with Cargo environment variables and linker overrides
    injected seamlessly from the CMake orchestrator.

## Build Guide

This project targets a modern high-performance C++ environment. Currently, the
build system is strictly verified for the clang toolchain on Debian-based Linux
(e.g., Ubuntu 24.04).

#### Prerequisites

Ensure your local environment meets these minimum version requirements to avoid
compilation errors related to C++20 standards and build-time orchestration.

-   **Compiler**: clang / clang++ 17+ (Required for stable C++20 feature
    support).

-   **Build Tools**: cmake (3.25+) and make.

-   **Python**: 3.12+ (Required development scripts).

-   **Java**: openjdk-17-jre-headless or newer (Required for ANTLR 4).

-   **Rust**: The Rust toolchain is required for specific sub-components.

-   **System Libraries**: zlib1g-dev, libssl-dev libcurl4-openssl-dev.

--------------------------------------------------------------------------------

**1. Configuration**

Configure the project using CMake Presets, which automatically handle the build
directory creation, generators, and toolchain settings. Note that while the
build system is currently hard-coded to C++20 within the presets, future updates
will transition this to a configurable variable.

```bash
# Native Linux build
cmake --preset default

# Android cross-compilation build
cmake --preset android-x86_64
```

**2. Executing the Build**

Once configured, initiate the compilation by pointing the standard `cmake
--build` command to the output directory defined by your chosen preset. The
build directory path automatically generated by the preset will follow the
format `cmake/build/<preset_name>`.

Parallel execution is highly recommended to manage the complex dependency tree,
but it must be balanced against available system resources.

**WARNING** High Memory Usage: Allocating excessive parallel jobs can cause a
SEGFAULT or an OOM-kill (Signal 9). To ensure a stable build, use a conservative
job count.

**Recommended Formula**: Available RAM / 8GB = Max parallel jobs.

```bash
# Example for 32GB RAM using "default" preset
cmake --build cmake/build/default -j4
```

**3. Verification**

Verify the binary integrity and Package Infrastructure mapping via an inference
test. This confirms that all internal symbols and external shims (Abseil,
Protobuf, etc.) are correctly linked and functional.

```bash
cd cmake/build/default/litert_lm/build/
./litert_lm_main \
  --model_path=/path/to/gemma-3n-E2B-it-int4.litertlm \
  --backend=cpu \
  --input_prompt="What is the tallest building in the world?"
```

*(Note: Adjust the executable path based on your preset's binary directory
output).*

**Expected Output**: A successful build will initialize the XNNPACK delegate and
return the model response along with benchmark metrics:

-   *Init Phases*: Executor and Tokenizer initialization times.

-   *Prefill/Decode Speeds*: Performance stats (tokens/sec) indicating the
    backend is optimized.

*Example*

```
dev-sh@:LiteRT-LM$ cmake/build/litert_lm_main --model_path=$model_path/gemma-3n-E2B-it-int4.litertlm --backend=cpu --input_prompt="What is the tallest building in the world?"
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
input_prompt: What is the tallest building in the world?
The tallest building in the world is the **Burj Khalifa** in Dubai, United Arab Emirates.

It stands at a staggering **828 meters (2,717 feet)** tall.

It was completed in 2010 and continues to hold the record.

BenchmarkInfo:
  Init Phases (2):
    - Executor initialization: 844.54 ms
    - Tokenizer initialization: 66.70 ms
    Total init time: 911.25 ms
--------------------------------------------------
  Time to first token: 2.40 s
--------------------------------------------------
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 18 tokens in 2.311920273s duration.
      Prefill Speed: 7.79 tokens/sec.
--------------------------------------------------
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 62 tokens in 5.53092314s duration.
      Decode Speed: 11.21 tokens/sec.
--------------------------------------------------
--------------------------------------------------
```

<br>

## Getting Started: Running the LiteRT-LM Container

To get the environment up and running, follow these steps from the root
directory of the project. The process is divided into build, create, and attach
phases to ensure container persistence is handled correctly.

### 1. Build the Image

First, we'll build the image using the configuration in the cmake/ directory.
This might take a moment if it's your first time, as it pulls in our build
dependencies.

```bash
podman build -f /path/to/repo/cmake/Containerfile -t litert_lm /path/to/repo
```

### 2. Create the Persistent Container

Instead of executing a one-off run, create a named container to preserve the
workspace state for future sessions. Using interactive mode ensures the
container is prepared for a functional terminal.

```bash
podman container create --interactive --tty --name litert_lm litert_lm:latest
```

### 3. Start and Join the Session

Finally, start the container and attach your shell to it.

```bash
podman start --attach litert_lm
```

**Note:** If you exit the container and want to get back in later, you don't
need to rebuild or recreate it. Just run the podman start --attach litert_lm
command again and you'll be right back where you left off.

<br>

## Troubleshooting

The LiteRT-LM Super-Build orchestrates a complex pipeline involving multi-phase
cross-compilation, dynamic dependency harvesting, and cross-language (FFI)
boundaries. Because of this architecture, build failures can stem from a variety
of sources—ranging from a simple omitted internal target to a silent upstream
layout change or a cross-language ABI mismatch.

This guide provides a structured approach to diagnosing and resolving build
pipeline issues. It is organized into three sections:

1.  **Failure Modes:** An overview of the 6 core categories of build errors and
    their common symptoms.
2.  **Debugging Techniques:** Standard diagnostic techniques for analyzing build
    state, tracing variable lifecycles, and inspecting binaries.
3.  **Standard Remediation Patterns:** Proven, repeatable fixes for patching
    upstream targets, synchronizing FFI boundaries, and restoring a healthy
    build state.

--------------------------------------------------------------------------------

### Part 1: Failure Modes (The 6 Categories)

**1. Internal Omissions**

*   A new C++ file is added to a module but omitted from the
    `add_litertlm_library` macro, causing "undefined reference" errors during
    the final linkage.
*   A new third-party header is included in the source, but no corresponding
    `FetchContent` or `ExternalProject` pipeline was established in
    `//cmake/packages/`.
*   Code generation fails to trigger because a new Rust module (`.rs`), Protobuf
    schema (`.proto`), or Flatbuffers schema (`.fbs`) was not added to its
    corresponding tracking variable.

**2. Upstream Parity Failures**

*   Upstream maintainers update their primary build system (e.g., Bazel) but
    forget to update their secondary `CMakeLists.txt` to include new source
    files, causing missing symbols downstream.
*   An upstream dependency explicitly hardcodes architecture-specific flags
    (e.g., `-mavx2` or `-Werror`) that break the cross-compilation toolchain.
*   An upstream dependency restructures its build output directories, causing
    compiled archives to be placed outside of our dynamic target map detection
    script's configured search paths.
*   Refactoring in upstream `CMakeLists.txt` (like renaming targets or
    variables) cause our `patch_file_content` regex to silently miss its target,
    leading to conflicting sub-dependencies or ODR violations.

**3. Build State & Concurrency Corruption**

*   A parallel build races ahead and attempts to compile a target before the
    host tools (`flatc`, `protoc`) have finished generating the required
    headers.
*   A manual build abort leaves corrupted stamp files in the `ExternalProject`
    directory, causing CMake to falsely believe configuration is complete and
    fail during compilation.

**4. Host-Target Contamination (Cross-Compilation Leaks)**

*   CMake's `find_package` accidentally locates a host machine's library (e.g.,
    `/usr/lib/libz.so`) instead of the target NDK sysroot archive, causing an
    `Exec format error`.
*   Phase routing logic fails, causing the build system to link or execute
    artifacts built with the wrong toolchain (e.g., host clang++ instead of NDK
    clang++).

**5. Environment & Toolchain Entropy**

*   Windows builds fail during dependency extraction because nested directories
    exceed the `MAX_PATH` limit (260 characters).
*   Differences between the local host environment and supported CI baselines
    (such as an obsolete Clang version or incompatible NDK) cause unexplainable
    linker errors, missing standard library features, or sudden syntax failures.
*   Host antivirus software locks a newly generated archive during the
    `POST_BUILD` harvesting phase, causing a "Permission denied" error.

**6. Cross-Language Boundary (FFI) Fractures**

*   ABI flags (like `-D_GLIBCXX_USE_CXX11_ABI=1`) are applied to C++ targets but
    not passed to Cargo, causing a successful build that segfaults at runtime.
*   Cargo defaults to the host system linker instead of the specifically
    injected NDK cross-compilation linker, making the resulting crate
    incompatible with the C++ payload.

--------------------------------------------------------------------------------

### Part 2: Debugging Techniques

**Build State & Log Analysis**

*   **Isolate Concurrency Issues:** Force a sequential build using `cmake
    --build . --parallel 1`. If it succeeds sequentially but fails in parallel,
    there is a missing `generator_complete` dependency.
*   **Investigate External Builds:** Always inspect `CMakeError.log` and
    `CMakeOutput.log` within the specific dependency’s build folder
    (`<BINARY_DIR>/external/<pkg>-build/CMakeFiles/`) to check if the compiler
    silently rejected injected flags.
*   **Reset Dependency State:** Delete the `stamps/` folder
    (`<BINARY_DIR>/external/<pkg>-prefix/src/<pkg>-stamp/`) to force CMake to
    re-run the download, patch, and configure steps for that specific package
    without wiping the entire workspace.

**Variable & Configuration Tracing**

*   **Trace the Variable Lifecycle:** Verify that configuration variables are
    defined in `<pkg>_config.cmake` and `<pkg>_shim.cmake`, handled properly by
    `<pkg>_patcher.cmake`, and successfully passed via `CMAKE_ARGS` or
    `PATCH_COMMAND` in `<pkg>.cmake`.
*   **Interrogate the Cache:** Open
    `<BINARY_DIR>/external/<pkg>-build/CMakeCache.txt` to verify that your
    injected variables actually expanded to their expected values, and to
    identify internal upstream variables that may need overriding.
*   **Compare Build Systems:** Cross-reference the upstream dependency’s primary
    build file (e.g., Bazel `BUILD`) against their `CMakeLists.txt` to quickly
    spot omitted source files or default `option()` discrepancies.

**Binary & Architecture Inspection**

*   **Enable Verbose Output:** Use `make VERBOSE=1` (or `ninja -v`) or set
    `CARGO_TERM_VERBOSE=true` to confirm exact linker paths and ensure
    cross-compilation toolchains are used instead of host system binaries.
*   **Inspect Object Files:** Use `nm -C`, `readelf -h`, or `otool` on the
    failing executable or object file to explicitly verify its compiled
    architecture and check for mismatched ABI tags across FFI boundaries.

**Environment Verification**

*   **Check Path Constraints:** Verify that host environment variables (like
    `ANDROID_NDK_HOME`) are pointing to the correct directories, and on Windows,
    ensure the project root is shallow enough to avoid `MAX_PATH` limitations.
*   **Verify Toolchain Versions:** Verify that the compiler and toolchain
    versions active in the failing environment (such as Clang or the Android
    NDK) meet the project's minimum requirements and align with the known-good
    baselines used in CI.

--------------------------------------------------------------------------------

### Part 3: Standard Remediation Patterns

**Upstream Patching & Overrides**

*   **Use the Patcher:** Use `patch_file_content` or `patch_delete_block` inside
    `<pkg>_patcher.cmake` to modify the upstream `CMakeLists.txt`. This allows
    you to bypass faulty logic, define missing targets, strip hardcoded flags,
    or inject missing `.cc` files directly into existing configurations.
*   **Expand Target Map Search Paths:** If an upstream dependency restructured
    its outputs, update the search directories in the dynamic target map script
    so it can successfully discover and harvest the relocated archives.
*   **Force Upstream Variables:** Explicitly pass toolchain sysroot paths and
    target constraints via `CMAKE_ARGS` in `<pkg>.cmake` (e.g.,
    `-DCMAKE_SYSROOT=...` and `-DCMAKE_FIND_ROOT_PATH_MODE_*=ONLY`).

**Internal Wiring & Code Generation**

*   **Update Target Links:** Check the `target_link_libraries` definition in
    your local `CMakeLists.txt`. Add any missing internal targets to ensure they
    link properly into the directory's facade, parent interfaces, or the
    `LITERTLM_DEPS` aggregate.
*   **Trigger Code Generators:** Ensure newly added files requiring generation
    (Rust `.rs`, Protobuf `.proto`, Flatbuffers `.fbs`) are added to their
    respective tracking variables so the host tools compile and alias the resulting headers.
*   **Lock Generator Execution Order:** Add `add_dependencies(<target>
    generator_complete)` to targets consuming generated files to prevent
    parallel build race conditions.

**Cross-Language Boundaries & Toolchain Adjustments**

*   **Synchronize ABI Boundaries:** Pass matching C++ ABI definitions to Cargo
    via `corrosion_set_env_vars` to prevent runtime segfaults across the FFI
    boundary.
*   **Force the Rust Linker:** Ensure `corrosion_set_linker` explicitly points
    Cargo to the correct cross-compilation Clang binary from the active
    toolchain.

**Host Environment & Toolchain Alignment**

*   **Enforce Supported Toolchains:** Upgrade local toolchains to match CI
    baselines, or explicitly set environment variables (e.g., `export CXX=...`
    or `ANDROID_NDK_HOME=...`) to force the build to utilize the correct
    compiler rather than defaulting to an outdated system-wide binary.
*   **Bypass Host System Constraints:** Relocate the project to a shallow root
    directory on Windows (e.g., `C:\src`) to avoid `MAX_PATH` extraction
    errors during dependency resolution, and configure local antivirus
    exclusions for the build output directory to prevent archive-locking.

--------------------------------------------------------------------------------

This project is licensed under the
[Apache 2.0 License.](https://github.com/google-ai-edge/LiteRT-LM/blob/main/LICENSE)
