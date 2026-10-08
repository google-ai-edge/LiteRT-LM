# LiteRT-LM Core Orchestration

This directory contains the primary `CMakeLists.txt` engine for LiteRT-LM.
However, this script is not meant to be run directly. Instead, it is invoked
**twice** by the root orchestrator (`//CMakeLists.txt`) as part of a hermetic,
Two-Stage Super-Build environment.

The script adapts its behavior entirely based on the injected
`LITERTLM_ORCHESTRATION_PHASE` variable, acting first as a host-tool
bootstrapper, and second as the primary cross-compilation build system.

--------------------------------------------------------------------------------

## Phase 1: Prebuild (Host Tools)

When invoked with `-DLITERTLM_ORCHESTRATION_PHASE=prebuild`, the script operates
in a minimized capacity to establish the host environment.

*   **Host Compilation:** It completely ignores cross-compilation toolchains
    (like the Android NDK) and uses the native host's C/C++ compilers.
*   **Tool Bootstrapping:** It bypasses the main LiteRT-LM source tree and
    strictly builds build-time dependencies, specifically focusing on generating
    native binaries for `protoc` (Protobuf) and `flatc` (FlatBuffers).

Once Phase 1 completes, the root orchestrator captures the absolute paths of
these generated host binaries and prepares the sandbox for Phase 2.

--------------------------------------------------------------------------------

## Phase 2: Target Build (LiteRT-LM)

When invoked with `-DLITERTLM_ORCHESTRATION_PHASE=litert_lm`, the script
transforms into the actual project build system. It ingests the target
toolchain, isolated Python environment, and the host tools generated in Phase 1.

### 1. Code Generation & Compiler Gating

LiteRT-LM relies heavily on C++ headers generated from `.proto` and `.fbs`
files. To prevent race conditions during parallel builds, the script maps all
schema files and establishes a strict compiler gate (`generator_complete`). No
C++ translation units are allowed to compile until the host tools have
successfully written all generated headers to disk.

### 2. Comprehensive Static Linking Strategy

Linking massive Machine Learning frameworks (like TensorFlow Lite and LiteRT)
alongside a heavily patched dependency tree requires a highly robust static
linking strategy to ensure ABI stability and functional binaries.

Instead of relying on fragile, manually sorted dependency graphs, the build
system aggregates the global payload and applies targeted linker flags:

*   **Cyclic Resolution (`--start-group` / `--end-group`):** Forces the linker
    to exhaustively loop through the provided static archives. This guarantees
    that complex, cyclical dependencies across the ML frameworks and custom Rust
    bridges are resolved automatically.
*   **Operator Preservation (`--whole-archive`):** Machine learning frameworks
    rely on self-registering C++ operators (via static initialization). Because
    these operators aren't explicitly called in the main execution path,
    standard linkers will aggressively strip them as "dead code." This flag
    forces the inclusion of these objects, preventing "Operator Not Found"
    runtime crashes.
*   **Converging Symbol Resolution (`--allow-multiple-definition`):** Safely
    merges overlapping symbols that occur when forcing external dependencies to
    link against our centralized, strictly-versioned core libraries (like
    Abseil), neutralizing One Definition Rule (ODR) violations.

### 3. Final Assembly

Once the dependency payloads are aggregated and the linker flags are dynamically
mapped to the target platform (Apple Mach-O, Android Bionic, Linux ELF, or
Windows MSVC), the script compiles `litert_lm_main` into a statically linked,
standalone executable.
