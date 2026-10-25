# LiteRT-LM Package Orchestration

This directory contains the orchestration modules for external third-party
dependencies.

Because LiteRT-LM operates as a highly hermetic, cross-platform Super-Build, we
cannot rely on standard `find_package()` or naive `FetchContent` integrations
for complex dependencies. Doing so inevitably leads to One Definition Rule (ODR)
violations, diamond dependency conflicts (e.g., multiple libraries fetching
conflicting versions of Abseil), and host/target architecture leakage during
Android cross-compilation.

Instead, this directory implements a **Dependency Hijacking and Transformation
Pattern**. We fetch upstream source code but patch its build logic,
forcing the dependency to consume *our* centralized sub-dependencies, respect
*our* linker strategies, and export targets that fit seamlessly into the
LiteRT-LM graph.

## The Transformation Pattern

Every dependency in this directory follows a standardized orchestration
lifecycle, typically divided into the following files (using `<name>` as the
package placeholder):

| File                     | Purpose                                           |
| :----------------------- | :------------------------------------------------ |
| `<name>_config.cmake`    | Defines all paths (source, build, install) and    |
:                          : handles phase-routing (e.g., swapping host vs.    :
:                          : target binaries during cross-compilation).        :
| `<name>.cmake`           | The primary entry point. Orchestrates the         |
:                          : `ExternalProject_Add` download, patches the       :
:                          : source, and triggers the build.                   :
| `<name>_patcher.cmake`   | A source-transformation script. It executes       |
:                          : *before* the external project configures, using   :
:                          : regex to remove nested dependencies and inject   :
:                          : our shims.                                        :
| `<name>_shim.cmake`      | An injected script that runs *inside* the         |
:                          : external project's CMake context. It forces ABI   :
:                          : compliance, overrides optimization flags, and     :
:                          : maps expected aliases to our global payloads.     :
| `<name>_aggregate.cmake` | Runs back in the LiteRT-LM context after the      |
:                          : external build finishes. It creates global        :
:                          : `ALIAS` targets (e.g., `protobuf\:\:libprotobuf`) :
:                          : to trap downstream dependencies and prevent them  :
:                          : from re-fetching the library.                     :
| `*_target_map.cmake`     | Auto-generated dictionaries mapping standard      |
:                          : target names to the physical, absolute paths of   :
:                          : the generated static archives (`.a` files).       :

--------------------------------------------------------------------------------

## Case Study: Protobuf

To understand how this pattern is applied in practice, consider the Protobuf
integration (`//cmake/packages/protobuf/`). Protobuf is particularly challenging
because it generates a build-time tool (`protoc`) that must run on the Host,
while its runtime libraries (`libprotobuf`) must be compiled for the Target.
Furthermore, Protobuf internally depends on Abseil.

Here is how the LiteRT-LM package infrastructure successfully ingests it:

### 1. Phase Routing (`protobuf_config.cmake`)

LiteRT-LM uses a two-stage Super-Build (`prebuild` for host tools, `litert_lm`
for target binaries). The config file dynamically routes the `protoc` binary
path based on the `LITERTLM_ORCHESTRATION_PHASE`. If compiling for an Android
target, it ensures the build system invokes the x86 Linux `protoc` generated
during the prebuild phase, preventing architecture execution errors.

### 2. Source Hijacking (`patcher.cmake`)

Before Protobuf configures, our patcher modifies its root `CMakeLists.txt`:

*   **Injection:** It injects `include(protobuf_shim.cmake)` directly into the
    upstream project.
*   **Purging:** It comments out Protobuf's internal `abseil-cpp.cmake` logic
    and recursively deletes its `cmake/abseil-cpp` directory. Protobuf is no
    longer allowed to fetch its own Abseil.
*   **Redirection:** It regex-replaces all instances of `absl::[target]` across
    the codebase with our central `LiteRTLM::absl::shim`.

### 3. Environment Forcing (`protobuf_shim.cmake`)

Now running *inside* Protobuf's build context, the shim enforces LiteRT-LM's
rules:

*   It provides the `LiteRTLM::absl::shim` target that the patcher redirected
    everything to, pointing it at our pre-built, ODR-compliant Abseil
    installation.
*   It overrides aggressive Android NDK optimization flags (downgrading `-O3` to
    `-O2` to prevent linker stripping bugs).
*   It wraps `CMAKE_CXX_STANDARD_LIBRARIES` in Brutalist linker flags (e.g.,
    `-Wl,--allow-multiple-definition` and `--start-group`), ensuring Protobuf
    links perfectly against our aggregated payloads.

### 4. Target Trapping (`protobuf_aggregate.cmake`)

Once Protobuf installs, LiteRT-LM scans the output directory and generates a
target map. The aggregate script reads this map and creates global interfaces.
Crucially, it generates an `ALIAS` target named exactly what downstream
dependencies expect (e.g., `protobuf::libprotobuf`).

When a downstream library (like TensorFlow Lite) calls `find_package(Protobuf)`,
CMake immediately resolves it to our `ALIAS` target. The downstream library
silently links against our hermetic, patched version of Protobuf, completely
bypassing its own dependency resolution logic.
