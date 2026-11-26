# LiteRT-LM CMake Modules

This directory serves as the functional backbone of the LiteRT-LM build system.
Rather than cluttering the primary `CMakeLists.txt` files with complex
implementation details, the core mechanics—architectural macros, dependency
aggregators, source-code patchers, and compiler gating logic—are modularized
here.

By isolating these mechanics, the build system separates the *what* (the targets
being built) from the *how* (the complex ODR-enforcement and cross-compilation
routing), keeping the main project files clean and declarative.

## Module Breakdown

### `collect_dependencies.cmake`

**Purpose:** Global dependency aggregation and linker payload construction.

*   Aggregates system include paths across all disparate packages and
    components.
*   Defines a primary `INTERFACE` target named `LITERTLM_DEPS`.
*   Links all external project shims (e.g., `LiteRTLM::absl::shim`),
    fetched libraries, and Rust bridge targets to `LITERTLM_DEPS`. This single
    target serves as the unified dependency payload that `litert_lm_main`
    ultimately links against.

### `external_project.cmake`

**Purpose:** Strict dependency sorting and initialization.

*   Defines `LITERTLM_DEPENDENCY_ORDER`, establishing the exact topological
    sorting required to safely initialize the complex build graph (e.g.,
    ensuring `absl` is fully orchestrated before `protobuf` attempts to build).
*   Iterates through this list, invoking the `load_package` macro to trigger the
    internal build or locate system installations for each critical component.

### `fetch_content.cmake`

**Purpose:** Standard CMake `FetchContent` management for well-behaved
dependencies.

*   Pulls in dependencies that do not require aggressive source-patching or ODR
    hijacking (e.g., ANTLR, Corrosion, llguidance, zlib).
*   Orchestrates the download of necessary pre-compiled binaries (like the ANTLR
    Java Tool).
*   Groups all generated include directories and source paths into centralized
    variables (`LITERTLM_FETCHCONTENT_MODULE_SRC_DIRS`, etc.) and establishes a
    synchronization target (`fetch_content_complete`).

### `generators.cmake`

**Purpose:** Code generation orchestration and central compilation gating.

*   Provides the `generate_src_files` function, which copies and normalizes
    source paths from the main tree into the hermetic build directory, stripping
    out nested legacy paths to match the internal structure.
*   Builds the `litertlm_protobuf` static library, ensuring it links properly
    against the patched Protobuf shims.
*   Maps out the entire C++, Rust, and Schema file tree, grouping them into
    master lists (`ALL_SOURCE_FILES`, etc.) to feed the final executable while
    enforcing compiler gates that prevent C++ compilation before headers exist.

### `macros.cmake`

**Purpose:** Core architectural CMake macros defining the LiteRT-LM build
vocabulary.

*   `load_package`: Evaluates whether to build a package internally or use a
    system-provided one via user override flags (e.g.,
    `LITERTLM_USE_SYSTEM_ABSL`).
*   `add_litertlm_library`: The standard wrapper used across the codebase to
    compile libraries. It automatically redirects source files to the hermetic
    build directory, registers artifacts to global properties, and binds the
    target to the `generator_complete` compiler gate.
*   `import_absl_lib` / `import_proto_lib`: Helpers for generating the initial
    namespace aliases during orchestration.
*   `literlm_configure_component_interface`: Wraps specific component targets in
    `--start-group` / `--end-group` linker flags to safely resolve cyclical
    sub-dependencies.

### `utils.cmake`

**Purpose:** Low-level utility functions and fallback logic.

*   `cmake_checkpoint_target`: A robust function that enforces the existence of
    a target. If a target doesn't exist, it dynamically creates a "Shim" (an
    Imported Target) to satisfy CMake's configuration phase, effectively
    separating dependency *declaration* from *definition*.
*   `patch_file_content` / `patch_delete_block`: Functions used heavily by the
    package orchestrators to read, regex-replace, and rewrite upstream
    dependency source code on the fly.
*   `setup_external_install_structure`: Pre-creates `include/`, `lib/`, and
    `bin/` directories for external projects. This tricks CMake's strict path
    validation, allowing imported targets to reference paths that will not
    physically exist until the build phase executes.
*   `kvp_parse_map`: A parser that breaks down the dynamically generated
    `TargetName=PhysicalPath` dictionaries into lists usable by CMake target
    properties.
