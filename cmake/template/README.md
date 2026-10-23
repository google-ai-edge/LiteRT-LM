# LiteRT-LM CMake Target Template

This directory contains the standard template for defining CMake targets within
the LiteRT-LM project. It enforces consistent naming conventions, visibility
rules, and dependency aggregation across the codebase.

## Target Definitions

The template demonstrates three primary target structures used in LiteRT-LM:

### 1. Static Libraries

Standard compiled libraries utilize the custom `add_litertlm_library` macro,
which automatically registers the target into the global build payload (e.g.,
for ODR enforcement and graph flattening).

*   **Macro Execution:** `add_litertlm_library(<target_name> STATIC <sources>)`
    defines the library and registers it for global collection.
*   **Namespacing:** An `ALIAS` library (e.g.,
    `LiteRTLM::Template::StaticLibExample`) is immediately created. This
    provides fail-fast typo protection and enforces a consistent consumer
    interface that maps to the directory structure.
*   **Include Directories:** Handled via `target_include_directories`. Global
    paths (`LITERTLM_INCLUDE_PATHS`) are grouped alongside any strictly local
    target-specific paths.
*   **Linking:** Uses `target_link_libraries` with `PUBLIC` or `PRIVATE` scopes.
    External dependencies are linked via the `LITERTLM_DEPS` aggregate, while
    internal dependencies strictly use their namespace aliases (e.g.,
    `LiteRTLM::Runtime::Utils`).

### 2. Interface Libraries (Header-Only)

Used for header-only components, aligning closely with standard Bazel `BUILD`
file conventions.

*   **Definition:** Declared using `add_litertlm_library(<target_name>
    INTERFACE)`. The target name should mirror the primary header file, though
    the file itself is not explicitly listed in the sources.
*   **Namespacing:** Aliased identically to static libraries (e.g.,
    `LiteRTLM::Template::InterfaceExample`).
*   **Scoping:** Both include directories and linked libraries must use the
    `INTERFACE` scope, as there is no compilation step for this specific target.

### 3. Folder Facades

A pattern used to bundle all sub-targets within a directory into a single,
convenient target for consumers.

*   **Definition:** Declared as a standard CMake `INTERFACE` library (e.g.,
    `add_library(<folder_name>_libs INTERFACE)`). It avoids the
    `add_litertlm_library` macro because it does not produce a compiled artifact
    itself.
*   **Namespacing:** Aliased to the directory's functional name (e.g.,
    `LiteRTLM::Template`).
*   **Aggregation:** Links all constituent targets within the folder using
    `target_link_libraries(... INTERFACE ...)`. Downstream consumers can link
    against this single alias to inherit the entire module.

## Core Conventions

*   **`add_litertlm_library`**: A project-specific macro that replaces the
    standard `add_library` for compiled artifacts, injecting them into the
    global ODR-compliant linker payload.
*   **`LITERTLM_DEPS`**: A global target aggregating all external third-party
    dependencies to simplify linking.
*   **Namespace Aliases (`LiteRTLM::*`)**: All internal linking must use the
    alias to ensure strict dependency resolution, prevent silent naming
    collisions, and improve build-file readability.
