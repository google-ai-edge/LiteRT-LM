# LiteRT-LM Rust Orchestration & C++ Bridge

This directory manages the integration between the primary C++ build system
(CMake) and the Rust ecosystem (Cargo). Because LiteRT-LM enforces a strict,
hermetic cross-compilation environment, Cargo cannot be allowed to resolve
targets, linkers, or ABIs autonomously.

Instead, this orchestration uses **Corrosion** and the **`cxx`** crate to force
Cargo to act as a subservient CMake target, ensuring all compiled Rust artifacts
respect the global One Definition Rule (ODR) and linker configurations.

## The Rust Workspace Architecture

The Rust integration avoids standard Cargo layout conventions (e.g., placing all
code in `src/`) to maintain tight source-locality with the C++ components they
interact with.

*   **The Umbrella Crate (`litert_lm_deps`)**: The root `Cargo.toml` defines a
    single `staticlib`. It pins core external Rust dependencies (such as
    `llguidance`, `tokenizers`, and `minijinja`) and orchestrates the internal
    workspace.
*   **Explicit Path Overrides**: Instead of keeping Rust source files siloed in
    the `cmake/rust/` directory, the crate uses explicit path directives (e.g.,
    `#[path = "../runtime/components/rust/..."]`) to consume Rust code directly
    from the `//runtime` C++ tree.
*   **Specialized Sub-Crates**: Distinct parsers and tool-use components (e.g.,
    `python_parser`, `json_parser`) are isolated into their own crates under
    `cmake/rust/` but map directly back to the execution engine's source files,
    keeping the dependency graph modular.

## The CMake-Cargo Bridge Handshake

The C++ build system ingests the Rust workspace through a carefully synchronized
sequence defined in `generators.cmake` and `generate_cxxbridge.cmake`.

### 1. Environment & Linker Hijacking

When `corrosion_import_crate` pulls the `litert_lm_deps` workspace into the
build graph, CMake aggressively overrides Cargo's default environment. It
injects specific Android NDK Clang paths and enforces C++20 standard flags via
`corrosion_set_linker` and `corrosion_set_env_vars`. This guarantees that
Cargo's cross-compilation matches the CMake environment identically.

### 2. Header Generation & Aliasing

The `cxx` crate generates C++ header bindings for the Rust bridges (e.g.,
`minijinja_template.rs`). Because internal C++ consumers expect a specific
naming convention to distinguish standard headers from Rust bridges, CMake
implements a custom build step to intercept the generated `.h` files and
explicitly alias them to `.rs.h` (e.g., `minijinja_template.rs.h`).

### 3. Artifact Harvesting & Aggregation

Cargo generates its static libraries deep within a temporary `target/`
directory. To seamlessly integrate this into the C++ linking phase:

*   A custom `POST_BUILD` script (`find_and_copy_cxxbridge.cmake`) hunts the
    Cargo output tree for the required archives (including the elusive
    `libcxxbridge1.a`) and extracts them into the CMake binary directory.
*   These artifacts, alongside the primary `liblitert_lm_deps.a`, are bundled
    into a single, global CMake target: `LiteRTLM::CxxBridge::Aggregate`.
*   The primary `litert_lm_main` executable simply links against this aggregate,
    inheriting the entire Rust ecosystem safely without requiring manual linker
    tracking.
