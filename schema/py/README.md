# Internal Development & Package Creation

This document outlines how to build, test, and package the
`litert-lm-builder` Python tools.

## Project Structure

- `litertlm_builder.py`: Core logic for building LiteRT-LM files.
- `litertlm_builder_cli.py`: Command-line interface for the builder.
- `litertlm_peek.py`: Core logic for inspecting LiteRT-LM files.
- `litertlm_peek_main.py`: Command-line interface for the peek tool.
- `pyproject.toml`: Build configurations.
- `bundle_pypi_package.sh`: Script to bundle the package into a PyPI-ready wheel.

## Building and Packaging

To build the package and create a `.whl` distribution, use the helper script:

```bash
./bundle_pypi_package.sh
```

### What happens during the build?

1.  **Bazel Bindings**: It delegates to Bazel to generate Protobuf (`.pb2`) and
    FlatBuffer bindings for the LiteRT-LM schemas.
2.  **Staging**: Files are copied into a temporary directory
    (`/tmp/litertlm_builder/`). All python files and generated bindings are put
    into a `litert_lm_builder` sub-directory to form the package structure.
    `pyproject.toml` is copied to the root.
3.  **Import Rewriting**: It searches and rewrites Protobuf/FlatBuffer internal
    imports and imports in all python files so they match the expected
    `litert_lm_builder` packaged namespace.
4.  **Wheel Generation**: It sets up a virtual environment, installs `uv`, and
    invokes `uv build --no-build-isolation` to generate a self-contained `.whl`
    artifact in `/tmp/litertlm_builder/dist/`.
5.  **Verification**: The script installs the generated wheel and runs `--help`
    on CLI tools to verify installation.

## Testing locally

Once built, you can test the wheel by creating a virtual environment, installing
the wheel directly, and verifying the CLIs:

```bash
uv venv
source .venv/bin/activate
uv pip install /tmp/litertlm_builder/dist/*.whl
litert-lm-builder --help
litert-lm-peek --help
```
