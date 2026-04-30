# External User Guide & API Walkthrough

The `litert-lm-builder` library provides Python tools for building, inspecting,
and unpacking LiteRT-LM files seamlessly across platforms.

## Installation

You can install the package directly from PyPI. We recommend using a virtual
environment:

```bash
uv venv
source .venv/bin/activate
uv pip install litert-lm-builder
```

## CLI Operations

The easiest way for most users to interact with LiteRT-LM files is through
the command-line interface. Once installed, you have access to two terminal
commands:

- `litert-lm-builder`
- `litert-lm-peek`

### litert-lm-builder (Chaining Subcommands)

key feature for CLI users is chaining multiple configuration arguments linearly.

```bash
litert-lm-builder \
  system_metadata --str Authors "ODML Team" \
  tflite_model --path schema/testdata/attention.tflite --model_type prefill_decode \
  sp_tokenizer --path runtime/components/testdata/sentencepiece.model \
  output --path demo.litertlm
```

You can optionally drive tool dynamically by reading standard TOML configuration

```bash
litert-lm-builder toml --path example.toml output --path real_via_toml.litertlm
```

### litert-lm-peek (Inspection & Extraction)

You can query metadata or unpack artifacts within an existing archive into an
output destination on disk natively:

```bash
# Dump diagnostic info to stdout
litert-lm-peek --litertlm_file demo.litertlm

# Extract byte-for-byte components directly
litert-lm-peek --litertlm_file demo.litertlm --dump_files_dir ./extracted_files
```

## Python API Demo Walkthrough

For more advanced use cases, you can use the LiteRT-LM builder directly
through the fully exposed Python programmatic API.

Below is a complete, self-contained walkthrough demonstrating how to bundle a
`.litertlm` file:

```python
import os
import sys

# Core classes directly importable from the top level
from litert_lm_builder import (
    LitertLmFileBuilder,
    Metadata,
    DType,
    TfLiteModelType,
    Backend,
    peek_litertlm_file
)

def build_demo_model():
    """Builds a .litertlm file programmatically using the Python API."""
    # Paths to your assets
    model_path = "schema/testdata/attention.tflite"
    sp_path = "runtime/components/testdata/sentencepiece.model"
    output_path = "demo_api.litertlm"

    # Initialize the core builder object
    builder = LitertLmFileBuilder()

    # Add metadata
    builder.add_system_metadata(Metadata(key="Authors", value="ODML Team", dtype=DType.STRING))
    builder.add_system_metadata(Metadata(key="TargetBackend", value=Backend.CPU.name, dtype=DType.STRING))

    # Add main TfLite model
    builder.add_tflite_model(
        tflite_model_path=model_path,
        model_type=TfLiteModelType.PREFILL_DECODE
    )

    # Add auxiliary tokens & tokenizers
    builder.add_sentencepiece_tokenizer(
        sp_tokenizer_path=sp_path
    )

    # Serialize stream to your output file
    with open(output_path, "wb") as f:
        builder.build(f)
    print(f"Successfully built {output_path}")

    # You can also use the peek programmatic API identically
    print(f"\n--- Peeking at {output_path} ---")
    peek_litertlm_file(output_path, None, sys.stdout)

if __name__ == "__main__":
    build_demo_model()
```
