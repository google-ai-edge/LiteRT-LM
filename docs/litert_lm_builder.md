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

The easiest way for most users to interact with LiteRT-LM files is through the
command-line interface. Once installed, you have access to two terminal
commands:

-   `litert-lm-builder`
-   `litert-lm-peek`

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

### Unpacking with litert-lm-builder

You can unpack an existing `.litertlm` file into a directory containing its
extracted sections and a reconstructed `model.toml`:

```bash
litert-lm-builder unpack --input demo.litertlm --output ./unpacked_dir
```

### litert-lm-peek (Inspection & Extraction)

This tool is designed to inspect the contents of a `.litertlm` file. It reads
the file's header, system metadata, and section information, and prints them to
the console.

You can query metadata or unpack artifacts within an existing archive into an
output destination on disk natively:

```bash
# Dump diagnostic info to stdout
litert-lm-peek --litertlm_file demo.litertlm

# Extract byte-for-byte components directly
litert-lm-peek --litertlm_file demo.litertlm --dump_files_dir ./extracted_files
```

## Python API Demo Walkthrough

For more advanced use cases, you can use the LiteRT-LM builder directly through
the fully exposed Python programmatic API.

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
    Backend,
    peek_litertlm_file
)
from litert_lm_builder.runtime.proto import llm_metadata_pb2

def build_demo_model():
    """Builds a .litertlm file programmatically using the Python API."""
    # Paths to your assets
    model_path = "schema/testdata/attention.tflite"
    sp_path = "runtime/components/testdata/sentencepiece.model"
    llm_metadata_path = "runtime/components/testdata/llm_metadata.pb"
    output_path = "demo_api.litertlm"

    # Initialize the core builder object
    builder = LitertLmFileBuilder()

    # Add metadata
    builder.add_system_metadata(Metadata(key="Authors", value="ODML Team", dtype=DType.STRING))
    builder.add_system_metadata(Metadata(key="TargetBackend", value=Backend.CPU.name, dtype=DType.STRING))
    builder.add_llm_metadata(llm_metadata_path=llm_metadata_path)

    # Add main TfLite model
    builder.add_tflite_model(
        tflite_model_path=model_path,
        model_type=llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE,
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

    # You can unpack the archive programmatically
    unpacked_builder = LitertLmFileBuilder.unpack(output_path, "./unpacked_output")
    print(f"Unpacked {output_path} into ./unpacked_output")

if __name__ == "__main__":
    build_demo_model()
```
