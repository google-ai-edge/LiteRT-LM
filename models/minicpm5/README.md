# MiniCPM 5

Canonical LiteRT-LM prompt templates and metadata configuration for the MiniCPM
5 model family.

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [MiniCPM GitHub Documentation](https://github.com/OpenBMB/MiniCPM)

### Features & Standardization

1.  **Tool Calling**:

    -   Supports function/tool signatures enclosed in `<tools>` and `</tools>`
        XML tags in the system prompt.
    -   Assistant function calls are formatted as XML within `<function
        name="...">` tags.
    -   Multi-step tool responses are wrapped within `<tool_response>` tags
        under the `tool` role.

2.  **Thinking Mode Toggle**:

    -   Adheres to the LiteRT-LM Chat Template Standard `enable_thinking`
        configuration (defaults to `false`).
    -   When thinking mode is disabled (`enable_thinking=false`), the template
        appends `<think>\n\n</think>\n\n` to skip reasoning tokens.

3.  **Thinking Channel**:

    -   `LlmMetadataProto.pbtext` defines the `thought` channel with delimiter
        tokens `<think>\n` and `</think>`.

## Model Conversion

To convert and package a MiniCPM 5 Hugging Face model into `.litertlm` format
using `litert-torch`:

```bash
litert-torch export_hf \
  --model=openbmb/MiniCPM5-1B \
  --litert_lm_llm_metadata_override=/tmp/minicpm_metadata.textpb \
  --quantization_recipe=dynamic_wi8_afp32 \
  --use_bool_mask=True \
  --apply_gpu_composites=True \
  --output_dir=/tmp/minicpm_test \
  --cache_length=32771 \
  --experimental_lightweight_conversion
```
