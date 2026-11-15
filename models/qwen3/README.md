# Qwen 3

Canonical LiteRT-LM prompt templates and metadata configuration for the Qwen 3
model family.

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [Qwen Chat Template Documentation](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#chat-template)

### Features & Standardization

1.  **Tool Calling**:

    -   Supports function/tool signatures enclosed in `<tools>` and `</tools>`
        XML tags in the system prompt.
    -   Assistant function calls are formatted as JSON within `<tool_call>` and
        `</tool_call>` tags.
    -   Multi-step tool responses are wrapped within `<tool_response>` tags
        under the `tool` role.

2.  **Thinking Mode Toggle**:

    -   Adheres to the LiteRT-LM Chat Template Standard `enable_thinking`
        configuration (defaults to `true`).
    -   When thinking mode is disabled (`enable_thinking=false`), the template
        appends `<think>\n\n</think>\n\n` to skip reasoning tokens.

3.  **Thinking Channel**:

    -   `LlmMetadataProto.pbtext` defines the `thought` channel with delimiter
        tokens `<think>\n` and `\n</think>`.

## Model Conversion

To convert and package a Qwen 3 Hugging Face model into `.litertlm` format using
`litert-torch`:

```bash
python -m litert_torch.generative.export_hf \
  --model=/path/to/qwen3_checkpoint \
  --litert_lm_llm_metadata_override=models/qwen3/LlmMetadataProto.pbtext \
  --quantization_recipe=dynamic_wi4_afp32 \
  --output_dir=/path/to/output_dir
```
