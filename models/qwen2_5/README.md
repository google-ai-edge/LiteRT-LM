# Qwen 2.5

Canonical LiteRT-LM prompt templates and metadata configuration for the Qwen 2.5
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

