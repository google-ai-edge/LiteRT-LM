# Liquid AI LFM

Canonical LiteRT-LM prompt templates and metadata configuration for the Liquid
AI LFM (Liquid Foundation Model) family (e.g. LFM 2.5 1.2B Instruct).

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [LiquidAI LFM2.5 Chat Template](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/chat_template.jinja)

### Features & Standardization

1.  **Role Markers**:

    -   Uses `<|im_start|>` and `<|im_end|>` delimiters for message roles:
        `system`, `user`, `assistant`, `tool`.

2.  **Tool Calling**:

    -   Function/tool signatures are enclosed in `List of tools: [...]` within
        the `<|im_start|>system` turn.
    -   Assistant function calls are formatted as Python-like invocations within
        `<|tool_call_start|>` and `<|tool_call_end|>` tags:
        `<|tool_call_start|>[function_name(arg1='val1',
        arg2=123)]<|tool_call_end|>`
    -   Tool outputs are supplied under the `tool` role enclosed in brackets
        `[...]`: `<|im_start|>tool\n[<content>]<|im_end|>\n`.

3.  **Thinking Mode Toggle**:

    -   Adheres to the LiteRT-LM Chat Template Standard `enable_thinking`
        configuration (defaults to `false`).
    -   When thinking mode is enabled (`enable_thinking=true`), the generation
        prompt appends `<think>\n`.
    -   Historical reasoning thoughts are preserved within `<think>` and
        `</think>` tags.

4.  **Thinking Channel**:

    -   `LlmMetadataProto.pbtext` defines the `thought` channel with start
        `<think>\n` and end `</think>`.
    -   It requires the `Thinking` variant to work. e.g., LFM2.5-1.2B-Thinking
