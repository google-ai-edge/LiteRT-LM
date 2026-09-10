# Liquid AI LFM

Canonical LiteRT-LM prompt templates and metadata configuration for the Liquid
AI LFM (Liquid Foundation Model) family (e.g. LFM 2.5 1.2B Instruct).

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [LiquidAI LFM2.5 Chat Template](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/chat_template.jinja)

### Features & Standardization

1.  **Role Markers & BOS Token**:

    -   Emits `{{- bos_token -}}` at the start of the prompt (e.g.
        `<|startoftext|>`).
    -   Uses `<|im_start|>` and `<|im_end|>` delimiters for message roles:
        `system`, `user`, `assistant`, `tool`.

2.  **Tool Calling**:

    -   Function/tool signatures are enclosed in `List of tools: [...]` with
        standard declarations (`[{"type": "function", "function": {...}}]`)
        within the `<|im_start|>system` turn.
    -   Assistant function calls are formatted as Python-like invocations within
        `<|tool_call_start|>` and `<|tool_call_end|>` tags:
        `<|tool_call_start|>[function_name(arg1='val1',
        arg2=123)]<|tool_call_end|>`
    -   Tool outputs are supplied under the `tool` role enclosed in brackets
        `[...]`: `<|im_start|>tool\n[<content>]<|im_end|>\n`.

3.  **Thinking Mode**:

    -   LFM 2.5 models do not support dynamic thinking toggles (the
        `enable_thinking` API).
    -   To use thinking, use the dedicated thinking model
        [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking).
    -   To disable thinking, use the instruct model
        [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

4.  **Thinking Channel**:

    -   `LlmMetadataProto.pbtext` defines the `thought` channel with start
        `<think>` and end `</think>`.
    -   This channel is used when running the `Thinking` variant (e.g.
        [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)),
        where the model outputs `<think>` and `</think>` tokens.
