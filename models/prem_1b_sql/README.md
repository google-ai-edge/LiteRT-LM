# Prem-1B-SQL

Canonical LiteRT-LM prompt templates and metadata configuration for Prem-1B-SQL.

## Chat Template

-   Canonical template: `chat_template.jinja`
-   Specification reference:
    [Prem-1B-SQL](https://huggingface.co/prem-research/prem-1B-SQL)

### Features & Standardization

1.  **Role Markers & BOS Token**:

    -   Emits `{{- bos_token -}}` at the start of the prompt (e.g. `<｜begin of
        sentence｜>`).
    -   Uses `### Instruction:\n` prefix for user turns and `### Response:\n`
        prefix for assistant turns, with `<|EOT|>\n` delimiter between turns.
    -   Falls back to the default DeepSeek Coder system instruction when no
        explicit system role is provided in the conversation history.

2.  **Multimodal Parts**:

    -   Content is formatted as standard multimodal parts (`type: "text"`).

