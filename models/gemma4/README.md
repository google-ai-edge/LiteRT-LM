# Gemma 4

Canonical LiteRT-LM prompt templates and metadata configuration for the Gemma 4
model family.

## Chat Template

-   Canonical template: `chat_template.jinja` (12B, 26B, 31B)
-   Canonical template (2B, 4B): `chat_template_e2b_e4b.jinja`
-   Specification reference:
    [chat_template.jinja](https://huggingface.co/google/gemma-4-E2B-it/blob/main/chat_template.jinja)
    from Hugging Face.

### Modifications

1.  **Tool Response Handling**:

    -   `chat_template.jinja` / `chat_template_e2b_e4b.jinja`: Expects tool
        responses to be contained within the `content` field of a message with
        role `tool`.
    -   Hugging Face: Expects tool responses to be a `tool_responses` field. The
        role is not checked.

2.  **System Message**:

    -   `chat_template.jinja` / `chat_template_e2b_e4b.jinja`: Supports both
        string and sequence (list of parts) for system message content.
    -   Hugging Face: Only works if system message content is a string.

3.  **Turn Termination (`<turn|>`)**:

    -   `chat_template.jinja` / `chat_template_e2b_e4b.jinja`: Uses the previous
        message type to determine whether to append the turn terminator
        `<turn|>`.
    -   Hugging Face: Uses a condition based on the presence of `tool_responses`
        and `content`.
