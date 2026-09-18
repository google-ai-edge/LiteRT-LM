# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI /v1/responses endpoint handler for LiteRT-LM."""

from __future__ import annotations

import dataclasses
import datetime
import traceback
from typing import Any

# Migrate to built-in "typing" when min python version is 3.12.
from typing_extensions import override

import litert_lm
from litert_lm_cli.commands.serve import openai_common


class _OpenAIV1ResponsesFormatter(openai_common.OpenAIStreamFormatter):
  """A formatter for Server-Sent Events in the OpenAI v1/responses API."""

  def __init__(self, now_str: str, created_ts: int, model_id: str):
    super().__init__(now_str, created_ts, model_id)
    self._resp_id = f"resp_{now_str}"

  def format_initial(self) -> bytes:
    """Formats the initial response.created event."""
    return openai_common.sse_data(
        openai_common.dump_json({"id": self._resp_id, "status": "in_progress"}),
        event="response.created",
    )

  def format_delta(self, text_output: str) -> bytes:
    """Formats a response.output_text.delta event."""
    return openai_common.sse_data(
        openai_common.dump_json({"delta": {"text": text_output}}),
        event="response.output_text.delta",
    )

  @override
  def format_complete(self, finish_reason: str = "stop") -> bytes:
    """Formats the response.completed event."""
    del finish_reason
    return openai_common.sse_data(
        openai_common.dump_json({"id": self._resp_id, "status": "completed"}),
        event="response.completed",
    )


@dataclasses.dataclass
class OutputContent:
  """Content metadata structure modeling generated output payload chunks.

  Attributes:
    type: The output content format identifier string.
    text: The generated raw string fragment.
    annotations: List of structural layout attachment stubs.
  """

  type: str
  text: str
  annotations: list[Any]


@dataclasses.dataclass
class ResponseOutput:
  """Message container segment tracking generation roles and status states.

  Attributes:
    id: Unique string identifier representing this specific generation output.
    type: The output container segment type descriptor.
    role: The entity role executing this specific output generation.
    status: The current processing lifecycle status identifier string.
    content: List of concrete generated content chunk models.
  """

  id: str
  type: str
  role: str
  status: str
  content: list[OutputContent]


@dataclasses.dataclass
class OpenAIResponse:
  """Top-level custom schema envelope wrapping compatible OpenAI outputs.

  Attributes:
    id: Unique string identifier for the overall response transaction.
    output: List of top-level output container segments.
  """

  id: str
  output: list[ResponseOutput]


def _handle_responses(
    handler: Any,
    conv: litert_lm.Conversation,
    prompt: str,
    stream: bool,
    *,
    now_str: str,
    created_ts: int,
    model_id: str,
    response_format: litert_lm.ResponseFormat | None = None,
) -> None:
  """Generates responses for the v1/responses endpoint.

  Endpoint: `/v1/responses`.
  - Request: Expects a JSON body with a "model" field and an "input" string.
    A "stream" field (boolean) can be included.
  - Response (Non-streaming): A custom JSON format containing the generated
    text.
  - Response (Streaming): SSEs with custom event types (`response.created`,
    `response.output_text.delta`, `response.completed`), terminated by
    `data: [DONE]`.

  Args:
    handler: The HTTP request handler instance.
    conv: The active LiteRT-LM conversation session.
    prompt: The input prompt string.
    stream: Whether to stream the response via Server-Sent Events.
    now_str: Timestamp string for unique identifier generation.
    created_ts: Epoch timestamp for creation metadata.
    model_id: The target model identifier.
    response_format: Optional response format for constrained decoding.
  """
  if not stream:
    response = conv.send_message(prompt, response_format=response_format)
    text_output = "".join(
        item.get("text", "")
        for item in response.get("content", [])
        if item.get("type") == "text"
    )
    resp_body = OpenAIResponse(
        id=f"resp_{now_str}",
        output=[
            ResponseOutput(
                id=f"msg_{now_str}",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    OutputContent(
                        type="output_text",
                        text=text_output,
                        annotations=[],
                    )
                ],
            )
        ],
    )
    handler.headers_sent = True
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (
            openai_common.dump_json(dataclasses.asdict(resp_body), indent=2)
            + "\n"
        ).encode("utf-8")
    )
    return

  formatter = _OpenAIV1ResponsesFormatter(now_str, created_ts, model_id)
  handler.stream_response(
      conv, prompt, formatter, response_format=response_format
  )


def handle_post_responses(handler: Any) -> None:
  """Handles POST requests to responses endpoint."""
  body = handler.parse_request_body()
  if body is None:
    return

  raw_model_str = body.get("model")
  prompt = body.get("input")

  if not raw_model_str or not prompt:
    handler.send_error(400, "Missing model or input")
    return

  try:
    model_id, backend, max_num_tokens = openai_common.parse_model_parameter(
        raw_model_str
    )
  except ValueError as e:
    handler.send_error(400, f"Invalid model parameter: {e}")
    return

  if isinstance(prompt, dict):
    try:
      prompt = openai_common.translate_openai_message(prompt)
    except ValueError as e:
      handler.send_error(400, f"Invalid prompt: {e}")
      return

  engine = handler.get_engine(
      model_id,
      prompt=prompt,
      backend=backend,
      max_num_tokens=max_num_tokens,
  )
  if engine is None:
    return

  try:
    thinking_config = openai_common.parse_thinking_config(
        body, model_id=model_id
    )
  except ValueError as e:
    handler.send_error(
        400,
        "Invalid thinking parameters: "
        + "".join(traceback.format_exception_only(e)),
    )
    return

  try:
    response_format = openai_common.parse_response_format(body)
  except ValueError as e:
    handler.send_error(
        400,
        "Invalid response_format parameters: "
        + "".join(traceback.format_exception_only(e)),
    )
    return

  stream = body.get("stream", False)

  try:
    provider = (
        litert_lm.LiteRtLmConstraintProviderType.LL_GUIDANCE
        if response_format is not None
        else None
    )
    constrained_decoding_config = litert_lm.ConstrainedDecodingConfig(
        enable=True,
        provider=provider,
    )
    with engine.create_conversation(
        messages=[],
        automatic_tool_calling=False,
        sampler_config=None,
        thinking_config=thinking_config,
        constrained_decoding_config=constrained_decoding_config,
    ) as conv:
      now = datetime.datetime.now(datetime.timezone.utc)
      now_str = now.strftime("%Y%m%d%H%M%S%f")
      created_ts = int(now.timestamp())

      _handle_responses(
          handler,
          conv,  # pyrefly: ignore[bad-argument-type]
          prompt,
          stream,
          now_str=now_str,
          created_ts=created_ts,
          model_id=raw_model_str,
          response_format=response_format,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    handler.handle_inference_error(e, raw_model_str, prompt)
