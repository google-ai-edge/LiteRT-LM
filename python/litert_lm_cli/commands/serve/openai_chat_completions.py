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

"""OpenAI /v1/chat/completions endpoint handler for LiteRT-LM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import json
import traceback
from typing import Any

# Migrate to built-in "typing" when min python version is 3.12.
from typing_extensions import override

import litert_lm
from litert_lm_cli.commands.serve import openai_common


def parse_sampler_config(
    body: dict[str, Any],
) -> litert_lm.SamplerConfig | None:
  """Parses and validates sampler parameters from the request body."""
  temperature = body.get("temperature")
  top_p = body.get("top_p")
  # Note: 'top_k' is not officially supported by the OpenAI API spec,
  # but we support it here as a custom parameter passed in the request body.
  top_k = body.get("top_k")
  seed = body.get("seed")

  if all(v is None for v in (temperature, top_p, top_k, seed)):
    return None

  return litert_lm.SamplerConfig(
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      seed=seed,
  )


class _OpenAIChatCompletionsFormatter(openai_common.OpenAIStreamFormatter):
  """A formatter for Server-Sent Events in the OpenAI Chat Completions API."""

  def __init__(
      self,
      now_str: str,
      created_ts: int,
      model_id: str,
      *,
      include_usage: bool = False,
  ):
    super().__init__(now_str, created_ts, model_id)
    self._chunk_id = f"chatcmpl_{now_str}"
    self._include_usage = include_usage

  def _make_payload(
      self,
      choices: Sequence[Mapping[str, Any]],
      usage: Mapping[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Creates a standardized payload for chat completion chunks."""
    payload: dict[str, Any] = {
        "id": self._chunk_id,
        "object": "chat.completion.chunk",
        "created": self._created_ts,
        "model": self._model_id,
        "choices": list(choices),
    }
    if usage is not None:
      payload["usage"] = dict(usage)
    elif self._include_usage:
      payload["usage"] = None
    return payload

  def format_initial(self) -> bytes:
    """Formats the initial chunk."""
    return openai_common.sse_data(
        openai_common.dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }])
        )
    )

  def format_delta(self, text_output: str) -> bytes:
    """Formats a delta chunk with text content."""
    return openai_common.sse_data(
        openai_common.dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {"content": text_output},
                "finish_reason": None,
            }])
        )
    )

  def format_tool_call_delta(
      self, tool_calls: Sequence[Mapping[str, Any]]
  ) -> bytes:
    """Formats a delta chunk with tool calls.

    Args:
      tool_calls: A sequence of tool calls returned by the model, where each
        tool call is a mapping containing 'function' details.

    Returns:
      A Server-Sent Event (SSE) message containing the formatted tool calls.
    """
    openai_tool_calls = [
        {
            "index": i,
            "id": f"call_{self._now_str}_{i}",
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name"),
                "arguments": json.dumps(
                    tc.get("function", {}).get("arguments", {})
                ),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]

    return openai_common.sse_data(
        openai_common.dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {"tool_calls": openai_tool_calls},
                "finish_reason": None,
            }])
        )
    )

  @override
  def format_complete(self, finish_reason: str = "stop") -> bytes:
    """Formats the final chunk indicating completion."""
    return openai_common.sse_data(
        openai_common.dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }])
        )
    )

  def format_usage(self, usage: Mapping[str, Any]) -> bytes:
    """Formats a chunk containing only token usage statistics."""
    return openai_common.sse_data(
        openai_common.dump_json(self._make_payload([], usage=usage))
    )


@dataclasses.dataclass
class _ProxyTool(litert_lm.Tool):
  """A proxy tool for OpenAPI definitions without implementation.

  Attributes:
    definition: A dictionary representing the OpenAPI tool definition.
  """

  definition: dict[str, Any]

  @override
  def get_tool_description(self) -> dict[str, Any]:
    """See base class."""
    return self.definition

  @override
  def execute(self, param: Any) -> Any:
    """Raises NotImplementedError as proxy tools are not executable."""
    raise NotImplementedError("Proxy tools are not executable.")


def _handle_chat_completions(
    handler: Any,
    conv: litert_lm.Conversation,
    prompt: str | dict[str, Any],
    model_id: str,
    stream: bool,
    *,
    now_str: str,
    created_ts: int,
    max_completion_tokens: int | None = None,
    stream_options: dict[str, Any] | None = None,
    response_format: litert_lm.ResponseFormat | None = None,
) -> None:
  """Generates responses for the OpenAI Chat Completions endpoint.

  Endpoint: `/v1/chat/completions` (and `/chat/completions`).
  - Request: Expects a JSON body with at least a "model" field and a
    "messages" array. The last message's "content" is used as the prompt.
    A "stream" field (boolean) can be included.
  - Response (Non-streaming): A JSON object in the OpenAI chat completion
    format, containing the model's text response.
  - Response (Streaming): Server-Sent Events (SSE) with
    `chat.completion.chunk` objects, including an initial role delta,
    content deltas, and a final delta with "stop" finish reason,
    terminated by `data: [DONE]`.

  Args:
    handler: The HTTP request handler instance.
    conv: The active LiteRT-LM conversation session.
    prompt: The input prompt extracted from the request messages.
    model_id: The target model identifier.
    stream: Whether to stream the response via Server-Sent Events.
    now_str: Timestamp string for unique identifier generation.
    created_ts: Epoch timestamp for creation metadata.
    max_completion_tokens: The maximum number of tokens to generate.
    stream_options: Options for streaming, such as include_usage.
    response_format: Optional response format for constrained decoding.
  """
  if not stream:
    text_parts = []
    tool_calls = []
    reasoning_tokens = 0
    for chunk in conv.send_message_async(
        prompt,
        max_output_tokens=max_completion_tokens,
        response_format=response_format,
    ):
      if chunk.get("channels"):
        reasoning_tokens += 1
      for item in chunk.get("content", []):
        if item.get("type") == "text":
          text_parts.append(item.get("text", ""))
      if chunk.get("tool_calls"):
        tool_calls.extend(chunk.get("tool_calls", []))

    text_output = "".join(text_parts)

    openai_tool_calls = [
        {
            "id": f"call_{now_str}_{i}",
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name"),
                "arguments": json.dumps(
                    tc.get("function", {}).get("arguments", {})
                ),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]

    resp_body = {
        "id": f"chatcmpl_{now_str}",
        "object": "chat.completion",
        "created": created_ts,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_output or None,
                **(
                    {"tool_calls": openai_tool_calls}
                    if openai_tool_calls
                    else {}
                ),
            },
            "finish_reason": "tool_calls" if openai_tool_calls else "stop",
        }],
        "usage": openai_common.compute_token_usage(
            conv, reasoning_tokens=reasoning_tokens
        ),
    }
    handler.headers_sent = True
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (openai_common.dump_json(resp_body, indent=2) + "\n").encode("utf-8")
    )
    return

  include_usage = bool(
      stream_options and stream_options.get("include_usage", False)
  )
  formatter = _OpenAIChatCompletionsFormatter(
      now_str, created_ts, model_id, include_usage=include_usage
  )
  handler.stream_response(
      conv,
      prompt,
      formatter,
      max_completion_tokens=max_completion_tokens,
      include_usage=include_usage,
      response_format=response_format,
  )


def handle_post_chat_completions(handler: Any) -> None:
  """Handles POST requests to chat completions endpoints."""
  body = handler.parse_request_body()
  if body is None:
    return

  raw_model_str = body.get("model")
  if not raw_model_str:
    handler.send_error(400, "Missing model")
    return

  try:
    model_id, backend, max_num_tokens = openai_common.parse_model_parameter(
        raw_model_str
    )
  except ValueError as e:
    handler.send_error(400, f"Invalid model parameter: {e}")
    return

  messages = body.get("messages")
  if isinstance(messages, list) and messages:
    name_by_tool_call_id = openai_common.build_name_by_tool_call_id_map(
        messages
    )

    try:
      translated_messages = [
          openai_common.translate_openai_message(m, name_by_tool_call_id)
          for m in messages
      ]
    except ValueError as e:
      handler.send_error(400, f"Invalid messages: {e}")
      return
  else:
    name_by_tool_call_id = None
    translated_messages = []

  if translated_messages:
    last_msg = translated_messages[-1]
    prompt = last_msg if isinstance(last_msg, dict) else body.get("input")
  else:
    prompt = body.get("input")

  if isinstance(prompt, dict):
    try:
      if not translated_messages or prompt is not last_msg:  # pyrefly: ignore[unbound-name]
        prompt = openai_common.translate_openai_message(
            prompt, name_by_tool_call_id
        )
    except ValueError as e:
      handler.send_error(400, f"Invalid prompt: {e}")
      return

  if not prompt:
    handler.send_error(400, "Missing input or messages")
    return

  engine = handler.get_engine(
      model_id,
      translated_messages,
      prompt,
      backend=backend,
      max_num_tokens=max_num_tokens,
  )
  if engine is None:
    return

  stream = body.get("stream", False)
  max_completion_tokens = body.get("max_completion_tokens")
  # bool is a subclass of int in Python, so isinstance(True, int) is True.
  # We must explicitly check for bool to prevent boolean values from passing.
  if max_completion_tokens is not None and (
      isinstance(max_completion_tokens, bool)
      or not isinstance(max_completion_tokens, int)
  ):
    handler.send_error(400, "max_completion_tokens must be an integer")
    return

  try:
    sampler_config = parse_sampler_config(body)
  except ValueError as e:
    handler.send_error(
        400,
        "Invalid sampler parameters: "
        + "".join(traceback.format_exception_only(e)),
    )
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

  # Parse tools if this is a chat completions request.
  tools_data = body.get("tools")
  tools = (
      [_ProxyTool(t) for t in tools_data if t.get("type") == "function"]
      if tools_data
      else []
  )

  try:
    context_messages = translated_messages[:-1] if translated_messages else []
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
        messages=context_messages,
        tools=tools or None,
        automatic_tool_calling=False,
        sampler_config=sampler_config,
        thinking_config=thinking_config,
        constrained_decoding_config=constrained_decoding_config,
    ) as conv:
      now = datetime.datetime.now(datetime.timezone.utc)
      now_str = now.strftime("%Y%m%d%H%M%S%f")
      created_ts = int(now.timestamp())

      stream_options = body.get("stream_options")
      if not isinstance(stream_options, dict):
        stream_options = {}

      _handle_chat_completions(
          handler,
          conv,  # pyrefly: ignore[bad-argument-type]
          prompt,
          raw_model_str,
          stream,
          now_str=now_str,
          created_ts=created_ts,
          max_completion_tokens=max_completion_tokens,
          stream_options=stream_options,
          response_format=response_format,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    handler.handle_inference_error(e, raw_model_str, prompt)
