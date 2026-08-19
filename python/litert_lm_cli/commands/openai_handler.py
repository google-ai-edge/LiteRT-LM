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

"""OpenAI API compatible HTTP request handler for LiteRT-LM.

References:
* Responses API:
https://developers.openai.com/api/reference/resources/responses/methods/create
* Chat Completions API:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
"""

from __future__ import annotations

import abc
import base64
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import http.server
import json
import math
import os
import struct
import traceback
from typing import Any
import urllib.request

import click

# Migrate to built-in "typing" when min python version is 3.12.
from typing_extensions import override

import litert_lm
from litert_lm_cli import config as cli_config
from litert_lm_cli import model as cli_model
from litert_lm_cli.commands import serve_util


def _dump_json(data: Any, *, indent: int | None = None) -> str:
  """Dumps data to a JSON string, ensuring non-ASCII characters are handled."""
  return json.dumps(data, ensure_ascii=False, indent=indent)


def _embedding_to_base64(embedding: list[float]) -> str:
  """Converts a float embedding vector to IEEE 754 float32 base64 string."""
  raw_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
  return base64.b64encode(raw_bytes).decode("utf-8")


def _l2_normalize(vec: list[float]) -> list[float]:
  """Applies L2 normalization to a float vector (after dimension slicing)."""
  norm = math.sqrt(sum(x * x for x in vec))
  if norm > 0:
    return [x / norm for x in vec]
  return vec


def _parse_embedding_content_part(part: Any) -> str | litert_lm.Content:
  """Parses a content part (str, int, dict, or Content) into a LiteRT-LM input item."""
  if isinstance(part, str):
    return part
  if isinstance(part, int):
    return str(part)
  if isinstance(part, litert_lm.Content):
    return part
  if isinstance(part, dict):
    part_type = part.get("type")
    if part_type == "text":
      return part.get("text", "")
    if part_type == "image_url":
      image_url = part.get("image_url", {})
      url = image_url.get("url", "")
      if url.startswith("data:"):
        header, data = url.split(",", 1)
        if "base64" in header:
          return litert_lm.Content.ImageBytes(base64.b64decode(data))
        raise ValueError(
            "Unsupported data URL format (only base64 is supported)"
        )
      if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=10) as response:
          return litert_lm.Content.ImageBytes(response.read())
      path = url
      if path.startswith("file://"):
        path = path[7:]
      return litert_lm.Content.ImageFile(path)
    if part_type == "input_audio":
      input_audio = part.get("input_audio", {})
      data = input_audio.get("data", "")
      return litert_lm.Content.AudioBytes(base64.b64decode(data))
    if part_type == "image":
      if "blob" in part:
        return litert_lm.Content.ImageBytes(base64.b64decode(part["blob"]))
      if "path" in part:
        return litert_lm.Content.ImageFile(part["path"])
    if part_type == "audio":
      if "blob" in part:
        return litert_lm.Content.AudioBytes(base64.b64decode(part["blob"]))
      if "path" in part:
        return litert_lm.Content.AudioFile(part["path"])
    raise ValueError(f"Unsupported content part type: {part_type!r}")
  raise ValueError(f"Unsupported input element type: {type(part).__name__}")


def _normalize_embedding_input(
    input_data: Any,
) -> list[str | litert_lm.Content | list[str | litert_lm.Content]]:
  """Normalizes OpenAI embedding 'input' parameter into a batch of items."""
  if not input_data:
    raise ValueError("input cannot be empty")

  if isinstance(input_data, (str, dict, litert_lm.Content)):
    return [_parse_embedding_content_part(input_data)]

  if isinstance(input_data, list):
    if all(isinstance(x, int) and not isinstance(x, bool) for x in input_data):
      return [[_parse_embedding_content_part(x) for x in input_data]]

    if all(isinstance(x, str) for x in input_data):
      return [x for x in input_data]

    if all(isinstance(x, dict) for x in input_data):
      if all("type" in x for x in input_data):
        return [[_parse_embedding_content_part(x) for x in input_data]]
      return [_parse_embedding_content_part(x) for x in input_data]

    if all(isinstance(x, list) for x in input_data):
      return [
          [_parse_embedding_content_part(part) for part in item]
          for item in input_data
      ]

    return [_parse_embedding_content_part(x) for x in input_data]

  raise ValueError(f"Unsupported input type: {type(input_data).__name__}")


def _sse_data(data: str, event: str | None = None) -> bytes:
  """Formats data into a Server-Sent Event (SSE) message."""
  if event:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")
  return f"data: {data}\n\n".encode("utf-8")


def _format_sse_final() -> bytes:
  """Formats the final [DONE] event for Server-Sent Events."""
  return b"data: [DONE]\n\n"


def _parse_model_parameter(
    model_param: str,
) -> tuple[str, str | None, int | None]:
  """Parses a model parameter string into (model_id, backend, max_num_tokens).

  Format: "<model-id>[,<backend>[,<max-token>]]"

  Note:
    This syntax is supported mainly for backward compatibility. We will not
    add extra config values to it and might remove this support in the future.

  Args:
    model_param: The model string from the request.

  Returns:
    A tuple of (model_id, backend, max_num_tokens).

  Raises:
    ValueError: If model_param is not a string, max_num_tokens is not a positive
      integer, or format is invalid.
  """
  if not isinstance(model_param, str):
    raise ValueError(
        f"model parameter must be a string, got {type(model_param).__name__}"
    )

  parts = [p.strip() for p in model_param.split(",")]
  if len(parts) > 3:
    raise ValueError(
        "Too many comma-separated components in model parameter:"
        f" {model_param!r}"
    )

  model_id = parts[0]
  if not model_id:
    raise ValueError("model_id cannot be empty")

  backend = parts[1] if len(parts) > 1 and parts[1] else None

  max_num_tokens = None
  if len(parts) > 2 and parts[2]:
    try:
      max_num_tokens = int(parts[2])
    except ValueError:
      raise ValueError(
          f"Invalid max_num_tokens in model parameter: {parts[2]!r}"
      ) from None
    if max_num_tokens <= 0:
      raise ValueError(
          f"max_num_tokens must be a positive integer, got {max_num_tokens}"
      )

  return model_id, backend, max_num_tokens


def _parse_sampler_config(
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


def _parse_thinking_config(
    body: dict[str, Any],
    model_id: str | None = None,
) -> litert_lm.ThinkingConfig | None:
  """Parses and validates thinking/reasoning parameters from the request body or config.json."""
  reasoning_effort = body.get("reasoning_effort")
  if reasoning_effort is not None:
    if not isinstance(reasoning_effort, str):
      raise ValueError(
          "reasoning_effort must be a string, got"
          f" {type(reasoning_effort).__name__}"
      )

    effort_lower = reasoning_effort.lower()
    if effort_lower == "none":
      return litert_lm.ThinkingConfig(
          enable_thinking=False,
          thinking_token_budget=0,
      )

    supported_efforts = ("minimal", "low", "medium", "high", "xhigh")
    if effort_lower in supported_efforts:
      # TODO: b/514760339 - Support fine-grained reasoning effort token budget
      # mappings.
      return litert_lm.ThinkingConfig(
          enable_thinking=True,
          thinking_token_budget=-1,
      )

    raise ValueError(
        f"Invalid reasoning_effort value: {reasoning_effort!r}. "
        "Supported strings: none, minimal, low, medium, high, xhigh."
    )

  if model_id is None and isinstance(body.get("model"), str):
    try:
      model_id = _parse_model_parameter(body["model"])[0]
    except ValueError:
      model_id = body["model"]

  model_cfg = (
      cli_config.get_model_config(model_id)
      if model_id
      else cli_config.load_config().default
  )

  thinking = model_cfg.thinking
  thinking_budget = model_cfg.thinking_budget

  if thinking is None and thinking_budget is None:
    return None

  if thinking is None:
    thinking = thinking_budget != 0
  if thinking_budget is None:
    thinking_budget = -1 if thinking else 0

  return litert_lm.ThinkingConfig(
      enable_thinking=thinking,
      thinking_token_budget=thinking_budget,
  )


def _parse_response_format(
    body: dict[str, Any],
) -> litert_lm.ResponseFormat | None:
  """Parses and validates response_format parameters from the request body."""
  response_format = body.get("response_format")
  if response_format is None:
    return None

  if not isinstance(response_format, dict):
    raise ValueError(
        f"response_format must be a dict, got {type(response_format).__name__}"
    )

  response_format_type = response_format.get("type")
  if not response_format_type or response_format_type == "text":
    return None

  if response_format_type == "json_object":
    schema = (
        response_format.get("schema")
        or response_format.get("json_schema")
        or {}
    )
    if isinstance(schema, dict) and "schema" in schema:
      schema = schema["schema"]
    if not isinstance(schema, (dict, str)):
      raise ValueError("json_object schema must be a dict or str")
    return litert_lm.ResponseFormat.json(schema)

  if response_format_type == "json_schema":
    json_schema_obj = response_format.get("json_schema")
    schema = None
    if isinstance(json_schema_obj, dict):
      schema = json_schema_obj.get("schema", json_schema_obj)
    elif "schema" in response_format:
      schema = response_format.get("schema")

    if schema is None or not isinstance(schema, (dict, str)):
      raise ValueError(
          "json_schema response_format requires a dict or str schema"
      )
    return litert_lm.ResponseFormat.json(schema)

  if response_format_type == "regex":
    pattern = (
        response_format.get("regex")
        or response_format.get("pattern")
        or response_format.get("schema_or_pattern")
    )
    if not pattern or not isinstance(pattern, str):
      raise ValueError("regex response_format requires a string pattern/regex")
    return litert_lm.ResponseFormat.regex(pattern)

  raise ValueError(
      f"Unsupported response_format type: {response_format_type!r}"
  )


def _compute_token_usage(
    conv: litert_lm.Conversation,
    *,
    reasoning_tokens: int = 0,
) -> dict[str, Any]:
  """Computes token usage statistics for the completed conversation turn."""
  prompt_tokens = 0
  completion_tokens = 0

  try:
    info = conv.get_benchmark_info()
    prompt_tokens = info.last_prefill_token_count
    completion_tokens = info.last_decode_token_count
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  total_tokens = prompt_tokens + completion_tokens

  return {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "total_tokens": total_tokens,
      "completion_tokens_details": {
          "reasoning_tokens": reasoning_tokens,
      },
  }


class _OpenAIStreamFormatter(abc.ABC):
  """A formatter for OpenAI API compatible Server-Sent Events."""

  def __init__(self, now_str: str, created_ts: int, model_id: str):
    self._now_str = now_str
    self._created_ts = created_ts
    self._model_id = model_id

  @abc.abstractmethod
  def format_initial(self) -> bytes:
    """Formats the initial event(s) of the stream."""

  @abc.abstractmethod
  def format_delta(self, text_output: str) -> bytes:
    """Formats a delta event with new text output."""

  @abc.abstractmethod
  def format_complete(self, finish_reason: str = "stop") -> bytes:
    """Formats the completion event."""

  def format_error(self, error: Exception) -> bytes:
    """Formats an error event."""
    del self
    return _sse_data(
        _dump_json({"error": "".join(traceback.format_exception_only(error))}),
        event="response.error",
    )

  def format_final(self) -> bytes:
    """Formats the final [DONE] event."""
    del self
    return _format_sse_final()


class _OpenAIChatCompletionsFormatter(_OpenAIStreamFormatter):
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
    return _sse_data(
        _dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }])
        )
    )

  def format_delta(self, text_output: str) -> bytes:
    """Formats a delta chunk with text content."""
    return _sse_data(
        _dump_json(
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

    return _sse_data(
        _dump_json(
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
    return _sse_data(
        _dump_json(
            self._make_payload([{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }])
        )
    )

  def format_usage(self, usage: Mapping[str, Any]) -> bytes:
    """Formats a chunk containing only token usage statistics."""
    return _sse_data(_dump_json(self._make_payload([], usage=usage)))


class _OpenAIV1ResponsesFormatter(_OpenAIStreamFormatter):
  """A formatter for Server-Sent Events in the OpenAI v1/responses API."""

  def __init__(self, now_str: str, created_ts: int, model_id: str):
    super().__init__(now_str, created_ts, model_id)
    self._resp_id = f"resp_{now_str}"

  def format_initial(self) -> bytes:
    """Formats the initial response.created event."""
    return _sse_data(
        _dump_json({"id": self._resp_id, "status": "in_progress"}),
        event="response.created",
    )

  def format_delta(self, text_output: str) -> bytes:
    """Formats a response.output_text.delta event."""
    return _sse_data(
        _dump_json({"delta": {"text": text_output}}),
        event="response.output_text.delta",
    )

  @override
  def format_complete(self, finish_reason: str = "stop") -> bytes:
    """Formats the response.completed event."""
    del finish_reason
    return _sse_data(
        _dump_json({"id": self._resp_id, "status": "completed"}),
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


def _parse_tool_arguments(args_str: str) -> dict[str, Any]:
  """Parses a JSON string of arguments into a dictionary, returning empty dict on error."""
  try:
    return json.loads(args_str)
  except json.JSONDecodeError:
    return {}


def _build_name_by_tool_call_id_map(
    messages: Sequence[Any],
) -> dict[str, str]:
  """Builds a mapping from tool_call_id to function name from message history."""
  name_by_tool_call_id = {}
  for m in messages:
    if not isinstance(m, dict):
      continue
    if m.get("role") != "assistant":
      continue
    if "tool_calls" not in m:
      continue
    tool_calls = m.get("tool_calls")
    if not isinstance(tool_calls, list):
      continue
    for tc in tool_calls:
      if not isinstance(tc, dict):
        continue
      tc_id = tc.get("id")
      func = tc.get("function")
      if not isinstance(func, dict):
        continue
      name = func.get("name")
      if tc_id and name:
        name_by_tool_call_id[tc_id] = name
  return name_by_tool_call_id


def _translate_openai_message(
    msg: Any,
    name_by_tool_call_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Translates an OpenAI message to a LiteRT-LM message format.

  This function takes a message dictionary, typically from an OpenAI Chat
  Completions request, and transforms its content to a format understood
  by LiteRT-LM's `send_message_async`. Specifically, it handles multimodal
  inputs like image URLs and audio data.

  The input `msg` is expected to be a dictionary with at least a "role" and
  potentially a "content" field. The "content" field can be a string or
  a list of content parts. This function focuses on translating list-based
  content parts.

  Supported translations for `msg["content"]` items:
  -   `{"type": "text", "text": ...}`: Passed through as is.
  -   `{"type": "image_url", "image_url": {"url": "..."}}`:
      -   If `url` starts with "data:", it's assumed to be a base64 encoded
          image and translated to `{"type": "image", "blob": <base64_data>}`.
      -   If `url` starts with "http://" or "https://", the image is fetched,
          base64 encoded, and translated to
          `{"type": "image", "blob": <base64_data>}`.
      -   If `url` starts with "file://", it's translated to
          `{"type": "image", "path": <local_path>}`.
      -   Other URLs are treated as local paths.
  -   `{"type": "input_audio", "input_audio": {"data": "..."}}`:
      Translated to `{"type": "audio", "blob": <base64_data>}`.
  -   Other content part types are passed through without modification.

  Args:
    msg: The message object, expected to be a dictionary.
    name_by_tool_call_id: Optional mapping from tool_call_id to tool name.

  Returns:
    A dictionary representing the message in a LiteRT-LM compatible format,
    with multimodal content (like images/audio) transformed.

  Raises:
    ValueError: If `msg` is not a dictionary, or if an unsupported data URL
      format is provided for an image, or if a data URL is invalid.
    RuntimeError: If an error occurs while downloading an image from a URL.
  """
  if not isinstance(msg, dict):
    raise ValueError("Message must be an object")

  role = msg.get("role")
  content = msg.get("content")

  if role == "tool":
    tool_call_id = msg.get("tool_call_id")
    if not tool_call_id:
      raise ValueError("Tool message must have a tool_call_id")
    if not name_by_tool_call_id or tool_call_id not in name_by_tool_call_id:
      raise ValueError(
          f"No matching tool call found for tool_call_id: {tool_call_id!r}"
      )

    return {
        "role": "tool",
        "content": [{
            "type": "tool_response",
            "name": name_by_tool_call_id[tool_call_id],
            "response": content,
        }],
    }

  if role == "assistant" and "tool_calls" in msg:
    openai_tool_calls = msg.get("tool_calls", [])
    litert_tool_calls = [
        {
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name"),
                "arguments": _parse_tool_arguments(
                    tc.get("function", {}).get("arguments", "{}")
                ),
            },
        }
        for tc in openai_tool_calls
    ]
    return {
        "role": "assistant",
        "tool_calls": litert_tool_calls,
        **({"content": content} if content else {}),
    }

  if not isinstance(content, list):
    return msg

  translated_content = []
  for part in content:
    if not isinstance(part, dict):
      translated_content.append(part)
      continue

    part_type = part.get("type")
    if part_type == "text":
      translated_content.append(part)
    elif part_type == "image_url":
      image_url = part.get("image_url", {})
      url = image_url.get("url", "")
      if url.startswith("data:"):
        try:
          header, data = url.split(",", 1)
          if "base64" in header:
            translated_content.append({
                "type": "image",
                "blob": data,
            })
          else:
            raise ValueError(
                "Unsupported data URL format (only base64 is supported)"
            )
        except ValueError as e:
          if "Unsupported data URL format" in str(e):
            raise
          raise ValueError("Invalid data URL format") from e
      elif url.startswith(("http://", "https://")):
        try:
          with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            base64_data = base64.b64encode(data).decode("utf-8")
            translated_content.append({
                "type": "image",
                "blob": base64_data,
            })
        except Exception as e:
          raise RuntimeError(
              f"Failed to download image from {url}: {e!r}"
          ) from e
      else:
        path = url
        if path.startswith("file://"):
          path = path[7:]
        translated_content.append({
            "type": "image",
            "path": path,
        })
    elif part_type == "input_audio":
      # The OpenAI Chat Completions API protocol only supports audio input
      # inline via base64-encoded bytes in the 'data' field (no URL-based
      # audio).
      input_audio = part.get("input_audio", {})
      data = input_audio.get("data", "")
      translated_content.append({
          "type": "audio",
          "blob": data,
      })
    else:
      translated_content.append(part)

  return {
      "role": role,
      "content": translated_content,
  }


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


class OpenAIHandler(serve_util.CORSRequestHandler):
  """Handler for OpenAI API requests.

  Responses API:
  https://developers.openai.com/api/reference/resources/responses/methods/create

  Chat Completions API:
  https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

  Attributes:
    _headers_sent: Boolean flag tracking if HTTP response status headers have
      already been transmitted.
  """

  def __init__(
      self,
      request: Any,
      client_address: Any,
      server: http.server.HTTPServer,
  ):
    """Pre-assigns internal routing state flags before standard lifecycle execution."""
    self._headers_sent = False
    super().__init__(request, client_address, server)

  def _stream_response(
      self,
      conv: litert_lm.Conversation,
      prompt: str | dict[str, Any],
      formatter: _OpenAIStreamFormatter,
      *,
      max_completion_tokens: int | None = None,
      include_usage: bool = False,
      response_format: litert_lm.ResponseFormat | None = None,
  ) -> None:
    """Streams server-sent events using the provided formatter.

    Args:
      conv: The active LiteRT-LM conversation session.
      prompt: The input prompt payload (string or dictionary).
      formatter: The protocol-specific stream formatter.
      max_completion_tokens: The maximum number of tokens to generate.
      include_usage: Whether to emit a token usage chunk right before [DONE].
      response_format: Optional response format for constrained decoding.
    """
    self._headers_sent = True
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()

    try:
      self.wfile.write(formatter.format_initial())
      self.wfile.flush()

      has_tool_calls = False
      reasoning_tokens = 0
      for chunk in conv.send_message_async(
          prompt,
          max_output_tokens=max_completion_tokens,
          response_format=response_format,
      ):
        if chunk.get("channels"):
          reasoning_tokens += 1
        text_output = "".join(
            item.get("text", "")
            for item in chunk.get("content", [])
            if item.get("type") == "text"
        )
        if text_output:
          self.wfile.write(formatter.format_delta(text_output))
          self.wfile.flush()

        tool_calls = chunk.get("tool_calls", [])
        if tool_calls:
          has_tool_calls = True
          if hasattr(formatter, "format_tool_call_delta"):
            self.wfile.write(formatter.format_tool_call_delta(tool_calls))
            self.wfile.flush()

      finish_reason = "tool_calls" if has_tool_calls else "stop"
      self.wfile.write(formatter.format_complete(finish_reason=finish_reason))
      self.wfile.flush()
      if include_usage and hasattr(formatter, "format_usage"):
        usage_dict = _compute_token_usage(
            conv, reasoning_tokens=reasoning_tokens
        )
        self.wfile.write(formatter.format_usage(usage_dict))
        self.wfile.flush()
      self.wfile.write(formatter.format_final())
      self.wfile.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      click.echo(
          click.style(
              f"Error during streaming with prompt {prompt!r}: {e!r}\n"
              f"{traceback.format_exc()}",
              fg="red",
          )
      )
      conv.cancel_process()
      try:
        self.wfile.write(formatter.format_error(e))
        self.wfile.flush()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def _handle_chat_completions(
      self,
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
          "usage": _compute_token_usage(
              conv, reasoning_tokens=reasoning_tokens
          ),
      }
      setattr(self, "_headers_sent", True)
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write((_dump_json(resp_body, indent=2) + "\n").encode("utf-8"))
      return

    include_usage = bool(
        stream_options and stream_options.get("include_usage", False)
    )
    formatter = _OpenAIChatCompletionsFormatter(
        now_str, created_ts, model_id, include_usage=include_usage
    )
    self._stream_response(
        conv,
        prompt,
        formatter,
        max_completion_tokens=max_completion_tokens,
        include_usage=include_usage,
        response_format=response_format,
    )

  def _handle_responses(
      self,
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
      self._headers_sent = True
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          (_dump_json(dataclasses.asdict(resp_body), indent=2) + "\n").encode(
              "utf-8"
          )
      )
      return

    formatter = _OpenAIV1ResponsesFormatter(now_str, created_ts, model_id)
    self._stream_response(
        conv, prompt, formatter, response_format=response_format
    )

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Handles GET requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)
    if path_without_query != "/v1/models":
      self.send_error(404, "Not Found")
      return

    try:
      models = cli_model.Model.get_all_models()
      data = []
      for m in models:
        try:
          created_ts = int(os.path.getmtime(m.model_path))
        except OSError:
          created_ts = 0
        data.append({
            "id": m.model_id,
            "object": "model",
            "created": created_ts,
            "owned_by": "litert-lm",
        })

      resp_body = {
          "object": "list",
          "data": data,
      }

      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write((_dump_json(resp_body, indent=2) + "\n").encode("utf-8"))
    except Exception as e:  # pylint: disable=broad-exception-caught
      click.echo(
          click.style(
              f"Error listing models: {e!r}\n{traceback.format_exc()}",
              fg="red",
          )
      )
      if not self.wfile.closed:
        try:
          self.send_error(500, "".join(traceback.format_exception_only(e)))
        except BrokenPipeError:
          pass

  def _get_post_data(self) -> dict[str, Any] | None:
    """Extracts and parses the JSON payload safely."""
    try:
      content_length = int(self.headers.get("Content-Length", 0))
      raw_data = self.rfile.read(content_length)
      return json.loads(raw_data.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
      return None

  def _parse_request_body(self) -> dict[str, Any] | None:
    """Parses the request body as JSON, sending a 400 error if invalid."""
    body = self._get_post_data()
    if body is None:
      self.send_error(400, "Invalid JSON")
      return None
    return body

  def _get_engine(
      self,
      model_id: str,
      translated_messages: list[dict[str, Any]] | None = None,
      prompt: Any = None,
      backend: str | None = None,
      max_num_tokens: int | None = None,
  ) -> litert_lm.Engine | None:
    """Retrieves or initializes the engine for the given model ID.

    Args:
      model_id: The model identifier string.
      translated_messages: Optional list of already translated messages.
      prompt: Optional prompt payload.
      backend: Optional requested backend override.
      max_num_tokens: Optional requested max_num_tokens override.

    Returns:
      The LiteRT-LM Engine instance, or None if initialization failed.
    """
    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      return serve_util.get_or_initialize_server_engine(
          self.server,
          model_id=model_id,
          backend=backend,
          max_num_tokens=max_num_tokens,
      )
    except FileNotFoundError as e:
      self.send_error(404, "".join(traceback.format_exception_only(e)))
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load engine: {e!r}")
      return None

  def _handle_inference_error(
      self, e: Exception, model_id: str, prompt: Any
  ) -> None:
    """Handles errors occurring during inference by logging and sending 500.

    Args:
      e: The caught exception.
      model_id: The model identifier.
      prompt: The prompt payload.
    """
    click.echo(
        click.style(
            f"Error during inference for model {model_id!r} with prompt "
            f"{prompt!r}: {e!r}\n{traceback.format_exc()}",
            fg="red",
        )
    )
    if not self.wfile.closed and not self._headers_sent:
      try:
        self.send_error(500, "".join(traceback.format_exception_only(e)))
      except BrokenPipeError:
        pass

  def _handle_chat_completions_endpoint(self) -> None:
    """Handles POST requests to chat completions endpoints."""
    body = self._parse_request_body()
    if body is None:
      return

    raw_model_str = body.get("model")
    if not raw_model_str:
      self.send_error(400, "Missing model")
      return

    try:
      model_id, backend, max_num_tokens = _parse_model_parameter(raw_model_str)
    except ValueError as e:
      self.send_error(400, f"Invalid model parameter: {e}")
      return

    messages = body.get("messages")
    if isinstance(messages, list) and messages:
      name_by_tool_call_id = _build_name_by_tool_call_id_map(messages)

      try:
        translated_messages = [
            _translate_openai_message(m, name_by_tool_call_id) for m in messages
        ]
      except ValueError as e:
        self.send_error(400, f"Invalid messages: {e}")
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
          prompt = _translate_openai_message(prompt, name_by_tool_call_id)
      except ValueError as e:
        self.send_error(400, f"Invalid prompt: {e}")
        return

    if not prompt:
      self.send_error(400, "Missing input or messages")
      return

    engine = self._get_engine(
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
      self.send_error(400, "max_completion_tokens must be an integer")
      return

    try:
      sampler_config = _parse_sampler_config(body)
    except ValueError as e:
      self.send_error(
          400,
          "Invalid sampler parameters: "
          + "".join(traceback.format_exception_only(e)),
      )
      return

    try:
      thinking_config = _parse_thinking_config(body, model_id=model_id)
    except ValueError as e:
      self.send_error(
          400,
          "Invalid thinking parameters: "
          + "".join(traceback.format_exception_only(e)),
      )
      return

    try:
      response_format = _parse_response_format(body)
    except ValueError as e:
      self.send_error(
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

        self._handle_chat_completions(
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
      self._handle_inference_error(e, raw_model_str, prompt)

  def _handle_responses_endpoint(self) -> None:
    """Handles POST requests to responses endpoint."""
    body = self._parse_request_body()
    if body is None:
      return

    raw_model_str = body.get("model")
    prompt = body.get("input")

    if not raw_model_str or not prompt:
      self.send_error(400, "Missing model or input")
      return

    try:
      model_id, backend, max_num_tokens = _parse_model_parameter(raw_model_str)
    except ValueError as e:
      self.send_error(400, f"Invalid model parameter: {e}")
      return

    if isinstance(prompt, dict):
      try:
        prompt = _translate_openai_message(prompt)
      except ValueError as e:
        self.send_error(400, f"Invalid prompt: {e}")
        return

    engine = self._get_engine(
        model_id,
        prompt=prompt,
        backend=backend,
        max_num_tokens=max_num_tokens,
    )
    if engine is None:
      return

    try:
      thinking_config = _parse_thinking_config(body, model_id=model_id)
    except ValueError as e:
      self.send_error(
          400,
          "Invalid thinking parameters: "
          + "".join(traceback.format_exception_only(e)),
      )
      return

    try:
      response_format = _parse_response_format(body)
    except ValueError as e:
      self.send_error(
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

        self._handle_responses(
            conv,  # pyrefly: ignore[bad-argument-type]
            prompt,
            stream,
            now_str=now_str,
            created_ts=created_ts,
            model_id=raw_model_str,
            response_format=response_format,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, raw_model_str, prompt)

  def _get_embedding_engine(
      self,
      model_id: str,
      backend: str | None = None,
  ) -> litert_lm.EmbeddingEngine | None:
    """Retrieves or initializes the embedding engine for the given model ID.

    Args:
      model_id: The model identifier string.
      backend: Optional requested backend override.

    Returns:
      The LiteRT-LM EmbeddingEngine instance, or None if initialization failed.
    """
    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      return serve_util.get_or_initialize_server_embedding_engine(
          self.server,
          model_id=model_id,
          backend=backend,
      )
    except FileNotFoundError as e:
      self.send_error(404, "".join(traceback.format_exception_only(e)))
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load embedding engine: {e!r}")
      return None

  def _handle_embeddings_endpoint(self) -> None:
    """Handles POST requests to embeddings endpoints (/v1/embeddings)."""
    body = self._parse_request_body()
    if body is None:
      return

    raw_model_str = body.get("model")
    if not raw_model_str:
      self.send_error(400, "Missing model")
      return

    try:
      model_id, backend, _ = _parse_model_parameter(raw_model_str)
    except ValueError as e:
      self.send_error(400, f"Invalid model parameter: {e}")
      return

    input_data = body.get("input")
    try:
      contents_batch = _normalize_embedding_input(input_data)
    except (ValueError, RuntimeError) as e:
      self.send_error(400, f"Invalid input parameter: {e}")
      return

    encoding_format = body.get("encoding_format", "float")
    if encoding_format not in ("float", "base64"):
      self.send_error(
          400,
          f"Invalid encoding_format: {encoding_format!r}. Supported formats:"
          " 'float', 'base64'.",
      )
      return

    dimensions = body.get("dimensions")
    if dimensions is not None:
      if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        self.send_error(400, "dimensions must be an integer")
        return
      if dimensions <= 0:
        self.send_error(400, "dimensions must be a positive integer")
        return

    engine = self._get_embedding_engine(model_id, backend=backend)
    if engine is None:
      return

    options = litert_lm.EmbeddingOptions(
        normalize=body.get("normalize"),
        insert_special_tokens=body.get("insert_special_tokens"),
    )

    try:
      responses = engine.compute_embedding_batch(
          contents_batch, options=options
      )
      data_list = []
      for i, resp in enumerate(responses):
        vec = resp.embedding
        if dimensions is not None:
          if dimensions > len(vec):
            self.send_error(
                400,
                f"dimensions must be between 1 and {len(vec)}, got"
                f" {dimensions}",
            )
            return
          vec = _l2_normalize(vec[:dimensions])

        if encoding_format == "base64":
          formatted_embedding = _embedding_to_base64(vec)
        else:
          formatted_embedding = vec

        data_list.append({
            "object": "embedding",
            "index": i,
            "embedding": formatted_embedding,
        })

      # TODO: Populate prompt_tokens accurately once EmbeddingEngine exposes token
      # count metadata from tokenization.
      resp_body = {
          "object": "list",
          "data": data_list,
          "model": raw_model_str,
          "usage": {
              "prompt_tokens": 0,
              "total_tokens": 0,
          },
      }

      setattr(self, "_headers_sent", True)
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write((_dump_json(resp_body, indent=2) + "\n").encode("utf-8"))
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, raw_model_str, input_data)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles POST requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)

    router = {
        "/v1/chat/completions": self._handle_chat_completions_endpoint,
        "/v1/responses": self._handle_responses_endpoint,
        "/v1/embeddings": self._handle_embeddings_endpoint,
    }

    if path_without_query in router:
      router[path_without_query]()
    else:
      self.send_error(404, "Not Found")
