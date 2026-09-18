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

"""Common utilities, formatters, and message translators for OpenAI endpoints."""

from __future__ import annotations

import abc
import base64
from collections.abc import Mapping, Sequence
import json
import traceback
from typing import Any
import urllib.request

import litert_lm
from litert_lm_cli import config as cli_config


def dump_json(data: Any, *, indent: int | None = None) -> str:
  """Dumps data to a JSON string, ensuring non-ASCII characters are handled."""
  return json.dumps(data, ensure_ascii=False, indent=indent)


def sse_data(data: str, event: str | None = None) -> bytes:
  """Formats data into a Server-Sent Event (SSE) message."""
  if event:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")
  return f"data: {data}\n\n".encode("utf-8")


def format_sse_final() -> bytes:
  """Formats the final [DONE] event for Server-Sent Events."""
  return b"data: [DONE]\n\n"


def parse_model_parameter(
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


def parse_thinking_config(
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
      model_id = parse_model_parameter(body["model"])[0]
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


def parse_response_format(
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


def compute_token_usage(
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


class OpenAIStreamFormatter(abc.ABC):
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
    return sse_data(
        dump_json({"error": "".join(traceback.format_exception_only(error))}),
        event="response.error",
    )

  def format_final(self) -> bytes:
    """Formats the final [DONE] event."""
    del self
    return format_sse_final()


def _parse_tool_arguments(args_str: str) -> dict[str, Any]:
  """Parses a JSON string of arguments into a dictionary, returning empty dict on error."""
  try:
    return json.loads(args_str)
  except json.JSONDecodeError:
    return {}


def build_name_by_tool_call_id_map(
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


def translate_openai_message(
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
