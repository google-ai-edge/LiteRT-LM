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
import os
import traceback
from typing import Any, override
import urllib.request

import click

import litert_lm
from litert_lm_cli import (
    model as cli_model,
)
from litert_lm_cli.commands import serve_util


def _dump_json(data: Any) -> str:
  """Dumps data to a JSON string, ensuring non-ASCII characters are handled."""
  return json.dumps(data, ensure_ascii=False)


def _sse_data(data: str, event: str | None = None) -> bytes:
  """Formats data into a Server-Sent Event (SSE) message."""
  if event:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")
  return f"data: {data}\n\n".encode("utf-8")


def _format_sse_final() -> bytes:
  """Formats the final [DONE] event for Server-Sent Events."""
  return b"data: [DONE]\n\n"


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


def _is_gpu_only_model(model_path: str) -> bool:
  """Returns True if the model is GPU-only."""
  return serve_util._is_gpu_only_model(model_path)


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

  def __init__(self, now_str: str, created_ts: int, model_id: str):
    super().__init__(now_str, created_ts, model_id)
    self._chunk_id = f"chatcmpl_{now_str}"

  def format_initial(self) -> bytes:
    """Formats the initial chunk."""
    return _sse_data(
        _dump_json({
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": self._created_ts,
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })
    )

  def format_delta(self, text_output: str) -> bytes:
    """Formats a delta chunk with text content."""
    return _sse_data(
        _dump_json({
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": self._created_ts,
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": text_output},
                "finish_reason": None,
            }],
        })
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
        _dump_json({
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": self._created_ts,
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": openai_tool_calls},
                "finish_reason": None,
            }],
        })
    )

  @override
  def format_complete(self, finish_reason: str = "stop") -> bytes:
    """Formats the final chunk indicating completion."""
    return _sse_data(
        _dump_json({
            "id": self._chunk_id,
            "object": "chat.completion.chunk",
            "created": self._created_ts,
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        })
    )


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


class OpenAIHandler(http.server.BaseHTTPRequestHandler):
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
      max_completion_tokens: int | None = None,
  ) -> None:
    """Streams server-sent events using the provided formatter.

    Args:
      conv: The active LiteRT-LM conversation session.
      prompt: The input prompt payload (string or dictionary).
      formatter: The protocol-specific stream formatter.
      max_completion_tokens: The maximum number of tokens to generate.
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
      for chunk in conv.send_message_async(
          prompt, max_output_tokens=max_completion_tokens
      ):
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
    """
    if not stream:
      response = conv.send_message(
          prompt, max_output_tokens=max_completion_tokens
      )
      text_output = "".join(
          item.get("text", "")
          for item in response.get("content", [])
          if item.get("type") == "text"
      )

      tool_calls = response.get("tool_calls", [])
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
      }
      setattr(self, "_headers_sent", True)
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(
          json.dumps(resp_body, ensure_ascii=False).encode("utf-8")
      )
      return

    formatter = _OpenAIChatCompletionsFormatter(now_str, created_ts, model_id)
    self._stream_response(
        conv, prompt, formatter, max_completion_tokens=max_completion_tokens
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
    """
    if not stream:
      response = conv.send_message(prompt)
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
          json.dumps(dataclasses.asdict(resp_body), ensure_ascii=False).encode(
              "utf-8"
          )
      )
      return

    formatter = _OpenAIV1ResponsesFormatter(now_str, created_ts, model_id)
    self._stream_response(conv, prompt, formatter)

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
        if not _is_gpu_only_model(m.model_path):
          data.append({
              "id": m.model_id,
              "object": "model",
              "created": created_ts,
              "owned_by": "litert-lm",
          })
        data.append({
            "id": f"{m.model_id},gpu",
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
      self.wfile.write(
          json.dumps(resp_body, ensure_ascii=False).encode("utf-8")
      )
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

  def _get_engine_for_spec(
      self,
      model_spec: str,
      translated_messages: list[dict[str, Any]] | None = None,
      prompt: Any = None,
  ) -> litert_lm.Engine | None:
    """Parses the model spec and retrieves or initializes the engine.

    Args:
      model_spec: The model specification string.
      translated_messages: Optional list of already translated messages.
      prompt: Optional prompt payload.

    Returns:
      The LiteRT-LM Engine instance, or None if initialization failed.
    """
    try:
      spec = serve_util.parse_model_spec(model_spec)
      model_id = spec.model_id
    except ValueError as e:
      self.send_error(400, "".join(traceback.format_exception_only(e)))
      return None

    messages_to_scan = []
    if translated_messages:
      messages_to_scan.extend(translated_messages)
    if prompt and isinstance(prompt, dict) and prompt not in messages_to_scan:
      messages_to_scan.append(prompt)

    need_vision = False
    need_audio = False
    for msg in messages_to_scan:
      if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
          for part in content:
            if isinstance(part, dict):
              part_type = part.get("type")
              if part_type == "image":
                need_vision = True
              elif part_type == "audio":
                need_audio = True
      if need_vision and need_audio:
        break

    # TODO: b/515805503 - Make the backend customizable..
    vision_backend = litert_lm.Backend.CPU() if need_vision else None
    audio_backend = litert_lm.Backend.CPU() if need_audio else None

    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      return serve_util.get_or_initialize_server_engine(
          self.server,
          model_id=model_id,
          backend=spec.backend,
          max_num_tokens=spec.max_num_tokens,
          vision_backend=vision_backend,
          audio_backend=audio_backend,
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

    model_spec = body.get("model")
    if not model_spec:
      self.send_error(400, "Missing model")
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
        if not translated_messages or prompt is not last_msg:
          prompt = _translate_openai_message(prompt, name_by_tool_call_id)
      except ValueError as e:
        self.send_error(400, f"Invalid prompt: {e}")
        return

    if not prompt:
      self.send_error(400, "Missing input or messages")
      return

    engine = self._get_engine_for_spec(model_spec, translated_messages, prompt)
    if engine is None:
      return

    stream = body.get("stream", False)
    max_completion_tokens = body.get("max_completion_tokens")

    try:
      sampler_config = _parse_sampler_config(body)
    except ValueError as e:
      self.send_error(
          400,
          "Invalid sampler parameters: "
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
      with engine.create_conversation(
          messages=context_messages,
          tools=tools or None,
          automatic_tool_calling=False,
          sampler_config=sampler_config,
      ) as conv:
        now = datetime.datetime.now(datetime.timezone.utc)
        now_str = now.strftime("%Y%m%d%H%M%S%f")
        created_ts = int(now.timestamp())

        self._handle_chat_completions(
            conv,
            prompt,
            model_spec,
            stream,
            now_str=now_str,
            created_ts=created_ts,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, model_spec, prompt)

  def _handle_responses_endpoint(self) -> None:
    """Handles POST requests to responses endpoint."""
    body = self._parse_request_body()
    if body is None:
      return

    model_spec = body.get("model")
    prompt = body.get("input")

    if not model_spec or not prompt:
      self.send_error(400, "Missing model or input")
      return

    if isinstance(prompt, dict):
      try:
        prompt = _translate_openai_message(prompt)
      except ValueError as e:
        self.send_error(400, f"Invalid prompt: {e}")
        return

    engine = self._get_engine_for_spec(model_spec, prompt=prompt)
    if engine is None:
      return

    stream = body.get("stream", False)

    try:
      with engine.create_conversation(
          messages=[],
          automatic_tool_calling=False,
          sampler_config=None,
      ) as conv:
        now = datetime.datetime.now(datetime.timezone.utc)
        now_str = now.strftime("%Y%m%d%H%M%S%f")
        created_ts = int(now.timestamp())

        self._handle_responses(
            conv,
            prompt,
            stream,
            now_str=now_str,
            created_ts=created_ts,
            model_id=model_spec,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, model_spec, prompt)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles POST requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)

    router = {
        "/v1/chat/completions": self._handle_chat_completions_endpoint,
        "/v1/responses": self._handle_responses_endpoint,
    }

    if path_without_query in router:
      router[path_without_query]()
    else:
      self.send_error(404, "Not Found")
