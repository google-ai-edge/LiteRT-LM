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
import dataclasses
import datetime
import http.server
import json
import traceback
from typing import Any

import click

import litert_lm
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
  def format_complete(self) -> bytes:
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

  def format_complete(self) -> bytes:
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
                "finish_reason": "stop",
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

  def format_complete(self) -> bytes:
    """Formats the response.completed event."""
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
  ) -> None:
    """Streams server-sent events using the provided formatter.

    Args:
      conv: The active LiteRT-LM conversation session.
      prompt: The input prompt payload (string or dictionary).
      formatter: The protocol-specific stream formatter.
    """
    self._headers_sent = True
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()

    try:
      self.wfile.write(formatter.format_initial())
      self.wfile.flush()

      for chunk in conv.send_message_async(prompt):
        text_output = "".join(
            item.get("text", "")
            for item in chunk.get("content", [])
            if item.get("type") == "text"
        )
        if text_output:
          self.wfile.write(formatter.format_delta(text_output))
          self.wfile.flush()

      self.wfile.write(formatter.format_complete())
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
    """
    if not stream:
      response = conv.send_message(prompt)
      text_output = "".join(
          item.get("text", "")
          for item in response.get("content", [])
          if item.get("type") == "text"
      )
      resp_body = {
          "id": f"chatcmpl_{now_str}",
          "object": "chat.completion",
          "created": created_ts,
          "model": model_id,
          "choices": [{
              "index": 0,
              "message": {
                  "role": "assistant",
                  "content": text_output,
              },
              "finish_reason": "stop",
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
    self._stream_response(conv, prompt, formatter)

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

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles POST requests for OpenAI API compatible endpoints."""
    path_without_query, *_ = self.path.split("?", 1)
    is_chat_completions = path_without_query in (
        "/v1/chat/completions",
        "/chat/completions",
    )
    if path_without_query != "/v1/responses" and not is_chat_completions:
      self.send_error(404, "Not Found")
      return

    content_length = int(self.headers.get("Content-Length", 0))
    try:
      body = json.loads(self.rfile.read(content_length))
    except json.JSONDecodeError:
      self.send_error(400, "Invalid JSON")
      return

    model_spec = body.get("model")
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
      last_msg = messages[-1]
      prompt = last_msg if isinstance(last_msg, dict) else body.get("input")
    else:
      prompt = body.get("input")

    if not model_spec or not prompt:
      self.send_error(400, "Missing model or input/messages")
      return

    try:
      spec = serve_util.parse_model_spec(model_spec)
      model_id = spec.model_id
    except ValueError as e:
      self.send_error(400, "".join(traceback.format_exception_only(e)))
      return

    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      engine = serve_util.get_or_initialize_server_engine(
          self.server,
          model_id=model_id,
          backend=spec.backend,
          max_num_tokens=spec.max_num_tokens,
      )
    except FileNotFoundError as e:
      self.send_error(404, "".join(traceback.format_exception_only(e)))
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load engine: {e!r}")
      return

    stream = body.get("stream", False)

    try:
      context_messages = (
          messages[:-1]
          if is_chat_completions and isinstance(messages, list)
          else []
      )
      with engine.create_conversation(
          messages=context_messages,
          automatic_tool_calling=False,
      ) as conv:
        now = datetime.datetime.now(datetime.timezone.utc)
        now_str = now.strftime("%Y%m%d%H%M%S%f")
        created_ts = int(now.timestamp())

        if is_chat_completions:
          self._handle_chat_completions(
              conv,
              prompt,
              model_spec,
              stream,
              now_str=now_str,
              created_ts=created_ts,
          )
        else:
          self._handle_responses(
              conv,
              prompt,
              stream,
              now_str=now_str,
              created_ts=created_ts,
              model_id=model_spec,
          )

    except Exception as e:  # pylint: disable=broad-exception-caught
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
