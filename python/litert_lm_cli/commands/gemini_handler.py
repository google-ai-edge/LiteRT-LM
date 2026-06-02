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

"""HTTP request handler for LiteRT-LM with Gemini-compatible API.

Reference: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

import collections.abc
import dataclasses
import http.server
import json
import re
import traceback
from typing import Any, NamedTuple

import click

import litert_lm
from litert_lm_cli.commands import serve_util

GEN_CONTENT_RE = re.compile(r"/v1beta/models/([^/\\:]+):generateContent")
STREAM_GEN_CONTENT_RE = re.compile(
    r"/v1beta/models/([^/\\:]+):streamGenerateContent"
)


class ConversationInputs(NamedTuple):
  """Inputs prepared for creating a conversation.

  Attributes:
    context_messages: List of context messages.
    tools: List of proxy tools.
    last_msg: The last message (prompt) dictionary, or None if empty.
  """

  context_messages: list[dict[str, Any]]
  tools: list[_ProxyTool]
  last_msg: dict[str, Any] | None


class _ProxyTool(litert_lm.Tool):
  """A tool that proxies OpenAPI definitions without implementation."""

  def __init__(self, definition: dict[str, Any]):
    self._definition = definition

  def get_tool_description(self) -> dict[str, Any]:
    return self._definition

  def execute(self, param: collections.abc.Mapping[str, Any]) -> Any:
    del self, param
    raise NotImplementedError("Proxy tools are not executable.")


def litertlm_message_from_gemini(
    gemini_content: collections.abc.Mapping[str, Any],
) -> dict[str, Any]:
  """Converts a Gemini API content object to a LiteRT-LM message."""
  role = gemini_content.get("role")
  if role == "model":
    role = "assistant"
  elif not role:
    role = "user"

  parts = gemini_content.get("parts", [])
  litertlm_parts = []
  tool_calls = []
  for p in parts:
    if "text" in p:
      litertlm_parts.append({"type": "text", "text": p["text"]})
    if "functionCall" in p:
      fc = p["functionCall"]
      tool_calls.append(
          {
              "function": {
                  "name": fc.get("name"),
                  "arguments": fc.get("args"),
              }
          }
      )
    if "functionResponse" in p:
      fr = p["functionResponse"]
      litertlm_parts.append({
          "type": "tool_response",
          "name": fr.get("name"),
          "response": fr.get("response"),
      })
      # LiteRT-LM uses "tool" as role for the function response.
      role = "tool"

  return {
      "role": role,
      **({"content": litertlm_parts} if litertlm_parts else {}),
      **({"tool_calls": tool_calls} if tool_calls else {}),
  }


def gemini_response_from_litertlm(
    litertlm_response: collections.abc.Mapping[str, Any],
    finish_reason: str = "STOP",
) -> dict[str, Any]:
  """Converts a LiteRT-LM response to a Gemini API response."""
  parts = []
  for item in litertlm_response.get("content", []):
    if item.get("type") == "text":
      parts.append({"text": item.get("text")})

  for tc in litertlm_response.get("tool_calls", []):
    f = tc.get("function", {})
    parts.append(
        {
            "functionCall": {
                "name": f.get("name"),
                "args": f.get("arguments"),
            }
        }
    )

  candidate: dict[str, Any] = {
      "content": {"role": "model", "parts": parts},
      "index": 0,
      **({"finishReason": finish_reason} if finish_reason else {}),
  }

  return {"candidates": [candidate]}


@dataclasses.dataclass(frozen=True)
class ParsedRequest:
  """A parsed Gemini API request from the URL path.

  Attributes:
    model_id: The identifier for the model.
    backend: The hardware backend to use, or None for auto-selection.
    max_num_tokens: The maximum number of tokens, or None for model default.
    is_stream: Whether the request is for streaming.
    error_msg: Error message if parsing failed, or None if successful.
  """

  model_id: str
  backend: litert_lm.Backend | None = None
  max_num_tokens: int | None = None
  is_stream: bool = False
  error_msg: str | None = None


def parse_model_and_backend(path: str) -> ParsedRequest:
  """Parses model spec, backend, max_tokens and stream flag from URL path.

  Args:
    path: The URL path.

  Returns:
    A ParsedRequest object.
  """
  path_without_query = path.split("?")[0]
  gen_match = GEN_CONTENT_RE.fullmatch(path_without_query)
  stream_match = STREAM_GEN_CONTENT_RE.fullmatch(path_without_query)

  match = gen_match or stream_match
  if not match:
    return ParsedRequest(model_id="", error_msg="Not Found")

  is_stream = bool(stream_match)
  model_spec = match.group(1)
  try:
    spec = serve_util.parse_model_spec(model_spec)
  except ValueError as e:
    return ParsedRequest(model_id="", is_stream=is_stream, error_msg=str(e))

  return ParsedRequest(
      model_id=spec.model_id,
      backend=spec.backend,
      max_num_tokens=spec.max_num_tokens,
      is_stream=is_stream,
  )


class GeminiHandler(http.server.BaseHTTPRequestHandler):
  """Handler for Gemini API requests."""

  def __init__(
      self,
      request: Any,
      client_address: Any,
      server: http.server.HTTPServer,
  ):
    """Pre-assigns internal routing state flags before standard lifecycle execution."""
    self._headers_sent = False
    super().__init__(request, client_address, server)

  def send_response(self, code, message=None):
    super().send_response(code, message)
    self._headers_sent = True

  def send_header(self, keyword, value):
    super().send_header(keyword, value)
    self._headers_sent = True

  def end_headers(self):
    super().end_headers()
    self._headers_sent = True

  def _get_post_data(self) -> dict[str, Any] | None:
    """Extracts and parses the JSON payload safely."""
    try:
      content_length = int(self.headers.get("Content-Length", 0))
      raw_data = self.rfile.read(content_length)
      return json.loads(raw_data.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
      return None

  def _parse_request_body(self, model_id: str) -> dict[str, Any] | None:
    """Parses request body, logs it, and handles invalid JSON.

    Args:
      model_id: The model identifier for logging.

    Returns:
      The parsed JSON body as a dictionary, or None if invalid.
    """
    body = self._get_post_data()
    if body is None:
      if not self._headers_sent:
        self.send_error(400, "Invalid JSON")
      return None
    click.echo(click.style(f"Request Body ({model_id}):", fg="magenta"))
    click.echo(json.dumps(body, indent=2, ensure_ascii=False))
    return body

  def _get_engine(self, req: ParsedRequest) -> litert_lm.Engine | None:
    """Retrieves or initializes the engine for the request.

    Args:
      req: The parsed request metadata.

    Returns:
      The LiteRT-LM Engine instance, or None if failed.
    """
    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      return serve_util.get_or_initialize_server_engine(
          self.server,
          model_id=req.model_id,
          backend=req.backend,
          max_num_tokens=req.max_num_tokens,
      )
    except FileNotFoundError as e:
      if not self._headers_sent:
        self.send_error(404, "".join(traceback.format_exception_only(e)))
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      if not self._headers_sent:
        self.send_error(500, f"Failed to load engine: {e!r}")
      return None

  def _handle_inference_error(
      self, e: Exception, model_id: str, prompt: Any
  ) -> None:
    """Handles inference errors by logging and sending 500.

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
    if not self._headers_sent and not self.wfile.closed:
      try:
        self.send_error(500, "".join(traceback.format_exception_only(e)))
      except BrokenPipeError:
        pass

  def _prepare_conversation_inputs(
      self, body: dict[str, Any]
  ) -> ConversationInputs:
    """Extracts system instruction, messages, and tools from body.

    Args:
      body: The parsed request body.

    Returns:
      The prepared ConversationInputs.
    """
    system_instruction = None
    si_data = body.get("systemInstruction") or body.get("system_instruction")
    if si_data:
      si_parts = si_data.get("parts", [])
      system_instruction = "".join(p.get("text", "") for p in si_parts)

    messages = [
        litertlm_message_from_gemini(c) for c in body.get("contents", [])
    ]

    tools = []
    tools_data = body.get("tools")
    if tools_data:
      for tool_entry in tools_data:
        for fd in tool_entry.get("functionDeclarations", []):
          tools.append(
              _ProxyTool({
                  "type": "function",
                  "function": fd,
              })
          )

    if not messages:
      return ConversationInputs([], [], None)

    # Last message is the prompt.
    last_msg = messages.pop()

    # Prefix messages (context)
    context_messages = []
    if system_instruction:
      context_messages.append({
          "role": "system",
          "content": [{"type": "text", "text": system_instruction}],
      })
    context_messages.extend(messages)

    return ConversationInputs(context_messages, tools, last_msg)

  def _parse_request_path(
      self, match: re.Match[str], is_stream: bool
  ) -> ParsedRequest | None:
    """Parses model spec and backend from regex match of the URL path.

    Args:
      match: The regex match object of the path.
      is_stream: Whether this is a streaming endpoint.

    Returns:
      The ParsedRequest metadata, or None if parsing failed.
    """
    model_spec = match.group(1)
    try:
      spec = serve_util.parse_model_spec(model_spec)
      return ParsedRequest(
          model_id=spec.model_id,
          backend=spec.backend,
          max_num_tokens=spec.max_num_tokens,
          is_stream=is_stream,
      )
    except ValueError as e:
      self.send_error(400, "".join(traceback.format_exception_only(e)))
      return None

  def _handle_generate_content(self, match: re.Match[str]) -> None:
    """Handles non-streaming generateContent requests.

    Args:
      match: The regex match object containing the model specification.
    """
    req = self._parse_request_path(match, is_stream=False)
    if req is None:
      return

    body = self._parse_request_body(req.model_id)
    if body is None:
      return

    engine = self._get_engine(req)
    if engine is None:
      return

    inputs = self._prepare_conversation_inputs(body)
    if not inputs.last_msg:
      if not self._headers_sent:
        self.send_error(400, "No contents provided")
      return

    try:
      with engine.create_conversation(
          messages=inputs.context_messages,
          tools=inputs.tools or None,
          automatic_tool_calling=False,
      ) as conv:
        response = conv.send_message(inputs.last_msg)
        click.echo(click.style("Raw Engine Response:", fg="magenta"))
        click.echo(json.dumps(response, ensure_ascii=False))

        resp_body = gemini_response_from_litertlm(response)
        click.echo(click.style("Gemini Response Body:", fg="magenta"))
        click.echo(json.dumps(resp_body, indent=2, ensure_ascii=False))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(resp_body, ensure_ascii=False).encode("utf-8")
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, req.model_id, inputs.last_msg)

  def _handle_stream_generate_content(self, match: re.Match[str]) -> None:
    """Handles streaming streamGenerateContent requests.

    Args:
      match: The regex match object containing the model specification.
    """
    req = self._parse_request_path(match, is_stream=True)
    if req is None:
      return

    body = self._parse_request_body(req.model_id)
    if body is None:
      return

    engine = self._get_engine(req)
    if engine is None:
      return

    inputs = self._prepare_conversation_inputs(body)
    if not inputs.last_msg:
      if not self._headers_sent:
        self.send_error(400, "No contents provided")
      return

    try:
      with engine.create_conversation(
          messages=inputs.context_messages,
          tools=inputs.tools or None,
          automatic_tool_calling=False,
      ) as conv:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        for chunk in conv.send_message_async(inputs.last_msg):
          click.echo(click.style("Stream Chunk:", fg="magenta"))
          click.echo(json.dumps(chunk, ensure_ascii=False))

          resp = gemini_response_from_litertlm(chunk, finish_reason="")
          self.wfile.write(
              f"data: {json.dumps(resp, ensure_ascii=False)}\n\n".encode(
                  "utf-8"
              )
          )
          self.wfile.flush()

        # Final chunk to signal completion
        final_resp = gemini_response_from_litertlm(
            {"content": []}, finish_reason="STOP"
        )
        click.echo(click.style("Final Stream Response:", fg="magenta"))
        click.echo(json.dumps(final_resp, ensure_ascii=False))

        self.wfile.write(
            f"data: {json.dumps(final_resp, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            )
        )
        self.wfile.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._handle_inference_error(e, req.model_id, inputs.last_msg)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles POST requests for generateContent and streamGenerateContent."""
    path_without_query = self.path.split("?")[0]

    router = {
        GEN_CONTENT_RE: self._handle_generate_content,
        STREAM_GEN_CONTENT_RE: self._handle_stream_generate_content,
    }

    for regex, handler in router.items():
      match = regex.fullmatch(path_without_query)
      if match:
        handler(match)
        return

    if not self._headers_sent:
      self.send_error(404, "Not Found")
