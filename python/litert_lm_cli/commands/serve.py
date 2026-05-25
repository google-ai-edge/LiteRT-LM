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

"""HTTP server for LiteRT-LM with Gemini-compatible API.

Reference: https://ai.google.dev/api/generate-content
"""

import collections.abc
import dataclasses
import http.server
import json
import re
from typing import Any

import click

import litert_lm
from litert_lm_cli import help_formatter
from litert_lm_cli.commands import openai_handler
from litert_lm_cli.commands import serve_util

GEN_CONTENT_RE = re.compile(r"^/v1beta/models/([^/\\:]+):generateContent$")
STREAM_GEN_CONTENT_RE = re.compile(
    r"^/v1beta/models/([^/\\:]+):streamGenerateContent$"
)


class _ProxyTool(litert_lm.Tool):
  """A tool that proxies OpenAPI definitions without implementation."""

  def __init__(self, definition: dict[str, Any]):
    self._definition = definition

  def get_tool_description(self) -> dict[str, Any]:
    return self._definition

  def execute(self, param: collections.abc.Mapping[str, Any]) -> Any:
    raise NotImplementedError("Proxy tools are not executable.")


def gemini_to_litertlm_message(
    gemini_content: dict[str, Any],
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


def litertlm_to_gemini_response(
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
  """Represents a parsed Gemini API request from the URL path.

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
  gen_match = GEN_CONTENT_RE.match(path_without_query)
  stream_match = STREAM_GEN_CONTENT_RE.match(path_without_query)

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

  # do_POST is the method name expected by http.server.BaseHTTPRequestHandler
  # to handle POST requests.
  def do_POST(self):  # pylint: disable=invalid-name
    """Handles POST requests for generateContent and streamGenerateContent."""
    req = parse_model_and_backend(self.path)
    if req.error_msg == "Not Found":
      self.send_error(404, "Not Found")
      return
    elif req.error_msg:
      self.send_error(400, req.error_msg)
      return

    model_id = req.model_id
    backend = req.backend
    max_tokens = req.max_num_tokens
    is_stream = req.is_stream

    content_length = int(self.headers.get("Content-Length", 0))
    try:
      body = json.loads(self.rfile.read(content_length))
    except json.JSONDecodeError:
      self.send_error(400, "Invalid JSON")
      return

    click.echo(click.style(f"Request Body ({model_id}):", fg="magenta"))
    click.echo(json.dumps(body, indent=2, ensure_ascii=False))

    try:
      assert isinstance(self.server, serve_util.LiteRTLMServer)
      engine = serve_util.get_or_initialize_server_engine(
          self.server,
          model_id=model_id,
          backend=backend,
          max_num_tokens=max_tokens,
      )
    except FileNotFoundError as e:
      self.send_error(404, str(e))
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Failed to load engine: {e}")
      return

    system_instruction = None
    si_data = body.get("systemInstruction") or body.get("system_instruction")
    if si_data:
      si_parts = si_data.get("parts", [])
      system_instruction = "".join(p.get("text", "") for p in si_parts)

    messages = [gemini_to_litertlm_message(c) for c in body.get("contents", [])]

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
      self.send_error(400, "No contents provided")
      return

    # Last message is the prompt.
    # Note: send_message expects Mapping[str, Any] with 'role' and 'content'.
    last_msg = messages.pop()

    # Prefix messages (context)
    context_messages = []
    if system_instruction:
      context_messages.append({
          "role": "system",
          "content": [{"type": "text", "text": system_instruction}],
      })
    context_messages.extend(messages)

    try:
      with engine.create_conversation(
          messages=context_messages,
          tools=tools or None,
          automatic_tool_calling=False,
      ) as conv:
        if is_stream:
          self.send_response(200)
          self.send_header("Content-Type", "text/event-stream")
          self.send_header("Cache-Control", "no-cache")
          self.end_headers()

          for chunk in conv.send_message_async(last_msg):
            click.echo(click.style("Stream Chunk:", fg="magenta"))
            click.echo(json.dumps(chunk, ensure_ascii=False))

            resp = litertlm_to_gemini_response(chunk, finish_reason="")
            self.wfile.write(
                f"data: {json.dumps(resp, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
            )
            self.wfile.flush()

          # Final chunk to signal completion
          final_resp = litertlm_to_gemini_response(
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
        else:
          response = conv.send_message(last_msg)
          click.echo(click.style("Raw Engine Response:", fg="magenta"))
          click.echo(json.dumps(response, ensure_ascii=False))

          resp_body = litertlm_to_gemini_response(response)
          click.echo(click.style("Gemini Response Body:", fg="magenta"))
          click.echo(json.dumps(resp_body, indent=2, ensure_ascii=False))

          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(
              json.dumps(resp_body, ensure_ascii=False).encode("utf-8")
          )

    except Exception as e:  # pylint: disable=broad-exception-caught
      click.echo(click.style(f"Error during inference: {e}", fg="red"))
      if not self.wfile.closed:
        try:
          self.send_error(500, str(e))
        except BrokenPipeError:
          pass


def run_server(
    host: str,
    port: int,
    handler_class: type[http.server.BaseHTTPRequestHandler],
) -> None:
  """Starts the HTTP server.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    handler_class: The HTTP handler class to use.
  """
  server_address = (host, port)
  try:
    with serve_util.LiteRTLMServer(server_address, handler_class) as server:
      click.echo(
          click.style(
              f"Starting LiteRT-LM API server on {host}:{port}...",
              fg="green",
              bold=True,
          )
      )
      try:
        server.serve_forever()
      finally:
        if server.litert_lm_engine is not None:
          server.litert_lm_engine.__exit__(None, None, None)
  except KeyboardInterrupt:
    click.echo(click.style("\nShutting down server...", fg="cyan"))


@click.command(
    cls=help_formatter.ColorCommand,
    help=(
        "Start a server with a Gemini or OpenAI compatible API (alpha feature)"
    ),
)
@click.option("--host", default="localhost", type=str, help="Host to listen on")
@click.option("--port", default=9379, type=int, help="Port to listen on")
@click.option(
    "--api",
    type=click.Choice(["gemini", "openai"], case_sensitive=False),
    default="gemini",
    help="The API protocol to use.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def serve(host: str, port: int, *, api: str, verbose: bool) -> None:
  """Starts a local HTTP server speaking the Gemini or OpenAI API protocol.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    api: The API protocol to use (gemini or openai).
    verbose: Whether to enable verbose logging.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  api_lower = api.lower()
  if api_lower == "gemini":
    handler_class = GeminiHandler
  elif api_lower == "openai":
    handler_class = openai_handler.OpenAIHandler
  else:
    raise click.BadParameter(f"Unsupported API: {api}")

  run_server(host, port, handler_class)


def register(cli: click.Group) -> None:
  """Registers the serve command."""
  cli.add_command(serve)
