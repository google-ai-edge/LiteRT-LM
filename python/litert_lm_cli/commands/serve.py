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

from __future__ import annotations

import http.server

import click

import litert_lm
from litert_lm_cli import help_formatter
from litert_lm_cli.commands import gemini_handler
from litert_lm_cli.commands import openai_handler
from litert_lm_cli.commands import serve_util


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
@click.option("--host", default="0.0.0.0", type=str, help="Host to listen on")
@click.option("--port", default=9379, type=int, help="Port to listen on")
@click.option(
    "--api",
    type=click.Choice(["openai", "gemini"], case_sensitive=False),
    default="openai",
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
    handler_class = gemini_handler.GeminiHandler
  elif api_lower == "openai":
    handler_class = openai_handler.OpenAIHandler
  else:
    raise click.BadParameter(f"Unsupported API: {api}")

  run_server(host, port, handler_class)


def register(cli: click.Group) -> None:
  """Registers the serve command."""
  cli.add_command(serve)
