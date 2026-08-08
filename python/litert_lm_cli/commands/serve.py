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
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli.commands import gemini_handler
from litert_lm_cli.commands import openai_handler
from litert_lm_cli.commands import serve_util


def run_server(
    host: str,
    port: int,
    handler_class: type[http.server.BaseHTTPRequestHandler],
    api_name: str,
    cors_origins: tuple[str, ...],
) -> None:
  """Starts the HTTP server.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    handler_class: The HTTP handler class to use.
    api_name: The API protocol name (e.g., "OpenAI", "Gemini").
    cors_origins: Allowed CORS origins.
  """
  server_address = (host, port)
  try:
    with serve_util.LiteRTLMServer(
        server_address, handler_class, cors_origins
    ) as server:
      click.echo(
          click.style(
              f"Starting {api_name}-compatible API server on {host}:{port}...",
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
        "Start an OpenAI-compatible API server.\n\n"
        "Supported OpenAI endpoints:\n"
        "  - /v1/models\n"
        "  - /v1/chat/completions\n\n"
        "Tips:\n"
        '  - Use "litert-lm import" to import a new model.\n'
        '  - Use "litert-lm list" to view already imported models.\n\n'
    ),
)
@common.config_option
@click.option("--host", default="0.0.0.0", type=str, help="Host to listen on")
@click.option("--port", default=9379, type=int, help="Port to listen on")
@click.option(
    "--api",
    hidden=True,  # Hidden until Gemini implementation is ready.
    type=click.Choice(["openai", "gemini"], case_sensitive=False),
    default="openai",
    help="The API protocol to use.",
)
@click.option(
    "--cors-origin",
    multiple=True,
    default=[],
    help=(
        "Allowed CORS origins. Can be specified multiple times. Defaults to"
        " none (CORS disabled)."
    ),
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def serve(
    host: str,
    port: int,
    *,
    api: str,
    cors_origin: tuple[str, ...],
    verbose: bool,
) -> None:
  """Starts a local HTTP server speaking the OpenAI or Gemini API protocol.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    api: The API protocol to use (openai or gemini).
    cors_origin: Allowed CORS origins.
    verbose: Whether to enable verbose logging.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  api_lower = api.lower()
  if api_lower == "gemini":
    handler_class = gemini_handler.GeminiHandler
    api_name = "Gemini"
  elif api_lower == "openai":
    handler_class = openai_handler.OpenAIHandler
    api_name = "OpenAI"
  else:
    raise click.BadParameter(f"Unsupported API: {api}")

  run_server(host, port, handler_class, api_name, cors_origin)


def register(cli: click.Group) -> None:
  """Registers the serve command."""
  cli.add_command(serve)
