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
from litert_lm_cli.commands.serve import gemini_handler
from litert_lm_cli.commands.serve import openai_common
from litert_lm_cli.commands.serve import openai_handler
from litert_lm_cli.commands.serve import util


def run_server(
    host: str,
    port: int,
    handler_class: type[http.server.BaseHTTPRequestHandler],
    api_name: str,
    cors_origins: tuple[str, ...],
    *,
    workers: int = 1,
    idle_unload: int = 0,
    preload: tuple[str, str | None, int | None] | None = None,
    verbose: bool = False,
) -> None:
  """Starts the HTTP server.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    handler_class: The HTTP handler class to use.
    api_name: The API protocol name (e.g., "OpenAI", "Gemini").
    cors_origins: Allowed CORS origins.
    workers: Number of engines that generate in parallel.
    idle_unload: Seconds without requests before an engine is unloaded (0
      keeps engines loaded).
    preload: `(model_id, backend, max_num_tokens)` to load on every worker at
      start-up, or None to load on the first request.
    verbose: Whether to log health-check requests.
  """
  server_address = (host, port)
  try:
    with util.LiteRTLMServer(
        server_address,
        handler_class,
        cors_origins,
        workers=workers,
        idle_unload_seconds=idle_unload,
        verbose=verbose,
    ) as server:
      click.echo(
          click.style(
              f"Starting {api_name}-compatible API server on {host}:{port}"
              f" ({workers} worker{'s' if workers != 1 else ''})...",
              fg="green",
              bold=True,
          )
      )
      server.start_background_tasks(preload=preload)
      try:
        server.serve_forever()
      finally:
        server.close_all_engines()
  except KeyboardInterrupt:
    click.echo(click.style("\nShutting down server...", fg="cyan"))


@click.command(
    cls=help_formatter.ColorCommand,
    help=(
        "Start an OpenAI-compatible API server.\n\n"
        "Supported OpenAI endpoints:\n"
        "  - /v1/models\n"
        "  - /v1/chat/completions\n"
        "  - /v1/embeddings\n\n"
        "Tips:\n"
        '  - Use "litert-lm import" to import a new model.\n'
        '  - Use "litert-lm list" to view already imported models.\n\n'
    ),
)
@common.config_option
@click.option("--host", default="127.0.0.1", type=str, help="Host to listen on")
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
@click.option(
    "--workers",
    default=1,
    type=click.IntRange(1, 64),
    show_default=True,
    help=(
        "Number of engines that generate in parallel. Each worker holds its"
        " own engine and KV cache, so memory grows with this number."
    ),
)
@click.option(
    "--idle-unload",
    "idle_unload",
    default=0,
    type=click.IntRange(0),
    show_default=True,
    help=(
        "Unload an engine after this many seconds without requests (0 keeps"
        " it loaded). Health checks do not count; the next request loads the"
        " engine again."
    ),
)
@click.option(
    "--preload",
    default=None,
    type=str,
    help=(
        "Load this model on every worker at start-up, in the same format as"
        ' the request\'s "model" field (for example "gemma3-1b,cpu,4096").'
    ),
)
def serve(
    host: str,
    port: int,
    *,
    api: str,
    cors_origin: tuple[str, ...],
    verbose: bool,
    workers: int = 1,
    idle_unload: int = 0,
    preload: str | None = None,
) -> None:
  """Starts a local HTTP server speaking the OpenAI or Gemini API protocol.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    api: The API protocol to use (openai or gemini).
    cors_origin: Allowed CORS origins.
    verbose: Whether to enable verbose logging.
    workers: Number of engines that generate in parallel.
    idle_unload: Seconds without requests before an engine is unloaded.
    preload: Model to load on every worker at start-up.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  preload_spec = None
  if preload:
    try:
      preload_spec = openai_common.parse_model_parameter(preload)
    except ValueError as e:
      raise click.BadParameter(str(e), param_hint="--preload") from e

  api_lower = api.lower()
  if api_lower == "gemini":
    handler_class = gemini_handler.GeminiHandler
    api_name = "Gemini"
  elif api_lower == "openai":
    handler_class = openai_handler.OpenAIHandler
    api_name = "OpenAI"
  else:
    raise click.BadParameter(f"Unsupported API: {api}")

  run_server(
      host,
      port,
      handler_class,
      api_name,
      cors_origin,
      workers=workers,
      idle_unload=idle_unload,
      preload=preload_spec,
      verbose=verbose,
  )


def register(cli: click.Group) -> None:
  """Registers the serve command."""
  cli.add_command(serve)
