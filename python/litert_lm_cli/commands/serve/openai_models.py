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

"""OpenAI /v1/models endpoint handler for LiteRT-LM."""

from __future__ import annotations

import http.server
import os
import traceback

import click

from litert_lm_cli import model as cli_model
from litert_lm_cli.commands.serve import openai_common


def handle_get_models(handler: http.server.BaseHTTPRequestHandler) -> None:
  """Handles GET /v1/models requests."""
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

    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (openai_common.dump_json(resp_body, indent=2) + "\n").encode("utf-8")
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(
        click.style(
            f"Error listing models: {e!r}\n{traceback.format_exc()}",
            fg="red",
        )
    )
    if not handler.wfile.closed:
      try:
        handler.send_error(500, "".join(traceback.format_exception_only(e)))
      except BrokenPipeError:
        pass
