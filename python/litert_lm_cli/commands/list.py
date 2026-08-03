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

"""List subcommand for LiteRT-LM CLI."""

import datetime
import os

import click

from litert_lm_cli import help_formatter
from litert_lm_cli import model


@click.command(cls=help_formatter.ColorCommand, name="list")
def list_models():
  """Lists all imported LiteRT-LM models."""
  base_dir = model.get_converted_models_base_dir()
  click.echo(f"Listing models in: {base_dir}")

  models = sorted(model.Model.get_all_models(), key=lambda m: m.model_id)

  # Calculate dynamic width for ID column
  id_width = max([len(m.model_id) for m in models] + [len("ID"), 25]) + 2

  click.echo(
      click.style(f"{'ID':<{id_width}} {'SIZE':<15} {'MODIFIED'}", bold=True)
  )

  for model_item in models:
    path = model_item.model_path
    try:
      stat = os.stat(path)
      size_bytes = stat.st_size
      if size_bytes >= 1024 * 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
      else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
      modified_date = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
          "%Y-%m-%d %H:%M:%S"
      )
    except FileNotFoundError:
      size_str = "Unknown"
      modified_date = "Unknown"

    click.echo(
        f"{model_item.model_id:<{id_width}} {size_str:<15} {modified_date}"
    )


def register(cli: click.Group) -> None:
  """Registers the list command."""
  cli.add_command(list_models)
