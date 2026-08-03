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

"""Rename subcommand for LiteRT-LM CLI."""

import os
import shutil

import click

from litert_lm_cli import help_formatter
from litert_lm_cli import model


@click.command(
    cls=help_formatter.ColorCommand,
    help="Renames a model.",
)
@click.argument("old_model_id")
@click.argument("new_model_id")
def rename(old_model_id, new_model_id):
  """Renames a model.

  Args:
    old_model_id: The current model ID.
    new_model_id: The new model ID.
  """
  old_model = model.Model.from_model_id(old_model_id)
  if not old_model.exists():
    click.echo(click.style(f"Model not found: {old_model_id}", fg="red"))
    return

  new_model = model.Model.from_model_id(new_model_id)
  if new_model.exists():
    click.echo(
        click.style(f"Target model ID already exists: {new_model_id}", fg="red")
    )
    return

  old_dir = os.path.dirname(old_model.model_path)
  new_dir = os.path.dirname(new_model.model_path)

  os.makedirs(os.path.dirname(new_dir), exist_ok=True)
  shutil.move(old_dir, new_dir)
  click.echo(
      click.style(
          f'Renamed model "{old_model_id}" to "{new_model_id}"', fg="green"
      )
  )


def register(cli: click.Group) -> None:
  """Registers the rename command."""
  cli.add_command(rename)
