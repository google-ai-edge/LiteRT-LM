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

"""Delete subcommand for LiteRT-LM CLI."""

import os
import shutil

import click

from litert_lm_cli import help_formatter
from litert_lm_cli import model


@click.command(
    cls=help_formatter.ColorCommand,
    help="Deletes a model from the local storage.",
)
@click.argument("model_id")
def delete(model_id):
  """Deletes a model from the local storage.

  Args:
    model_id: The ID of the model to delete.
  """
  model_obj = model.Model.from_model_id(model_id)
  model_dir = os.path.dirname(model_obj.model_path)
  if os.path.exists(model_dir) and model_dir.startswith(
      model.get_converted_models_base_dir()
  ):
    shutil.rmtree(model_dir)
    click.echo(click.style(f"Deleted model: {model_id}", fg="green"))
  else:
    click.echo(click.style(f"Model not found: {model_id}", fg="red"))


def register(cli: click.Group) -> None:
  """Registers the delete command."""
  cli.add_command(delete)
