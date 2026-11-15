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

"""Unpack subcommand for LiteRT-LM CLI."""

import os
import sys
import textwrap

import click

from litert_lm_builder import litertlm_builder
from litert_lm_cli import help_formatter
from litert_lm_cli import model


def _is_interactive() -> bool:
  """Returns True if the current input stream is an interactive terminal."""
  return sys.stdin.isatty()


@click.command(
    cls=help_formatter.ColorCommand,
    help="Unpacks a LiteRT-LM file into an output directory.",
)
@click.argument("model_reference")
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help=textwrap.dedent("""\
        \b
        The directory where the unpacked files and model.toml will be saved.
          - If MODEL_REFERENCE is a model file, defaults to a subdirectory
            named after the model inside the same directory as the model file
            (e.g., my-model.litertlm -> my-model/).
          - If MODEL_REFERENCE is a model ID, setting --output-dir is required.
        """),
)
@click.option(
    "--allow-overwrite",
    is_flag=True,
    default=False,
    help=(
        "Allow overwriting files inside the output directory if it already"
        " exists."
    ),
)
@click.option(
    "--chat-template",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, path_type=str),
    help=(
        "Path to save the extracted jinja_prompt_template from LlmMetadata as"
        " a Jinja file. Raises an error if jinja_prompt_template is missing in"
        " the model."
    ),
)
def unpack(
    model_reference: str,
    output_dir: str | None = None,
    allow_overwrite: bool = False,
    chat_template: str | None = None,
):
  """Unpacks a LiteRT-LM file into an output directory.

  Args:
    model_reference: The model ID or local path to the LiteRT-LM file to unpack.
    output_dir: The directory where the unpacked files and model.toml will be
      saved. Required if model_reference is a model ID. Defaults to a directory
      named after the model inside the same directory as the model file if
      model_reference is a model file path.
    allow_overwrite: Whether to allow overwriting existing files inside
      output_dir.
    chat_template: Optional path to save the unpacked jinja_prompt_template.
  """
  model_reference = os.path.expanduser(model_reference)
  if output_dir is not None:
    output_dir = os.path.expanduser(output_dir)
  if chat_template is not None:
    chat_template = os.path.expanduser(chat_template)

  is_local_file = os.path.exists(model_reference)
  model_obj = model.Model.from_model_reference(model_reference)
  if not model_obj.exists():
    click.echo(
        click.style(
            f"Error: Model or file not found: '{model_reference}'", fg="red"
        )
    )
    return

  if output_dir is None:
    if not is_local_file:
      click.echo(
          click.style(
              "Error: --output-dir is required when unpacking a model ID.",
              fg="red",
          )
      )
      return
    model_basename = os.path.splitext(os.path.basename(model_reference))[0]
    parent_dir = os.path.dirname(model_reference) or "."
    output_dir = os.path.join(parent_dir, model_basename)

  if os.path.isfile(output_dir) or os.path.abspath(
      output_dir
  ) == os.path.abspath(model_reference):
    click.echo(
        click.style(
            f"Error: Cannot unpack into '{output_dir}' because it conflicts"
            " with an existing file or the model path. Please use --output-dir"
            " to specify the output directory.",
            fg="red",
        )
    )
    return

  if os.path.exists(output_dir) and not allow_overwrite:
    if _is_interactive():
      click.echo(f"Output directory '{output_dir}' already exists.")
      click.echo(
          "To avoid this prompt, use --allow-overwrite or specify a different"
          " --output-dir.\n"
      )
      if not click.confirm("Overwrite files inside?"):
        click.echo("Aborted.")
        return
    else:
      click.echo(
          click.style(
              f"Error: Output directory '{output_dir}' already exists. Please"
              " use a different --output-dir or pass --allow-overwrite to"
              " overwrite.",
              fg="red",
          )
      )
      return

  try:
    os.makedirs(output_dir, exist_ok=True)
    if chat_template is not None:
      toml_path = litertlm_builder.unpack(
          model_obj.model_path,
          output_dir,
          jinja_prompt_template_path=chat_template,
      )
    else:
      toml_path = litertlm_builder.unpack(model_obj.model_path, output_dir)
    click.echo(
        click.style(
            f"Successfully unpacked {model_reference} into {output_dir}\n"
            f"TOML configuration saved at {toml_path}",
            fg="green",
        )
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(click.style(f"Error unpacking model: {e!r}", fg="red"))


def register(cli: click.Group) -> None:
  """Registers the unpack command."""
  cli.add_command(unpack)
