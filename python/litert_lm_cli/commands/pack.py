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

"""Pack subcommand for LiteRT-LM CLI."""

import os
import sys
import textwrap

import click

from litert_lm_builder import litertlm_builder
from litert_lm_cli import help_formatter


def _is_interactive() -> bool:
  """Returns True if the current input stream is an interactive terminal."""
  return sys.stdin.isatty()


@click.command(
    cls=help_formatter.ColorCommand,
    help=textwrap.dedent("""\
        Packs a LiteRT-LM file using a TOML configuration file or a directory.

        Default output behaviors:
          - If CONFIG_OR_DIR is omitted, it defaults to 'model.toml' in the
            current directory, and the output defaults to 'output.litertlm' in
            the current directory.
          - If a TOML file is specified, the output defaults to 'output.litertlm'
            in the current directory.
          - If a directory is specified, it packs 'model.toml' inside it, and
            the output defaults to '{directory_name}.litertlm' (or
            '../{current_directory_name}.litertlm' if '.' is specified).
        """),
)
@click.argument(
    "config_or_dir",
    required=False,
    default="model.toml",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=str),
)
@click.option(
    "--output",
    default=None,
    type=click.Path(
        dir_okay=False, file_okay=True, writable=True, path_type=str
    ),
    help=textwrap.dedent("""\
        \b
        Path to output .litertlm file.
          - Defaults to 'output.litertlm' in the current directory when packing
            a TOML file (or when no argument is specified).
          - Defaults to '{directory_name}.litertlm' when packing a directory.
        """),
)
@click.option(
    "--allow-overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting the output file if it already exists.",
)
@click.option(
    "--chat-template",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    help=(
        "Path to a Jinja file. If provided, overwrites the"
        " jinja_prompt_template field in LlmMetadata."
    ),
)
def pack(
    config_or_dir: str = "model.toml",
    output: str | None = None,
    allow_overwrite: bool = False,
    chat_template: str | None = None,
):
  """Packs a TOML configuration file or directory into a LiteRT-LM file.

  Args:
    config_or_dir: Optional path to a TOML configuration file or directory
      containing model.toml. Defaults to the current directory.
    output: Optional path to output .litertlm file. Defaults to inferred path
      from config file name or directory name.
    allow_overwrite: Whether to allow overwriting an existing output file.
    chat_template: Optional path to a Jinja file to overwrite
      jinja_prompt_template.
  """
  config_or_dir = os.path.expanduser(config_or_dir)
  if chat_template is not None:
    chat_template = os.path.expanduser(chat_template)

  if os.path.isfile(config_or_dir):
    config_path = config_or_dir
  elif os.path.isdir(config_or_dir):
    config_path = os.path.join(config_or_dir, "model.toml")
  else:
    click.echo(
        click.style(
            f"Error: Invalid path type for '{config_or_dir}'.", fg="red"
        )
    )
    return

  if not os.path.isfile(config_path):
    msg = f"Error: TOML configuration file not found at '{config_path}'."
    click.echo(click.style(msg, fg="red"))
    return

  if output is not None:
    output_path = os.path.abspath(os.path.expanduser(output))
  else:
    if os.path.isfile(config_or_dir):
      output_path = os.path.abspath("output.litertlm")
    elif (
        config_or_dir == "."
        or not config_or_dir
        or os.path.abspath(config_or_dir) == os.path.abspath(".")
    ):
      model_name = os.path.basename(os.path.abspath("."))
      output_path = os.path.abspath(
          os.path.join("..", f"{model_name}.litertlm")
      )
    else:
      clean_path = os.path.normpath(config_or_dir)
      output_path = os.path.abspath(f"{clean_path}.litertlm")

  if os.path.isdir(output_path):
    click.echo(
        click.style(
            f"Error: Output path '{output_path}' is a directory. The output"
            " path must be a file.",
            fg="red",
        )
    )
    return

  if os.path.exists(output_path) and not allow_overwrite:
    if _is_interactive():
      click.echo(f"Output file '{output_path}' already exists.")
      click.echo(
          "To avoid this prompt, use --allow-overwrite or specify a different"
          " --output.\n"
      )
      if not click.confirm("Overwrite?"):
        click.echo("Aborted.")
        return
    else:
      click.echo(
          click.style(
              f"Error: Output file '{output_path}' already exists. Please use a"
              " different --output or pass --allow-overwrite to overwrite.",
              fg="red",
          )
      )
      return

  try:
    if chat_template is not None:
      litertlm_builder.pack(
          config_path, output_path, jinja_prompt_template_path=chat_template
      )
    else:
      litertlm_builder.pack(config_path, output_path)
    click.echo(
        click.style(
            f"Successfully packed LiteRT-LM file to {output_path}", fg="green"
        )
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(click.style(f"Error packing model: {e!r}", fg="red"))


def register(cli: click.Group) -> None:
  """Registers the pack command."""
  cli.add_command(pack)
