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

"""Helper functions for LiteRT-LM CLI user interaction."""

import click
from prompt_toolkit.keys import Keys
import questionary

from litert_lm_cli import huggingface_download
from litert_lm_cli import model

_QUESTIONARY_STYLE = questionary.Style([
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("question", "bold"),
])


def prompt_selection(
    choices: list[str],
    message: str = "Please select an option",
) -> str:
  """Prompts the user to select an option from a list.

  Args:
    choices: The list of choices to present.
    message: The message to display.

  Returns:
    The selected choice.

  Raises:
    click.Abort: If the user cancels the selection.
  """
  q = questionary.select(
      message,
      choices=choices,
      style=_QUESTIONARY_STYLE,
  )

  # Users can quit the menu with "esc", "ctrl+d", and "ctrl+c" (already the
  # default).
  @q.application.key_bindings.add(Keys.Escape, eager=True)  # pyrefly: ignore[missing-attribute]
  @q.application.key_bindings.add(Keys.ControlD, eager=True)  # pyrefly: ignore[missing-attribute]
  def _(event):
    event.app.exit(exception=KeyboardInterrupt())

  val = q.ask()
  if val is None:
    raise click.Abort()
  return val


def resolve_model_file(
    from_huggingface_repo: str | None,
    huggingface_token: str | None,
) -> str:
  """Resolves the model file when it is missing, prompting the user.

  Args:
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.

  Returns:
    The resolved model file name/path.

  Raises:
    click.ClickException: If no files are found in the HF repo, or no local
      models are found.
    click.Abort: If the user cancels the selection.
  """

  if from_huggingface_repo:
    files = huggingface_download.list_litertlm_files(
        repo_id=from_huggingface_repo,
        token=huggingface_token,
    )
    if not files:
      raise click.ClickException(
          f"No .litertlm files found in repository {from_huggingface_repo!r}"
      )
    if len(files) == 1:
      click.echo(f"Found single .litertlm file: {files[0]!r}. Selecting it.")
      return files[0]
    return prompt_selection(
        choices=files,
        message="Please select a model file",
    )

  models = sorted(model.Model.get_all_models(), key=lambda m: m.model_id)
  if not models:
    raise click.ClickException(
        "No imported models found. Please import a model first using"
        " 'litert-lm import'."
    )
  if len(models) == 1:
    click.echo(
        f"Found single imported model: {models[0].model_id!r}. Selecting it."
    )
    return models[0].model_id

  return prompt_selection(
      choices=[m.model_id for m in models],
      message="Please select a model",
  )
