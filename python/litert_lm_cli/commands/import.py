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

"""Import subcommand for LiteRT-LM CLI."""

import os
import pathlib
import shutil
import ssl
import textwrap
import urllib.error
import urllib.parse
import urllib.request

import click

from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import huggingface_download
from litert_lm_cli import model


def download_experimental_model(
    *,
    model_id: str,
    user_agent: str,
    ssl_context: ssl.SSLContext | None = None,
) -> str:
  """Downloads an experimental model.

  Args:
    model_id: The unique ID of the experimental model to download.
    user_agent: The secret passcode (User-Agent) for authentication.
    ssl_context: The SSL context to use for the connection.

  Returns:
    The absolute path to the downloaded temporary model file.

  Raises:
    click.ClickException: If the download fails.
  """
  url = f"https://dl.google.com/litert-lm/experimental/{urllib.parse.quote(model_id)}/model.litertlm"

  req = urllib.request.Request(url, headers={"User-Agent": user_agent})

  try:
    with urllib.request.urlopen(req, context=ssl_context) as response:
      total_size = common.parse_total_size(response.getheader("Content-Length"))
      size_suffix = common.download_size_suffix(total_size)

      click.echo(f"Downloading {model_id!r}{size_suffix}...")

      format_progress = lambda pos: common.format_download_progress(
          pos, total_size
      )
      downloading_dir = pathlib.Path(model.get_cli_base_dir()) / "downloading"
      return str(
          common.stream_download(
              response,
              download_dir=downloading_dir,
              length=total_size,
              format_progress=format_progress,
          )
      )

  except urllib.error.URLError as e:
    raise click.ClickException(
        f"Failed to download model {model_id!r}: {e!r}"
    ) from e


def _copy_source(
    source: str,
    dest: str,
    *,
    model_file: str,
    user_agent: str | None,
    ssl_context: ssl.SSLContext | None = None,
) -> str | None:
  """Copies the source file to dest, falling back to download if needed.

  If the source is a local file (equal to model_file) and is not found, and a
  user_agent is provided, it attempts to download it as an experimental model
  and
  then copies it.

  Args:
    source: The resolved source path (might be HF downloaded file or local
      file).
    dest: The destination path to copy to.
    model_file: The original model file argument (used for download ID).
    user_agent: The user agent for experimental model download.
    ssl_context: The SSL context to use for experimental download.

  Returns:
    The path to the temporary file if one was created and needs cleanup,
    otherwise None.

  Raises:
    click.ClickException: If the `source` file is not found and, if
      `user_agent` is provided, the attempt to download it as an experimental
      model
      also fails.
  """
  try:
    shutil.copy(source, dest)
    return None
  except FileNotFoundError as e:
    if source == model_file and user_agent:
      downloaded_file = download_experimental_model(
          model_id=model_file,
          user_agent=user_agent,
          ssl_context=ssl_context,
      )
      try:
        shutil.copy(downloaded_file, dest)
        return downloaded_file
      except BaseException:
        try:
          os.remove(downloaded_file)
        except OSError:
          pass
        raise
    raise click.ClickException(f"Source file not found: {source}") from e


@click.command(
    cls=help_formatter.ColorCommand,
    name="import",
    help=textwrap.dedent("""\
        Imports a model from a local path or HuggingFace hub.
        \b
        Examples:
          # Import from a local path
          litert-lm import ./model.litertlm my-model

          # Import from a HuggingFace repository
          litert-lm import --from-huggingface-repo org/repo model.litertlm my-model

          # Import and use the default model ID
          litert-lm import ./model.litertlm"""),
)
@common.huggingface_options
@click.option(
    "--user-agent",
    hidden=True,
    envvar="LITERT_LM_USER_AGENT",
    default=None,
    help="""The user agent used to download experimental models.""",
)
@click.argument("model_file")
@click.argument("model_ref", required=False)
def import_model(
    from_huggingface_repo: str | None,
    huggingface_token: str | None,
    user_agent: str | None,
    model_file: str,
    model_ref: str | None,
) -> None:
  """Imports a model from a local path or HuggingFace hub.

  Args:
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: HuggingFace API token.
    user_agent: The user agent used to download experimental models (internal).
    model_file: The path in the repo (if from-huggingface-repo is set) or local
      path.
    model_ref: The reference ID to store the model as. Defaults to the filename
      of MODEL_FILE.
  """
  effective_model_ref = model_ref or os.path.basename(model_file)
  temporary_file = None

  if from_huggingface_repo:
    source = huggingface_download.download_from_huggingface(
        repo_id=from_huggingface_repo,
        filename=model_file,
        token=huggingface_token,
    )
  else:
    source = model_file

  model_obj = model.Model.from_model_id(effective_model_ref)
  model_path = model_obj.model_path
  model_dir = os.path.dirname(model_path)

  os.makedirs(model_dir, exist_ok=True)

  ssl_context = None

  try:
    temporary_file = _copy_source(
        source,
        model_path,
        model_file=model_file,
        user_agent=user_agent,
        ssl_context=ssl_context,
    )
    click.echo(
        click.style(f"Successfully imported model to {model_path}", fg="green")
    )
    click.echo(
        click.style(
            "You can now run the model with 'litert-lm run"
            f" {effective_model_ref}'",
            fg="green",
        )
    )
  finally:
    if temporary_file is not None:
      try:
        os.remove(temporary_file)
      except OSError as e:
        click.echo(
            click.style(
                f"Failed to remove temporary file {temporary_file}: {e!r}",
                fg="yellow",
            )
        )


def register(cli: click.Group) -> None:
  """Registers the import command."""
  cli.add_command(import_model)
