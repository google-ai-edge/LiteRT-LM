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

import collections.abc
import http.client
import os
import shutil
import ssl
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request

import click

from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import model

_DOWNLOAD_CHUNK_SIZE = 1024 * 64


def _stream_download(
    response: http.client.HTTPResponse,
    *,
    length: int | None,
    format_progress: collections.abc.Callable[[int], str],
) -> str:
  """Streams the response body to a temporary file with a progress bar.

  If any exception occurs during the download or file writing, the temporary
  file is guaranteed to be cleaned up.

  Args:
    response: The HTTPResponse object to read the body from.
    length: The total expected length of the download in bytes, if known. Used
      for the progress bar's total size.
    format_progress: A callable that takes the current progress position (in
      bytes) and returns a formatted string for the progress bar.

  Returns:
    The absolute path to the temporary file where the response body was written.
  """
  # Use a dedicated download directory within the user's home to avoid potential
  # space limitations or quota issues that might be present in /tmp.
  download_dir = os.path.join(model.get_cli_base_dir(), "downloading")
  os.makedirs(download_dir, exist_ok=True)
  tmp_file = tempfile.NamedTemporaryFile(dir=download_dir, delete=False)
  tmp_file_path = tmp_file.name

  try:
    with tmp_file:
      with click.progressbar(
          length=length,
          show_pos=False,
          show_percent=False,
          show_eta=False,
          item_show_func=lambda item: item,
          bar_template="[%(bar)s]  %(info)s",
          width=20,
      ) as bar:
        current_pos = 0
        for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK_SIZE), b""):
          tmp_file.write(chunk)
          current_pos += len(chunk)
          bar.update(len(chunk), current_item=format_progress(current_pos))
    return tmp_file_path
  except BaseException:
    # Ensure the file is closed before attempting to remove it.
    try:
      os.remove(tmp_file_path)
    except OSError:
      pass
    raise


def _format_size(size_in_bytes: int) -> str:
  """Formats bytes to a human-readable string (e.g., 18.2GB)."""
  for unit in ["B", "KB", "MB", "GB", "TB"]:
    if size_in_bytes < 1024.0:
      if unit == "B":
        return f"{int(size_in_bytes)}{unit}"
      return f"{size_in_bytes:.1f}{unit}"
    size_in_bytes /= 1024.0
  return f"{size_in_bytes:.1f}PB"


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
    response = urllib.request.urlopen(req, context=ssl_context)
  except urllib.error.URLError as e:
    raise click.ClickException(
        f"Failed to download model '{model_id}': {e!r}"
    ) from e

  with response:
    content_length = response.getheader("Content-Length")
    if content_length is None:
      total_size = None
      size_suffix = ""
    else:
      try:
        total_size = int(content_length)
        size_suffix = f" ({_format_size(total_size)})"
      except ValueError:
        total_size = None
        size_suffix = ""

    click.echo(f"Downloading {model_id!r}{size_suffix}...")

    def format_progress(current_pos_bytes: int) -> str:
      if total_size and total_size > 0:
        pct = int((current_pos_bytes / total_size) * 100)
        return f"{pct}%"

      if current_pos_bytes > 1024 * 1024:
        return f"{current_pos_bytes / (1024 * 1024):.1f} MB"
      return f"{current_pos_bytes / 1024:.1f} KB"

    return _stream_download(
        response,
        length=total_size,
        format_progress=format_progress,
    )


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
    downloaded_file = common.download_from_huggingface(
        from_huggingface_repo, model_file, huggingface_token
    )
    if not downloaded_file:
      raise click.ClickException(
          f"Failed to download model file '{model_file}' from HuggingFace"
          f" repository '{from_huggingface_repo}'."
      )
    source = downloaded_file
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
