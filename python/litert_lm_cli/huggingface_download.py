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

"""Hugging Face model downloader and cacher for LiteRT-LM CLI."""

import http.client
import pathlib
import shutil
import urllib.error
import urllib.parse
import urllib.request

import click

from litert_lm_cli import common
from litert_lm_cli import model


def _download_and_save_file(
    response: http.client.HTTPResponse,
    *,
    filename: str,
    repo_id: str,
    target_path: pathlib.Path,
) -> None:
  """Downloads the file and saves it to target_path.

  Args:
    response: The HTTPResponse object to read the body from.
    filename: The name of the file being downloaded.
    repo_id: The HuggingFace repository ID.
    target_path: The local path where the file should be saved.

  Raises:
    click.ClickException: If downloading or saving the file fails.
  """
  total_size = common.parse_total_size(response.getheader("Content-Length"))
  size_suffix = common.download_size_suffix(total_size)

  click.echo(f"Downloading {filename!r} from {repo_id!r}{size_suffix}...")

  format_progress = lambda pos: common.format_download_progress(pos, total_size)
  tmp_file_path = None
  try:
    downloading_dir = pathlib.Path(model.get_cli_base_dir()) / "downloading"
    tmp_file_path = common.stream_download(
        response,
        download_dir=downloading_dir,
        length=total_size,
        format_progress=format_progress,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp_file_path, target_path)
  except Exception as e:
    if tmp_file_path is not None:
      try:
        tmp_file_path.unlink(missing_ok=True)
      except OSError:
        pass
    raise click.ClickException(
        f"Failed to save downloaded file {filename!r} to"
        f" {str(target_path)!r}: {e!r}"
    ) from e


def download_from_huggingface(
    *,
    repo_id: str,
    filename: str,
    token: str | None,
) -> str:
  """Downloads a file from HuggingFace Hub.

  Args:
    repo_id: The HuggingFace repository ID.
    filename: The filename to download.
    token: The HuggingFace API token.

  Returns:
    The local path to the downloaded file.

  Raises:
    click.ClickException: If the download fails.
  """
  quoted_filename = urllib.parse.quote(filename, safe="/")
  url = f"https://huggingface.co/{repo_id}/resolve/main/{quoted_filename}"

  cache_dir = (
      pathlib.Path(model.get_cli_base_dir()) / "cache" / "huggingface" / repo_id
  )
  target_path = cache_dir / filename

  try:
    # Try to open the file to check if it is already cached.
    with target_path.open("rb"):
      pass
  except FileNotFoundError:
    pass
  except OSError:
    pass
  else:
    click.echo(f"Using cached model: {str(target_path)!r}")
    return str(target_path)

  headers = {"Authorization": f"Bearer {token}"} if token is not None else {}

  req = urllib.request.Request(url, headers=headers)

  try:
    with urllib.request.urlopen(req) as response:
      _download_and_save_file(
          response, filename=filename, repo_id=repo_id, target_path=target_path
      )
      return str(target_path)

  except urllib.error.HTTPError as e:
    if e.code == 401 or e.code == 403:
      if token is None:
        click.echo(
            click.style(
                "HuggingFace token not found. If this is a private or gated"
                " repository, you can provide the token via the"
                " --huggingface-token option, or by setting the"
                " HF_TOKEN environment variable.",
                fg="yellow",
            )
        )
      raise click.ClickException(
          f"Error downloading {filename!r} from HuggingFace repo"
          f" {repo_id!r} (HTTP {e.code}): {e.reason!r}"
      ) from e

    raise click.ClickException(
        f"Error downloading {filename!r} from HuggingFace repo"
        f" {repo_id!r}: {e!r}"
    ) from e
  except urllib.error.URLError as e:
    raise click.ClickException(
        f"Error downloading {filename!r} from HuggingFace repo {repo_id!r}:"
        f" {e!r}"
    ) from e
