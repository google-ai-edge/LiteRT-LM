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

"""Shared options and helper functions for LiteRT-LM CLI."""

import collections.abc
import http.client
import pathlib
import tempfile
import textwrap

import click


def parse_speculative_decoding(unused_ctx, unused_param, value):
  """Click callback to parse speculative decoding mode strings into bool | None.

  Args:
    unused_ctx: The click context.
    unused_param: The click parameter.
    value: The value to parse ("auto", "true", or "false").

  Returns:
    True for "true", False for "false", and None for "auto".
  """
  if value is None:
    return None
  value_lower = value.lower()
  if value_lower == "auto":
    return None
  elif value_lower == "true":
    return True
  elif value_lower == "false":
    return False
  return value


def cache_dir_value_from_cache_mode(cache: str) -> str:
  """Returns the cache directory value for the given cache mode.

  Args:
    cache: The cache mode. Valid values are "disk", "memory", "no".

  Returns:
    The cache directory value as a string.

  Raises:
    ValueError: If the cache mode is invalid.
  """
  if cache == "disk":
    return ""
  elif cache == "memory":
    return ":memory"
  elif cache == "no":
    return ":nocache"
  else:
    raise ValueError(f"Invalid cache mode: {cache}")


def huggingface_options(f):
  """Decorator for HuggingFace-related options."""
  f = click.option(
      "--huggingface-token",
      default=None,
      envvar="HF_TOKEN",
      help=(
          "The HuggingFace API token to use when downloading from an"
          " access-gated HuggingFace repository. This can also be set via the"
          " HF_TOKEN environment variable."
      ),
  )(f)
  f = click.option(
      "--from-huggingface-repo",
      default=None,
      help="The HuggingFace repository ID to download the model from, if set.",
  )(f)
  return f


def common_inference_options(f):
  """Decorator for common options shared across commands."""
  f = huggingface_options(f)
  f = click.option(
      "--verbose",
      is_flag=True,
      default=False,
      help="Whether to enable verbose logging.",
  )(f)
  f = click.option(
      "--cache",
      type=click.Choice(["disk", "memory", "no"]),
      default="disk",
      help=textwrap.dedent("""\
          \b
          Caching mode for compiled model artifacts to speed up startup.
            - disk: Persists compiled artifacts to a file next to the model.
            - memory: Caches compiled artifacts in RAM (CPU backend only, not available on Windows).
            - no: Disables caching (recompiles on every run).
          """),
  )(f)
  f = click.option(
      "--enable-speculative-decoding",
      type=click.Choice(["auto", "true", "false"], case_sensitive=False),
      default="auto",
      callback=parse_speculative_decoding,
      help=textwrap.dedent("""\
          \b
          Speculative decoding mode ("auto", "true", "false").
            - auto: Automatically determine the speculative decoding behavior from the model metadata.
            - true: Force enable speculative decoding. It will throw an error if the model does not support it.
            - false: Force disable speculative decoding.
          """),
  )(f)
  f = click.option(
      "--backend",
      type=click.Choice(["cpu", "gpu", "npu"], case_sensitive=False),
      default="cpu",
      help="The backend to use.",
  )(f)
  return f


_DOWNLOAD_CHUNK_SIZE = 1024 * 64


def _size_string_from_bytes(size_in_bytes: int) -> str:
  """Formats bytes to a human-readable string (e.g., 18.2GiB)."""
  size = float(size_in_bytes)
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if size < 1024.0:
      if unit == "B":
        return f"{int(size)}{unit}"
      return f"{size:.1f}{unit}"
    size /= 1024.0
  return f"{size:.1f}PiB"


def stream_download(
    response: http.client.HTTPResponse,
    *,
    download_dir: pathlib.Path,
    length: int | None,
    format_progress: collections.abc.Callable[[int], str],
) -> pathlib.Path:
  """Streams the download response body to a temporary file.

  If any exception occurs during the download or file writing, the temporary
  file is guaranteed to be cleaned up.

  Args:
    response: The HTTPResponse object to read the body from.
    download_dir: The directory where the temporary file should be created.
    length: The expected length of the download in bytes, if known.
    format_progress: A callable returning the formatted progress string.

  Returns:
    The path to the temporary file where the response body was written.
  """
  download_dir.mkdir(parents=True, exist_ok=True)
  tmp_file = tempfile.NamedTemporaryFile(dir=download_dir, delete=False)
  tmp_file_path = pathlib.Path(tmp_file.name)
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
  except Exception:
    try:
      tmp_file_path.unlink(missing_ok=True)
    except OSError:
      pass
    raise
  else:
    return tmp_file_path


def parse_total_size(content_length: str | None) -> int | None:
  """Parses the Content-Length header value into an integer."""
  if content_length is None:
    return None
  try:
    return int(content_length)
  except ValueError:
    return None


def download_size_suffix(total_size: int | None) -> str:
  """Generates a formatted size suffix string for download logs."""
  return (
      f" ({_size_string_from_bytes(total_size)})"
      if total_size is not None
      else ""
  )


def format_download_progress(
    current_pos_bytes: int, total_size: int | None
) -> str:
  """Formats the download progress into a human-readable string."""
  if total_size is not None and total_size > 0:
    pct = int((current_pos_bytes / total_size) * 100)
    return f"{pct}%"
  if current_pos_bytes > 1_048_576:
    return f"{current_pos_bytes / 1_048_576:.1f} MiB"
  return f"{current_pos_bytes / 1024:.1f} KiB"
