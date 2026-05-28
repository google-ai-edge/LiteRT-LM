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
      help=(
          "The HuggingFace API token to use when downloading from a access"
          " gated HuggingFace repository. This can also be set via the"
          " HUGGING_FACE_HUB_TOKEN or HF_TOKEN environment variables, or by"
          " running `hf auth login`."
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


def download_from_huggingface(repo_id, filename, token):
  """Downloads a file from HuggingFace Hub.

  Args:
    repo_id: The HuggingFace repository ID.
    filename: The filename to download.
    token: The HuggingFace API token.

  Returns:
    The local path to the downloaded file, or None if download failed.
  """
  try:
    # pylint: disable=g-import-not-at-top
    from huggingface_hub import get_token
    from huggingface_hub import hf_hub_download
  except ImportError:
    click.echo(
        click.style(
            "Error: huggingface_hub is not installed. Please install it to"
            " download from HuggingFace.",
            fg="red",
        )
    )
    return None

  effective_token = token or get_token()

  click.echo(f"Downloading {filename} from {repo_id}...")
  try:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=effective_token,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(
        click.style(f"Error downloading from HuggingFace: {e}", fg="red")
    )
    if not effective_token:
      click.echo(
          click.style(
              "HuggingFace token not found. If this is a private or gated"
              " repository, you can provide the token via the"
              " --huggingface-token option, setting the"
              " HUGGING_FACE_HUB_TOKEN environment variable, or by running"
              " 'hf auth login'.",
              fg="yellow",
          )
      )
    return None
