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

"""Main script for litert-lm binary."""

import datetime
import os
import shutil
import subprocess
import sys

import click

import litert_lm
from litert_lm_cli import help_formatter
from litert_lm_cli import model
from litert_lm_cli import serve as _serve_module
from litert_lm_cli import venv_manager
from litert_lm_cli import version


@click.group(
    cls=help_formatter.ColorGroup,
    name="litert-lm",
    context_settings=dict(
        show_default=True,
        max_content_width=120,
        help_option_names=["-h", "--help"],
    ),
)
@click.version_option(version=version.VERSION)
def cli():
  """CLI tool for LiteRT-LM models."""


_serve_module.register(cli)


@cli.command(name="list")
def list_models():
  """Lists all imported LiteRT-LM models."""
  base_dir = model.get_converted_models_base_dir()
  click.echo(f"Listing models in: {base_dir}")

  models = sorted(model.Model.get_all_models(), key=lambda m: m.model_id)

  # Calculate dynamic width for ID column
  id_width = max([len(m.model_id) for m in models] + [len("ID"), 25]) + 2

  click.echo(
      click.style(f"{'ID':<{id_width}} {'SIZE':<15} {'MODIFIED'}", bold=True)
  )

  for model_item in models:
    path = model_item.model_path
    try:
      stat = os.stat(path)
      size_bytes = stat.st_size
      if size_bytes >= 1024 * 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
      else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
      modified_date = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
          "%Y-%m-%d %H:%M:%S"
      )
    except FileNotFoundError:
      size_str = "Unknown"
      modified_date = "Unknown"

    click.echo(
        f"{model_item.model_id:<{id_width}} {size_str:<15} {modified_date}"
    )


def _download_from_huggingface(repo_id, filename, token):
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


@cli.command(
    name="import",
    help="""Imports a model from a local path or HuggingFace hub.

  MODEL_FILE: The local path to the model file, or the path in the HuggingFace
  repo if --from-huggingface-repo is set.
  MODEL_REF: The ID to store the model as. Defaults to the filename of
  MODEL_FILE.

  \b
  Examples:
    # Import from a local path
    litert-lm import ./model.litertlm my-model

    # Import from a HuggingFace repository
    litert-lm import --from-huggingface-repo org/repo model.litertlm my-model

    # Import and use the default model ID
    litert-lm import ./model.litertlm""",
)
@huggingface_options
@click.argument("model_file")
@click.argument("model_ref", required=False)
def import_model(
    from_huggingface_repo, huggingface_token, model_file, model_ref
):
  """Imports a model from a local path or HuggingFace hub.

  Args:
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: HuggingFace API token.
    model_file: The path in the repo (if from-huggingface-repo is set) or local
      path.
    model_ref: The reference ID to store the model as. Defaults to the filename
      of MODEL_FILE.
  """
  effective_model_ref = model_ref or os.path.basename(model_file)

  if from_huggingface_repo:
    source = _download_from_huggingface(
        from_huggingface_repo, model_file, huggingface_token
    )
    if not source:
      return
  else:
    source = model_file
    if not os.path.exists(source):
      click.echo(click.style(f"Source file not found: {source}", fg="red"))
      return

  model_obj = model.Model.from_model_id(effective_model_ref)
  model_path = model_obj.model_path
  model_dir = os.path.dirname(model_path)

  os.makedirs(model_dir, exist_ok=True)

  shutil.copy(source, model_path)
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


@cli.command(help="Deletes a model from the local storage.")
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


@cli.command(help="Renames a model.")
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
      "--enable-speculative-decoding",
      type=click.Choice(["auto", "true", "false"], case_sensitive=False),
      default="auto",
      callback=parse_speculative_decoding,
      help="""\b
Speculative decoding mode ("auto", "true", "false").
  - auto: Automatically determine the speculative decoding behavior from the model metadata.
  - true: Force enable speculative decoding. It will throw an error if the model does not support it.
  - false: Force disable speculative decoding.
""",
  )(f)
  f = click.option(
      "--backend",
      type=click.Choice(["cpu", "gpu"], case_sensitive=False),
      default="cpu",
      help="The backend to use.",
  )(f)
  return f


@cli.command(
    help="""Benchmarks a LiteRT-LM model.

  \b
  Examples:
    # Benchmark using a model ID from 'litert-lm list'
    litert-lm benchmark my-model

    # Benchmark using a local path
    litert-lm benchmark ./model.litertlm

    # Benchmark directly from a HuggingFace repository
    litert-lm benchmark --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference")
@click.option(
    "-p",
    "--prefill_tokens",
    default=256,
    type=int,
    help="The number of tokens to prefill.",
)
@click.option(
    "-d",
    "--decode_tokens",
    default=256,
    type=int,
    help="The number of tokens to decode.",
)
@common_inference_options
def benchmark(
    model_reference: str,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    backend: str = "cpu",
    android: bool = False,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
):
  """Benchmarks a LiteRT-LM model.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prefill_tokens: The number of tokens to prefill.
    decode_tokens: The number of tokens to decode.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  if from_huggingface_repo:
    model_path = _download_from_huggingface(
        from_huggingface_repo, model_reference, huggingface_token
    )
    if not model_path:
      return
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)

  model_obj.benchmark(
      prefill_tokens=prefill_tokens,
      decode_tokens=decode_tokens,
      is_android=android,
      backend=backend,
      enable_speculative_decoding=enable_speculative_decoding,
  )


@cli.command(
    help="""Runs a LiteRT-LM model interactively or with a single prompt.

  \b
  Examples:
    # Run interactively using a model ID from 'litert-lm list'
    litert-lm run my-model

    # Run with a single prompt using a local path
    litert-lm run ./model.litertlm --prompt "Hi there!"

    # Run directly from a HuggingFace repository
    litert-lm run --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference")
@click.option(
    "--prompt", default=None, help="A single prompt to run once and exit."
)
@click.option(
    "--preset",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a Python file containing tool functions and system"
        " instructions."
    ),
)
@click.option(
    "--no-template",
    is_flag=True,
    default=False,
    help=(
        "Interact with the model directly without applying prompt templates."
        " That means the input should include all control tokens for the model"
        " expected."
    ),
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=None,
    help="Maximum number of tokens for the KV cache.",
)
@click.option(
    "--filter-channel-content-from-kv-cache",
    is_flag=True,
    default=False,
    help="Whether to filter channel content from the KV cache.",
)
@click.option(
    "--vision-backend",
    type=click.Choice(["cpu", "gpu", ""], case_sensitive=False),
    default=None,
    help="The backend to use for vision encoding.",
)
@click.option(
    "--audio-backend",
    type=click.Choice(["cpu", "gpu", ""], case_sensitive=False),
    default=None,
    help="The backend to use for audio encoding.",
)
@click.option(
    "--attachment",
    multiple=True,
    type=click.Path(dir_okay=False),
    help=(
        "Path to an attachment (image or audio only). Can be specified multiple"
        " times. Attachements are placed before the first user text prompt."
    ),
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "The number of top logits used during sampling. If not set, use the"
        " default from the model or engine."
    ),
)
@click.option(
    "--top-p",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "The cumulative probability threshold for nucleus sampling. If not set,"
        " use the default from the model or engine."
    ),
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=None,
    help=(
        "The temperature to use for sampling. If not set, use the default from"
        " the model or engine."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=(
        "The seed to use for randomization. If not set, use the default from"
        " the model or engine."
    ),
)
@common_inference_options
def run(
    model_reference,
    prompt=None,
    preset=None,
    backend="cpu",
    android=False,
    enable_speculative_decoding=None,
    verbose=False,
    no_template=False,
    from_huggingface_repo=None,
    huggingface_token=None,
    max_num_tokens=None,
    filter_channel_content_from_kv_cache=False,
    vision_backend=None,
    audio_backend=None,
    attachment=(),
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
):
  r"""Runs a LiteRT-LM model interactively or with a single prompt.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prompt: A single prompt to run once and exit.
    preset: Path to a Python file containing tool functions and system
      instructions.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    no_template: Interact with the model directly without applying prompt
      templates or stripping stop tokens.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    filter_channel_content_from_kv_cache: Whether to filter channel content from
      the KV cache.
    vision_backend: The backend to use for vision tasks.
    audio_backend: The backend to use for audio tasks.
    attachment: Path to an attachment (e.g., image or audio).
    top_k: The number of top logits used during sampling.
    top_p: The cumulative probability threshold for nucleus sampling.
    temperature: The temperature to use for sampling.
    seed: The seed to use for randomization.
  """
  if attachment and no_template:
    click.echo(
        click.style(
            "Error: Attachments are not supported with --no-template.",
            fg="red",
        )
    )
    return

  expanded_attachments = []
  has_audio = False
  has_image = False

  for a in attachment:
    expanded = os.path.expanduser(a)
    if not os.path.exists(expanded):
      raise click.BadParameter(f"File '{a}' does not exist.")
    expanded_attachments.append(expanded)

    try:
      a_type = model.get_attachment_type(expanded)
      if a_type == "audio":
        has_audio = True
      elif a_type == "image":
        has_image = True
    except ValueError as e:
      raise click.BadParameter(str(e)) from e

  if has_audio and not audio_backend:
    click.echo(
        click.style(
            "Error: Audio attachments require --audio-backend to be set.",
            fg="red",
        )
    )
    return

  if has_image and not vision_backend:
    click.echo(
        click.style(
            "Error: Image attachments require --vision-backend to be set.",
            fg="red",
        )
    )
    return

  # If the stdin is not connected to the terminal, e.g., piped or redirected
  # input, then handle the input as the one-shot prompt.
  #
  # # Redirected input:
  # $ litert-lm run < prompt.txt
  # $ litert-lm run --prompt="Explain this error log" < error.log
  #
  # # Piped input:
  # $ cat text.txt | litert-lm run --prompt="Summarize the content."
  if not sys.stdin.isatty():
    piped_input = sys.stdin.read().strip()
    if piped_input:
      prompt = f"{prompt}\n\n{piped_input}" if prompt else piped_input
    elif not prompt:
      # If no prompt is provided and it's not a TTY, we can't be interactive.
      return

  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  if from_huggingface_repo:
    model_path = _download_from_huggingface(
        from_huggingface_repo, model_reference, huggingface_token
    )
    if not model_path:
      return
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)
    if not model_obj.exists():
      # Only auto-convert if it looks like a HuggingFace repo ID (account/repo)
      # and is not a local path.
      parts = model_reference.split("/")
      if len(parts) == 2 and all(parts) and not os.path.exists(model_reference):
        click.echo(
            click.style(
                f"Model '{model_reference}' not found. Attempting to convert"
                f" from https://huggingface.co/{model_reference} ...",
                fg="yellow",
            )
        )
        convert.callback(source=model_reference)
        model_obj = model.Model.from_model_reference(model_reference)

      if not model_obj.exists():
        click.echo(
            click.style(
                f"Failed to find or convert model '{model_reference}'.",
                fg="red",
            )
        )
        return

  model_obj.run_interactive(
      prompt=prompt,
      is_android=android,
      backend=backend,
      preset=preset,
      enable_speculative_decoding=enable_speculative_decoding,
      no_template=no_template,
      max_num_tokens=max_num_tokens,
      filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
      vision_backend=vision_backend,
      audio_backend=audio_backend,
      attachments=tuple(expanded_attachments),
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      seed=seed,
  )


def main():
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
  cli()


if __name__ == "__main__":
  main()
