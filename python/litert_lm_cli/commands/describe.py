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
"""Describe subcommand for LiteRT-LM CLI."""

import click

import litert_lm
from litert_lm_cli import cli_helpers
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import huggingface_download
from litert_lm_cli import model


@click.command(cls=help_formatter.ColorCommand, name="describe")
@click.argument("model_reference", required=False)
@common.config_option
@common.huggingface_options
def describe_model(
    model_reference: str | None = None,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
):
  """Describes a LiteRT-LM model and prints its capabilities.

  MODEL_REFERENCE can be a local file path to a .litertlm model file,
  or an imported model ID (e.g. 'gemma3-1b').

  Example usage:

    litert-lm describe my-model-id

    litert-lm describe /path/to/model.litertlm

    # Describe directly from a HuggingFace repository
    litert-lm describe --from-huggingface-repo org/repo model.litertlm
  """

  model_reference = model_reference or cli_helpers.resolve_model_file(
      from_huggingface_repo,
      huggingface_token,
  )

  if from_huggingface_repo:
    model_path = huggingface_download.download_from_huggingface(
        repo_id=from_huggingface_repo,
        filename=model_reference,
        token=huggingface_token,
    )
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)
    if not model_obj.exists():
      raise click.ClickException(f"Failed to find model '{model_reference}'.")

  try:
    capabilities = litert_lm.Capabilities(model_obj.model_path)
  except (FileNotFoundError, RuntimeError) as e:
    raise click.ClickException(
        f"Failed to load capabilities for model '{model_reference}': {e}"
    )

  modalities = []
  if capabilities.input_modalities.text:
    modalities.append("Text")
  if capabilities.input_modalities.vision:
    modalities.append("Vision")
  if capabilities.input_modalities.audio:
    modalities.append("Audio")
  if capabilities.input_modalities.video:
    modalities.append("Video")
  modalities_str = " ".join(modalities) if modalities else "None"

  click.echo("========================================")
  click.echo(" LiteRT-LM Model Capabilities Report")
  click.echo("========================================")
  click.echo(f"File: {model_obj.model_path}\n")
  click.echo("[LLM Capabilities]")
  click.echo(
      "  Supports Function Call: "
      f"{'YES' if capabilities.supports_function_calling() else 'NO'}"
  )
  click.echo(
      "  Supports Thinking:      "
      f"{'YES' if capabilities.supports_thinking() else 'NO'}"
  )
  click.echo(
      "  Speculative Decoding:   "
      f"{'YES' if capabilities.has_speculative_decoding_support() else 'NO'}"
  )
  click.echo(
      f"  Max Vision Token Budget: {capabilities.max_vision_token_budget}"
  )

  sampler_config = capabilities.default_sampler_params
  click.echo(f"  Sampler Temp:           {sampler_config.temperature:.2f}")
  click.echo(f"  Sampler Top K:          {sampler_config.top_k}")
  click.echo(f"  Sampler Top P:          {sampler_config.top_p:.2f}")
  click.echo(f"  Input Modalities:       {modalities_str}")
  click.echo("========================================")


def register(cli: click.Group) -> None:
  """Registers the describe command."""
  cli.add_command(describe_model)
