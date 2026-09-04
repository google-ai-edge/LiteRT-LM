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


def _format_npu_soc_and_brand(
    brand: litert_lm.LiteRtLmNpuBrand, soc_name: str | None
) -> str | None:
  """Combines NPU brand and SoC name (e.g. 'Qualcomm QNN SM8850')."""
  brand_names = {
      litert_lm.LiteRtLmNpuBrand.QUALCOMM: "Qualcomm QNN",
      litert_lm.LiteRtLmNpuBrand.GOOGLE_TENSOR: "Google Tensor TPU",
      litert_lm.LiteRtLmNpuBrand.MEDIATEK: "MediaTek Neuron",
      litert_lm.LiteRtLmNpuBrand.INTEL: "Intel NPU",
      litert_lm.LiteRtLmNpuBrand.SAMSUNG: "Samsung Exynos NPU",
  }
  brand_name = brand_names.get(brand)
  if brand_name and soc_name:
    return f"{brand_name} {soc_name}"
  elif brand_name:
    return brand_name
  elif soc_name:
    return soc_name
  return None


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
    model_info = litert_lm.ModelInfo(model_obj.model_path)
  except (FileNotFoundError, RuntimeError) as e:
    raise click.ClickException(
        f"Failed to load capabilities for model '{model_reference}': {e}"
    )

  modalities = []
  if model_info.input_modalities.text:
    modalities.append("Text")
  if model_info.input_modalities.vision:
    modalities.append("Vision")
  if model_info.input_modalities.audio:
    modalities.append("Audio")
  if model_info.input_modalities.video:
    modalities.append("Video")
  modalities_str = " ".join(modalities) if modalities else "None"

  click.echo("========================================")
  click.echo(" LiteRT-LM Model Info Report")
  click.echo("========================================")
  click.echo(f"File: {model_obj.model_path}\n")
  click.echo("[LLM Capabilities]")
  click.echo(
      "  Supports Function Call: "
      f"{'YES' if model_info.supports_function_calling() else 'NO'}"
  )
  click.echo(
      "  Supports Thinking:      "
      f"{'YES' if model_info.supports_thinking() else 'NO'}"
  )
  click.echo(
      "  Speculative Decoding:   "
      f"{'YES' if model_info.has_speculative_decoding_support() else 'NO'}"
  )
  click.echo(f"  Max Vision Token Budget: {model_info.max_vision_token_budget}")
  click.echo(
      f"  Min Runtime Version:    {model_info.min_runtime_version or '-1'}"
  )
  lengths = model_info.vision_signature_selection
  lengths_str = str(lengths) if lengths is not None else "-1"
  click.echo(f"  Vision Signature Selection: {lengths_str}")

  sampler_config = model_info.default_sampler_params
  top_k_val = sampler_config.top_k if sampler_config.top_k is not None else 0
  click.echo(f"  Sampler Temp:           {sampler_config.temperature:.2f}")
  click.echo(f"  Sampler Top K:          {top_k_val}")
  click.echo(f"  Sampler Top P:          {sampler_config.top_p:.2f}")
  click.echo(f"  Max Context Tokens:     {model_info.max_context_tokens}")
  click.echo(
      "  Is Dynamic Context:     "
      f"{'YES' if model_info.is_dynamic_context else 'NO'}"
  )
  click.echo(f"  Input Modalities:       {modalities_str}")

  # Report supported backends, default backend, and target NPU SoC per
  # active modality.
  for mod_name, mod_enum in [
      ("Text", litert_lm.LiteRtLmModality.TEXT),
      ("Vision", litert_lm.LiteRtLmModality.VISION),
      ("Audio", litert_lm.LiteRtLmModality.AUDIO),
      ("Video", litert_lm.LiteRtLmModality.VIDEO),
  ]:
    if getattr(model_info.input_modalities, mod_name.lower()):
      supported_backends = model_info.supported_backends_for_modality(mod_enum)
      backends_str = " ".join([b.upper() for b in supported_backends]) or "None"
      click.echo(f"  {mod_name} Backends:".ljust(26) + backends_str)
      if supported_backends:
        click.echo(
            f"  {mod_name} Default Backend:".ljust(26)
            + supported_backends[0].upper()
        )
      soc_desc = _format_npu_soc_and_brand(
          model_info.npu_brand_for_modality(mod_enum),
          model_info.soc_name_for_modality(mod_enum),
      )
      if soc_desc:
        click.echo(f"  {mod_name} SoC Name:".ljust(26) + soc_desc)
  click.echo("========================================")


def register(cli: click.Group) -> None:
  """Registers the describe command."""
  cli.add_command(describe_model)
