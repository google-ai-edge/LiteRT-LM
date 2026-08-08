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

"""Utility functions for litert-lm models."""

from __future__ import annotations

import collections
import dataclasses
import glob
import importlib.util
import inspect
import io
import mimetypes
import os
import pathlib
from typing import Any

import click

import litert_lm
from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_peek
from litert_lm_cli import config

# The default model types representing the main text model components.
_DEFAULT_TARGET_MODEL_TYPES = frozenset({
    litertlm_builder.TfLiteModelType.ARTISAN_TEXT_DECODER.value,
    litertlm_builder.TfLiteModelType.PREFILL_DECODE.value,
})


def get_attachment_type(path: str) -> str:
  """Returns the attachment type (audio or image) from the file path.

  Args:
    path: Path to the attachment.

  Returns:
    'audio' or 'image'.

  Raises:
    ValueError: If the file type cannot be determined or is unsupported.
  """
  mime_type, _ = mimetypes.guess_type(path)
  if mime_type:
    if mime_type.startswith("audio/"):
      return "audio"
    elif mime_type.startswith("image/"):
      return "image"
    else:
      raise ValueError(f"Unsupported attachment type for '{path}': {mime_type}")
  else:
    raise ValueError(f"Could not determine file type for attachment '{path}'.")


def load_preset(preset: str):
  """Loads a preset file and returns the tools, messages and extra_context."""
  click.echo(click.style(f"Loading preset from {preset}:", dim=True))
  if not os.path.exists(preset):
    click.echo(click.style(f"Preset file not found: {preset}", fg="red"))
    return None, None, None

  spec = importlib.util.spec_from_file_location("user_tools", preset)
  if not spec or not spec.loader:
    click.echo(click.style(f"Failed to load tools from {preset}", fg="red"))
    return None, None, None

  user_tools = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(user_tools)

  tools = getattr(user_tools, "tools", None)
  if tools is None:
    tools = [
        obj
        for name, obj in inspect.getmembers(user_tools, inspect.isfunction)
        if obj.__module__ == "user_tools"
    ]

  messages = None
  system_instruction = getattr(user_tools, "system_instruction", None)
  if system_instruction:
    click.echo(
        click.style(f"- System instruction: {system_instruction}", dim=True)
    )
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}],
    }]

  click.echo(click.style("- Tools:", dim=True))
  for tool in tools:
    click.echo(
        click.style(f"  - {getattr(tool, '__name__', str(tool))}", dim=True)
    )

  extra_context = getattr(user_tools, "extra_context", None)
  if extra_context:
    click.echo(click.style(f"- Extra context: {extra_context}", dim=True))

  return tools, messages, extra_context


def model_default_backend(
    model_path: str,
    target_model_types: collections.abc.Container[
        str
    ] = _DEFAULT_TARGET_MODEL_TYPES,
) -> str | None:
  """Inspects the .litertlm file metadata to detect the default backend.

  Args:
    model_path: The path to the .litertlm model file.
    target_model_types: The model types to look for. Defaults to main model
      types (ARTISAN_TEXT_DECODER, PREFILL_DECODE).

  Returns:
    The default backend name (e.g., 'gpu', 'cpu') or None if not found.
    Returning None for optional adapters (audio/vision) when they are not
    present in the model is crucial. It signals the CLI to pass None to the
    Engine, safely disabling them and preventing C++ initialization crashes.
  """
  try:
    with io.StringIO() as dummy_out:
      metadata = litertlm_peek.read_litertlm_header(model_path, dummy_out)
      if metadata:
        section_metadata = metadata.SectionMetadata()
        if section_metadata:
          for i in range(section_metadata.ObjectsLength()):
            section = section_metadata.Objects(i)
            if not section:
              continue
            model_type = litertlm_peek.get_model_type(section)
            if model_type:
              model_type_lower = model_type.lower()
              if model_type_lower in target_model_types:
                if (
                    model_type_lower
                    == litertlm_builder.TfLiteModelType.ARTISAN_TEXT_DECODER.value
                ):
                  return "gpu"
                if section.ItemsLength() > 0:
                  for j in range(section.ItemsLength()):
                    item = section.Items(j)
                    if item is None:
                      continue
                    item_dict = litertlm_peek.kvp_to_dict(item)
                    if item_dict.get("key") == "backend_constraint":
                      val = item_dict.get("value")
                      if val:
                        backends = [b.strip().lower() for b in val.split(",")]
                        if backends:
                          return backends[0]
                return "cpu"
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(
        click.style(f"Failed to inspect model metadata: {e!r}", fg="yellow")
    )

  # Fallback for main model if not found in metadata or on error.
  # Optional adapters return None if not found, to disable them.
  if target_model_types == _DEFAULT_TARGET_MODEL_TYPES:
    return "cpu"
  return None


def _create_backend_obj(
    backend_name: str | None,
    cpu_thread_count: int | None = None,
    gpu_decode_steps_per_sync: int | None = None,
) -> litert_lm.Backend | None:
  """Creates a litert_lm.Backend object from name, or returns None."""
  if backend_name is None:
    return None
  elif backend_name == "gpu":
    return litert_lm.Backend.GPU(
        gpu_decode_steps_per_sync=gpu_decode_steps_per_sync
    )
  elif backend_name == "npu":
    return litert_lm.Backend.NPU()
  else:
    return litert_lm.Backend.CPU(thread_count=cpu_thread_count)


def parse_backend(
    backend: str | None = None,
    *,
    model_obj: Model,
    cpu_thread_count: int | None = None,
    target_model_types: collections.abc.Container[
        str
    ] = _DEFAULT_TARGET_MODEL_TYPES,
    label: str | None = None,
    gpu_decode_steps_per_sync: int | None = None,
) -> litert_lm.Backend | None:
  """Parses the backend string and resolves it against model constraints.

  Args:
    backend: The backend requested by the user (e.g., "cpu", "gpu", "npu").
    model_obj: Model instance to check for constraints.
    cpu_thread_count: Optional thread count for CPU backend.
    target_model_types: Container of model types to look for when resolving
      default backend. Defaults to main model types.
    label: Optional label for the backend (e.g., "audio", "vision") used in log
      messages.
    gpu_decode_steps_per_sync: Optional decode steps per sync for GPU backend.

  Returns:
    The resolved litert_lm.Backend to use, or None if not supported.
  """
  model_cfg = config.get_model_config(model_obj.model_id)
  is_main_model = target_model_types == _DEFAULT_TARGET_MODEL_TYPES

  # 1. Resolve Backend
  resolved_backend = backend
  backend_from_config = False
  if resolved_backend is None:
    if is_main_model and model_cfg.backend is not None:
      resolved_backend = model_cfg.backend
      backend_from_config = True
    elif label == "vision" and model_cfg.vision_backend is not None:
      resolved_backend = model_cfg.vision_backend
      backend_from_config = True
    elif label == "audio" and model_cfg.audio_backend is not None:
      resolved_backend = model_cfg.audio_backend
      backend_from_config = True
    else:
      resolved_backend = model_default_backend(
          model_obj.model_path, target_model_types
      )
      # Print info message for model's default backend if it is not CPU
      if resolved_backend and resolved_backend != "cpu":
        label_str = f" for {label}" if label else ""
        click.echo(
            click.style(
                f"Using model's default backend{label_str}: {resolved_backend}",
                fg="bright_black",
            )
        )

  if resolved_backend is None:
    return None

  # Print info message if backend came from config
  if backend_from_config:
    label_str = f" for {label}" if label else ""
    click.echo(
        click.style(
            f"Using backend{label_str} from config for model"
            f" '{model_obj.model_id}': {resolved_backend}",
            fg="bright_black",
        )
    )

  # 2. Resolve CPU Thread Count
  resolved_threads = cpu_thread_count
  threads_from_config = False
  if resolved_threads is None:
    if is_main_model and model_cfg.cpu_thread_count is not None:
      resolved_threads = model_cfg.cpu_thread_count
      threads_from_config = True

  # Print info message if threads came from config
  if threads_from_config:
    click.echo(
        click.style(
            "Using cpu_thread_count from config for model"
            f" '{model_obj.model_id}': {resolved_threads}",
            fg="bright_black",
        )
    )

  # 3. Resolve GPU Decode Steps Per Sync
  resolved_gpu_decode_steps_per_sync = gpu_decode_steps_per_sync
  gpu_decode_steps_from_config = False
  if resolved_gpu_decode_steps_per_sync is None:
    if is_main_model and model_cfg.gpu_decode_steps_per_sync is not None:
      resolved_gpu_decode_steps_per_sync = model_cfg.gpu_decode_steps_per_sync
      gpu_decode_steps_from_config = True

  # Print info message if gpu_decode_steps_per_sync came from config
  if gpu_decode_steps_from_config:
    click.echo(
        click.style(
            "Using gpu_decode_steps_per_sync from config for model"
            f" '{model_obj.model_id}': {resolved_gpu_decode_steps_per_sync}",
            fg="bright_black",
        )
    )

  return _create_backend_obj(
      resolved_backend.lower(),
      resolved_threads,
      resolved_gpu_decode_steps_per_sync,
  )


def resolve_config_option(
    value: Any,
    model_obj: Model | None,
    config_key: str,
    label: str | None = None,
) -> Any:
  """Resolves an option value, falling back to config if value is None.

  Args:
    value: Explicit value passed by CLI/user (if any).
    model_obj: Model instance (if available).
    config_key: Attribute name on ModelConfig (e.g. "cache", "max_num_tokens").
    label: Display label for logging (defaults to config_key).

  Returns:
    The explicit value if set, otherwise value from ModelConfig if set,
    otherwise None.
  """
  if value is not None or model_obj is None:
    return value

  model_id = getattr(model_obj, "model_id", None)
  if not model_id:
    return value

  model_cfg = config.get_model_config(model_id)
  config_val = getattr(model_cfg, config_key, None)
  if config_val is not None:
    display_label = label or config_key
    click.echo(
        click.style(
            f"Using {display_label} from config for model"
            f" '{model_id}': {config_val}",
            fg="bright_black",
        )
    )
    return config_val

  return None


@dataclasses.dataclass
class Model:
  """Represents a LiteRT-LM model.

  Attributes:
    model_id: The ID of the model.
    model_path: The local path to the model file.
  """

  model_id: str
  model_path: str

  def exists(self) -> bool:
    """Returns True if the model file exists locally."""
    return os.path.isfile(self.model_path)

  def to_str(self) -> str:
    """Returns a string representation of the model."""
    return self.model_id

  @classmethod
  def get_all_models(cls):
    """Returns a list of all locally available models."""
    model_paths = glob.glob(
        "*/model.litertlm",
        root_dir=get_converted_models_base_dir(),
        recursive=True,
    )

    return [
        Model.from_model_id(pathlib.Path(path).parent.name.replace("--", "/"))
        for path in model_paths
    ]

  @classmethod
  def from_model_reference(cls, model_reference):
    """Creates a Model instance from a model reference."""
    if os.path.exists(model_reference):
      return cls.from_model_path(model_reference)
    else:
      # assume the reference is model_id
      return cls.from_model_id(model_reference)

  @classmethod
  def from_model_path(cls, model_path):
    """Creates a Model instance from a model path."""
    abs_path = os.path.abspath(model_path)
    if os.path.basename(abs_path) == "model.litertlm":
      parent_name = pathlib.Path(abs_path).parent.name
      model_id = parent_name.replace("--", "/")
    else:
      model_id = os.path.basename(abs_path)
    return cls(
        model_id=model_id,
        model_path=abs_path,
    )

  @classmethod
  def from_model_id(cls, model_id):
    """Creates a Model instance from a model ID."""
    return cls(
        model_id=model_id,
        model_path=os.path.join(
            get_converted_models_base_dir(),
            model_id.replace("/", "--"),
            "model.litertlm",
        ),
    )


# Just to use the huggingface convention. Likely to change.
def model_id_dir_name(model_id):
  """Converts a model ID to a directory name."""
  return model_id.replace("/", "--")


get_cli_base_dir = config.get_cli_base_dir


# ~/.litert-lm/models
def get_converted_models_base_dir():
  """Gets the base directory for all converted models."""
  return os.path.join(get_cli_base_dir(), "models")


# ~/.litert-lm/models/<model_id>
def get_model_dir(model_id):
  """Gets the model directory for a given model ID."""
  return os.path.join(
      get_converted_models_base_dir(),
      model_id_dir_name(model_id),
  )
