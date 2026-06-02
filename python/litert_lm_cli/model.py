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

import dataclasses
import glob
import importlib.util
import inspect
import io
import mimetypes
import os
import pathlib
import traceback

import click

import litert_lm
from litert_lm_builder import litertlm_peek


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


def _backend_constraint(model_path: str) -> litert_lm.Backend:
  """Inspects the .litertlm file metadata to detect the required backend.

  Args:
    model_path: The path to the .litertlm model file.

  Returns:
    Backend.GPU() if the model metadata specifies 'gpu_artisan' as the backend
    constraint, otherwise Backend.CPU().
  """
  try:
    with io.StringIO() as dummy_out:
      metadata = litertlm_peek.read_litertlm_header(model_path, dummy_out)
      section_metadata = metadata.SectionMetadata()
      if not section_metadata:
        return litert_lm.Backend.CPU()
      for i in range(section_metadata.ObjectsLength()):
        section = section_metadata.Objects(i)
        if not section:
          continue
        if (
            litertlm_peek.get_model_type(section)
            == "tf_lite_artisan_text_decoder"
        ):
          return litert_lm.Backend.GPU()
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(
        click.style(f"Failed to inspect model metadata: {e!r}", fg="yellow")
    )
  return litert_lm.Backend.CPU()


def parse_backend(
    backend: str, *, model_obj: Model | None = None
) -> litert_lm.Backend:
  """Parses the backend string and resolves it against model constraints.

  If the user requests 'cpu' (or defaults to it) but the model metadata
  specifies a 'gpu_artisan' constraint, this will automatically upgrade
  the backend to GPU and print a notification.

  Args:
    backend: The backend requested by the user (e.g., "cpu", "gpu", "npu").
    model_obj: Optional Model instance to check for constraints.

  Returns:
    The resolved litert_lm.Backend to use.
  """
  backend_lower = backend.lower()
  if backend_lower == "gpu":
    requested = litert_lm.Backend.GPU()
  elif backend_lower == "npu":
    requested = litert_lm.Backend.NPU()
  else:
    requested = litert_lm.Backend.CPU()

  # Force GPU if the model requires it (CPU is unsupported for artisan models).
  if model_obj is not None:
    if isinstance(
        _backend_constraint(model_obj.model_path), litert_lm.Backend.GPU
    ):
      click.echo(
          click.style(
              "Using GPU backend for this model because CPU is unsupported.",
              fg="cyan",
          )
      )
      return litert_lm.Backend.GPU()

  return requested


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
    return cls(
        model_id=os.path.basename(model_path),
        model_path=os.path.abspath(model_path),
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


def get_cli_base_dir() -> str:
  """Gets the base directory for LiteRT-LM CLI."""
  env_override = os.environ.get("LITERT_LM_DIR")
  if env_override:
    return os.path.abspath(env_override)
  return os.path.join(os.path.expanduser("~"), ".litert-lm")


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
