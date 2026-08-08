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
"""Configuration manager for LiteRT-LM CLI."""

from __future__ import annotations

import dataclasses
import json
import os
import pkgutil
from typing import Any

import click

KEY_DEFAULT = "default"
KEY_MODELS = "models"
KEY_BACKEND = "backend"
KEY_CPU_THREAD_COUNT = "cpu_thread_count"
KEY_AUDIO_BACKEND = "audio_backend"
KEY_VISION_BACKEND = "vision_backend"
KEY_CACHE = "cache"
KEY_MAX_NUM_TOKENS = "max_num_tokens"
KEY_TEMPERATURE = "temperature"
KEY_TOP_P = "top_p"
KEY_TOP_K = "top_k"
KEY_SEED = "seed"
KEY_GPU_DECODE_STEPS_PER_SYNC = "gpu_decode_steps_per_sync"
KEY_SPECULATIVE_DECODING = "speculative_decoding"
KEY_THINKING = "thinking"
KEY_THINKING_BUDGET = "thinking_budget"
KEY_ACTIVATION_DATA_TYPE = "activation_data_type"


@dataclasses.dataclass
class ModelConfig:
  """Configuration settings for a LiteRT-LM model or default settings."""

  backend: str | None = None
  cpu_thread_count: int | None = None
  audio_backend: str | None = None
  vision_backend: str | None = None
  cache: str | None = None
  max_num_tokens: int | None = None
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None
  seed: int | None = None
  gpu_decode_steps_per_sync: int | None = None
  speculative_decoding: bool | None = None
  thinking: bool | None = None
  thinking_budget: int | None = None
  activation_data_type: str | None = None

  def merge_from(self, other: ModelConfig) -> None:
    """Merges values from another config, overwriting if set."""
    if other.backend is not None:
      self.backend = other.backend
    if other.cpu_thread_count is not None:
      self.cpu_thread_count = other.cpu_thread_count
    if other.audio_backend is not None:
      self.audio_backend = other.audio_backend
    if other.vision_backend is not None:
      self.vision_backend = other.vision_backend
    if other.cache is not None:
      self.cache = other.cache
    if other.max_num_tokens is not None:
      self.max_num_tokens = other.max_num_tokens
    if other.temperature is not None:
      self.temperature = other.temperature
    if other.top_p is not None:
      self.top_p = other.top_p
    if other.top_k is not None:
      self.top_k = other.top_k
    if other.seed is not None:
      self.seed = other.seed
    if other.gpu_decode_steps_per_sync is not None:
      self.gpu_decode_steps_per_sync = other.gpu_decode_steps_per_sync
    if other.speculative_decoding is not None:
      self.speculative_decoding = other.speculative_decoding
    if other.thinking is not None:
      self.thinking = other.thinking
    if other.thinking_budget is not None:
      self.thinking_budget = other.thinking_budget
    if other.activation_data_type is not None:
      self.activation_data_type = other.activation_data_type


@dataclasses.dataclass
class AppConfig:
  """Top-level application configuration containing default and per-model configs."""

  default: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  models: dict[str, ModelConfig] = dataclasses.field(default_factory=dict)


_CACHED_CONFIG: AppConfig | None = None
_SCHEMA: dict[str, Any] | None = None
_CUSTOM_CONFIG_PATH: str | None = None


def get_cli_base_dir() -> str:
  """Gets the base directory for LiteRT-LM CLI."""
  env_override = os.environ.get("LITERT_LM_DIR")
  if env_override:
    return os.path.abspath(env_override)
  return os.path.join(os.path.expanduser("~"), ".litert-lm")


def set_config_path(config_path: str | None) -> None:
  """Sets a custom config file path and invalidates cached config."""
  global _CUSTOM_CONFIG_PATH, _CACHED_CONFIG
  if config_path is not None:
    _CUSTOM_CONFIG_PATH = os.path.abspath(os.path.expanduser(config_path))
  else:
    _CUSTOM_CONFIG_PATH = None
  _CACHED_CONFIG = None


def get_config_path() -> str:
  """Gets the path to the config.json file."""
  if _CUSTOM_CONFIG_PATH is not None:
    return _CUSTOM_CONFIG_PATH
  return os.path.join(get_cli_base_dir(), "config.json")


def _load_schema() -> dict[str, Any]:
  """Loads the JSON schema for configuration validation."""
  global _SCHEMA
  if _SCHEMA is not None:
    return _SCHEMA
  try:
    data = pkgutil.get_data(__name__, "config_schema.json")
    if data is not None:
      _SCHEMA = json.loads(data.decode("utf-8"))
      return _SCHEMA
    schema_path = os.path.join(os.path.dirname(__file__), "config_schema.json")
    if os.path.exists(schema_path):
      with open(schema_path, "r", encoding="utf-8") as f:
        _SCHEMA = json.load(f)
        return _SCHEMA
  except Exception as e:
    raise click.ClickException(f"Failed to load config schema: {e}") from e

  raise click.ClickException(
      "Config schema file 'config_schema.json' not found."
  )


def _parse_model_config(data: dict[str, Any]) -> ModelConfig:
  """Parses a ModelConfig from a dict."""
  return ModelConfig(
      backend=data.get(KEY_BACKEND),
      cpu_thread_count=data.get(KEY_CPU_THREAD_COUNT),
      audio_backend=data.get(KEY_AUDIO_BACKEND),
      vision_backend=data.get(KEY_VISION_BACKEND),
      cache=data.get(KEY_CACHE),
      max_num_tokens=data.get(KEY_MAX_NUM_TOKENS),
      temperature=data.get(KEY_TEMPERATURE),
      top_p=data.get(KEY_TOP_P),
      top_k=data.get(KEY_TOP_K),
      seed=data.get(KEY_SEED),
      gpu_decode_steps_per_sync=data.get(KEY_GPU_DECODE_STEPS_PER_SYNC),
      speculative_decoding=data.get(KEY_SPECULATIVE_DECODING),
      thinking=data.get(KEY_THINKING),
      thinking_budget=data.get(KEY_THINKING_BUDGET),
      activation_data_type=data.get(KEY_ACTIVATION_DATA_TYPE),
  )


# NOTE: We intentionally implement a lightweight pure-Python schema validator
# below rather than depending on the `jsonschema` library. `jsonschema` (via its
# transitive dependency `referencing` / `rpds-py`) requires compiled Rust
# C-extensions that are unsupported or fail to build on Python 3.10 and Android
# environments. To maintain cross-platform compatibility across all LiteRT-LM
# edge targets, keep this validation logic pure-Python without third-party
# dependencies.


class _SchemaValidationError(Exception):
  """Internal error raised during JSON schema validation."""

  def __init__(self, message: str, path: list[str | int]):
    super().__init__(message)
    self.message = message
    self.path = path


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: list[str | int],
) -> None:
  """Validates a JSON instance against a schema definition in pure Python."""
  if "$ref" in schema:
    ref = schema["$ref"]
    if ref.startswith("#/definitions/"):
      def_name = ref.split("/")[-1]
      target_schema = root_schema.get("definitions", {}).get(def_name)
      if target_schema is not None:
        _validate_instance(instance, target_schema, root_schema, path)
        return

  expected_type = schema.get("type")
  if expected_type:
    if expected_type == "object":
      if not isinstance(instance, dict):
        raise _SchemaValidationError(
            f"{instance!r} is not of type 'object'", path
        )
    elif expected_type == "string":
      if not isinstance(instance, str):
        raise _SchemaValidationError(
            f"{instance!r} is not of type 'string'", path
        )
    elif expected_type == "integer":
      if not isinstance(instance, int) or isinstance(instance, bool):
        raise _SchemaValidationError(
            f"{instance!r} is not of type 'integer'", path
        )
    elif expected_type == "number":
      if not isinstance(instance, (int, float)) or isinstance(instance, bool):
        raise _SchemaValidationError(
            f"{instance!r} is not of type 'number'", path
        )
    elif expected_type == "boolean":
      if not isinstance(instance, bool):
        raise _SchemaValidationError(
            f"{instance!r} is not of type 'boolean'", path
        )

  if "enum" in schema:
    if instance not in schema["enum"]:
      raise _SchemaValidationError(
          f"{instance!r} is not one of {schema['enum']!r}", path
      )

  if "minimum" in schema:
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
      if instance < schema["minimum"]:
        raise _SchemaValidationError(
            f"{instance} is less than the minimum of {schema['minimum']}", path
        )

  if "maximum" in schema:
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
      if instance > schema["maximum"]:
        raise _SchemaValidationError(
            f"{instance} is greater than the maximum of {schema['maximum']}",
            path,
        )

  if isinstance(instance, dict):
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
      if prop_name in instance:
        _validate_instance(
            instance[prop_name], prop_schema, root_schema, path + [prop_name]
        )

    additional_props = schema.get("additionalProperties", True)
    if isinstance(additional_props, dict):
      for key, val in instance.items():
        if key not in properties:
          _validate_instance(val, additional_props, root_schema, path + [key])


def _validate_schema(
    config_data: Any, schema: dict[str, Any], config_name: str = "config.json"
) -> None:
  """Validates config_data against schema in pure Python.

  Note: We intentionally do not use `jsonschema` here because `jsonschema`
  depends on `rpds-py`, which does not support Python 3.10 and Android.

  Args:
    config_data: The parsed JSON config dictionary to validate.
    schema: The JSON schema dictionary to validate against.
    config_name: The config filename or path for error reporting.
  """
  try:
    _validate_instance(config_data, schema, schema, [])
  except _SchemaValidationError as e:
    path_str = ".".join(str(p) for p in e.path)
    prefix = f"{path_str}: " if path_str else ""
    raise click.ClickException(
        f"{config_name} validation error: {prefix}{e.message}"
    ) from e


def load_config(config_path: str | None = None) -> AppConfig:
  """Loads and validates the configuration file."""
  global _CACHED_CONFIG
  if config_path is None and _CACHED_CONFIG is not None:
    return _CACHED_CONFIG

  resolved_path = (
      os.path.abspath(os.path.expanduser(config_path))
      if config_path is not None
      else get_config_path()
  )
  if not os.path.exists(resolved_path):
    return AppConfig()

  config_name = os.path.basename(resolved_path)
  try:
    with open(resolved_path, "r", encoding="utf-8") as f:
      content = f.read().strip()
      if not content:
        config_data = {}
      else:
        config_data = json.loads(content)
  except json.JSONDecodeError as e:
    raise click.ClickException(f"Failed to parse {config_name}: {e}") from e
  except Exception as e:
    raise click.ClickException(f"Failed to read {config_name}: {e}") from e

  if not isinstance(config_data, dict):
    raise click.ClickException(
        f"{config_name}: Config must be a JSON object (dict)."
    )

  schema = _load_schema()
  _validate_schema(config_data, schema, config_name=config_name)

  app_config = AppConfig()
  if KEY_DEFAULT in config_data:
    app_config.default = _parse_model_config(config_data[KEY_DEFAULT])
  if KEY_MODELS in config_data:
    for model_id, model_data in config_data[KEY_MODELS].items():
      app_config.models[model_id] = _parse_model_config(model_data)

  if config_path is None:
    _CACHED_CONFIG = app_config
  return app_config


def get_model_config(model_id: str) -> ModelConfig:
  """Returns the configuration for a specific model, falling back to defaults."""
  config_data = load_config()
  merged = ModelConfig()

  merged.merge_from(config_data.default)

  model_cfg = config_data.models.get(model_id)
  if model_cfg is None:
    alt_id = (
        model_id.replace("/", "--")
        if "/" in model_id
        else model_id.replace("--", "/")
    )
    model_cfg = config_data.models.get(alt_id)

  if model_cfg is not None:
    merged.merge_from(model_cfg)

  return merged


def _clear_cache() -> None:
  """Clears the cached configuration and custom config path (primarily for testing)."""
  global _CACHED_CONFIG, _CUSTOM_CONFIG_PATH
  _CACHED_CONFIG = None
  _CUSTOM_CONFIG_PATH = None
