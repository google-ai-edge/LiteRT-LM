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
"""Tests for the LiteRT-LM CLI configuration manager."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import click
from click.testing import CliRunner
import jsonschema

from litert_lm_cli import common
from litert_lm_cli import config


class ConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config._clear_cache()
    # Save original env
    self._original_env = os.environ.get("LITERT_LM_DIR")
    # Create temp dir and set env
    self.temp_dir = self.create_tempdir()
    os.environ["LITERT_LM_DIR"] = self.temp_dir.full_path

  def tearDown(self):
    # Restore original env
    if self._original_env is not None:
      os.environ["LITERT_LM_DIR"] = self._original_env
    else:
      os.environ.pop("LITERT_LM_DIR", None)
    super().tearDown()

  def _write_config(self, content: str) -> None:
    """Helper to write content to the config.json file."""
    config_path = config.get_config_path()
    with open(config_path, "w") as f:
      f.write(content)

  def test_get_config_no_file(self):
    # No file exists in the temp dir
    self.assertEqual(config.load_config(), config.AppConfig())

  def test_get_config_valid(self):
    self._write_config(
        '{"default": {"backend": "gpu", "cpu_thread_count": 4}, "models":'
        ' {"m1": {"cpu_thread_count": 8}}}'
    )
    self.assertEqual(
        config.load_config(),
        config.AppConfig(
            default=config.ModelConfig(backend="gpu", cpu_thread_count=4),
            models={
                "m1": config.ModelConfig(cpu_thread_count=8),
            },
        ),
    )

  def test_get_config_valid_all_fields(self):
    self._write_config(
        '{"default": {"audio_backend": "cpu", "vision_backend": "gpu",'
        ' "cache": "memory", "max_num_tokens": 1024, "temperature": 0.7,'
        ' "top_p": 0.9, "top_k": 40, "seed": 12345,'
        ' "gpu_decode_steps_per_sync": 4, "speculative_decoding": true,'
        ' "thinking": true, "thinking_budget": 5, "activation_data_type":'
        ' "fp16"}}'
    )
    self.assertEqual(
        config.load_config(),
        config.AppConfig(
            default=config.ModelConfig(
                audio_backend="cpu",
                vision_backend="gpu",
                cache="memory",
                max_num_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=12345,
                gpu_decode_steps_per_sync=4,
                speculative_decoding=True,
                thinking=True,
                thinking_budget=5,
                activation_data_type="fp16",
            )
        ),
    )

  def test_get_config_invalid_json(self):
    self._write_config("invalid json")
    with self.assertRaises(click.ClickException) as ctx:
      config.load_config()
    self.assertIn("Failed to parse config.json", str(ctx.exception))

  @parameterized.named_parameters(
      ("not_dict", "[]", "Config must be a JSON object"),
      (
          "default_not_dict",
          '{"default": 123}',
          "default: 123 is not of type 'object'",
      ),
      (
          "backend_invalid",
          '{"default": {"backend": "invalid"}}',
          "default.backend: 'invalid' is not one of ['cpu', 'gpu', 'npu']",
      ),
      (
          "default_cpu_thread_count_not_int",
          '{"default": {"cpu_thread_count": "four"}}',
          "default.cpu_thread_count: 'four' is not of type 'integer'",
      ),
      (
          "default_cpu_thread_count_invalid",
          '{"default": {"cpu_thread_count": 0}}',
          "default.cpu_thread_count: 0 is less than the minimum of 1",
      ),
      (
          "default_cpu_thread_count_negative",
          '{"default": {"cpu_thread_count": -1}}',
          "default.cpu_thread_count: -1 is less than the minimum of 1",
      ),
      (
          "models_not_dict",
          '{"models": []}',
          "models: [] is not of type 'object'",
      ),
      (
          "model_entry_not_dict",
          '{"models": {"m": 123}}',
          "models.m: 123 is not of type 'object'",
      ),
      (
          "model_backend_invalid",
          '{"models": {"m": {"backend": "invalid"}}}',
          "models.m.backend: 'invalid' is not one of ['cpu', 'gpu', 'npu']",
      ),
      (
          "model_cpu_thread_count_not_int",
          '{"models": {"m": {"cpu_thread_count": "four"}}}',
          "models.m.cpu_thread_count: 'four' is not of type 'integer'",
      ),
      (
          "model_cpu_thread_count_invalid",
          '{"models": {"m": {"cpu_thread_count": 0}}}',
          "models.m.cpu_thread_count: 0 is less than the minimum of 1",
      ),
      (
          "audio_backend_invalid",
          '{"default": {"audio_backend": "invalid"}}',
          (
              "default.audio_backend: 'invalid' is not one of ['cpu', 'gpu',"
              " 'npu']"
          ),
      ),
      (
          "vision_backend_invalid",
          '{"default": {"vision_backend": "tpu"}}',
          "default.vision_backend: 'tpu' is not one of ['cpu', 'gpu', 'npu']",
      ),
      (
          "cache_invalid",
          '{"default": {"cache": "redis"}}',
          "default.cache: 'redis' is not one of ['disk', 'memory', 'no']",
      ),
      (
          "max_num_tokens_invalid",
          '{"default": {"max_num_tokens": 0}}',
          "default.max_num_tokens: 0 is less than the minimum of 1",
      ),
      (
          "temperature_negative",
          '{"default": {"temperature": -0.5}}',
          "default.temperature: -0.5 is less than the minimum of 0.0",
      ),
      (
          "top_p_out_of_range",
          '{"default": {"top_p": 1.5}}',
          "default.top_p: 1.5 is greater than the maximum of 1.0",
      ),
      (
          "top_k_invalid",
          '{"default": {"top_k": 0}}',
          "default.top_k: 0 is less than the minimum of 1",
      ),
      (
          "gpu_decode_steps_per_sync_invalid",
          '{"default": {"gpu_decode_steps_per_sync": 0}}',
          "default.gpu_decode_steps_per_sync: 0 is less than the minimum of 1",
      ),
      (
          "speculative_decoding_not_bool",
          '{"default": {"speculative_decoding": "true"}}',
          "default.speculative_decoding: 'true' is not of type 'boolean'",
      ),
      (
          "thinking_not_bool",
          '{"default": {"thinking": "true"}}',
          "default.thinking: 'true' is not of type 'boolean'",
      ),
      (
          "thinking_budget_not_int",
          '{"default": {"thinking_budget": "ten"}}',
          "default.thinking_budget: 'ten' is not of type 'integer'",
      ),
      (
          "thinking_budget_invalid",
          '{"default": {"thinking_budget": -2}}',
          "default.thinking_budget: -2 is less than the minimum of -1",
      ),
      (
          "activation_data_type_invalid",
          '{"default": {"activation_data_type": "invalid"}}',
          (
              "default.activation_data_type: 'invalid' is not one of ['fp32',"
              " 'fp16', 'int16', 'int8']"
          ),
      ),
  )
  def test_get_config_invalid_schema(self, json_data, expected_error):
    self._write_config(json_data)
    with self.assertRaises(click.ClickException) as ctx:
      config.load_config()
    self.assertIn(expected_error, str(ctx.exception))

  def test_get_model_config_no_file(self):
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertIsNone(result.backend)

  def test_get_model_config_valid(self):
    self._write_config('{"models": {"my-model": {"backend": "gpu"}}}')
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertEqual(result.backend, "gpu")

  def test_get_model_config_not_configured(self):
    self._write_config('{"models": {"other-model": {"backend": "gpu"}}}')
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertIsNone(result.backend)

  def test_get_model_config_with_fallback(self):
    self._write_config(
        '{"default": {"backend": "cpu", "cpu_thread_count": 4}, "models":'
        ' {"gpu-model": {"backend": "gpu"}, "custom-cpu-model":'
        ' {"cpu_thread_count": 8}, "empty-model": {}}}'
    )

    # 1. Model not in config -> should fall back to default (cpu, 4 threads)
    result1 = config.get_model_config("not-configured-model")
    self.assertEqual(result1.backend, "cpu")
    self.assertEqual(result1.cpu_thread_count, 4)

    # 2. Model in config with backend (gpu) only -> should use model backend
    # (gpu) and default threads (4)
    result2 = config.get_model_config("gpu-model")
    self.assertEqual(result2.backend, "gpu")
    self.assertEqual(result2.cpu_thread_count, 4)

    # 3. Model in config with threads (8) only -> should fall back to default
    # backend (cpu) and use model threads (8)
    result3 = config.get_model_config("custom-cpu-model")
    self.assertEqual(result3.backend, "cpu")
    self.assertEqual(result3.cpu_thread_count, 8)

    # 4. Model in config but empty -> should fall back to default (cpu, 4
    # threads)
    result4 = config.get_model_config("empty-model")
    self.assertEqual(result4.backend, "cpu")
    self.assertEqual(result4.cpu_thread_count, 4)

  def test_get_model_config_id_normalization(self):
    self._write_config(
        '{"models": {"gemma4--2b": {"backend": "gpu"}, "other/model":'
        ' {"cpu_thread_count": 8}, "google--gemma3-1b-it": {"backend":'
        ' "npu"}}}'
    )
    result1 = config.get_model_config("gemma4/2b")
    self.assertEqual(result1.backend, "gpu")

    result2 = config.get_model_config("other--model")
    self.assertEqual(result2.cpu_thread_count, 8)

    result3 = config.get_model_config("google/gemma3-1b-it")
    self.assertEqual(result3.backend, "npu")

  def test_get_model_config_with_fallback_new_fields(self):
    self._write_config(
        '{"default": {"audio_backend": "cpu", "temperature": 0.7},'
        ' "models": {"m1": {"vision_backend": "gpu", "temperature": 0.2}}}'
    )
    result = config.get_model_config("m1")
    self.assertEqual(result.audio_backend, "cpu")
    self.assertEqual(result.vision_backend, "gpu")
    self.assertEqual(result.temperature, 0.2)

  def test_get_config_unknown_key_ignored(self):
    self._write_config(
        '{"unknown_top_level": 999, "default": {"backend": "gpu",'
        ' "unknown_field": 123}}'
    )
    app_cfg = config.load_config()
    self.assertEqual(
        app_cfg,
        config.AppConfig(
            default=config.ModelConfig(backend="gpu"),
        ),
    )

  @parameterized.named_parameters(
      ("valid_empty", {}),
      (
          "valid_default",
          {"default": {"backend": "gpu", "cpu_thread_count": 4}},
      ),
      (
          "valid_full",
          {
              "default": {
                  "audio_backend": "cpu",
                  "vision_backend": "gpu",
                  "cache": "memory",
                  "max_num_tokens": 1024,
                  "temperature": 0.7,
                  "top_p": 0.9,
                  "top_k": 40,
                  "seed": 12345,
                  "gpu_decode_steps_per_sync": 4,
                  "speculative_decoding": True,
                  "thinking": True,
                  "thinking_budget": 5,
              },
              "models": {
                  "m1": {"cpu_thread_count": 8, "cache": "disk"},
              },
          },
      ),
      ("invalid_default_type", {"default": 123}),
      ("invalid_backend_enum", {"default": {"backend": "tpu"}}),
      ("invalid_thread_count_type", {"default": {"cpu_thread_count": "four"}}),
      ("invalid_thread_count_min", {"default": {"cpu_thread_count": 0}}),
      ("invalid_cache_enum", {"default": {"cache": "redis"}}),
      ("invalid_max_tokens_min", {"default": {"max_num_tokens": 0}}),
      ("invalid_temp_min", {"default": {"temperature": -0.5}}),
      ("invalid_top_p_max", {"default": {"top_p": 1.5}}),
      ("invalid_top_k_min", {"default": {"top_k": 0}}),
      (
          "invalid_gpu_decode_steps_min",
          {"default": {"gpu_decode_steps_per_sync": 0}},
      ),
      (
          "invalid_speculative_bool",
          {"default": {"speculative_decoding": "true"}},
      ),
      (
          "invalid_thinking_bool",
          {"default": {"thinking": "true"}},
      ),
      (
          "invalid_thinking_budget_type",
          {"default": {"thinking_budget": "five"}},
      ),
      (
          "invalid_thinking_budget_min",
          {"default": {"thinking_budget": -2}},
      ),
      ("invalid_models_type", {"models": []}),
      ("invalid_model_entry_type", {"models": {"m1": 123}}),
  )
  def test_differential_validation_with_jsonschema(self, config_data):
    """Diff test verifying our pure-Python validator matches jsonschema."""
    schema = config._load_schema()

    jsonschema_error = None
    jsonschema_path = None
    try:
      jsonschema.validate(instance=config_data, schema=schema)
    except jsonschema.ValidationError as e:
      jsonschema_error = e.message
      jsonschema_path = list(e.path)

    custom_error = None
    try:
      config._validate_schema(config_data, schema)
    except click.ClickException as e:
      custom_error = str(e)

    if jsonschema_error is None:
      self.assertIsNone(
          custom_error,
          f"Expected config_data {config_data} to be valid, but got:"
          f" {custom_error}",
      )
    else:
      self.assertIsNotNone(
          custom_error,
          f"Expected config_data {config_data} to fail validation with"
          f" {jsonschema_error}, but it passed custom validation.",
      )
      path_str = ".".join(str(p) for p in jsonschema_path)  # pyrefly: ignore[not-iterable]
      prefix = f"{path_str}: " if path_str else ""
      expected_err_prefix = f"config.json validation error: {prefix}"
      self.assertIn(expected_err_prefix, custom_error)

  def test_set_and_get_config_path(self):
    custom_path = os.path.join(self.temp_dir.full_path, "my_config.json")
    config.set_config_path(custom_path)
    self.assertEqual(config.get_config_path(), custom_path)

    config.set_config_path(None)
    self.assertEqual(
        config.get_config_path(),
        os.path.join(self.temp_dir.full_path, "config.json"),
    )

  def test_clear_cache_resets_custom_config_path(self):
    custom_path = os.path.join(self.temp_dir.full_path, "my_config.json")
    config.set_config_path(custom_path)
    self.assertEqual(config.get_config_path(), custom_path)

    config._clear_cache()
    self.assertEqual(
        config.get_config_path(),
        os.path.join(self.temp_dir.full_path, "config.json"),
    )

  def test_load_config_with_custom_path_arg(self):
    custom_path = os.path.join(self.temp_dir.full_path, "custom.json")
    with open(custom_path, "w", encoding="utf-8") as f:
      f.write('{"default": {"backend": "npu"}}')

    app_cfg = config.load_config(custom_path)
    self.assertEqual(app_cfg.default.backend, "npu")

  def test_load_config_with_set_config_path(self):
    custom_path = os.path.join(self.temp_dir.full_path, "custom.json")
    with open(custom_path, "w", encoding="utf-8") as f:
      f.write('{"models": {"my-model": {"backend": "gpu"}}}')

    config.set_config_path(custom_path)
    model_cfg = config.get_model_config("my-model")
    self.assertEqual(model_cfg.backend, "gpu")

  def test_custom_config_invalid_json_error_message(self):
    custom_path = os.path.join(self.temp_dir.full_path, "my_custom.json")
    with open(custom_path, "w", encoding="utf-8") as f:
      f.write("invalid json")

    config.set_config_path(custom_path)
    with self.assertRaises(click.ClickException) as ctx:
      config.load_config()
    self.assertIn("Failed to parse my_custom.json", str(ctx.exception))

  def test_load_config_empty_file(self):
    empty_path = os.path.join(self.temp_dir.full_path, "empty.json")
    with open(empty_path, "w", encoding="utf-8") as f:
      f.write("")

    app_cfg = config.load_config(empty_path)
    self.assertEqual(app_cfg, config.AppConfig())

  def test_load_config_dev_null(self):
    if not os.path.exists("/dev/null"):
      self.skipTest("/dev/null not available")
    config.set_config_path("/dev/null")
    app_cfg = config.load_config()
    self.assertEqual(app_cfg, config.AppConfig())
    model_cfg = config.get_model_config("some-model")
    self.assertEqual(model_cfg, config.ModelConfig())

  def test_parse_config_opt(self):
    common.parse_config_opt(None, None, "/path/to/custom.json")
    self.assertEqual(config.get_config_path(), "/path/to/custom.json")

    common.parse_config_opt(None, None, None)
    self.assertEqual(config.get_config_path(), "/path/to/custom.json")

  def test_config_option_sets_custom_config_path(self):
    @click.command()
    @common.config_option
    def dummy_cmd():
      click.echo(f"config_path: {config.get_config_path()}")

    custom_file = os.path.join(self.temp_dir.full_path, "custom.json")
    with open(custom_file, "w", encoding="utf-8") as f:
      f.write("{}")

    runner = CliRunner()
    result = runner.invoke(dummy_cmd, ["--config", custom_file])
    self.assertEqual(result.exit_code, 0)
    self.assertIn(f"config_path: {custom_file}", result.output)

  def test_config_option_non_existent_file(self):
    @click.command()
    @common.config_option
    def dummy_cmd():
      pass

    runner = CliRunner()
    result = runner.invoke(
        dummy_cmd, ["--config", "/nonexistent/path/config.json"]
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("does not exist", result.output)

  def test_config_option_directory_rejected(self):
    @click.command()
    @common.config_option
    def dummy_cmd():
      pass

    runner = CliRunner()
    result = runner.invoke(dummy_cmd, ["--config", self.temp_dir.full_path])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("is a directory", result.output)

  def test_config_option_is_eager(self):
    events = []

    def normal_callback(ctx, param, value):
      events.append("normal_callback")
      return value

    @click.command()
    @click.option("--normal", callback=normal_callback)
    @common.config_option
    def dummy_cmd(normal):
      pass

    custom_file = os.path.join(self.temp_dir.full_path, "custom.json")
    with open(custom_file, "w", encoding="utf-8") as f:
      f.write("{}")

    runner = CliRunner()
    result = runner.invoke(
        dummy_cmd, ["--normal", "val", "--config", custom_file]
    )
    self.assertEqual(result.exit_code, 0)
    self.assertEqual(config.get_config_path(), custom_file)


if __name__ == "__main__":
  absltest.main()
