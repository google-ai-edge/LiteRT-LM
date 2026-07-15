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

"""Unit tests for the main litert-lm CLI."""

import os
import unittest.mock

from absl.testing import absltest
from click.testing import CliRunner
from prompt_toolkit import key_binding

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import main
from litert_lm_cli.commands import run as run_cmd


class MainTest(absltest.TestCase):

  def test_help_shorthand(self):
    runner = CliRunner()
    result_help = runner.invoke(main.cli, ["--help"])
    result_h = runner.invoke(main.cli, ["-h"])
    self.assertEqual(result_help.exit_code, 0)
    self.assertEqual(result_h.exit_code, 0)
    self.assertEqual(result_help.output, result_h.output)

  def test_cache_dir_value_from_cache_mode(self):
    self.assertEqual(common.cache_dir_value_from_cache_mode("no"), ":nocache")
    self.assertEqual(
        common.cache_dir_value_from_cache_mode("memory"), ":memory"
    )
    self.assertEqual(common.cache_dir_value_from_cache_mode("disk"), "")
    with self.assertRaises(ValueError):
      common.cache_dir_value_from_cache_mode("invalid")

  def test_subcommand_help_shorthand(self):
    runner = CliRunner()
    result_help = runner.invoke(main.cli, ["list", "--help"])
    result_h = runner.invoke(main.cli, ["list", "-h"])
    self.assertEqual(result_help.exit_code, 0)
    self.assertEqual(result_h.exit_code, 0)
    self.assertEqual(result_help.output, result_h.output)

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_piped_input(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    # Mocking stdin by providing input to the runner
    result = runner.invoke(
        main.cli, ["run", "my-model"], input="Hello from pipe\n"
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertEqual(kwargs["prompt"], "Hello from pipe")

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_prompt_and_piped_input(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    # Mocking stdin by providing input to the runner
    result = runner.invoke(
        main.cli,
        ["run", "my-model", "--prompt", "Prompt arg"],
        input="Hello from pipe\n",
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertEqual(kwargs["prompt"], "Prompt arg\n\nHello from pipe")

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_non_tty_no_input(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    # No input provided, isatty will be False in CliRunner
    result = runner.invoke(main.cli, ["run", "my-model"])

    self.assertEqual(result.exit_code, 0)
    # Should return early and not start the interactive session
    mock_run_interactive.assert_not_called()

  def test_create_keybindings(self):
    kb = run_cmd._create_keybindings()
    self.assertIsInstance(kb, key_binding.KeyBindings)
    # Check if expected keys are added.
    keys = [str(b.keys) for b in kb.bindings]
    # Check if enter (ControlM), c-j (ControlJ), esc+enter, c-c (ControlC).
    self.assertTrue(any("ControlM" in k and "Escape" not in k for k in keys))
    self.assertTrue(any("ControlJ" in k for k in keys))
    self.assertTrue(any("Escape" in k and "ControlM" in k for k in keys))
    self.assertTrue(any("ControlC" in k for k in keys))

  def test_run_sampling_flags(self):
    with unittest.mock.patch(
        "litert_lm_cli.model.Model.from_model_reference",
        autospec=True,
    ) as mock_from_model_ref, unittest.mock.patch(
        "litert_lm_cli.commands.run.run_interactive",
        autospec=True,
    ) as mock_run_interactive:
      mock_model = unittest.mock.MagicMock()
      mock_from_model_ref.return_value = mock_model
      mock_model.exists.return_value = True

      runner = CliRunner()
      result = runner.invoke(
          main.cli,
          [
              "run",
              "my-model",
              "--prompt",
              "hi",
              "--top-k",
              "10",
              "--top-p",
              "0.9",
              "--temperature",
              "0.8",
              "--seed",
              "42",
          ],
      )

      self.assertEqual(result.exit_code, 0)
      mock_run_interactive.assert_called_once()
      kwargs = mock_run_interactive.call_args.kwargs
      self.assertEqual(kwargs["top_k"], 10)
      self.assertEqual(kwargs["top_p"], 0.9)
      self.assertEqual(kwargs["temperature"], 0.8)
      self.assertEqual(kwargs["seed"], 42)

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_activation_data_type_flag(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    for choice in ["fp32", "fp16", "int16", "int8"]:
      mock_run_interactive.reset_mock()
      runner = CliRunner()
      result = runner.invoke(
          main.cli,
          [
              "run",
              "my-model",
              "--prompt",
              "hi",
              f"--activation-data-type={choice}",
          ],
      )

      self.assertEqual(result.exit_code, 0)
      mock_run_interactive.assert_called_once()
      kwargs = mock_run_interactive.call_args.kwargs
      self.assertEqual(
          kwargs["activation_data_type"],
          litert_lm.ActivationDataType.from_str(choice),
      )

  def test_run_activation_data_type_flag_invalid(self):
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--prompt",
            "hi",
            "--activation-data-type=invalid",
        ],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Invalid value for '--activation-data-type'", result.output)

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.benchmark.run_benchmark"
  )
  def test_benchmark_activation_data_type_flag(
      self, mock_run_benchmark, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    for choice in ["fp32", "fp16", "int16", "int8"]:
      mock_run_benchmark.reset_mock()
      runner = CliRunner()
      result = runner.invoke(
          main.cli,
          [
              "benchmark",
              "my-model",
              f"--activation-data-type={choice}",
          ],
      )

      self.assertEqual(result.exit_code, 0)
      mock_run_benchmark.assert_called_once()
      kwargs = mock_run_benchmark.call_args.kwargs
      self.assertEqual(
          kwargs["activation_data_type"],
          litert_lm.ActivationDataType.from_str(choice),
      )

  def test_benchmark_activation_data_type_flag_invalid(self):
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "benchmark",
            "my-model",
            "--activation-data-type=invalid",
        ],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Invalid value for '--activation-data-type'", result.output)

  def test_run_thinking_budget_flag(self):
    with unittest.mock.patch(
        "litert_lm_cli.model.Model.from_model_reference",
        autospec=True,
    ) as mock_from_model_ref, unittest.mock.patch(
        "litert_lm_cli.commands.run.run_interactive",
        autospec=True,
    ) as mock_run_interactive:
      mock_model = unittest.mock.MagicMock()
      mock_from_model_ref.return_value = mock_model
      mock_model.exists.return_value = True

      runner = CliRunner()
      result = runner.invoke(
          main.cli,
          [
              "run",
              "my-model",
              "--prompt",
              "hi",
              "--thinking-budget",
              "10",
          ],
      )

      self.assertEqual(result.exit_code, 0)
      mock_run_interactive.assert_called_once()
      kwargs = mock_run_interactive.call_args.kwargs
      self.assertEqual(kwargs["thinking_budget"], 10)

  def test_run_no_template_flag(self):
    runner = CliRunner()
    # Test that --no-template is a valid option for the run command.
    # We use --help to avoid actually running the model.
    result = runner.invoke(main.cli, ["run", "--help"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("--no-template", result.output)

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_vision_and_audio_backends(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--vision-backend",
            "gpu",
            "--audio-backend",
            "cpu",
            "--prompt",
            "Hi",
        ],
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertEqual(kwargs["vision_backend"], "gpu")
    self.assertEqual(kwargs["audio_backend"], "cpu")

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_default_backends(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--prompt",
            "Hi",
        ],
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertIsNone(kwargs["vision_backend"])
    self.assertIsNone(kwargs["audio_backend"])

  @unittest.mock.patch("os.path.expanduser")
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_attachments(
      self, mock_run_interactive, mock_from_model_ref, mock_expanduser
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    # Mock expanduser to return the path as is, or a fake expanded path
    mock_expanduser.side_effect = lambda x: x.replace("~", "/home/user")

    runner = CliRunner()
    with runner.isolated_filesystem():
      # We need to make sure the "expanded" path exists for the check in main.py
      # Since we are in an isolated filesystem, we'll just use simple names
      with open("image.jpg", "w") as f:
        f.write("image content")

      # For tilde expansion test, we mock os.path.exists as well if needed,
      # or just use paths that will exist.
      with unittest.mock.patch("os.path.exists", return_value=True):
        result = runner.invoke(
            main.cli,
            [
                "run",
                "my-model",
                "--vision-backend",
                "gpu",
                "--audio-backend",
                "cpu",
                "--attachment",
                "~/audio.wav",
                "--attachment",
                "image.jpg",
                "--prompt",
                "Hi",
            ],
        )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertEqual(kwargs["vision_backend"], "gpu")
    self.assertEqual(kwargs["audio_backend"], "cpu")
    self.assertEqual(
        kwargs["attachments"], ("/home/user/audio.wav", "image.jpg")
    )

  @unittest.mock.patch("os.path.exists", return_value=True)
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_audio_attachment_default_backend(
      self, mock_run_interactive, mock_from_model_ref, mock_exists
  ):
    del mock_exists  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--attachment",
            "audio.wav",
            "--prompt",
            "Hi",
        ],
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertIsNone(kwargs["audio_backend"])
    self.assertIsNone(kwargs["vision_backend"])

  @unittest.mock.patch("os.path.exists", return_value=True)
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_image_attachment_default_backend(
      self, mock_run_interactive, mock_from_model_ref, mock_exists
  ):
    del mock_exists  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--attachment",
            "image.jpg",
            "--prompt",
            "Hi",
        ],
    )

    self.assertEqual(result.exit_code, 0)
    mock_run_interactive.assert_called_once()
    kwargs = mock_run_interactive.call_args.kwargs
    self.assertIsNone(kwargs["audio_backend"])
    self.assertIsNone(kwargs["vision_backend"])

  @unittest.mock.patch("os.path.exists", return_value=True)
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_unsupported_attachment_type(
      self, mock_run_interactive, mock_from_model_ref, mock_exists
  ):
    del mock_exists  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--attachment",
            "test.txt",
            "--prompt",
            "Hi",
        ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Unsupported attachment type", result.output)
    mock_run_interactive.assert_not_called()

  @unittest.mock.patch("os.path.exists", return_value=False)
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_non_existent_attachment(
      self, mock_run_interactive, mock_from_model_ref, mock_exists
  ):
    del mock_exists  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--attachment",
            "ghost.jpg",
            "--prompt",
            "Hi",
        ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("File 'ghost.jpg' does not exist.", result.output)
    mock_run_interactive.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  @unittest.mock.patch(
      "litert_lm_cli.commands.run.run_interactive"
  )
  def test_run_with_attachments_and_no_template(
      self, mock_run_interactive, mock_from_model_ref
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True

    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        [
            "run",
            "my-model",
            "--attachment",
            "image.jpg",
            "--no-template",
            "--prompt",
            "Hi",
        ],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn(
        "Error: Attachments are not supported with --no-template.",
        result.output,
    )
    mock_run_interactive.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_cli.commands.list.os.stat"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.get_all_models"
  )
  def test_list_models(self, mock_get_all_models, mock_stat):
    mock_model1 = unittest.mock.MagicMock()
    mock_model1.model_id = "gemma3-1b"
    mock_model1.model_path = "/path/to/gemma3-1b/model.litertlm"

    mock_model2 = unittest.mock.MagicMock()
    mock_model2.model_id = "custom-model"
    mock_model2.model_path = "/path/to/custom-model/model.litertlm"

    mock_get_all_models.return_value = [mock_model1, mock_model2]

    mock_stat_result = unittest.mock.MagicMock()
    mock_stat_result.st_size = 1024 * 1024 * 500  # 500 MB
    mock_stat_result.st_mtime = 1741212053  # 2026-03-05 17:00:53
    mock_stat.return_value = mock_stat_result

    runner = CliRunner()
    result = runner.invoke(main.cli, ["list"])

    self.assertEqual(result.exit_code, 0)
    self.assertIn("gemma3-1b", result.output)
    self.assertIn("500.0 MB", result.output)
    self.assertIn("custom-model", result.output)
    self.assertNotIn("Unknown", result.output)

  def test_run_invalid_cpu_thread_count(self):
    runner = CliRunner()
    result = runner.invoke(
        main.cli,
        ["run", "my-model", "--cpu-thread-count", "0"],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn(
        "Invalid value for '--cpu-thread-count': 0 is not in the range x>=1.",
        result.output,
    )

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_with_output_dir(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"
    mock_builder_unpack.return_value = "/path/to/unpacked/model.toml"

    runner = CliRunner()
    with runner.isolated_filesystem():
      result = runner.invoke(
          main.cli, ["unpack", "my-model", "--output-dir", "unpacked"]
      )
      self.assertEqual(result.exit_code, 0)
      mock_builder_unpack.assert_called_once_with(
          "/path/to/my-model/model.litertlm", "unpacked"
      )

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_with_file_output_dir_fails(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("existing_file", "w") as f:
        f.write("dummy")
      result = runner.invoke(
          main.cli, ["unpack", "my-model", "--output-dir", "existing_file"]
      )
      self.assertEqual(result.exit_code, 2)
      self.assertIn(
          "Directory 'existing_file' is a file.",
          result.output,
      )
      mock_builder_unpack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_default_dir(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"
    mock_builder_unpack.return_value = "my-model.litertlm.unpacked/model.toml"

    runner = CliRunner()
    with runner.isolated_filesystem():
      # Create a dummy source file to simulate unpacking a local .litertlm file
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      result = runner.invoke(main.cli, ["unpack", "my-model.litertlm"])
      self.assertEqual(result.exit_code, 0)
      mock_builder_unpack.assert_called_once_with(
          "/path/to/my-model/model.litertlm", "./my-model"
      )

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_default_dir_conflict_fails(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("my-model", "w") as f:
        f.write("existing file")
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      result = runner.invoke(main.cli, ["unpack", "my-model.litertlm"])
      self.assertEqual(result.exit_code, 0)
      self.assertIn(
          "Error: Cannot unpack into './my-model' because it conflicts with an"
          " existing file",
          result.output,
      )
      mock_builder_unpack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_default_dir_in_subfolder(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      os.makedirs("sub/dir", exist_ok=True)
      with open("sub/dir/model.litertlm", "w") as f:
        f.write("dummy")
      result = runner.invoke(main.cli, ["unpack", "sub/dir/model.litertlm"])
      self.assertEqual(result.exit_code, 0)
      mock_builder_unpack.assert_called_once_with(
          "/path/to/my-model/model.litertlm", "sub/dir/model"
      )

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_model_id_requires_output_dir(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      result = runner.invoke(main.cli, ["unpack", "my-model"])
      self.assertEqual(result.exit_code, 0)
      self.assertIn(
          "Error: --output-dir is required when unpacking a model ID.",
          result.output,
      )
      mock_builder_unpack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_command_expands_user(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"
    mock_builder_unpack.return_value = "/path/to/unpacked/model.toml"

    runner = CliRunner()
    with runner.isolated_filesystem():
      result = runner.invoke(
          main.cli, ["unpack", "my-model", "--output-dir", "~/unpacked"]
      )
      self.assertEqual(result.exit_code, 0)
      expected_dir = os.path.expanduser("~/unpacked")
      mock_builder_unpack.assert_called_once_with(
          "/path/to/my-model/model.litertlm", expected_dir
      )

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      result = runner.invoke(
          main.cli,
          ["pack", ".", "--output", "out.litertlm"],
      )
      self.assertEqual(result.exit_code, 0)
      expected_config = os.path.join(".", "model.toml")
      expected_out = os.path.abspath("out.litertlm")
      mock_builder_pack.assert_called_once_with(expected_config, expected_out)

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_default_output(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      result = runner.invoke(main.cli, ["pack"])
      self.assertEqual(result.exit_code, 0)
      expected_config = "model.toml"
      expected_out = os.path.abspath("output.litertlm")
      mock_builder_pack.assert_called_once_with(expected_config, expected_out)

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_directory_fails(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      os.makedirs("my_dir", exist_ok=True)
      result = runner.invoke(main.cli, ["pack", "my_dir"])
      self.assertEqual(result.exit_code, 0)
      self.assertIn("Error: TOML configuration file not found", result.output)
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_default_config_not_found(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      result = runner.invoke(main.cli, ["pack"])
      self.assertEqual(result.exit_code, 2)
      self.assertIn(
          "Path 'model.toml' does not exist.",
          result.output,
      )
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_with_directory(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      os.makedirs("my-model", exist_ok=True)
      with open("my-model/model.toml", "w") as f:
        f.write("dummy")
      result = runner.invoke(main.cli, ["pack", "my-model"])
      self.assertEqual(result.exit_code, 0)
      expected_config = os.path.join("my-model", "model.toml")
      expected_out = os.path.abspath("my-model.litertlm")
      mock_builder_pack.assert_called_once_with(expected_config, expected_out)

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_with_current_directory(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      os.makedirs("my-model", exist_ok=True)
      with open("my-model/model.toml", "w") as f:
        f.write("dummy")
      # Change working directory to inside my-model
      original_cwd = os.getcwd()
      try:
        os.chdir("my-model")
        result = runner.invoke(main.cli, ["pack", "."])
        self.assertEqual(result.exit_code, 0)
        expected_config = os.path.join(".", "model.toml")
        expected_out = os.path.abspath(os.path.join("..", "my-model.litertlm"))
        mock_builder_pack.assert_called_once_with(expected_config, expected_out)
      finally:
        os.chdir(original_cwd)

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_with_config_file_succeeds(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("test-model.toml", "w") as f:
        f.write("dummy")
      result = runner.invoke(main.cli, ["pack", "test-model.toml"])
      self.assertEqual(result.exit_code, 0)
      expected_out = os.path.abspath("output.litertlm")
      mock_builder_pack.assert_called_once_with("test-model.toml", expected_out)

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_with_non_existent_path_fails(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      result = runner.invoke(main.cli, ["pack", "invalid_path"])
      self.assertEqual(result.exit_code, 2)
      self.assertIn("Path 'invalid_path' does not exist.", result.output)
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_with_directory_output_fails(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      os.makedirs("output_dir", exist_ok=True)
      result = runner.invoke(main.cli, ["pack", "--output", "output_dir"])
      self.assertEqual(result.exit_code, 2)
      self.assertIn(
          "File 'output_dir' is a directory.",
          result.output,
      )
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_command_default_output_conflict_with_directory_fails(
      self, mock_builder_pack
  ):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      os.makedirs("output.litertlm", exist_ok=True)
      result = runner.invoke(main.cli, ["pack"])
      self.assertEqual(result.exit_code, 0)
      self.assertIn(
          "is a directory. The output path must be a file.",
          result.output,
      )
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_existing_output_non_interactive_fails(self, mock_builder_pack):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      with open("out.litertlm", "w") as f:
        f.write("existing")
      result = runner.invoke(
          main.cli, ["pack", ".", "--output", "out.litertlm"]
      )
      self.assertEqual(result.exit_code, 0)
      self.assertIn(
          "already exists. Please use a different --output", result.output
      )
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_existing_output_with_allow_overwrite_succeeds(
      self, mock_builder_pack
  ):
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      with open("out.litertlm", "w") as f:
        f.write("existing")
      result = runner.invoke(
          main.cli,
          ["pack", ".", "--output", "out.litertlm", "--allow-overwrite"],
      )
      self.assertEqual(result.exit_code, 0)
      mock_builder_pack.assert_called_once()

  @unittest.mock.patch(
      "litert_lm_cli.commands.pack._is_interactive",
      return_value=True,
  )
  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_existing_output_interactive_confirm(
      self, mock_builder_pack, mock_is_interactive
  ):
    del mock_is_interactive  # unused
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      with open("out.litertlm", "w") as f:
        f.write("existing")
      result = runner.invoke(
          main.cli,
          ["pack", ".", "--output", "out.litertlm"],
          input="y\n",
      )
      self.assertEqual(result.exit_code, 0)
      mock_builder_pack.assert_called_once()

  @unittest.mock.patch(
      "litert_lm_cli.commands.pack._is_interactive",
      return_value=True,
  )
  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.pack"
  )
  def test_pack_existing_output_interactive_abort(
      self, mock_builder_pack, mock_is_interactive
  ):
    del mock_is_interactive  # unused
    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("model.toml", "w") as f:
        f.write('[[section]]\nsection_type = "TFLiteModel"')
      with open("out.litertlm", "w") as f:
        f.write("existing")
      result = runner.invoke(
          main.cli,
          ["pack", ".", "--output", "out.litertlm"],
          input="n\n",
      )
      self.assertEqual(result.exit_code, 0)
      self.assertIn("Aborted.", result.output)
      mock_builder_pack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_existing_output_dir_non_interactive_fails(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      os.makedirs("my-model", exist_ok=True)
      result = runner.invoke(main.cli, ["unpack", "my-model.litertlm"])
      self.assertEqual(result.exit_code, 0)
      self.assertIn(
          "already exists. Please use a different --output-dir", result.output
      )
      mock_builder_unpack.assert_not_called()

  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_existing_output_dir_with_allow_overwrite_succeeds(
      self, mock_from_model_ref, mock_builder_unpack
  ):
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      os.makedirs("my-model", exist_ok=True)
      result = runner.invoke(
          main.cli, ["unpack", "my-model.litertlm", "--allow-overwrite"]
      )
      self.assertEqual(result.exit_code, 0)
      mock_builder_unpack.assert_called_once()

  @unittest.mock.patch(
      "litert_lm_cli.commands.unpack._is_interactive",
      return_value=True,
  )
  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_existing_output_dir_interactive_confirm(
      self, mock_from_model_ref, mock_builder_unpack, mock_is_interactive
  ):
    del mock_is_interactive  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      os.makedirs("my-model", exist_ok=True)
      result = runner.invoke(
          main.cli, ["unpack", "my-model.litertlm"], input="y\n"
      )
      self.assertEqual(result.exit_code, 0)
      mock_builder_unpack.assert_called_once()

  @unittest.mock.patch(
      "litert_lm_cli.commands.unpack._is_interactive",
      return_value=True,
  )
  @unittest.mock.patch(
      "litert_lm_builder.litertlm_builder.unpack"
  )
  @unittest.mock.patch(
      "litert_lm_cli.model.Model.from_model_reference"
  )
  def test_unpack_existing_output_dir_interactive_abort(
      self, mock_from_model_ref, mock_builder_unpack, mock_is_interactive
  ):
    del mock_is_interactive  # unused
    mock_model = unittest.mock.MagicMock()
    mock_from_model_ref.return_value = mock_model
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/my-model/model.litertlm"

    runner = CliRunner()
    with runner.isolated_filesystem():
      with open("my-model.litertlm", "w") as f:
        f.write("dummy")
      os.makedirs("my-model", exist_ok=True)
      result = runner.invoke(
          main.cli, ["unpack", "my-model.litertlm"], input="n\n"
      )
      self.assertEqual(result.exit_code, 0)
      self.assertIn("Aborted.", result.output)
      mock_builder_unpack.assert_not_called()


if __name__ == "__main__":
  absltest.main()
