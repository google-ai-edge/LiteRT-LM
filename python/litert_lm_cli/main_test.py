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
from litert_lm_cli import model


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
    from litert_lm_cli.commands import run as run_cmd

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


if __name__ == "__main__":
  absltest.main()
