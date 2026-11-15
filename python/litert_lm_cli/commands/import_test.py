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

"""Unit tests for the LiteRT-LM import command."""

import errno
import importlib
import os
from unittest import mock

from absl.testing import absltest
import click
from click.testing import CliRunner

from litert_lm_cli import cli_helpers
from litert_lm_cli import huggingface_download
from litert_lm_cli import model

# Import 'import' subcommand dynamically to bypass Python keyword restriction.
import_cmd = importlib.import_module(
    "litert_lm_cli.commands.import"
)


class ImportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    # Mock Model to use temp dir
    self.mock_model = mock.MagicMock()
    self.mock_model.model_path = os.path.join(
        self.temp_dir.full_path, "model.litertlm"
    )

    self.mock_from_model_id = self.enter_context(
        mock.patch.object(
            model.Model,
            "from_model_id",
            return_value=self.mock_model,
            autospec=True,
        )
    )

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  @mock.patch.object(
      huggingface_download, "download_from_huggingface", autospec=True
  )
  def test_import_hf_success(self, mock_download, mock_resolve):
    mock_resolve.return_value = "model2.litertlm"
    fake_source = os.path.join(self.temp_dir.full_path, "fake_source")
    with open(fake_source, "w") as f:
      f.write("fake content")
    mock_download.return_value = fake_source

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo"],
    )

    self.assertEqual(result.exit_code, 0)
    mock_resolve.assert_called_once_with("org/repo", None)
    mock_download.assert_called_once_with(
        repo_id="org/repo", filename="model2.litertlm", token=None
    )
    self.assertTrue(os.path.exists(self.mock_model.model_path))
    with open(self.mock_model.model_path, "r") as f:
      self.assertEqual(f.read(), "fake content")

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  def test_import_resolve_error(self, mock_resolve):
    mock_resolve.side_effect = click.ClickException("Resolve failed")

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo"],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Resolve failed", result.output)
    mock_resolve.assert_called_once()

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  def test_import_resolve_abort(self, mock_resolve):
    mock_resolve.side_effect = click.Abort()

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo"],
    )

    self.assertNotEqual(result.exit_code, 0)
    mock_resolve.assert_called_once()

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  @mock.patch.object(
      huggingface_download, "download_from_huggingface", autospec=True
  )
  def test_import_hf_with_model_file(self, mock_download, mock_resolve):
    mock_resolve.return_value = "model2.litertlm"
    fake_source = os.path.join(self.temp_dir.full_path, "fake_source")
    with open(fake_source, "w") as f:
      f.write("fake content")
    mock_download.return_value = fake_source

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo", "model2.litertlm"],
    )

    self.assertEqual(result.exit_code, 0)
    mock_resolve.assert_not_called()
    mock_download.assert_called_once_with(
        repo_id="org/repo", filename="model2.litertlm", token=None
    )
    self.assertTrue(os.path.exists(self.mock_model.model_path))
    with open(self.mock_model.model_path, "r") as f:
      self.assertEqual(f.read(), "fake content")

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  @mock.patch.object(
      huggingface_download, "download_from_huggingface", autospec=True
  )
  def test_import_hf_with_model_file_and_import_as_model_id(
      self, mock_download, mock_resolve
  ):
    mock_resolve.return_value = "model2.litertlm"
    fake_source = os.path.join(self.temp_dir.full_path, "fake_source")
    with open(fake_source, "w") as f:
      f.write("fake content")
    mock_download.return_value = fake_source

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo", "model2.litertlm", "custom-id"],
    )

    self.assertEqual(result.exit_code, 0)
    mock_resolve.assert_not_called()
    mock_download.assert_called_once_with(
        repo_id="org/repo", filename="model2.litertlm", token=None
    )
    self.mock_from_model_id.assert_called_once_with("custom-id")
    self.assertTrue(os.path.exists(self.mock_model.model_path))
    with open(self.mock_model.model_path, "r") as f:
      self.assertEqual(f.read(), "fake content")

  @mock.patch.object(cli_helpers, "resolve_model_file", autospec=True)
  @mock.patch.object(
      huggingface_download, "download_from_huggingface", autospec=True
  )
  def test_import_creates_hard_link(self, mock_download, mock_resolve):
    mock_resolve.return_value = "model.litertlm"
    fake_source = os.path.join(self.temp_dir.full_path, "fake_source")
    with open(fake_source, "w") as f:
      f.write("hard link test content")
    mock_download.return_value = fake_source

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        ["--from-huggingface-repo", "org/repo"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertTrue(
        os.path.samefile(fake_source, self.mock_model.model_path),
        "Imported model should be a hard link to the source file.",
    )
    self.assertGreaterEqual(
        os.stat(self.mock_model.model_path).st_nlink,
        2,
        "Hard-linked file should have a link count of at least 2.",
    )

  def test_import_local_model_creates_hard_link(self):
    local_source = os.path.join(self.temp_dir.full_path, "local_model.litertlm")
    with open(local_source, "w") as f:
      f.write("local model content")

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        [local_source, "my-local-model"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertTrue(
        os.path.samefile(local_source, self.mock_model.model_path),
        "Imported local model should be a hard link to the source file.",
    )
    self.assertGreaterEqual(
        os.stat(self.mock_model.model_path).st_nlink,
        2,
    )

  def test_import_destination_already_exists_overwritten(self):
    # Pre-create the destination file with old content
    os.makedirs(os.path.dirname(self.mock_model.model_path), exist_ok=True)
    with open(self.mock_model.model_path, "w") as f:
      f.write("old stale content")

    new_source = os.path.join(self.temp_dir.full_path, "new_source.litertlm")
    with open(new_source, "w") as f:
      f.write("new updated content")

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        [new_source, "my-model"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertTrue(
        os.path.samefile(new_source, self.mock_model.model_path),
    )
    with open(self.mock_model.model_path, "r") as f:
      self.assertEqual(f.read(), "new updated content")

  def test_import_destination_same_file_noop(self):
    os.makedirs(os.path.dirname(self.mock_model.model_path), exist_ok=True)
    with open(self.mock_model.model_path, "w") as f:
      f.write("same file content")

    runner = CliRunner()
    result = runner.invoke(
        import_cmd.import_model,
        [self.mock_model.model_path, "my-model"],
    )

    self.assertEqual(result.exit_code, 0)
    with open(self.mock_model.model_path, "r") as f:
      self.assertEqual(f.read(), "same file content")

  def test_import_fallback_to_copy_on_os_error(self):
    local_source = os.path.join(
        self.temp_dir.full_path, "local_source.litertlm"
    )
    with open(local_source, "w") as f:
      f.write("copy fallback content")

    with mock.patch.object(
        os,
        "link",
        side_effect=OSError(errno.EXDEV, "Invalid cross-device link"),
    ) as mock_link:
      runner = CliRunner()
      result = runner.invoke(
          import_cmd.import_model,
          [local_source, "my-model"],
      )

      self.assertEqual(result.exit_code, 0)
      mock_link.assert_called_once()
      self.assertTrue(os.path.exists(self.mock_model.model_path))
      with open(self.mock_model.model_path, "r") as f:
        self.assertEqual(f.read(), "copy fallback content")
      self.assertEqual(os.stat(self.mock_model.model_path).st_nlink, 1)


if __name__ == "__main__":
  absltest.main()
