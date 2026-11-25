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

import email.message
import errno
import http.client
import http.server
import importlib
import os
import threading
from unittest import mock
import urllib.error
import urllib.request

from absl.testing import absltest
from absl.testing import parameterized
import click
from click.testing import CliRunner

from litert_lm_cli import cli_helpers
from litert_lm_cli import huggingface_download
from litert_lm_cli import model

# Import 'import' subcommand dynamically to bypass Python keyword restriction.
import_cmd = importlib.import_module(
    "litert_lm_cli.commands.import"
)


class ImportTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.enter_context(
        mock.patch.object(
            model,
            "get_cli_base_dir",
            return_value=self.temp_dir.full_path,
            autospec=True,
        )
    )
    # Mock Model to use temp dir
    self.mock_model = mock.MagicMock()
    self.mock_model.model_path = os.path.join(
        self.temp_dir.full_path, "models", "model.litertlm"
    )

    def _from_model_id(model_id: str):
      self.mock_model.model_id = model_id
      return self.mock_model

    self.mock_from_model_id = self.enter_context(
        mock.patch.object(
            model.Model,
            "from_model_id",
            side_effect=_from_model_id,
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

  def _mock_http_response(
      self, payload: bytes, *, content_length: str | None = "auto"
  ) -> mock.MagicMock:
    response = mock.create_autospec(http.client.HTTPResponse, instance=True)
    response.__enter__.return_value = response
    response.getheader.return_value = (
        str(len(payload)) if content_length == "auto" else content_length
    )
    response.read.side_effect = [payload, b""]
    return response

  @parameterized.named_parameters(
      ("q", "https://example.com/dir/gemma.litertlm?id=1", "gemma.litertlm"),
      ("root_fallback", "https://example.com/?id=1", "model.litertlm"),
      ("dot_fallback", "https://example.com/.", "model.litertlm"),
      ("dotdot_fallback", "https://example.com/dir/..?id=1", "model.litertlm"),
      ("http_scheme", "http://example.com/model.litertlm", "model.litertlm"),
  )
  def test_import_url_default_model_id(self, url: str, expected_id: str):
    content_length = None if url.endswith("/?id=1") else "auto"
    with mock.patch.object(
        urllib.request.OpenerDirector,
        "open",
        return_value=self._mock_http_response(
            b"model", content_length=content_length
        ),
        autospec=True,
    ) as mock_open:
      result = CliRunner().invoke(import_cmd.import_model, [url])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertEqual(self.mock_model.model_id, expected_id)
    self.assertEqual(mock_open.call_args.kwargs["timeout"], 60)

  @parameterized.named_parameters(
      (
          "url_as_model_id",
          ["local.litertlm", "https://example.com/download?id=123"],
          "is a URL, not a model ID",
      ),
      (
          "url_with_hf_repo",
          [
              "--from-huggingface-repo",
              "org/repo",
              "https://example.com/model.litertlm",
              "my-model",
          ],
          "leave out --from-huggingface-repo",
      ),
      ("malformed_ipv6", ["http://[::1/model.litertlm"], "Invalid URL"),
      ("dot_id", ["local.litertlm", "."], "Invalid model ID '.'"),
      ("dotdot_id", ["local.litertlm", ".."], "Invalid model ID '..'"),
      ("empty_id", ["local.litertlm", ""], "Invalid model ID ''"),
      ("backslash_id", ["local.litertlm", "dir\\escape"], "Invalid model ID"),
  )
  def test_import_rejects_invalid_arguments(
      self, cli_args: list[str], expected_msg: str
  ):
    result = CliRunner().invoke(import_cmd.import_model, cli_args)
    self.assertEqual(result.exit_code, 2, msg=result.output)
    self.assertIn(expected_msg, result.output)

  def test_import_from_local_http_server(self):
    served = {"payload": b"url model bytes!"}

    class _Handler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        if self.path in ("/redirect", "/redirect-ftp"):
          target = (
              f"http://127.0.0.1:{server.server_port}/model.litertlm"
              if self.path == "/redirect"
              else "ftp://example.com/model.litertlm"
          )
          self.send_response(302)
          self.send_header("Location", target)
          self.end_headers()
          return
        body = served["payload"]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    base_url = f"http://127.0.0.1:{server.server_port}"

    result = CliRunner().invoke(
        import_cmd.import_model, [f"{base_url}/model.litertlm", "my-model"]
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("(16B)...", result.output)
    self.assertIn("litert-lm run my-model", result.output)
    with open(self.mock_model.model_path, "rb") as f:
      self.assertEqual(f.read(), b"url model bytes!")
    downloading_dir = os.path.join(self.temp_dir.full_path, "downloading")
    self.assertEmpty(os.listdir(downloading_dir))

    served["payload"] = b"updated url model bytes!"
    result = CliRunner().invoke(
        import_cmd.import_model, [f"{base_url}/redirect", "my-model"]
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    with open(self.mock_model.model_path, "rb") as f:
      self.assertEqual(f.read(), b"updated url model bytes!")

    result = CliRunner().invoke(
        import_cmd.import_model, [f"{base_url}/redirect-ftp", "my-model"]
    )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Redirect to unsupported scheme 'ftp'", result.output)

  def test_url_redirect_and_download_errors(self):
    handler = import_cmd._UrlRedirectHandler()
    headers = email.message.Message()
    url = "https://example.com/model.litertlm"
    with self.assertRaisesRegex(urllib.error.URLError, "unencrypted http://"):
      handler.redirect_request(
          urllib.request.Request("HTTPS://example.com/a"),
          None, 302, "Found", headers, "HTTP://example.com/b",
      )

    for outcome, expected_msg in [
        (
            self._mock_http_response(b"short", content_length="100"),
            f"Incomplete download from {url}: got 5 of 100 bytes.",
        ),
        (
            urllib.error.HTTPError(url, 404, "Not Found", headers, None),
            f"No model found at {url} (HTTP 404)",
        ),
        (
            urllib.error.HTTPError(url, 500, "Error", headers, None),
            f"Failed to download model from {url}",
        ),
        (
            http.client.RemoteDisconnected("closed"),
            f"Failed to download model from {url}",
        ),
    ]:
      is_exc = isinstance(outcome, Exception)
      with mock.patch.object(
          urllib.request.OpenerDirector,
          "open",
          return_value=mock.DEFAULT if is_exc else outcome,
          side_effect=outcome if is_exc else None,
          autospec=True,
      ):
        result = CliRunner().invoke(import_cmd.import_model, [url, "my-model"])
      self.assertEqual(result.exit_code, 1)
      self.assertIn(expected_msg, result.output)
      self.assertFalse(
          os.path.exists(os.path.dirname(self.mock_model.model_path))
      )
    downloading_dir = os.path.join(self.temp_dir.full_path, "downloading")
    self.assertEmpty(os.listdir(downloading_dir))


if __name__ == "__main__":
  absltest.main()
