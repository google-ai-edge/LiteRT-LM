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

import email.message
import http.client
import io
import pathlib
from unittest import mock
import urllib.error
import urllib.request

from absl.testing import absltest
import click

from litert_lm_cli import huggingface_download
from litert_lm_cli import model


class HuggingFaceDownloadTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.mock_get_cli_base_dir = self.enter_context(
        mock.patch.object(
            model,
            "get_cli_base_dir",
            return_value=self.temp_dir.full_path,
            autospec=True,
        )
    )

  def test_download_from_huggingface_cached(self):
    repo_id = "test-owner/test-model"
    filename = "model.litertlm"
    cached_path = (
        pathlib.Path(self.temp_dir.full_path)
        / "cache"
        / "huggingface"
        / repo_id
        / filename
    )
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text("cached content")

    with mock.patch.object(
        urllib.request, "urlopen", autospec=True
    ) as mock_urlopen:
      result = huggingface_download.download_from_huggingface(
          repo_id=repo_id, filename=filename, token=None
      )
      self.assertEqual(result, str(cached_path))
      mock_urlopen.assert_not_called()

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_from_huggingface_success(self, mock_urlopen):
    repo_id = "test-owner/test-model"
    filename = "model.litertlm"
    expected_content = b"model data"

    mock_response = mock.create_autospec(
        http.client.HTTPResponse, spec_set=True, instance=True
    )
    mock_response.getheader.return_value = str(len(expected_content))
    bytes_io = io.BytesIO(expected_content)
    mock_response.read.side_effect = bytes_io.read
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = huggingface_download.download_from_huggingface(
        repo_id=repo_id, filename=filename, token="fake-token"
    )

    expected_path = (
        pathlib.Path(self.temp_dir.full_path)
        / "cache"
        / "huggingface"
        / repo_id
        / filename
    )
    self.assertEqual(result, str(expected_path))
    self.assertTrue(expected_path.exists())
    self.assertEqual(expected_path.read_bytes(), expected_content)

    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    self.assertEqual(req.get_header("Authorization"), "Bearer fake-token")
    self.assertEqual(
        req.full_url,
        f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
    )

  @mock.patch.object(click, "echo", autospec=True)
  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_from_huggingface_http_error_gated(
      self, mock_urlopen, mock_echo
  ):
    repo_id = "test-owner/test-model"
    filename = "model.litertlm"

    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://fake",
        code=401,
        msg="Unauthorized",
        hdrs=email.message.Message(),
        fp=None,
    )

    with self.assertRaises(click.ClickException) as ctx:
      huggingface_download.download_from_huggingface(
          repo_id=repo_id, filename=filename, token=None
      )

    self.assertIn("HTTP 401", str(ctx.exception))
    mock_echo.assert_called_once()
    self.assertIn("token not found", mock_echo.call_args[0][0])

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_from_huggingface_http_error_generic(self, mock_urlopen):
    repo_id = "test-owner/test-model"
    filename = "model.litertlm"

    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://fake",
        code=404,
        msg="Not Found",
        hdrs=email.message.Message(),
        fp=None,
    )

    with self.assertRaises(click.ClickException) as ctx:
      huggingface_download.download_from_huggingface(
          repo_id=repo_id, filename=filename, token="some-token"
      )

    self.assertIn("404", str(ctx.exception))

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_from_huggingface_url_error(self, mock_urlopen):
    repo_id = "test-owner/test-model"
    filename = "model.litertlm"

    mock_urlopen.side_effect = urllib.error.URLError("connection refused")

    with self.assertRaises(click.ClickException) as ctx:
      huggingface_download.download_from_huggingface(
          repo_id=repo_id, filename=filename, token=None
      )

    self.assertIn("connection refused", str(ctx.exception))


if __name__ == "__main__":
  absltest.main()
