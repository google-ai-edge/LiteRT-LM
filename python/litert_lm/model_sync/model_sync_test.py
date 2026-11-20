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
"""Tests for LiteRT-LM ModelSync Python API."""

from __future__ import annotations

from collections import abc
import email.message
import hashlib
import http.server
import os
import pathlib
import struct
import threading
from typing import cast
from unittest import mock
import urllib.request

from absl import flags
from absl.testing import absltest
import flatbuffers

from litert_lm import model_sync
from litert_lm.model_sync import _model_sync
from litert_lm_builder import litertlm_header_schema_py_generated as schema


def _create_test_model(
    path: str,
    sections: list[tuple[int, str, bytes, int]],
    sha_override: str | None = None,
) -> None:
  """Constructs a .litertlm file for unit testing."""
  sec_objs = []
  for data_type, model_type, payload, begin_offset in sections:
    sha = sha_override or hashlib.sha256(payload).hexdigest()
    items = [
        schema.KeyValuePairT(
            key="sha256",
            valueType=schema.VData.StringValue,
            value=schema.StringValueT(sha),
        )
    ]
    if model_type:
      items.append(
          schema.KeyValuePairT(
              key="model_type",
              valueType=schema.VData.StringValue,
              value=schema.StringValueT(model_type),
          )
      )
    sec_objs.append(
        schema.SectionObjectT(
            items=items,
            dataType=data_type,
            beginOffset=begin_offset,
            endOffset=begin_offset + len(payload),
        )
    )

  root = schema.LiteRTLMMetaDataT(
      sectionMetadata=schema.SectionMetadataT(objects=sec_objs)
  )
  b = flatbuffers.Builder(1024)
  b.Finish(root.Pack(b))
  fb_bytes = b.Output()

  with open(path, "wb") as f:
    f.write(b"LITERTLM")
    f.write(struct.pack("<IIIIQ", 1, 9, 0, 0, 32 + len(fb_bytes)))
    f.write(fb_bytes)
    for _, _, payload, begin_offset in sections:
      if f.tell() < begin_offset:
        f.write(b"\0" * (begin_offset - f.tell()))
      f.write(payload)


class ModelSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    self.local_path = os.path.join(self.test_dir, "local.litertlm")
    self.remote_path = os.path.join(self.test_dir, "remote.litertlm")

    _create_test_model(
        self.local_path,
        [
            (4, "", b"tokenizer_v1", 4096),
            (3, "weights", b"weights_payload_shared", 8192),
        ],
    )
    _create_test_model(
        self.remote_path,
        [
            (4, "", b"tokenizer_v2_updated", 4096),
            (3, "weights", b"weights_payload_shared", 8192),
        ],
    )

  def _start_http_server(
      self, handler: type[http.server.BaseHTTPRequestHandler]
  ) -> int:
    """Serves `handler` on a free local port until the test ends."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    return server.server_port

  def test_plan_sync(self):
    with model_sync.plan_sync(self.local_path, self.remote_path) as plan:
      self.assertFalse(plan.is_up_to_date)
      self.assertGreater(plan.bytes_to_download, 0)
      self.assertEqual(plan.bytes_to_reuse, 22)
      self.assertEqual(
          plan.total_remote_size, os.path.getsize(self.remote_path)
      )
      self.assertIn("DIFFERENTIAL SLICE UPDATE", plan.summary())
    with self.assertRaises(RuntimeError):
      _ = plan.is_up_to_date

    with model_sync.plan_sync(self.local_path, self.local_path) as plan:
      self.assertTrue(plan.is_up_to_date)
      self.assertEqual(plan.bytes_to_download, 0)

    with model_sync.plan_sync(None, self.remote_path) as plan:
      self.assertFalse(plan.is_up_to_date)
      self.assertEqual(plan.bytes_to_reuse, 0)
      self.assertIn("FULL", plan.summary())

  def test_plan_sync_with_custom_callable(self):
    def custom_fetcher(start: int, end: int) -> bytes:
      with open(self.remote_path, "rb") as f:
        f.seek(start)
        return f.read(end - start + 1)

    with model_sync.plan_sync(self.local_path, custom_fetcher) as plan:
      self.assertFalse(plan.is_up_to_date)
      self.assertEqual(plan.bytes_to_reuse, 22)

  def test_update_model_modes(self):
    with open(self.remote_path, "rb") as f:
      remote_bytes = f.read()
    for name, local, opts in [
        ("differential", self.local_path, None),
        ("none_base", None, None),
        ("clean", self.local_path, model_sync.SyncOptions(clean=True)),
        (
            "no_disk_check",
            self.local_path,
            model_sync.SyncOptions(check_disk_space=False),
        ),
    ]:
      with self.subTest(name=name):
        output_path = os.path.join(self.test_dir, f"{name}.litertlm")
        model_sync.update_model(
            local, output_path, self.remote_path, options=opts
        )
        with open(output_path, "rb") as f_out:
          self.assertEqual(f_out.read(), remote_bytes)

  def test_update_model_threshold_fallback(self):
    output_path = os.path.join(self.test_dir, "threshold_synced.litertlm")
    opts = model_sync.SyncOptions(full_download_threshold=0.01)
    with model_sync.plan_sync(self.local_path, self.remote_path, opts) as plan:
      self.assertIn("FULL DOWNLOAD", plan.summary())
      self.assertEqual(plan.bytes_to_reuse, 0)
      self.assertEqual(plan.bytes_to_download, plan.total_remote_size)

    remote_size = os.path.getsize(self.remote_path)
    fetched_ranges: list[tuple[int, int]] = []

    def recording_fetcher(start: int, end: int) -> bytes:
      fetched_ranges.append((start, end))
      with open(self.remote_path, "rb") as f:
        f.seek(start)
        return f.read(end - start + 1)

    model_sync.update_model(
        self.local_path, output_path, recording_fetcher, options=opts
    )
    self.assertTrue(
        any(
            0 < start <= 4096 and end == remote_size - 1
            for start, end in fetched_ranges
        ),
        msg=f"Expected a full-payload fetch, got {fetched_ranges}",
    )

  def test_http_range_fetcher_integration(self):
    with open(self.remote_path, "rb") as f:
      remote_data = f.read()

    class _RangeHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
          parts = range_header[6:].split("-")
          start = int(parts[0])
          end = min(
              int(parts[1]) if parts[1] else len(remote_data) - 1,
              len(remote_data) - 1,
          )
          chunk = remote_data[start : end + 1]
          self.send_response(206)
          self.send_header(
              "Content-Range", f"bytes {start}-{end}/{len(remote_data)}"
          )
          self.send_header("Content-Length", str(len(chunk)))
          self.end_headers()
          self.wfile.write(chunk)
        else:
          self.send_response(200)
          self.send_header("Content-Length", str(len(remote_data)))
          self.end_headers()
          self.wfile.write(remote_data)

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_RangeHandler)
    with model_sync.plan_sync(
        self.local_path, f"http://127.0.0.1:{port}/model.litertlm"
    ) as plan:
      self.assertFalse(plan.is_up_to_date)
      self.assertEqual(plan.bytes_to_reuse, 22)

  def test_checksum_verification_failure_raises(self):
    corrupt_path = os.path.join(self.test_dir, "corrupt.litertlm")
    _create_test_model(
        corrupt_path,
        [(1, "", b"real_payload_here", 4096)],
        sha_override="1" * 64,
    )
    output_path = os.path.join(self.test_dir, "corrupt_out.litertlm")
    with self.assertRaisesRegex(
        RuntimeError, "DATA_LOSS.*Section SHA-256 verification failed"
    ):
      model_sync.update_model(None, output_path, corrupt_path)

  def test_real_model_sync(self):
    real_model_path = str(
        pathlib.Path(flags.FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

    with model_sync.plan_sync(real_model_path, real_model_path) as plan:
      self.assertNotEmpty(plan.summary())
      self.assertGreater(plan.total_remote_size, 0)

    output_path = os.path.join(self.test_dir, "real_model_out.litertlm")
    model_sync.update_model(real_model_path, output_path, real_model_path)
    self.assertEqual(
        os.path.getsize(output_path), os.path.getsize(real_model_path)
    )

  def test_error_message_propagation(self):
    def failing_fetcher(start: int, end: int) -> bytes:
      del start, end
      raise RuntimeError("boom from python fetcher")

    with self.assertRaisesRegex(RuntimeError, "boom from python fetcher"):
      model_sync.plan_sync(self.local_path, failing_fetcher)

  def test_cache_sync_and_offline_fallback(self):
    cache_dir = os.path.join(self.test_dir, "offline_cache")
    seeded = model_sync.cache_sync(cache_dir, "offline-model", self.remote_path)
    self.assertTrue(os.path.exists(seeded))
    self.assertIn("offline-model", seeded)

    def offline_fetcher(start: int, end: int) -> bytes:
      del start, end
      raise TimeoutError("simulated timeout")

    fallback_path = model_sync.cache_sync(
        cache_dir,
        "offline-model",
        offline_fetcher,
        options=model_sync.SyncOptions(),
    )
    self.assertEqual(fallback_path, seeded)

    strict_opts = model_sync.SyncOptions(allow_offline_fallback=False)
    with self.assertRaisesRegex(RuntimeError, "simulated timeout"):
      model_sync.cache_sync(
          cache_dir, "offline-model", offline_fetcher, options=strict_opts
      )

  def test_sync_options_forward_only_set_fields(self):
    lib = mock.Mock()
    with mock.patch.object(_model_sync, "_ffi", autospec=True) as mock_ffi:
      mock_ffi.create_checked.return_value = 1234
      # pylint: disable-next=protected-access
      with _model_sync._c_options(
          lib, model_sync.SyncOptions(clean=True)
      ) as handle:
        self.assertEqual(handle, 1234)
    mock_ffi.call_checked.assert_called_once_with(
        lib, "litert_lm_model_sync_options_set_clean", 1234, True
    )
    lib.litert_lm_model_sync_options_delete.assert_called_once_with(1234)

  def test_cache_sync_falls_back_when_connection_drops(self):
    class _DisconnectingHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        # Close the connection without sending a response.
        self.close_connection = True

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_DisconnectingHandler)
    cache_dir = os.path.join(self.test_dir, "disconnect_cache")
    cached_path = model_sync.cache_sync(cache_dir, "model", self.remote_path)

    self.assertEqual(
        model_sync.cache_sync(
            cache_dir, "model", f"http://127.0.0.1:{port}/model.litertlm"
        ),
        cached_path,
    )

  def test_cache_sync_falls_back_when_body_is_cut_short(self):
    cut_short: list[int] = []

    class _TruncatingHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        start, end = self.headers["Range"].removeprefix("bytes=").split("-")
        length = int(end) - int(start) + 1
        cut_short.append(length)
        self.send_response(206)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        # Close the connection after half of the announced body.
        self.wfile.write(b"\0" * (length // 2))
        self.close_connection = True

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_TruncatingHandler)
    cache_dir = os.path.join(self.test_dir, "truncated_cache")
    cached_path = model_sync.cache_sync(cache_dir, "model", self.remote_path)

    self.assertEqual(
        model_sync.cache_sync(
            cache_dir, "model", f"http://127.0.0.1:{port}/model.litertlm"
        ),
        cached_path,
    )
    self.assertNotEmpty(cut_short)

  def test_http_range_fetcher_accepts_body_without_content_length(self):
    class _NoLengthHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        # The body ends when the connection closes (HTTP/1.0).
        self.send_response(206)
        self.end_headers()
        self.wfile.write(b"data")

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_NoLengthHandler)
    fetcher = cast(
        abc.Callable[[int, int, abc.Callable[[bytes], bool]], None],
        model_sync.create_http_range_fetcher(
            f"http://127.0.0.1:{port}/model.litertlm"
        ),
    )
    chunks: list[bytes] = []
    fetcher(0, 3, lambda chunk: chunks.append(chunk) is None)
    self.assertEqual(b"".join(chunks), b"data")

  def test_http_range_fetcher_ignores_range_in_headers(self):
    seen_headers: list[tuple[str | None, str | None]] = []

    class _RecordingHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        seen_headers.append(
            (self.headers["Range"], self.headers["Authorization"])
        )
        self.send_response(206)
        self.send_header("Content-Length", "4")
        self.end_headers()
        self.wfile.write(b"data")

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_RecordingHandler)
    fetcher = cast(
        abc.Callable[[int, int, abc.Callable[[bytes], bool]], None],
        model_sync.create_http_range_fetcher(
            f"http://127.0.0.1:{port}/model.litertlm",
            headers={"range": "bytes=0-0", "Authorization": "Bearer tok"},
        ),
    )
    fetcher(4, 7, lambda chunk: True)
    self.assertEqual(seen_headers, [("bytes=4-7", "Bearer tok")])

  def test_cache_sync_reports_malformed_url(self):
    cache_dir = os.path.join(self.test_dir, "bad_url_cache")
    model_sync.cache_sync(cache_dir, "model", self.remote_path)

    # urllib rejects the space with `http.client.InvalidURL`, which must not
    # fall back to the cached model.
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      model_sync.cache_sync(
          cache_dir, "model", "http://127.0.0.1:1/bad name.litertlm"
      )

  def test_timeout_is_passed_to_http_range_fetcher(self):
    release = threading.Event()

    class _SlowHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        release.wait(timeout=10)
        self.close_connection = True

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_SlowHandler)
    # Cleanups run in reverse order, so this releases the handler before the
    # server shuts down.
    self.addCleanup(release.set)
    with self.assertRaisesRegex(RuntimeError, "DEADLINE_EXCEEDED"):
      model_sync.plan_sync(
          self.local_path,
          f"http://127.0.0.1:{port}/model.litertlm",
          timeout=0.1,
      )

  def test_sync_plan_is_released_when_garbage_collected(self):
    lib = mock.Mock()
    plan = model_sync.SyncPlan(1234, lib)
    del plan
    lib.litert_lm_model_sync_plan_delete.assert_called_once_with(1234)

    # An object whose `__init__` didn't run has nothing to release.
    model_sync.SyncPlan.__new__(model_sync.SyncPlan).close()

  def test_http_range_ignored_and_status_mapping(self):
    seen_auth_headers: list[str | None] = []

    class _EdgeHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        seen_auth_headers.append(self.headers.get("Authorization"))
        if self.path == "/big200":
          self.send_response(200)
          self.send_header("Content-Length", "1048576")
          self.end_headers()
          self.wfile.write(b"x" * 1024)
        elif self.path == "/small200":
          self.send_response(200)
          self.send_header("Content-Length", "10")
          self.end_headers()
          self.wfile.write(b"ABCDEFGHIJ")
        elif self.path == "/r416":
          self.send_response(416)
          self.end_headers()
        elif self.path == "/r404":
          self.send_response(404)
          self.end_headers()
        else:
          self.send_response(400)
          self.end_headers()

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_EdgeHandler)

    chunks: list[bytes] = []
    upper_fetcher = cast(
        abc.Callable[[int, int, abc.Callable[[bytes], bool]], None],
        model_sync.create_range_fetcher(
            f"HTTP://127.0.0.1:{port}/small200",
            headers={"Authorization": "Bearer tok"},
        ),
    )
    upper_fetcher(0, 9, lambda b: chunks.append(b) is None)
    self.assertEqual(b"".join(chunks), b"ABCDEFGHIJ")
    self.assertEqual(seen_auth_headers[-1], "Bearer tok")

    with self.assertRaises(NotImplementedError):
      upper_fetcher(5, 9, lambda b: True)

    with self.assertRaisesRegex(RuntimeError, "UNIMPLEMENTED"):
      model_sync.plan_sync(self.local_path, f"http://127.0.0.1:{port}/big200")

    with self.assertRaisesRegex(RuntimeError, "OUT_OF_RANGE"):
      model_sync.plan_sync(self.local_path, f"http://127.0.0.1:{port}/r416")

    cache_dir = os.path.join(self.test_dir, "http_404_cache")
    model_sync.cache_sync(cache_dir, "cached-m", self.remote_path)
    with self.assertRaisesRegex(RuntimeError, "NOT_FOUND"):
      model_sync.cache_sync(
          cache_dir,
          "cached-m",
          f"http://127.0.0.1:{port}/r404",
          options=model_sync.SyncOptions(allow_offline_fallback=True),
      )

  def test_http_range_fetcher_sends_credentials_to_one_server_only(self):
    data_requests: list[tuple[str | None, ...]] = []

    class _RedirectingHandler(http.server.BaseHTTPRequestHandler):
      """Redirects /to/<port> to /data on that port, and serves /data."""

      def do_GET(self):  # pylint: disable=invalid-name
        if self.path.startswith("/to/"):
          target_port = self.path.removeprefix("/to/")
          self.send_response(302)
          self.send_header("Location", f"http://127.0.0.1:{target_port}/data")
          self.end_headers()
          return
        data_requests.append((
            self.headers.get("Host"),
            self.headers.get("Authorization"),
            self.headers.get("Cookie"),
            self.headers.get("Range"),
        ))
        self.send_response(206)
        self.send_header("Content-Length", "4")
        self.end_headers()
        self.wfile.write(b"data")

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    first_port = self._start_http_server(_RedirectingHandler)
    other_port = self._start_http_server(_RedirectingHandler)
    chunks: list[bytes] = []

    for redirect, target_port, expected_credentials in [
        ("same server", first_port, ("Bearer tok", "session=1")),
        ("other server", other_port, (None, None)),
    ]:
      with self.subTest(redirect=redirect):
        chunks.clear()
        fetcher = cast(
            abc.Callable[[int, int, abc.Callable[[bytes], bool]], None],
            model_sync.create_http_range_fetcher(
                f"http://127.0.0.1:{first_port}/to/{target_port}",
                headers={"Authorization": "Bearer tok", "Cookie": "session=1"},
            ),
        )
        fetcher(0, 3, lambda chunk: chunks.append(chunk) is None)
        self.assertEqual(b"".join(chunks), b"data")
        self.assertEqual(
            data_requests[-1],
            (f"127.0.0.1:{target_port}", *expected_credentials, "bytes=0-3"),
        )

  def test_redirect_handler_keeps_credentials_only_on_the_same_server(self):
    # pylint: disable-next=protected-access
    handler = _model_sync._RedirectHandler()
    for new_url, keeps_credentials in [
        ("https://example.com:443/other", True),
        ("https://EXAMPLE.com/other", True),
        ("https://cdn.example.com/model", False),
        ("http://example.com/model", False),
        ("https://example.com:8443/model", False),
    ]:
      with self.subTest(new_url=new_url):
        req = urllib.request.Request(
            "https://example.com/model",
            headers={
                "Authorization": "Bearer tok",
                "Cookie": "session=1",
                "Proxy-Authorization": "Basic cHJveHk=",
                "Range": "bytes=0-3",
            },
        )
        new_req = handler.redirect_request(
            req, None, 302, "Found", email.message.Message(), new_url
        )
        if new_req is None:
          self.fail("The redirect was not followed.")
        credentials = [
            new_req.get_header(name)
            for name in ("Authorization", "Cookie", "Proxy-authorization")
        ]
        self.assertEqual(
            credentials,
            ["Bearer tok", "session=1", "Basic cHJveHk="]
            if keeps_credentials
            else [None, None, None],
        )
        self.assertEqual(new_req.get_header("Range"), "bytes=0-3")

  def test_http_range_fetcher_refuses_redirect_to_ftp(self):
    class _FtpRedirectHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        self.send_response(302)
        self.send_header("Location", "ftp://127.0.0.1:1/model.litertlm")
        self.end_headers()

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    port = self._start_http_server(_FtpRedirectHandler)
    fetcher = cast(
        abc.Callable[[int, int, abc.Callable[[bytes], bool]], None],
        model_sync.create_http_range_fetcher(
            f"http://127.0.0.1:{port}/model.litertlm"
        ),
    )
    with self.assertRaisesRegex(ValueError, "Unsupported redirect URL scheme"):
      fetcher(0, 3, lambda chunk: True)


if __name__ == "__main__":
  absltest.main()
