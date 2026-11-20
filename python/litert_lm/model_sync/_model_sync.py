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
"""Implementation of the `litert_lm.model_sync` package."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import contextlib
import ctypes
import dataclasses
import http.client
import inspect
import os
import socket
import ssl
import types
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .. import _ffi


@dataclasses.dataclass
class SyncOptions:
  """Options configuring differential sync behavior.

  Each field matches a `litert_lm_model_sync_options_set_*` function of the
  LiteRT-LM C API (`c/model_sync.h`). Fields left as `None` keep the defaults
  of the sync engine.

  Attributes:
    full_download_threshold: Ratio of changed section bytes, in `[0.0, 1.0]`,
      above which the whole model is downloaded instead of only the changed
      sections. `0.0` disables this fallback. Engine default: `0.8`.
    clean: Whether to delete the local model before downloading when there
      isn't enough disk space for both copies. Engine default: `False`.
    check_disk_space: Whether to check for enough free disk space before
      downloading. Engine default: `True`.
    allow_offline_fallback: Whether `cache_sync` returns the cached model when
      the remote model is unavailable or times out. Engine default: `True`.
  """

  full_download_threshold: float | None = None
  clean: bool | None = None
  check_disk_space: bool | None = None
  allow_offline_fallback: bool | None = None


@contextlib.contextmanager
def _c_options(
    lib: ctypes.CDLL, options: SyncOptions | None
) -> Iterator[int | None]:
  """Allocates and populates an opaque C options handle for the block.

  Only the fields of `options` that are set are passed to the C API.
  """
  if options is None:
    yield None
    return

  handle = _ffi.create_checked(lib, "litert_lm_model_sync_options_create")
  try:
    threshold = options.full_download_threshold
    for suffix, val in (
        (
            "full_download_threshold",
            None if threshold is None else float(threshold),
        ),
        ("clean", options.clean),
        ("check_disk_space", options.check_disk_space),
        ("allow_offline_fallback", options.allow_offline_fallback),
    ):
      if val is not None:
        _ffi.call_checked(
            lib, f"litert_lm_model_sync_options_set_{suffix}", handle, val
        )
    yield handle
  finally:
    lib.litert_lm_model_sync_options_delete(handle)


RangeFetcher = (
    Callable[[int, int, Callable[[bytes], bool]], None]
    | Callable[[int, int], bytes]
)

_HTTP_STATUS_TO_CODE = types.MappingProxyType({
    400: _ffi.StatusCode.INVALID_ARGUMENT,
    401: _ffi.StatusCode.UNAUTHENTICATED,
    403: _ffi.StatusCode.PERMISSION_DENIED,
    404: _ffi.StatusCode.NOT_FOUND,
    408: _ffi.StatusCode.DEADLINE_EXCEEDED,
    416: _ffi.StatusCode.OUT_OF_RANGE,
    429: _ffi.StatusCode.RESOURCE_EXHAUSTED,
})


def _exception_to_status_code(exc: Exception) -> int:
  """Maps a Python fetch exception to an appropriate LiteRtLmStatusCode."""
  if isinstance(exc, NotImplementedError):
    return int(_ffi.StatusCode.UNIMPLEMENTED)
  if isinstance(exc, (ValueError, http.client.InvalidURL)):
    return int(_ffi.StatusCode.INVALID_ARGUMENT)
  if isinstance(exc, urllib.error.HTTPError):
    if exc.code in _HTTP_STATUS_TO_CODE:
      return int(_HTTP_STATUS_TO_CODE[exc.code])
    if 400 <= exc.code < 500:
      return int(_ffi.StatusCode.FAILED_PRECONDITION)
    if exc.code >= 500:
      return int(_ffi.StatusCode.UNAVAILABLE)
    return int(_ffi.StatusCode.UNKNOWN)
  if isinstance(exc, (TimeoutError, socket.timeout)):
    return int(_ffi.StatusCode.DEADLINE_EXCEEDED)
  if isinstance(exc, urllib.error.URLError):
    if isinstance(exc.reason, (TimeoutError, socket.timeout)):
      return int(_ffi.StatusCode.DEADLINE_EXCEEDED)
    return int(_ffi.StatusCode.UNAVAILABLE)
  if isinstance(exc, (ConnectionError, http.client.HTTPException)):
    # urllib raises these without a `URLError` wrapper when the connection is
    # reset, or closed before the status line or inside a chunked body.
    return int(_ffi.StatusCode.UNAVAILABLE)
  return int(_ffi.StatusCode.UNKNOWN)


def _wrap_range_fetcher(fetcher: RangeFetcher, lib: ctypes.CDLL) -> Any:
  """Wraps a Python callable into a C-compatible range fetcher callback."""
  num_params = len(inspect.signature(fetcher).parameters)

  def c_callback(offset: int, length: int, writer_ptr: Any, _: Any) -> int:
    if length == 0:
      return int(_ffi.StatusCode.OK)

    start_byte = offset
    end_byte = offset + length - 1

    def write_chunk(chunk: bytes) -> bool:
      if not chunk:
        return True
      buf = (ctypes.c_char * len(chunk)).from_buffer_copy(chunk)
      status = lib.litert_lm_model_sync_range_writer_write(
          writer_ptr, buf, len(chunk)
      )
      return status == _ffi.StatusCode.OK

    f: Any = fetcher
    try:
      if num_params == 2:
        data = f(start_byte, end_byte)
        if isinstance(data, (bytes, bytearray)):
          if not write_chunk(bytes(data)):
            lib.litert_lm_model_sync_range_writer_set_error(
                writer_ptr, "Failed to write fetched range chunk."
            )
            return int(_ffi.StatusCode.DATA_LOSS)
          return int(_ffi.StatusCode.OK)
        lib.litert_lm_model_sync_range_writer_set_error(
            writer_ptr, "Range fetcher did not return bytes."
        )
        return int(_ffi.StatusCode.UNKNOWN)

      res = f(start_byte, end_byte, write_chunk)
      if res is not None and not res:
        lib.litert_lm_model_sync_range_writer_set_error(
            writer_ptr, "Range fetcher returned falsy status."
        )
        return int(_ffi.StatusCode.UNKNOWN)
      return int(_ffi.StatusCode.OK)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      lib.litert_lm_model_sync_range_writer_set_error(writer_ptr, str(exc))
      return _exception_to_status_code(exc)

  return _ffi.RANGE_FETCHER_CALLBACK_TYPE(c_callback)


class SyncPlan:
  """Represents a computed sync plan between local and remote models."""

  def __init__(self, plan_handle: int, lib: ctypes.CDLL | None = None):
    """Takes ownership of a native sync plan.

    Args:
      plan_handle: Handle returned by
        `litert_lm_model_sync_plan_create_from_file`. `close()` frees it.
      lib: The loaded LiteRT-LM C library that created `plan_handle`. `None`
        uses the library that this package loads.
    """
    self._lib = lib if lib is not None else _ffi.get_lib()
    self._handle: int | None = plan_handle

  def __enter__(self) -> SyncPlan:
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()

  def close(self) -> None:
    """Releases the underlying C sync plan resources."""
    # `__init__` may have failed before setting `_handle`.
    if getattr(self, "_handle", None):
      self._lib.litert_lm_model_sync_plan_delete(self._handle)
      self._handle = None

  def _check_closed(self) -> None:
    if not self._handle:
      raise RuntimeError("SyncPlan is already closed.")

  @property
  def is_up_to_date(self) -> bool:
    """Returns True if the local model is already completely up to date."""
    self._check_closed()
    return _ffi.get_checked(
        self._lib,
        "litert_lm_model_sync_plan_is_up_to_date",
        ctypes.c_bool,
        self._handle,
    )

  @property
  def bytes_to_download(self) -> int:
    """Returns total bytes required to download."""
    self._check_closed()
    return _ffi.get_checked(
        self._lib,
        "litert_lm_model_sync_plan_get_bytes_to_download",
        ctypes.c_uint64,
        self._handle,
    )

  @property
  def bytes_to_reuse(self) -> int:
    """Returns total bytes to reuse from local model."""
    self._check_closed()
    return _ffi.get_checked(
        self._lib,
        "litert_lm_model_sync_plan_get_bytes_to_reuse",
        ctypes.c_uint64,
        self._handle,
    )

  @property
  def total_remote_size(self) -> int:
    """Returns total size of the remote model in bytes."""
    self._check_closed()
    return _ffi.get_checked(
        self._lib,
        "litert_lm_model_sync_plan_get_total_remote_size",
        ctypes.c_uint64,
        self._handle,
    )

  def format(self) -> str:
    """Returns a human-readable formatted summary table of the sync plan."""
    self._check_closed()
    formatted = _ffi.get_checked(
        self._lib,
        "litert_lm_model_sync_plan_format",
        ctypes.c_char_p,
        self._handle,
    )
    return formatted.decode("utf-8") if formatted is not None else ""

  def summary(self) -> str:
    """Alias for format()."""
    return self.format()

  def __str__(self) -> str:
    return self.format()


def create_file_range_fetcher(path: str) -> RangeFetcher:
  """Creates a range fetcher reading from a local file."""
  abs_path = os.path.abspath(path)

  def fetcher(
      start_byte: int, end_byte: int, write_fn: Callable[[bytes], bool]
  ) -> None:
    with open(abs_path, "rb") as f:
      f.seek(start_byte)
      remaining = end_byte - start_byte + 1
      while remaining > 0:
        chunk = f.read(min(64 * 1024, remaining))
        if not chunk:
          break
        if not write_fn(chunk):
          raise OSError("write_fn returned False during range fetch")
        remaining -= len(chunk)

  return fetcher


def _is_http_url(source: str) -> bool:
  """Returns True if source uses an HTTP or HTTPS URL scheme."""
  return source.lower().startswith(("http://", "https://"))


_DEFAULT_PORTS = types.MappingProxyType({"http": 80, "https": 443})
# Headers with credentials for one server, spelled the way `Request.add_header`
# stores them (`str.capitalize()`), which `Request.remove_header` needs.
_CREDENTIAL_HEADERS = ("Authorization", "Cookie", "Proxy-authorization")


def _origin(url: str) -> tuple[str, str | None, int | None]:
  """Returns the scheme, host and port that a request to `url` goes to."""
  parts = urllib.parse.urlsplit(url)
  scheme = parts.scheme.lower()
  return scheme, parts.hostname, parts.port or _DEFAULT_PORTS.get(scheme)


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
  """Follows HTTP(S) redirects, but sends credentials to one server only."""

  def redirect_request(
      self,
      req: urllib.request.Request,
      fp: Any,
      code: int,
      msg: str,
      headers: Any,
      newurl: str,
  ) -> urllib.request.Request | None:
    # urllib would also follow a redirect to ftp://.
    if not _is_http_url(newurl):
      raise ValueError(f"Unsupported redirect URL scheme: {newurl}")
    new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
    if new_req is not None and _origin(newurl) != _origin(req.full_url):
      for name in _CREDENTIAL_HEADERS:
        new_req.remove_header(name)
    return new_req


def create_http_range_fetcher(
    url: str,
    headers: Mapping[str, str] | None = None,
    chunk_size: int = 64 * 1024,
    ssl_context: ssl.SSLContext | None = None,
    timeout: float | None = 30.0,
) -> RangeFetcher:
  """Creates a range fetcher using HTTP Range requests.

  Args:
    url: HTTP or HTTPS URL of the remote model.
    headers: Extra HTTP request headers, such as `Authorization`. `None` sends
      no extra headers. A `Range` header is replaced by the requested range,
      and credential headers are dropped on redirects to a different server.
    chunk_size: Maximum number of bytes passed to `write_fn` at a time.
    ssl_context: SSL context for HTTPS requests. `None` uses the default
      context.
    timeout: Timeout in seconds of each blocking HTTP operation, or `None` for
      no timeout.

  Returns:
    A streaming range fetcher that takes `(start_byte, end_byte, write_fn)`,
    where `end_byte` is inclusive.
  """
  opener = urllib.request.build_opener(
      urllib.request.HTTPSHandler(context=ssl_context), _RedirectHandler
  )

  def fetcher(
      start_byte: int, end_byte: int, write_fn: Callable[[bytes], bool]
  ) -> None:
    if not _is_http_url(url):
      raise ValueError(f"Invalid HTTP URL: {url}")
    if start_byte < 0 or end_byte < start_byte:
      raise ValueError(f"Invalid byte range: {start_byte}-{end_byte}")

    expected_bytes = end_byte - start_byte + 1
    req = urllib.request.Request(url, headers=dict(headers or {}))
    # `Request` stores header names with `str.capitalize()`, so this replaces a
    # `Range` header from `headers` in any letter case.
    req.add_header("Range", f"bytes={start_byte}-{end_byte}")
    with opener.open(req, timeout=timeout) as resp:
      final_url = resp.geturl()
      if final_url and not _is_http_url(final_url):
        raise ValueError(f"Unsupported redirect URL scheme: {final_url}")
      if start_byte > 0 and getattr(resp, "status", None) == 200:
        raise NotImplementedError(
            f"Server ignored Range header (returned HTTP 200 OK) for {url}"
        )
      try:
        body_length = int(resp.getheader("Content-Length", ""))
      except ValueError:
        body_length = None
      if body_length is not None and body_length > expected_bytes:
        raise NotImplementedError(
            f"Server ignored Range header (Content-Length {body_length}"
            f" > {expected_bytes}) for {url}"
        )

      total_read = 0
      while True:
        chunk = resp.read(chunk_size)
        if not chunk:
          break
        total_read += len(chunk)
        if total_read > expected_bytes:
          raise NotImplementedError(
              f"Server ignored Range header (exceeded {expected_bytes} bytes)"
              f" for {url}"
          )
        if not write_fn(chunk):
          raise OSError("write_fn returned False during HTTP stream")
      # `HTTPResponse.read` returns b"" instead of raising when the server
      # closes the connection before the whole body arrives.
      if body_length is not None and total_read < body_length:
        raise ConnectionError(
            f"Connection closed after {total_read} of {body_length} bytes"
            f" from {url}"
        )

  return fetcher


def create_range_fetcher(
    source: str | RangeFetcher,
    headers: Mapping[str, str] | None = None,
    ssl_context: ssl.SSLContext | None = None,
    timeout: float | None = 30.0,
) -> RangeFetcher:
  """Creates or normalizes a range fetcher from URL, file path, or callable.

  Args:
    source: An HTTP(S) URL, a local file path, or a `RangeFetcher`, which is
      returned unchanged.
    headers: Extra HTTP request headers for an HTTP(S) URL. `None` sends no
      extra headers.
    ssl_context: SSL context for an HTTPS URL. `None` uses the default context.
    timeout: Timeout in seconds of each blocking HTTP operation for an HTTP(S)
      URL, or `None` for no timeout.

  Returns:
    A `RangeFetcher` that reads from `source`.

  Raises:
    TypeError: If `source` is neither a string nor callable.
  """
  if callable(source):
    return source
  if isinstance(source, str):
    if _is_http_url(source):
      return create_http_range_fetcher(
          source, headers=headers, ssl_context=ssl_context, timeout=timeout
      )
    return create_file_range_fetcher(source)
  raise TypeError(
      f"Expected str path/url or callable fetcher, got {type(source)}"
  )


def plan_sync(
    local: str | None,
    remote: str | RangeFetcher,
    options: SyncOptions | None = None,
    ssl_context: ssl.SSLContext | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = 30.0,
) -> SyncPlan:
  """Computes a synchronization plan between a local file and a remote model.

  `ssl_context`, `headers` and `timeout` are used only when `remote` is an
  HTTP(S) URL.

  Args:
    local: Path of the local model, or `None` if there is no local copy.
    remote: URL, file path or range fetcher of the remote model.
    options: Sync options. `None` keeps the defaults of the sync engine.
    ssl_context: SSL context for HTTPS requests.
    headers: Extra HTTP request headers, such as `Authorization`.
    timeout: Timeout in seconds of each blocking HTTP operation, or `None` for
      no timeout.

  Returns:
    The plan. Call `close()`, or use it in a `with` statement, to free it.
  """
  if local is not None and not isinstance(local, str):
    raise TypeError(f"local must be str or None, got {type(local)}")

  lib = _ffi.get_lib()
  remote_fetcher = create_range_fetcher(
      remote, headers=headers, ssl_context=ssl_context, timeout=timeout
  )
  c_remote_cb = _wrap_range_fetcher(remote_fetcher, lib)

  with _c_options(lib, options) as c_opts:
    plan_ptr = _ffi.create_checked(
        lib,
        "litert_lm_model_sync_plan_create_from_file",
        local if local is not None else "",
        c_remote_cb,
        None,
        c_opts,
    )

  return SyncPlan(plan_ptr, lib)


def update_model(
    local_path: str | None,
    output_path: str,
    remote: str | RangeFetcher,
    options: SyncOptions | None = None,
    ssl_context: ssl.SSLContext | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = 30.0,
) -> None:
  """Synchronizes a local model with a remote model, writing to output_path.

  `ssl_context`, `headers` and `timeout` are used only when `remote` is an
  HTTP(S) URL.

  Args:
    local_path: Path of the local model whose unchanged sections are reused, or
      `None` to download the whole model.
    output_path: Path to write the synchronized model to. If it is
      `local_path`, the model is updated in place atomically.
    remote: URL, file path or range fetcher of the remote model.
    options: Sync options. `None` keeps the defaults of the sync engine.
    ssl_context: SSL context for HTTPS requests.
    headers: Extra HTTP request headers, such as `Authorization`.
    timeout: Timeout in seconds of each blocking HTTP operation, or `None` for
      no timeout.
  """
  lib = _ffi.get_lib()
  remote_fetcher = create_range_fetcher(
      remote, headers=headers, ssl_context=ssl_context, timeout=timeout
  )
  c_remote_cb = _wrap_range_fetcher(remote_fetcher, lib)

  with _c_options(lib, options) as c_opts:
    _ffi.call_checked(
        lib,
        "litert_lm_model_sync_update",
        local_path if local_path is not None else "",
        output_path,
        c_remote_cb,
        None,
        c_opts,
    )


def cache_sync(
    cache_dir: str,
    model_id: str,
    remote: str | RangeFetcher,
    options: SyncOptions | None = None,
    ssl_context: ssl.SSLContext | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = 30.0,
) -> str:
  """Synchronizes a model inside a versioned cache directory.

  `ssl_context`, `headers` and `timeout` are used only when `remote` is an
  HTTP(S) URL.

  Args:
    cache_dir: Directory of the cached models. It is created if needed.
    model_id: ID of the model in `cache_dir`.
    remote: URL, file path or range fetcher of the remote model.
    options: Sync options. `None` keeps the defaults of the sync engine.
    ssl_context: SSL context for HTTPS requests.
    headers: Extra HTTP request headers, such as `Authorization`.
    timeout: Timeout in seconds of each blocking HTTP operation, or `None` for
      no timeout.

  Returns:
    The path of the synchronized model in `cache_dir`. If the remote model is
    unavailable or times out, the path of the cached copy, unless
    `options.allow_offline_fallback` is false.
  """
  lib = _ffi.get_lib()
  remote_fetcher = create_range_fetcher(
      remote, headers=headers, ssl_context=ssl_context, timeout=timeout
  )
  c_remote_cb = _wrap_range_fetcher(remote_fetcher, lib)

  with _c_options(lib, options) as c_opts:
    result_handle = _ffi.create_checked(
        lib,
        "litert_lm_model_sync_cache_sync",
        cache_dir,
        model_id,
        c_remote_cb,
        None,
        c_opts,
    )
  try:
    cached_path = _ffi.get_checked(
        lib,
        "litert_lm_model_sync_result_get_cached_path",
        ctypes.c_char_p,
        result_handle,
    )
    return cached_path.decode("utf-8") if cached_path else ""
  finally:
    lib.litert_lm_model_sync_result_delete(result_handle)
