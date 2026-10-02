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
"""Session wrapper for LiteRT-LM."""

import collections.abc
import ctypes
import pathlib
import queue
import warnings
from . import interfaces
from ._ffi import call_checked
from ._ffi import check_status
from ._ffi import create_checked
from ._ffi import create_optional_checked
from ._ffi import get_checked
from ._ffi import InputDataType
from ._ffi import StatusCode
from ._ffi import STREAM_CALLBACK_TYPE


class Session(interfaces.AbstractSession):
  """Session wrapper for the LiteRT-LM C API."""

  def __init__(self, lib, session_ptr, engine=None):
    super().__init__()
    self._lib = lib
    self._ptr = session_ptr
    self._engine = engine  # Keep engine alive
    # Keep the active ctypes callback alive to prevent SIGSEGV if the C++ thread
    # calls it after the local variable is garbage collected during
    # cancellation.
    self._current_callback = None

  def close(self):
    if hasattr(self, "_ptr") and self._ptr and self._lib:
      try:
        self._lib.litert_lm_session_delete(self._ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._ptr = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def run_prefill(self, contents: list[str]) -> None:
    num_inputs = len(contents)
    inputs = (ctypes.c_void_p * num_inputs)()
    created_inputs = []
    try:
      for i, text in enumerate(contents):
        encoded_text = text.encode("utf-8")
        input_ptr = create_checked(
            self._lib,
            "litert_lm_input_data_create",
            InputDataType.TEXT,
            encoded_text,
            len(encoded_text),
        )
        created_inputs.append(input_ptr)
        inputs[i] = input_ptr

      res = self._lib.litert_lm_session_run_prefill(
          self._ptr, inputs, num_inputs
      )
      if res != 0:
        raise RuntimeError("litert_lm_session_run_prefill failed")
    finally:
      for input_ptr in created_inputs:
        self._lib.litert_lm_input_data_delete(input_ptr)

  def run_decode(self) -> interfaces.Responses:
    resp_ptr = create_checked(
        self._lib, "litert_lm_session_run_decode", self._ptr
    )
    return self._wrap_responses(resp_ptr)

  def run_decode_async(self) -> collections.abc.Iterator[interfaces.Responses]:
    q = queue.Queue()

    def callback(unused_data, chunk_ptr):
      # Runs on a C++ thread: report failures through the queue, not by raising.
      try:
        error_msg = get_checked(
            self._lib,
            "litert_lm_stream_chunk_get_error",
            ctypes.c_char_p,
            chunk_ptr,
        )
        if error_msg:
          q.put(RuntimeError(error_msg.decode("utf-8")))
        else:
          chunk = get_checked(
              self._lib,
              "litert_lm_stream_chunk_get_text",
              ctypes.c_char_p,
              chunk_ptr,
          )
          is_final = get_checked(
              self._lib,
              "litert_lm_stream_chunk_is_final",
              ctypes.c_bool,
              chunk_ptr,
          )
          q.put((chunk.decode("utf-8") if chunk else "", is_final))
      except Exception as e:  # pylint: disable=broad-exception-caught
        q.put(e)

    c_callback = STREAM_CALLBACK_TYPE(callback)
    self._current_callback = c_callback
    res = self._lib.litert_lm_session_run_decode_async(
        self._ptr, c_callback, None
    )
    if res != 0:
      raise RuntimeError("litert_lm_session_run_decode_async failed")

    while True:
      item = q.get()
      if isinstance(item, Exception):
        err_msg = str(item)
        if "CANCELLED" in err_msg or "Max number of tokens reached" in err_msg:
          break
        raise item
      chunk, is_final = item
      if chunk:
        yield interfaces.Responses(texts=[chunk], scores=[], token_lengths=[])
      if is_final:
        break

  def run_text_scoring(
      self, target_text: list[str], store_token_lengths: bool = False
  ) -> interfaces.Responses:
    num_targets = len(target_text)
    c_targets = (ctypes.c_char_p * num_targets)()
    for i, t in enumerate(target_text):
      c_targets[i] = t.encode("utf-8")

    resp_ptr = create_checked(
        self._lib,
        "litert_lm_session_run_text_scoring",
        self._ptr,
        c_targets,
        num_targets,
        store_token_lengths,
    )
    return self._wrap_responses(resp_ptr)

  def _get_response_text_at(self, resp_ptr, index: int) -> str | None:
    """Returns the response text at `index`, or None if there is none."""
    text = ctypes.c_char_p()
    status = self._lib.litert_lm_responses_get_response_text_at(
        resp_ptr, index, ctypes.byref(text)
    )
    if status == StatusCode.NOT_FOUND:
      return None
    check_status(self._lib, "litert_lm_responses_get_response_text_at", status)
    return text.value.decode("utf-8") if text.value is not None else None

  def _wrap_responses(self, resp_ptr) -> interfaces.Responses:
    try:
      lib = self._lib
      num = get_checked(
          lib, "litert_lm_responses_get_num_candidates", ctypes.c_int, resp_ptr
      )
      texts = []
      scores = []
      lengths = []
      token_scores = []
      for i in range(num):
        t = self._get_response_text_at(resp_ptr, i)
        if t is not None:
          texts.append(t)
        if get_checked(
            lib, "litert_lm_responses_has_score_at", ctypes.c_bool, resp_ptr, i
        ):
          scores.append(
              get_checked(
                  lib,
                  "litert_lm_responses_get_score_at",
                  ctypes.c_float,
                  resp_ptr,
                  i,
              )
          )
        if get_checked(
            lib,
            "litert_lm_responses_has_token_length_at",
            ctypes.c_bool,
            resp_ptr,
            i,
        ):
          lengths.append(
              get_checked(
                  lib,
                  "litert_lm_responses_get_token_length_at",
                  ctypes.c_int,
                  resp_ptr,
                  i,
              )
          )
        if get_checked(
            lib,
            "litert_lm_responses_has_token_scores_at",
            ctypes.c_bool,
            resp_ptr,
            i,
        ):
          num_scores = get_checked(
              lib,
              "litert_lm_responses_get_num_token_scores_at",
              ctypes.c_int,
              resp_ptr,
              i,
          )
          scores_ptr = get_checked(
              lib,
              "litert_lm_responses_get_token_scores_at",
              ctypes.POINTER(ctypes.c_float),
              resp_ptr,
              i,
          )
          if scores_ptr:
            token_scores.append([scores_ptr[j] for j in range(num_scores)])
      return interfaces.Responses(
          texts=texts,
          scores=scores,
          token_lengths=lengths,
          token_scores=token_scores,
      )
    finally:
      self._lib.litert_lm_responses_delete(resp_ptr)

  def get_benchmark_info(self) -> interfaces.BenchmarkInfo:
    """See base class."""
    if not self._ptr:
      raise RuntimeError("Session is closed.")
    info_ptr = create_checked(
        self._lib, "litert_lm_session_get_benchmark_info", self._ptr
    )
    try:
      return interfaces.create_benchmark_info(self._lib, info_ptr)
    finally:
      self._lib.litert_lm_benchmark_info_delete(info_ptr)

  def cancel_process(self) -> None:
    if self._ptr:
      call_checked(self._lib, "litert_lm_session_cancel_process", self._ptr)

  def get_debug_artifacts(self) -> interfaces.DebugArtifacts | None:
    """See base class."""
    if not get_checked(
        self._lib, "litert_lm_experimental_is_debugger_enabled", ctypes.c_bool
    ):
      warnings.warn(
          "LiteRT-LM Debugger is disabled in this runtime build. "
          "To enable artifact tracing, re-compile using "
          "'--define LITERT_LM_DEBUGGER_ENABLED=1'.",
          RuntimeWarning,
          stacklevel=2,
      )
      return None

    if not self._engine or not self._engine.cache_dir:
      return None

    if not self._ptr:
      return None

    debug_info_ptr = create_optional_checked(
        self._lib, "litert_lm_experimental_session_get_debug_info", self._ptr
    )
    if not debug_info_ptr:
      return None

    try:
      capture_dir_bytes = get_checked(
          self._lib,
          "litert_lm_experimental_session_debug_info_get_capture_dir",
          ctypes.c_char_p,
          debug_info_ptr,
      )
      if not capture_dir_bytes:
        return None
      capture_dir_str = capture_dir_bytes.decode("utf-8")
    finally:
      self._lib.litert_lm_experimental_session_debug_info_delete(debug_info_ptr)

    if not capture_dir_str:
      return None

    session_dir = pathlib.Path(self._engine.cache_dir) / capture_dir_str
    if not session_dir.exists() or not session_dir.is_dir():
      return None

    tensor_paths = sorted(session_dir.glob("*.safetensors"))
    trace_log = session_dir / "generated_tokens.jsonl"
    trace_log_path = trace_log if trace_log.exists() else None

    return interfaces.DebugArtifacts(
        tensor_paths=tensor_paths,
        trace_log_path=trace_log_path,
    )
