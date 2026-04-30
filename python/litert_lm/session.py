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
import queue
from . import interfaces
from ._ffi import InputDataType
from ._ffi import LiteRtLmInputData
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
    inputs = (LiteRtLmInputData * num_inputs)()
    # Keep strings alive during call
    keep_alive = []
    for i, text in enumerate(contents):
      encoded_text = text.encode("utf-8")
      keep_alive.append(encoded_text)
      inputs[i].type = InputDataType.TEXT
      inputs[i].data = ctypes.cast(
          ctypes.c_char_p(encoded_text), ctypes.c_void_p
      )
      inputs[i].size = len(encoded_text)

    res = self._lib.litert_lm_session_run_prefill(self._ptr, inputs, num_inputs)
    if res != 0:
      raise RuntimeError("litert_lm_session_run_prefill failed")

  def run_decode(self) -> interfaces.Responses:
    resp_ptr = self._lib.litert_lm_session_run_decode(self._ptr)
    if not resp_ptr:
      raise RuntimeError("litert_lm_session_run_decode failed")
    return self._wrap_responses(resp_ptr)

  def run_decode_async(self) -> collections.abc.Iterator[interfaces.Responses]:
    q = queue.Queue()

    def callback(unused_data, chunk, is_final, error_msg):
      if error_msg:
        q.put(RuntimeError(error_msg.decode("utf-8")))
      else:
        q.put((chunk.decode("utf-8") if chunk else "", is_final))

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

    resp_ptr = self._lib.litert_lm_session_run_text_scoring(
        self._ptr, c_targets, num_targets, store_token_lengths
    )
    if not resp_ptr:
      raise RuntimeError("litert_lm_session_run_text_scoring failed")
    return self._wrap_responses(resp_ptr)

  def _wrap_responses(self, resp_ptr) -> interfaces.Responses:
    try:
      num = self._lib.litert_lm_responses_get_num_candidates(resp_ptr)
      texts = []
      scores = []
      lengths = []
      token_scores = []
      for i in range(num):
        t = self._lib.litert_lm_responses_get_response_text_at(resp_ptr, i)
        if t is not None:
          texts.append(t.decode("utf-8"))
        if self._lib.litert_lm_responses_has_score_at(resp_ptr, i):
          scores.append(self._lib.litert_lm_responses_get_score_at(resp_ptr, i))
        if self._lib.litert_lm_responses_has_token_length_at(resp_ptr, i):
          lengths.append(
              self._lib.litert_lm_responses_get_token_length_at(resp_ptr, i)
          )
        if self._lib.litert_lm_responses_has_token_scores_at(resp_ptr, i):
          num_scores = self._lib.litert_lm_responses_get_num_token_scores_at(
              resp_ptr, i
          )
          scores_ptr = self._lib.litert_lm_responses_get_token_scores_at(
              resp_ptr, i
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

  def cancel_process(self) -> None:
    if self._ptr:
      self._lib.litert_lm_session_cancel_process(self._ptr)
