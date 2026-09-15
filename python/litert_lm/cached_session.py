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
"""CachedSession wrapper for LiteRT-LM."""

from __future__ import annotations


class CachedSession:
  """CachedSession wrapper for the LiteRT-LM C API."""

  def __init__(self, lib, session_ptr, engine=None):
    self._lib = lib
    self._ptr = session_ptr
    self._engine = engine  # Keep engine alive

  def close(self):
    if hasattr(self, "_ptr") and self._ptr and self._lib:
      try:
        self._lib.litert_lm_cached_session_delete(self._ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._ptr = None

  def __del__(self):
    self.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  @property
  def last_matched_tokens(self) -> int:
    """Returns the number of prefix-cached tokens matched in the last prefill."""
    if not self._ptr:
      return 0
    return self._lib.litert_lm_cached_session_get_last_matched_tokens(self._ptr)

  @property
  def last_total_prompt_tokens(self) -> int:
    """Returns the total prompt tokens in the last prefill."""
    if not self._ptr:
      return 0
    return self._lib.litert_lm_cached_session_get_last_total_prompt_tokens(
        self._ptr
    )
