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
"""Benchmark wrapper for LiteRT-LM."""

import ctypes
import dataclasses
import threading

from . import interfaces
from ._ffi import _get_lib
from ._ffi import call_checked
from ._ffi import create_checked
from ._ffi import InputDataType


@dataclasses.dataclass
class Benchmark(interfaces.AbstractBenchmark):
  """Benchmark wrapper for the LiteRT-LM C API."""

  _lib: ctypes.CDLL | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _engine_ptr: int | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  _in_context: bool = dataclasses.field(
      default=False, init=False, repr=False, compare=False
  )
  _lock: threading.RLock = dataclasses.field(
      default_factory=threading.RLock, init=False, repr=False, compare=False
  )

  def __enter__(self) -> "Benchmark":
    self._in_context = True
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    del exc_type, exc_val, exc_tb
    self._in_context = False
    self.close()

  def _get_or_create_engine(self) -> int:
    with self._lock:
      if self._engine_ptr is not None:
        return self._engine_ptr

      lib = _get_lib()
      self._lib = lib
      model_path = self.model_path
      backend_str = self.backend.get_name()

      settings = create_checked(
          lib,
          "litert_lm_engine_settings_create",
          model_path,
          backend_str,
          None,
          None,
      )

      try:
        if self.activation_data_type is not None:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_activation_data_type",
              settings,
              self.activation_data_type.value,
          )

        call_checked(
            lib, "litert_lm_engine_settings_enable_benchmark", settings
        )

        if (
            isinstance(self.backend, interfaces.CPU)
            and self.backend.thread_count is not None
        ):
          call_checked(
              lib,
              "litert_lm_engine_settings_set_num_threads",
              settings,
              self.backend.thread_count,
          )

        if self.max_num_tokens is not None:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_max_num_tokens",
              settings,
              self.max_num_tokens,
          )
        call_checked(
            lib,
            "litert_lm_engine_settings_set_num_prefill_tokens",
            settings,
            self.prefill_tokens,
        )
        call_checked(
            lib,
            "litert_lm_engine_settings_set_num_decode_tokens",
            settings,
            self.decode_tokens,
        )
        if self.cache_dir:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_cache_dir",
              settings,
              self.cache_dir,
          )
        if self.enable_speculative_decoding is not None:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_enable_speculative_decoding",
              settings,
              self.enable_speculative_decoding,
          )
        if isinstance(self.backend, interfaces.GPU):
          if self.backend.gpu_decode_steps_per_sync is not None:
            call_checked(
                lib,
                "litert_lm_engine_settings_set_gpu_decode_steps_per_sync",
                settings,
                self.backend.gpu_decode_steps_per_sync,
            )
          # When benchmarking, we should wait the initialization to complete to
          # make sure the timing of prefill is correct.
          call_checked(
              lib,
              "litert_lm_engine_settings_set_gpu_wait_for_weight_uploads",
              settings,
              True,
          )
        if self.use_ringbuffers_local_attention is not None:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_use_ringbuffers_local_attention",
              settings,
              self.use_ringbuffers_local_attention,
          )
        if self.enable_ynnpack is not None:
          call_checked(
              lib,
              "litert_lm_engine_settings_set_enable_ynnpack",
              settings,
              self.enable_ynnpack,
          )

        self._engine_ptr = create_checked(
            lib, "litert_lm_engine_create", settings
        )
      finally:
        lib.litert_lm_engine_settings_delete(settings)

      return self._engine_ptr

  def close(self) -> None:
    with self._lock:
      engine_ptr, self._engine_ptr = self._engine_ptr, None
      lib = self._lib
    if engine_ptr and lib:
      try:
        lib.litert_lm_engine_delete(engine_ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def run(self) -> interfaces.BenchmarkInfo:
    engine_ptr = self._get_or_create_engine()
    lib = self._lib
    assert lib is not None

    try:
      session_ptr = create_checked(
          lib, "litert_lm_engine_create_session", engine_ptr, None
      )
      try:
        prompt = self.prompt.encode("utf-8")
        input_ptr = create_checked(
            lib,
            "litert_lm_input_data_create",
            InputDataType.TEXT,
            prompt,
            len(prompt),
        )
        try:
          inputs = (ctypes.c_void_p * 1)(input_ptr)
          responses = create_checked(
              lib, "litert_lm_session_generate_content", session_ptr, inputs, 1
          )
          lib.litert_lm_responses_delete(responses)
          info_ptr = create_checked(
              lib, "litert_lm_session_get_benchmark_info", session_ptr
          )
          try:
            return interfaces.create_benchmark_info(lib, info_ptr)
          finally:
            lib.litert_lm_benchmark_info_delete(info_ptr)
        finally:
          lib.litert_lm_input_data_delete(input_ptr)
      finally:
        lib.litert_lm_session_delete(session_ptr)
    finally:
      if not self._in_context:
        self.close()
