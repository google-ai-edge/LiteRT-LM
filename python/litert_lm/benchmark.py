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

from . import interfaces
from ._ffi import _get_lib
from ._ffi import InputDataType
from ._ffi import LiteRtLmInputData


class Benchmark(interfaces.AbstractBenchmark):
  """Benchmark wrapper for the LiteRT-LM C API."""

  def run(self) -> interfaces.BenchmarkInfo:
    lib = _get_lib()
    model_path = self.model_path
    backend_str = self.backend.name.lower()

    settings = lib.litert_lm_engine_settings_create(
        model_path,
        backend_str,
        None,
        None,
    )
    if not settings:
      raise RuntimeError(
          "Failed to create engine settings for benchmark"
          f" (model_path={model_path}, backend={backend_str})"
      )

    lib.litert_lm_engine_settings_enable_benchmark(settings)
    lib.litert_lm_engine_settings_set_num_prefill_tokens(
        settings, self.prefill_tokens
    )
    lib.litert_lm_engine_settings_set_num_decode_tokens(
        settings, self.decode_tokens
    )
    if self.cache_dir:
      lib.litert_lm_engine_settings_set_cache_dir(settings, self.cache_dir)

    engine_ptr = lib.litert_lm_engine_create(settings)
    lib.litert_lm_engine_settings_delete(settings)

    if not engine_ptr:
      raise RuntimeError(
          f"Failed to create engine for benchmark (model_path={model_path},"
          f" backend={backend_str})"
      )

    session_ptr = lib.litert_lm_engine_create_session(engine_ptr, None)
    if not session_ptr:
      lib.litert_lm_engine_delete(engine_ptr)
      raise RuntimeError(
          f"Failed to create session for benchmark (model_path={model_path})"
      )

    dummy_prompt = b"benchmark"
    input_data = LiteRtLmInputData()
    input_data.type = InputDataType.TEXT
    input_data.data = ctypes.cast(
        ctypes.c_char_p(dummy_prompt), ctypes.c_void_p
    )
    input_data.size = len(dummy_prompt)

    responses = lib.litert_lm_session_generate_content(
        session_ptr, ctypes.byref(input_data), 1
    )
    if responses:
      lib.litert_lm_responses_delete(responses)

    info_ptr = lib.litert_lm_session_get_benchmark_info(session_ptr)
    if not info_ptr:
      lib.litert_lm_session_delete(session_ptr)
      lib.litert_lm_engine_delete(engine_ptr)
      raise RuntimeError("Failed to get benchmark info")

    info = interfaces.BenchmarkInfo(
        init_time_in_second=lib.litert_lm_benchmark_info_get_total_init_time_in_second(
            info_ptr
        ),
        time_to_first_token_in_second=lib.litert_lm_benchmark_info_get_time_to_first_token(
            info_ptr
        ),
        last_prefill_token_count=lib.litert_lm_benchmark_info_get_prefill_token_count_at(
            info_ptr, 0
        ),
        last_prefill_tokens_per_second=lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
            info_ptr, 0
        ),
        last_decode_token_count=lib.litert_lm_benchmark_info_get_decode_token_count_at(
            info_ptr, 0
        ),
        last_decode_tokens_per_second=lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
            info_ptr, 0
        ),
    )

    lib.litert_lm_benchmark_info_delete(info_ptr)
    lib.litert_lm_session_delete(session_ptr)
    lib.litert_lm_engine_delete(engine_ptr)
    return info
