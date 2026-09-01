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


class Benchmark(interfaces.AbstractBenchmark):
  """Benchmark wrapper for the LiteRT-LM C API."""

  def run(self) -> interfaces.BenchmarkInfo:
    lib = _get_lib()
    model_path = self.model_path
    backend_str = self.backend.get_name()

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

    if self.activation_data_type is not None:
      lib.litert_lm_engine_settings_set_activation_data_type(
          settings, self.activation_data_type.value
      )

    lib.litert_lm_engine_settings_enable_benchmark(settings)

    if (
        isinstance(self.backend, interfaces.CPU)
        and self.backend.thread_count is not None
    ):
      lib.litert_lm_engine_settings_set_num_threads(
          settings, self.backend.thread_count
      )

    if self.max_num_tokens is not None:
      lib.litert_lm_engine_settings_set_max_num_tokens(
          settings, self.max_num_tokens
      )
    lib.litert_lm_engine_settings_set_num_prefill_tokens(
        settings, self.prefill_tokens
    )
    lib.litert_lm_engine_settings_set_num_decode_tokens(
        settings, self.decode_tokens
    )
    if self.cache_dir:
      lib.litert_lm_engine_settings_set_cache_dir(settings, self.cache_dir)
    if self.enable_speculative_decoding is not None:
      lib.litert_lm_engine_settings_set_enable_speculative_decoding(
          settings, self.enable_speculative_decoding
      )
    if isinstance(self.backend, interfaces.GPU):
      if self.backend.gpu_decode_steps_per_sync is not None:
        lib.litert_lm_engine_settings_set_gpu_decode_steps_per_sync(
            settings, self.backend.gpu_decode_steps_per_sync
        )
      # When benchmarking, we should wait the initialization to complete to make
      # sure the timing of prefill is correct.
      # TODO(litertlm@): This should be set to True whenever benchmarking with
      # GPU backend.
      lib.litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(
          settings, True
      )
    if self.use_ringbuffers_local_attention is not None:
      lib.litert_lm_engine_settings_set_use_ringbuffers_local_attention(
          settings, self.use_ringbuffers_local_attention
      )
    if self.enable_ynnpack is not None:
      lib.litert_lm_engine_settings_set_enable_ynnpack(
          settings, self.enable_ynnpack
      )

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

    prompt = self.prompt.encode("utf-8")
    input_ptr = lib.litert_lm_input_data_create(
        InputDataType.TEXT, prompt, len(prompt)
    )
    if not input_ptr:
      lib.litert_lm_session_delete(session_ptr)
      lib.litert_lm_engine_delete(engine_ptr)
      raise RuntimeError("Failed to create input data")

    inputs = (ctypes.c_void_p * 1)(input_ptr)

    try:
      responses = lib.litert_lm_session_generate_content(session_ptr, inputs, 1)
      if responses:
        lib.litert_lm_responses_delete(responses)
    finally:
      lib.litert_lm_input_data_delete(input_ptr)

    info_ptr = lib.litert_lm_session_get_benchmark_info(session_ptr)
    if not info_ptr:
      lib.litert_lm_session_delete(session_ptr)
      lib.litert_lm_engine_delete(engine_ptr)
      raise RuntimeError("Failed to get benchmark info")

    info = interfaces.create_benchmark_info(lib, info_ptr)
    lib.litert_lm_benchmark_info_delete(info_ptr)
    lib.litert_lm_session_delete(session_ptr)
    lib.litert_lm_engine_delete(engine_ptr)
    return info
