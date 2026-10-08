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
"""Low-level C library loading and FFI signatures."""

from __future__ import annotations

import ctypes
import enum
from importlib import resources
import os
from typing import Any


class c_string_p(ctypes.c_char_p):  # pylint: disable=invalid-name
  """Custom ctypes type that automatically encodes Python strings to UTF-8 bytes."""

  @classmethod
  def from_param(cls, obj):
    if obj is None:
      return None
    if isinstance(obj, str):
      return obj.encode("utf-8")
    return obj


class InputDataType(enum.IntEnum):
  TEXT = 0
  IMAGE = 1
  IMAGE_END = 2
  AUDIO = 3
  AUDIO_END = 4


class TokenUnionType(enum.IntEnum):
  STRING = 0
  IDS = 1


class SamplerType(enum.IntEnum):
  UNSPECIFIED = 0
  TOP_K = 1
  TOP_P = 2
  GREEDY = 3


class LiteRtLmModality(enum.IntEnum):
  TEXT = 0
  VISION = 1
  AUDIO = 2
  VIDEO = 3


class LiteRtLmNpuBrand(enum.IntEnum):
  UNKNOWN = 0
  QUALCOMM = 1
  GOOGLE_TENSOR = 2
  MEDIATEK = 3
  INTEL = 4
  SAMSUNG = 5


class LiteRtLmBackendType(enum.IntEnum):
  UNSPECIFIED = 0
  CPU = 1
  GPU = 2
  NPU = 3


class LiteRtLmModelType(enum.IntEnum):
  UNKNOWN = 0
  LLM = 1
  EMBEDDING = 2


# C-compatible callback type that matches 'LiteRtLmStreamCallback' in engine.h.
STREAM_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class LogSeverity(enum.IntEnum):
  VERBOSE = 0
  DEBUG = 1
  INFO = 2
  WARNING = 3
  ERROR = 4
  FATAL = 5
  SILENT = 1000


class ActivationDataType(enum.IntEnum):
  """Activation data type for inference."""

  FLOAT32 = 0
  FLOAT16 = 1
  INT16 = 2
  INT8 = 3

  @classmethod
  def from_str(cls, val: str) -> ActivationDataType | None:
    mapping = {
        "fp32": cls.FLOAT32,
        "fp16": cls.FLOAT16,
        "int16": cls.INT16,
        "int8": cls.INT8,
    }
    return mapping.get(val.lower())


class LiteRtLmConstraintType(enum.IntEnum):
  NONE = 0
  REGEX = 1
  JSON_SCHEMA = 2


class LiteRtLmConstraintProviderType(enum.IntEnum):
  NONE = 0
  LL_GUIDANCE = 1


class StatusCode(enum.IntEnum):
  """Mirrors `LiteRtLmStatusCode` in c/error_reporter.h."""

  OK = 0
  CANCELLED = 1
  UNKNOWN = 2
  INVALID_ARGUMENT = 3
  DEADLINE_EXCEEDED = 4
  NOT_FOUND = 5
  ALREADY_EXISTS = 6
  PERMISSION_DENIED = 7
  RESOURCE_EXHAUSTED = 8
  FAILED_PRECONDITION = 9
  ABORTED = 10
  OUT_OF_RANGE = 11
  UNIMPLEMENTED = 12
  INTERNAL = 13
  UNAVAILABLE = 14
  DATA_LOSS = 15
  UNAUTHENTICATED = 16


_LIB: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
  """Loads and returns the LiteRT-LM C shared library."""
  global _LIB
  if _LIB is not None:
    return _LIB

  import sys
  if sys.platform == "win32":
    lib_name = "litert-lm.dll"
  else:
    extension = "dylib" if sys.platform == "darwin" else "so"
    lib_name = f"liblitert-lm.{extension}"

  # 1. Try loading using importlib.resources (handles .par and package files)
  try:
    ref = resources.files(__package__) / lib_name
    with resources.as_file(ref) as path:
      if path.exists():
        _LIB = ctypes.CDLL(str(path))
  except (ImportError, FileNotFoundError, TypeError):
    pass

  # 2. Fallback to direct path in runfiles for local development/Bazel
  if _LIB is None:
    path = os.path.join(os.path.dirname(__file__), "../../c", lib_name)
    if os.path.exists(path):
      _LIB = ctypes.CDLL(path)

  if _LIB is None:
    raise RuntimeError(
        f"Could not find {lib_name}. Ensure it is built and included in the"
        " package or runfiles."
    )

  _setup_lib_signatures(_LIB)
  return _LIB


def _setup_lib_signatures(lib):
  """Configures the argument and return types for C API functions."""
  # Error reporting
  lib.litert_lm_get_last_error_message.restype = ctypes.c_char_p
  lib.litert_lm_get_last_error_message.argtypes = []

  # Log level
  lib.litert_lm_set_min_log_level.restype = ctypes.c_int
  lib.litert_lm_set_min_log_level.argtypes = [ctypes.c_int]

  # Input Data
  lib.litert_lm_input_data_create.restype = ctypes.c_int
  lib.litert_lm_input_data_create.argtypes = [
      ctypes.c_int,
      ctypes.c_void_p,
      ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_input_data_delete.restype = None
  lib.litert_lm_input_data_delete.argtypes = [ctypes.c_void_p]

  # Engine Settings
  lib.litert_lm_engine_settings_create.restype = ctypes.c_int
  lib.litert_lm_engine_settings_create.argtypes = [
      c_string_p,
      c_string_p,
      c_string_p,
      c_string_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_engine_settings_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_settings_set_max_num_tokens.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_max_num_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_max_num_images.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_max_num_images.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_max_vision_tokens_per_image.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_max_vision_tokens_per_image.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_num_threads.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_num_threads.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_audio_num_threads.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_audio_num_threads.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_cache_dir.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_cache_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_engine_settings_set_litert_dispatch_lib_dir.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_litert_dispatch_lib_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_engine_settings_set_enable_speculative_decoding.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_enable_speculative_decoding.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_engine_settings_set_gpu_decode_steps_per_sync.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_gpu_decode_steps_per_sync.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_gpu_wait_for_weight_uploads.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_gpu_wait_for_weight_uploads.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_engine_settings_set_use_ringbuffers_local_attention.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_use_ringbuffers_local_attention.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_engine_settings_set_activation_data_type.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_activation_data_type.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_enable_ynnpack.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_enable_ynnpack.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]

  lib.litert_lm_engine_settings_set_lora_rank.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_lora_rank.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_supported_lora_ranks.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_supported_lora_ranks.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_size_t,
  ]
  lib.litert_lm_engine_settings_set_audio_lora_rank.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_audio_lora_rank.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_supported_audio_lora_ranks.restype = (
      ctypes.c_int
  )
  lib.litert_lm_engine_settings_set_supported_audio_lora_ranks.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_size_t,
  ]
  lib.litert_lm_engine_settings_enable_benchmark.restype = ctypes.c_int
  lib.litert_lm_engine_settings_enable_benchmark.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_settings_set_num_prefill_tokens.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_num_prefill_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_num_decode_tokens.restype = ctypes.c_int
  lib.litert_lm_engine_settings_set_num_decode_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Engine
  lib.litert_lm_engine_create.restype = ctypes.c_int
  lib.litert_lm_engine_create.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p]

  # Sampler Params
  lib.litert_lm_sampler_params_create.restype = ctypes.c_int
  lib.litert_lm_sampler_params_create.argtypes = [
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_sampler_params_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_sampler_params_set_top_k.restype = ctypes.c_int
  lib.litert_lm_sampler_params_set_top_k.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_sampler_params_set_top_p.restype = ctypes.c_int
  lib.litert_lm_sampler_params_set_top_p.argtypes = [
      ctypes.c_void_p,
      ctypes.c_float,
  ]
  lib.litert_lm_sampler_params_set_temperature.restype = ctypes.c_int
  lib.litert_lm_sampler_params_set_temperature.argtypes = [
      ctypes.c_void_p,
      ctypes.c_float,
  ]
  lib.litert_lm_sampler_params_set_seed.restype = ctypes.c_int
  lib.litert_lm_sampler_params_set_seed.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Session Config
  lib.litert_lm_session_config_create.restype = ctypes.c_int
  lib.litert_lm_session_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_session_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_config_set_max_output_tokens.restype = ctypes.c_int
  lib.litert_lm_session_config_set_max_output_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_session_config_set_apply_prompt_template.restype = ctypes.c_int
  lib.litert_lm_session_config_set_apply_prompt_template.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_session_config_set_sampler_params.restype = ctypes.c_int
  lib.litert_lm_session_config_set_sampler_params.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_session_config_set_lora_path.restype = ctypes.c_int
  lib.litert_lm_session_config_set_lora_path.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_session_config_set_audio_lora_path.restype = ctypes.c_int
  lib.litert_lm_session_config_set_audio_lora_path.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_session_config_set_enable_speculative_decoding.restype = (
      ctypes.c_int
  )
  lib.litert_lm_session_config_set_enable_speculative_decoding.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]

  # Session
  lib.litert_lm_engine_create_session.restype = ctypes.c_int
  lib.litert_lm_engine_create_session.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_session_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_cancel_process.restype = ctypes.c_int
  lib.litert_lm_session_cancel_process.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_run_prefill.restype = ctypes.c_int
  lib.litert_lm_session_run_prefill.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_size_t,
  ]
  lib.litert_lm_session_run_decode.restype = ctypes.c_int
  lib.litert_lm_session_run_decode.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_session_run_decode_async.restype = ctypes.c_int
  lib.litert_lm_session_run_decode_async.argtypes = [
      ctypes.c_void_p,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]
  lib.litert_lm_session_run_text_scoring.restype = ctypes.c_int
  lib.litert_lm_session_run_text_scoring.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
      ctypes.c_size_t,
      ctypes.c_bool,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_session_generate_content.restype = ctypes.c_int
  lib.litert_lm_session_generate_content.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_session_generate_content_stream.restype = ctypes.c_int
  lib.litert_lm_session_generate_content_stream.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_size_t,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]

  # Conversation Config
  lib.litert_lm_conversation_config_create.restype = ctypes.c_int
  lib.litert_lm_conversation_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_config_set_session_config.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_session_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_config_set_system_message.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_system_message.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_tools.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_tools.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_messages.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_messages.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_extra_context.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_extra_context.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_prompt_template.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_prompt_template.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_enable_constrained_decoding.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_config_set_enable_constrained_decoding.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_conversation_config_set_constraint_provider.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_config_set_constraint_provider.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_conversation_config_set_filter_channel_content_from_kv_cache.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_config_set_filter_channel_content_from_kv_cache.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_conversation_config_set_thinking_config.restype = ctypes.c_int
  lib.litert_lm_conversation_config_set_thinking_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]

  # Repetition Penalty Config
  lib.litert_lm_repetition_penalty_config_create.restype = ctypes.c_int
  lib.litert_lm_repetition_penalty_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_repetition_penalty_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_repetition_penalty_config_set_repetition_penalty.restype = (
      ctypes.c_int
  )
  lib.litert_lm_repetition_penalty_config_set_repetition_penalty.argtypes = [
      ctypes.c_void_p,
      ctypes.c_float,
  ]
  lib.litert_lm_repetition_penalty_config_set_presence_penalty.restype = (
      ctypes.c_int
  )
  lib.litert_lm_repetition_penalty_config_set_presence_penalty.argtypes = [
      ctypes.c_void_p,
      ctypes.c_float,
  ]
  lib.litert_lm_repetition_penalty_config_set_frequency_penalty.restype = (
      ctypes.c_int
  )
  lib.litert_lm_repetition_penalty_config_set_frequency_penalty.argtypes = [
      ctypes.c_void_p,
      ctypes.c_float,
  ]
  lib.litert_lm_repetition_penalty_config_set_window_size.restype = ctypes.c_int
  lib.litert_lm_repetition_penalty_config_set_window_size.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # No Repeat Ngram Config
  lib.litert_lm_no_repeat_ngram_config_create.restype = ctypes.c_int
  lib.litert_lm_no_repeat_ngram_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_no_repeat_ngram_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size.restype = (
      ctypes.c_int
  )
  lib.litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_no_repeat_ngram_config_set_window_size.restype = ctypes.c_int
  lib.litert_lm_no_repeat_ngram_config_set_window_size.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Suppress Tokens Config
  lib.litert_lm_suppress_tokens_config_create.restype = ctypes.c_int
  lib.litert_lm_suppress_tokens_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_suppress_tokens_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_suppress_tokens_config_set_suppress_tokens.restype = (
      ctypes.c_int
  )
  lib.litert_lm_suppress_tokens_config_set_suppress_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_size_t,
  ]

  # Thinking Config
  lib.litert_lm_thinking_config_create.restype = ctypes.c_int
  lib.litert_lm_thinking_config_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_thinking_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_thinking_config_set_enable_thinking.restype = ctypes.c_int
  lib.litert_lm_thinking_config_set_enable_thinking.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_thinking_config_set_thinking_token_budget.restype = ctypes.c_int
  lib.litert_lm_thinking_config_set_thinking_token_budget.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Conversation Optional Args
  lib.litert_lm_conversation_optional_args_create.restype = ctypes.c_int
  lib.litert_lm_conversation_optional_args_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_optional_args_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_optional_args_set_repetition_penalty_config.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_repetition_penalty_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_optional_args_set_no_repeat_ngram_config.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_no_repeat_ngram_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_optional_args_set_suppress_tokens_config.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_suppress_tokens_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_optional_args_set_visual_token_budget.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_visual_token_budget.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_conversation_optional_args_set_max_output_tokens.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_max_output_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_conversation_optional_args_set_thinking_config.restype = (
      ctypes.c_int
  )
  lib.litert_lm_conversation_optional_args_set_thinking_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]

  # Conversation
  lib.litert_lm_conversation_create.restype = ctypes.c_int
  lib.litert_lm_conversation_create.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_send_message.restype = ctypes.c_int
  lib.litert_lm_conversation_send_message.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      c_string_p,
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_send_message_stream.restype = ctypes.c_int
  lib.litert_lm_conversation_send_message_stream.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      c_string_p,
      ctypes.c_void_p,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_cancel_process.restype = ctypes.c_int
  lib.litert_lm_conversation_cancel_process.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_render_message_to_string.restype = ctypes.c_int
  lib.litert_lm_conversation_render_message_to_string.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.litert_lm_conversation_get_token_count.restype = ctypes.c_int
  lib.litert_lm_conversation_get_token_count.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]

  # Conversation Optional Args
  lib.litert_lm_conversation_optional_args_create.restype = ctypes.c_int
  lib.litert_lm_conversation_optional_args_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_optional_args_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_optional_args_set_constraint.restype = ctypes.c_int
  lib.litert_lm_conversation_optional_args_set_constraint.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      c_string_p,
  ]

  # interfaces.Responses
  lib.litert_lm_responses_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_responses_get_num_candidates.restype = ctypes.c_int
  lib.litert_lm_responses_get_num_candidates.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_responses_get_response_text_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_response_text_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.litert_lm_responses_has_score_at.restype = ctypes.c_int
  lib.litert_lm_responses_has_score_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_responses_get_score_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_score_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_float),
  ]
  lib.litert_lm_responses_has_token_length_at.restype = ctypes.c_int
  lib.litert_lm_responses_has_token_length_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_responses_get_token_length_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_token_length_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_responses_has_token_scores_at.restype = ctypes.c_int
  lib.litert_lm_responses_has_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_responses_get_num_token_scores_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_num_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_responses_get_token_scores_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
  ]

  # JSON Response
  lib.litert_lm_json_response_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_json_response_get_string.restype = ctypes.c_int
  lib.litert_lm_json_response_get_string.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]

  # Benchmark Info
  lib.litert_lm_session_get_benchmark_info.restype = ctypes.c_int
  lib.litert_lm_session_get_benchmark_info.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_conversation_get_benchmark_info.restype = ctypes.c_int
  lib.litert_lm_conversation_get_benchmark_info.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_benchmark_info_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_benchmark_info_get_time_to_first_token.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_time_to_first_token.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_double),
  ]
  lib.litert_lm_benchmark_info_get_total_init_time_in_second.restype = (
      ctypes.c_int
  )
  lib.litert_lm_benchmark_info_get_total_init_time_in_second.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_double),
  ]
  lib.litert_lm_benchmark_info_get_num_prefill_turns.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_num_prefill_turns.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_benchmark_info_get_num_decode_turns.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_num_decode_turns.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_benchmark_info_get_prefill_token_count_at.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_prefill_token_count_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_benchmark_info_get_decode_token_count_at.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_decode_token_count_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at.restype = (
      ctypes.c_int
  )
  lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_double),
  ]
  lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at.restype = (
      ctypes.c_int
  )
  lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_double),
  ]

  # Tokenizer
  lib.litert_lm_engine_tokenize.restype = ctypes.c_int
  lib.litert_lm_engine_tokenize.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_tokenize_result_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_tokenize_result_get_tokens.restype = ctypes.c_int
  lib.litert_lm_tokenize_result_get_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
  ]
  lib.litert_lm_tokenize_result_get_num_tokens.restype = ctypes.c_int
  lib.litert_lm_tokenize_result_get_num_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_size_t),
  ]

  lib.litert_lm_engine_detokenize.restype = ctypes.c_int
  lib.litert_lm_engine_detokenize.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_detokenize_result_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_detokenize_result_get_string.restype = ctypes.c_int
  lib.litert_lm_detokenize_result_get_string.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]

  # Token Union / Metadata
  lib.litert_lm_token_union_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_union_get_type.restype = ctypes.c_int
  lib.litert_lm_token_union_get_type.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_token_union_get_string.restype = ctypes.c_int
  lib.litert_lm_token_union_get_string.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.litert_lm_token_union_get_ids.restype = ctypes.c_int
  lib.litert_lm_token_union_get_ids.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
      ctypes.POINTER(ctypes.c_size_t),
  ]

  lib.litert_lm_token_unions_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_unions_get_num_tokens.restype = ctypes.c_int
  lib.litert_lm_token_unions_get_num_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_size_t),
  ]
  lib.litert_lm_token_unions_get_token_at.restype = ctypes.c_int
  lib.litert_lm_token_unions_get_token_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_void_p),
  ]

  lib.litert_lm_engine_get_start_token.restype = ctypes.c_int
  lib.litert_lm_engine_get_start_token.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_engine_get_stop_tokens.restype = ctypes.c_int
  lib.litert_lm_engine_get_stop_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]

  # Stream Chunk
  lib.litert_lm_stream_chunk_get_text.restype = ctypes.c_int
  lib.litert_lm_stream_chunk_get_text.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.litert_lm_stream_chunk_is_final.restype = ctypes.c_int
  lib.litert_lm_stream_chunk_is_final.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_stream_chunk_get_error.restype = ctypes.c_int
  lib.litert_lm_stream_chunk_get_error.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]

  # Embedding Engine Settings
  lib.litert_lm_embedding_engine_settings_create.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_settings_create.argtypes = [
      c_string_p,
      c_string_p,
      c_string_p,
      c_string_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_embedding_engine_settings_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_embedding_engine_settings_set_num_threads.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_settings_set_num_threads.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_engine_settings_set_audio_num_threads.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_audio_num_threads.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_engine_settings_set_cache_dir.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_settings_set_cache_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_embedding_engine_settings_set_min_input_length.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_min_input_length.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_engine_settings_set_max_input_length.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_max_input_length.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_engine_settings_set_vision_tokens_per_image.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_engine_settings_set_vision_tokens_per_image.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Embedding Options
  lib.litert_lm_embedding_options_create.restype = ctypes.c_int
  lib.litert_lm_embedding_options_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p)
  ]
  lib.litert_lm_embedding_options_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_embedding_options_set_normalize.restype = ctypes.c_int
  lib.litert_lm_embedding_options_set_normalize.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_embedding_options_get_normalize.restype = ctypes.c_int
  lib.litert_lm_embedding_options_get_normalize.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_embedding_options_set_insert_special_tokens.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_set_insert_special_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_embedding_options_get_insert_special_tokens.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_get_insert_special_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_embedding_options_set_input_overflow_strategy.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_set_input_overflow_strategy.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_options_get_input_overflow_strategy.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_get_input_overflow_strategy.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_embedding_options_set_output_size.restype = ctypes.c_int
  lib.litert_lm_embedding_options_set_output_size.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_options_get_output_size.restype = ctypes.c_int
  lib.litert_lm_embedding_options_get_output_size.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_embedding_options_set_vision_tokens_per_image.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_set_vision_tokens_per_image.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_embedding_options_get_vision_tokens_per_image.restype = (
      ctypes.c_int
  )
  lib.litert_lm_embedding_options_get_vision_tokens_per_image.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]

  # Embedding Response
  lib.litert_lm_embedding_response_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_embedding_response_get_size.restype = ctypes.c_int
  lib.litert_lm_embedding_response_get_size.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_size_t),
  ]
  lib.litert_lm_embedding_response_get_values.restype = ctypes.c_int
  lib.litert_lm_embedding_response_get_values.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
  ]

  # Embedding Responses
  lib.litert_lm_embedding_responses_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_embedding_responses_get_size.restype = ctypes.c_int
  lib.litert_lm_embedding_responses_get_size.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_size_t),
  ]
  lib.litert_lm_embedding_responses_get_at.restype = ctypes.c_int
  lib.litert_lm_embedding_responses_get_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_void_p),
  ]

  # Embedding Engine
  lib.litert_lm_embedding_engine_create.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_create.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_embedding_engine_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_embedding_engine_compute_embedding.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_compute_embedding.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_size_t,
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_embedding_engine_compute_embedding_batch.restype = ctypes.c_int
  lib.litert_lm_embedding_engine_compute_embedding_batch.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
      ctypes.POINTER(ctypes.c_size_t),
      ctypes.c_size_t,
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]

  # Experimental C API
  lib.litert_lm_experimental_is_debugger_enabled.restype = ctypes.c_int
  lib.litert_lm_experimental_is_debugger_enabled.argtypes = [
      ctypes.POINTER(ctypes.c_bool)
  ]
  lib.litert_lm_experimental_session_get_debug_info.restype = ctypes.c_int
  lib.litert_lm_experimental_session_get_debug_info.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_experimental_session_debug_info_delete.argtypes = [
      ctypes.c_void_p
  ]
  lib.litert_lm_experimental_session_debug_info_get_capture_dir.restype = (
      ctypes.c_int
  )
  lib.litert_lm_experimental_session_debug_info_get_capture_dir.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.litert_lm_experimental_conversation_get_session_debug_info.restype = (
      ctypes.c_int
  )
  lib.litert_lm_experimental_conversation_get_session_debug_info.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]

  # Loaded File / Model Info API
  lib.litert_lm_loaded_file_create.restype = ctypes.c_int
  lib.litert_lm_loaded_file_create.argtypes = [
      c_string_p,
      ctypes.POINTER(ctypes.c_void_p),
  ]
  lib.litert_lm_loaded_file_delete.restype = None
  lib.litert_lm_loaded_file_delete.argtypes = [ctypes.c_void_p]
  for bool_getter in (
      lib.litert_lm_loaded_file_has_speculative_decoding_support,
      lib.litert_lm_loaded_file_supports_thinking,
      lib.litert_lm_loaded_file_supports_function_calling,
      lib.litert_lm_loaded_file_is_dynamic_context,
  ):
    bool_getter.restype = ctypes.c_int
    bool_getter.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
  lib.litert_lm_loaded_file_sampler_type.restype = ctypes.c_int
  lib.litert_lm_loaded_file_sampler_type.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_loaded_file_sampler_temperature.restype = ctypes.c_int
  lib.litert_lm_loaded_file_sampler_temperature.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_float),
  ]
  lib.litert_lm_loaded_file_sampler_top_k.restype = ctypes.c_int
  lib.litert_lm_loaded_file_sampler_top_k.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int32),
  ]
  lib.litert_lm_loaded_file_sampler_top_p.restype = ctypes.c_int
  lib.litert_lm_loaded_file_sampler_top_p.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_float),
  ]
  lib.litert_lm_loaded_file_supports_input_modality.restype = ctypes.c_int
  lib.litert_lm_loaded_file_supports_input_modality.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_bool),
  ]
  lib.litert_lm_loaded_file_max_vision_token_budget.restype = ctypes.c_int
  lib.litert_lm_loaded_file_max_vision_token_budget.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int32),
  ]
  lib.litert_lm_loaded_file_vision_signature_selection.restype = ctypes.c_int
  lib.litert_lm_loaded_file_vision_signature_selection.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int32),
      ctypes.c_int32,
      ctypes.POINTER(ctypes.c_int32),
  ]

  lib.litert_lm_loaded_file_max_context_tokens.restype = ctypes.c_int
  lib.litert_lm_loaded_file_max_context_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_uint32),
  ]

  lib.litert_lm_loaded_file_min_runtime_version.restype = ctypes.c_int
  lib.litert_lm_loaded_file_min_runtime_version.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]

  lib.litert_lm_loaded_file_modality_supported_backends.restype = ctypes.c_int
  lib.litert_lm_loaded_file_modality_supported_backends.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_int32,
      ctypes.POINTER(ctypes.c_int32),
  ]
  lib.litert_lm_loaded_file_model_type.restype = ctypes.c_int
  lib.litert_lm_loaded_file_model_type.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_loaded_file_embedding_dimension.restype = ctypes.c_int
  lib.litert_lm_loaded_file_embedding_dimension.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int32),
  ]
  lib.litert_lm_loaded_file_embedding_signature_selection.restype = ctypes.c_int
  lib.litert_lm_loaded_file_embedding_signature_selection.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int32),
      ctypes.c_int32,
      ctypes.POINTER(ctypes.c_int32),
  ]

  lib.litert_lm_loaded_file_modality_npu_brand.restype = ctypes.c_int
  lib.litert_lm_loaded_file_modality_npu_brand.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_int),
  ]
  lib.litert_lm_loaded_file_modality_soc_name.restype = ctypes.c_int
  lib.litert_lm_loaded_file_modality_soc_name.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_char_p),
  ]


def check_status(lib: ctypes.CDLL, func_name: str, status: int) -> None:
  """Raises if a C API call returned a non-OK `LiteRtLmStatusCode`.

  Must be called on the same thread that made the C API call, right after it,
  because the C library stores the last error message in thread-local storage.

  Args:
    lib: The loaded C library instance.
    func_name: Name of the C API function that returned `status`.
    status: The `LiteRtLmStatusCode` returned by `func_name`.

  Raises:
    RuntimeError: If `status` is not `kLiteRtLmStatusOk`. The message contains
      `func_name`, the status code and the C library's last error message.
  """
  if status == StatusCode.OK:
    return
  try:
    code_name = StatusCode(status).name
  except ValueError:
    code_name = "UNKNOWN_STATUS_CODE"
  message = lib.litert_lm_get_last_error_message()
  message = (
      message.decode("utf-8", errors="replace")
      if message
      else "no error message"
  )
  raise RuntimeError(
      f"{func_name} failed with status {code_name} ({status}): {message}"
  )


def call_checked(lib: ctypes.CDLL, func_name: str, *args) -> None:
  """Calls the status-returning C API function `func_name` and checks it.

  Args:
    lib: The loaded C library instance.
    func_name: Name of a C API function that returns a `LiteRtLmStatusCode`.
    *args: Arguments forwarded to the C API function.

  Raises:
    RuntimeError: If the call does not return `kLiteRtLmStatusOk`.
  """
  check_status(lib, func_name, getattr(lib, func_name)(*args))


def create_checked(lib: ctypes.CDLL, func_name: str, *args) -> int:
  """Calls a C API constructor that returns its handle through an out-parameter.

  The C API constructors return a `LiteRtLmStatusCode` and write the new handle
  to a trailing `out_*` parameter. This helper supplies that out-parameter.

  Args:
    lib: The loaded C library instance.
    func_name: Name of a C API constructor that takes a trailing `void**`
      out-parameter and returns a `LiteRtLmStatusCode`.
    *args: Arguments forwarded to the C API function, excluding the
      out-parameter.

  Returns:
    The created handle, owned by the caller.

  Raises:
    RuntimeError: If the call does not return `kLiteRtLmStatusOk` or does not
      produce a handle.
  """
  out = ctypes.c_void_p()
  call_checked(lib, func_name, *args, ctypes.byref(out))
  if out.value is None:
    raise RuntimeError(f"{func_name} returned a null handle")
  return out.value


def create_optional_checked(
    lib: ctypes.CDLL, func_name: str, *args
) -> int | None:
  """Like `create_checked`, but a NULL result on success means "absent".

  For C API functions whose result is legitimately optional (e.g. no start
  token configured): they return `kLiteRtLmStatusOk` and set the trailing
  `out_*` parameter to NULL.

  Args:
    lib: The loaded C library instance.
    func_name: Name of a C API function that takes a trailing `void**`
      out-parameter and returns a `LiteRtLmStatusCode`.
    *args: Arguments forwarded to the C API function, excluding the
      out-parameter.

  Returns:
    The produced handle, owned by the caller, or None if the value is absent.

  Raises:
    RuntimeError: If the call does not return `kLiteRtLmStatusOk`.
  """
  out = ctypes.c_void_p()
  call_checked(lib, func_name, *args, ctypes.byref(out))
  return out.value


def get_checked(lib: ctypes.CDLL, func_name: str, out_type: type[Any], *args):
  """Calls a C API accessor that returns its result through an out-parameter.

  The C API accessors return a `LiteRtLmStatusCode` and write their result to
  a trailing `out_*` parameter. This helper supplies that out-parameter.

  Args:
    lib: The loaded C library instance.
    func_name: Name of a C API function that takes a trailing out-parameter of
      type `out_type*` and returns a `LiteRtLmStatusCode`.
    out_type: The ctypes type of the out-parameter's pointee, e.g.
      `ctypes.c_int`, `ctypes.c_char_p` or `ctypes.POINTER(ctypes.c_float)`.
    *args: Arguments forwarded to the C API function, excluding the
      out-parameter.

  Returns:
    The value written to the out-parameter. For simple ctypes types this is
    the Python value (e.g. `int`, `float`, `bool`, `bytes` or None for a NULL
    string); for pointer types it is the ctypes pointer instance.

  Raises:
    RuntimeError: If the call does not return `kLiteRtLmStatusOk`.
  """
  out = out_type()
  call_checked(lib, func_name, *args, ctypes.byref(out))
  return getattr(out, "value", out)


def get_optional_checked(
    lib: ctypes.CDLL, func_name: str, out_type: type[Any], *args
):
  """Like `get_checked`, but `kLiteRtLmStatusNotFound` means "absent".

  For C API accessors of optional values: they return `kLiteRtLmStatusNotFound`
  (and leave the out-parameter unwritten) when the value is not defined.

  Args:
    lib: The loaded C library instance.
    func_name: Name of a C API function that takes a trailing out-parameter of
      type `out_type*` and returns a `LiteRtLmStatusCode`.
    out_type: The ctypes type of the out-parameter's pointee.
    *args: Arguments forwarded to the C API function, excluding the
      out-parameter.

  Returns:
    The value written to the out-parameter (see `get_checked`), or None if the
    call returned `kLiteRtLmStatusNotFound`.

  Raises:
    RuntimeError: If the call returns a status other than `kLiteRtLmStatusOk`
      or `kLiteRtLmStatusNotFound`.
  """
  out = out_type()
  status = getattr(lib, func_name)(*args, ctypes.byref(out))
  if status == StatusCode.NOT_FOUND:
    return None
  check_status(lib, func_name, status)
  return getattr(out, "value", out)


def set_min_log_severity(severity: LogSeverity):
  """Sets the minimum logging severity for the C library."""
  lib = _get_lib()
  call_checked(lib, "litert_lm_set_min_log_level", int(severity))
