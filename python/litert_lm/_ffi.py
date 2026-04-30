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


class c_string_p(ctypes.c_char_p):  # pylint: disable=invalid-name
  """Custom ctypes type that automatically encodes Python strings to UTF-8 bytes."""

  @classmethod
  def from_param(cls, obj):
    if obj is None:
      return None
    if isinstance(obj, str):
      return obj.encode("utf-8")
    return obj


class LiteRtLmSamplerParams(ctypes.Structure):
  _fields_ = [
      ("type", ctypes.c_int),
      ("top_k", ctypes.c_int),
      ("top_p", ctypes.c_float),
      ("temperature", ctypes.c_float),
      ("seed", ctypes.c_int),
  ]


class LiteRtLmInputData(ctypes.Structure):
  _fields_ = [
      ("type", ctypes.c_int),
      ("data", ctypes.c_void_p),
      ("size", ctypes.c_size_t),
  ]


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


# C-compatible callback type that matches 'LiteRtLmStreamCallback' in engine.h.
STREAM_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_char_p
)


class LogSeverity(enum.IntEnum):
  VERBOSE = 0
  DEBUG = 1
  INFO = 2
  WARNING = 3
  ERROR = 4
  FATAL = 5
  SILENT = 1000


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
    path = os.path.join(os.path.dirname(__file__), lib_name)
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
  # Log level
  lib.litert_lm_set_min_log_level.argtypes = [ctypes.c_int]

  # Engine Settings
  lib.litert_lm_engine_settings_create.restype = ctypes.c_void_p
  lib.litert_lm_engine_settings_create.argtypes = [
      c_string_p,
      c_string_p,
      c_string_p,
      c_string_p,
  ]
  lib.litert_lm_engine_settings_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_settings_set_max_num_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_cache_dir.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_engine_settings_set_enable_speculative_decoding.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_engine_settings_enable_benchmark.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_settings_set_num_prefill_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_engine_settings_set_num_decode_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Engine
  lib.litert_lm_engine_create.restype = ctypes.c_void_p
  lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p]

  # Session Config
  lib.litert_lm_session_config_create.restype = ctypes.c_void_p
  lib.litert_lm_session_config_create.argtypes = []
  lib.litert_lm_session_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_config_set_max_output_tokens.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_session_config_set_apply_prompt_template.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_session_config_set_sampler_params.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(LiteRtLmSamplerParams),
  ]

  # Session
  lib.litert_lm_engine_create_session.restype = ctypes.c_void_p
  lib.litert_lm_engine_create_session.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_session_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_cancel_process.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_run_prefill.restype = ctypes.c_int
  lib.litert_lm_session_run_prefill.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(LiteRtLmInputData),
      ctypes.c_size_t,
  ]
  lib.litert_lm_session_run_decode.restype = ctypes.c_void_p
  lib.litert_lm_session_run_decode.argtypes = [ctypes.c_void_p]
  lib.litert_lm_session_run_decode_async.restype = ctypes.c_int
  lib.litert_lm_session_run_decode_async.argtypes = [
      ctypes.c_void_p,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]
  lib.litert_lm_session_run_text_scoring.restype = ctypes.c_void_p
  lib.litert_lm_session_run_text_scoring.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
      ctypes.c_size_t,
      ctypes.c_bool,
  ]
  lib.litert_lm_session_generate_content.restype = ctypes.c_void_p
  lib.litert_lm_session_generate_content.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(LiteRtLmInputData),
      ctypes.c_size_t,
  ]
  lib.litert_lm_session_generate_content_stream.restype = ctypes.c_int
  lib.litert_lm_session_generate_content_stream.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(LiteRtLmInputData),
      ctypes.c_size_t,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]

  # Conversation Config
  lib.litert_lm_conversation_config_create.restype = ctypes.c_void_p
  lib.litert_lm_conversation_config_create.argtypes = []
  lib.litert_lm_conversation_config_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_config_set_session_config.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_config_set_system_message.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_tools.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_messages.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_extra_context.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_config_set_enable_constrained_decoding.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]
  lib.litert_lm_conversation_config_set_filter_channel_content_from_kv_cache.argtypes = [
      ctypes.c_void_p,
      ctypes.c_bool,
  ]

  # Conversation
  lib.litert_lm_conversation_create.restype = ctypes.c_void_p
  lib.litert_lm_conversation_create.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_send_message.restype = ctypes.c_void_p
  lib.litert_lm_conversation_send_message.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      c_string_p,
  ]
  lib.litert_lm_conversation_send_message_stream.restype = ctypes.c_int
  lib.litert_lm_conversation_send_message_stream.argtypes = [
      ctypes.c_void_p,
      c_string_p,
      c_string_p,
      STREAM_CALLBACK_TYPE,
      ctypes.c_void_p,
  ]
  lib.litert_lm_conversation_cancel_process.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_render_message_to_string.restype = ctypes.c_char_p
  lib.litert_lm_conversation_render_message_to_string.argtypes = [
      ctypes.c_void_p,
      c_string_p,
  ]

  # interfaces.Responses
  lib.litert_lm_responses_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_responses_get_num_candidates.restype = ctypes.c_int
  lib.litert_lm_responses_get_num_candidates.argtypes = [ctypes.c_void_p]
  lib.litert_lm_responses_get_response_text_at.restype = ctypes.c_char_p
  lib.litert_lm_responses_get_response_text_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_has_score_at.restype = ctypes.c_bool
  lib.litert_lm_responses_has_score_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_get_score_at.restype = ctypes.c_float
  lib.litert_lm_responses_get_score_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_has_token_length_at.restype = ctypes.c_bool
  lib.litert_lm_responses_has_token_length_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_get_token_length_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_token_length_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_has_token_scores_at.restype = ctypes.c_bool
  lib.litert_lm_responses_has_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_get_num_token_scores_at.restype = ctypes.c_int
  lib.litert_lm_responses_get_num_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_responses_get_token_scores_at.restype = ctypes.POINTER(
      ctypes.c_float
  )
  lib.litert_lm_responses_get_token_scores_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # JSON Response
  lib.litert_lm_json_response_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_json_response_get_string.restype = ctypes.c_char_p
  lib.litert_lm_json_response_get_string.argtypes = [ctypes.c_void_p]

  # Benchmark Info
  lib.litert_lm_session_get_benchmark_info.restype = ctypes.c_void_p
  lib.litert_lm_session_get_benchmark_info.argtypes = [ctypes.c_void_p]
  lib.litert_lm_conversation_get_benchmark_info.restype = ctypes.c_void_p
  lib.litert_lm_conversation_get_benchmark_info.argtypes = [ctypes.c_void_p]
  lib.litert_lm_benchmark_info_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_benchmark_info_get_time_to_first_token.restype = ctypes.c_double
  lib.litert_lm_benchmark_info_get_time_to_first_token.argtypes = [
      ctypes.c_void_p
  ]
  lib.litert_lm_benchmark_info_get_total_init_time_in_second.restype = (
      ctypes.c_double
  )
  lib.litert_lm_benchmark_info_get_total_init_time_in_second.argtypes = [
      ctypes.c_void_p
  ]
  lib.litert_lm_benchmark_info_get_num_prefill_turns.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_num_prefill_turns.argtypes = [
      ctypes.c_void_p
  ]
  lib.litert_lm_benchmark_info_get_num_decode_turns.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_num_decode_turns.argtypes = [ctypes.c_void_p]
  lib.litert_lm_benchmark_info_get_prefill_token_count_at.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_prefill_token_count_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_benchmark_info_get_decode_token_count_at.restype = ctypes.c_int
  lib.litert_lm_benchmark_info_get_decode_token_count_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at.restype = (
      ctypes.c_double
  )
  lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]
  lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at.restype = (
      ctypes.c_double
  )
  lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
  ]

  # Tokenizer
  lib.litert_lm_engine_tokenize.restype = ctypes.c_void_p
  lib.litert_lm_engine_tokenize.argtypes = [ctypes.c_void_p, c_string_p]
  lib.litert_lm_tokenize_result_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_tokenize_result_get_tokens.restype = ctypes.POINTER(
      ctypes.c_int
  )
  lib.litert_lm_tokenize_result_get_tokens.argtypes = [ctypes.c_void_p]
  lib.litert_lm_tokenize_result_get_num_tokens.restype = ctypes.c_size_t
  lib.litert_lm_tokenize_result_get_num_tokens.argtypes = [ctypes.c_void_p]

  lib.litert_lm_engine_detokenize.restype = ctypes.c_void_p
  lib.litert_lm_engine_detokenize.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_int),
      ctypes.c_size_t,
  ]
  lib.litert_lm_detokenize_result_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_detokenize_result_get_string.restype = ctypes.c_char_p
  lib.litert_lm_detokenize_result_get_string.argtypes = [ctypes.c_void_p]

  # Token Union / Metadata
  lib.litert_lm_token_union_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_union_get_type.restype = ctypes.c_int
  lib.litert_lm_token_union_get_type.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_union_get_string.restype = ctypes.c_char_p
  lib.litert_lm_token_union_get_string.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_union_get_ids.restype = ctypes.c_int
  lib.litert_lm_token_union_get_ids.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
      ctypes.POINTER(ctypes.c_size_t),
  ]

  lib.litert_lm_token_unions_delete.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_unions_get_num_tokens.restype = ctypes.c_size_t
  lib.litert_lm_token_unions_get_num_tokens.argtypes = [ctypes.c_void_p]
  lib.litert_lm_token_unions_get_token_at.restype = ctypes.c_void_p
  lib.litert_lm_token_unions_get_token_at.argtypes = [
      ctypes.c_void_p,
      ctypes.c_size_t,
  ]

  lib.litert_lm_engine_get_start_token.restype = ctypes.c_void_p
  lib.litert_lm_engine_get_start_token.argtypes = [ctypes.c_void_p]
  lib.litert_lm_engine_get_stop_tokens.restype = ctypes.c_void_p
  lib.litert_lm_engine_get_stop_tokens.argtypes = [ctypes.c_void_p]


def set_min_log_severity(severity: LogSeverity):
  """Sets the minimum logging severity for the C library."""
  _get_lib().litert_lm_set_min_log_level(int(severity))
