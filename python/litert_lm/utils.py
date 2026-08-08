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
"""Utility functions for LiteRT-LM."""

import ctypes
from . import interfaces
from ._ffi import SamplerType
from ._ffi import TokenUnionType


def _sampler_config_to_params(
    lib,
    config: interfaces.SamplerConfig | None,
) -> ctypes.c_void_p:
  """Converts a SamplerConfig to a LiteRtLmSamplerParams opaque pointer."""
  params = lib.litert_lm_sampler_params_create(SamplerType.TOP_P)
  if not params:
    raise RuntimeError("Failed to create LiteRtLmSamplerParams")

  if config is not None:
    lib.litert_lm_sampler_params_set_top_k(
        params, config.top_k if config.top_k is not None else 1
    )
    lib.litert_lm_sampler_params_set_top_p(
        params, config.top_p if config.top_p is not None else 0.95
    )
    lib.litert_lm_sampler_params_set_temperature(
        params, config.temperature if config.temperature is not None else 1.0
    )
    lib.litert_lm_sampler_params_set_seed(
        params, config.seed if config.seed is not None else 0
    )
  return params


def thinking_config_to_params(
    lib,
    config: interfaces.ThinkingConfig | None,
) -> ctypes.c_void_p | None:
  """Converts a ThinkingConfig to a LiteRtLmThinkingConfig opaque pointer.

  Args:
      lib: The loaded C library instance.
      config: The thinking configuration object, or None.

  Returns:
      A new C pointer owned by the caller (caller must call
      litert_lm_thinking_config_delete when done), or None if config is None.

  Raises:
      RuntimeError: If pointer creation fails.
  """
  if config is None:
    return None
  params = lib.litert_lm_thinking_config_create()
  if not params:
    raise RuntimeError("Failed to create LiteRtLmThinkingConfig")
  lib.litert_lm_thinking_config_set_enable_thinking(
      params, config.enable_thinking
  )
  lib.litert_lm_thinking_config_set_thinking_token_budget(
      params, config.thinking_token_budget
  )
  return params


def _parse_token_union(lib, union_ptr):
  """Parses a C LiteRtLmTokenUnion into a Python string or list of IDs."""
  if not union_ptr:
    return None
  try:
    u_type = lib.litert_lm_token_union_get_type(union_ptr)
    if u_type == TokenUnionType.STRING:
      s = lib.litert_lm_token_union_get_string(union_ptr)
      return s.decode("utf-8") if s else None
    elif u_type == TokenUnionType.IDS:
      ids_ptr = ctypes.POINTER(ctypes.c_int)()
      num_ids = ctypes.c_size_t()
      if (
          lib.litert_lm_token_union_get_ids(
              union_ptr, ctypes.byref(ids_ptr), ctypes.byref(num_ids)
          )
          == 0
      ):
        return [ids_ptr[i] for i in range(num_ids.value)]
    return None
  finally:
    lib.litert_lm_token_union_delete(union_ptr)
