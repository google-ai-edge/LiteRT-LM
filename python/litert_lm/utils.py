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
import dataclasses
import warnings

from . import interfaces
from ._ffi import call_checked
from ._ffi import create_checked
from ._ffi import get_checked
from ._ffi import SamplerType
from ._ffi import TokenUnionType


def _sampler_config_to_params(
    lib,
    config: interfaces.SamplerConfig | None,
) -> int:
  """Converts a SamplerConfig to a LiteRtLmSamplerParams opaque pointer."""
  params = create_checked(
      lib, "litert_lm_sampler_params_create", SamplerType.TOP_P
  )

  if config is not None:
    try:
      call_checked(
          lib,
          "litert_lm_sampler_params_set_top_k",
          params,
          config.top_k if config.top_k is not None else 1,
      )
      call_checked(
          lib,
          "litert_lm_sampler_params_set_top_p",
          params,
          config.top_p if config.top_p is not None else 0.95,
      )
      call_checked(
          lib,
          "litert_lm_sampler_params_set_temperature",
          params,
          config.temperature if config.temperature is not None else 1.0,
      )
      call_checked(
          lib,
          "litert_lm_sampler_params_set_seed",
          params,
          config.seed if config.seed is not None else 0,
      )
    except BaseException:
      lib.litert_lm_sampler_params_delete(params)
      raise
  return params


def thinking_config_to_params(
    lib,
    config: interfaces.ThinkingConfig | None,
) -> int | None:
  """Converts a ThinkingConfig to a LiteRtLmThinkingConfig opaque pointer.

  Args:
      lib: The loaded C library instance.
      config: The thinking configuration object, or None.

  Returns:
      A new C pointer owned by the caller (caller must call
      litert_lm_thinking_config_delete when done), or None if config is None.

  Raises:
      RuntimeError: If pointer creation fails or a setter reports an error.
  """
  if config is None:
    return None
  params = create_checked(lib, "litert_lm_thinking_config_create")
  try:
    call_checked(
        lib,
        "litert_lm_thinking_config_set_enable_thinking",
        params,
        config.enable_thinking,
    )
    call_checked(
        lib,
        "litert_lm_thinking_config_set_thinking_token_budget",
        params,
        config.thinking_token_budget,
    )
  except BaseException:
    lib.litert_lm_thinking_config_delete(params)
    raise
  return params


def _parse_token_union(lib, union_ptr):
  """Parses a C LiteRtLmTokenUnion into a Python string or list of IDs.

  Args:
    lib: The loaded C library instance.
    union_ptr: Pointer to the LiteRtLmTokenUnion to parse, or None. The caller
      keeps ownership; this function does not delete it.

  Returns:
    The token string or list of token IDs, or None if there is no token.
  """
  if not union_ptr:
    return None
  u_type = get_checked(
      lib, "litert_lm_token_union_get_type", ctypes.c_int, union_ptr
  )
  if u_type == TokenUnionType.STRING:
    s = get_checked(
        lib, "litert_lm_token_union_get_string", ctypes.c_char_p, union_ptr
    )
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


def warn_enable_speculative_decoding_deprecated(
    enable_speculative_decoding: bool | None, *, stacklevel: int = 3
) -> None:
  """Warns if the deprecated `enable_speculative_decoding` is set.

  Args:
    enable_speculative_decoding: The deprecated flag.
    stacklevel: Passed to `warnings.warn`. The default of 3 points at the caller
      of the public method that calls this function directly.
  """
  if enable_speculative_decoding is not None:
    warnings.warn(
        "enable_speculative_decoding is deprecated. Use"
        " speculative_decoding_config=SpeculativeDecodingConfig(mode="
        "SpeculativeDecodingMode.ON) (or OFF) instead.",
        category=DeprecationWarning,
        stacklevel=stacklevel,
    )


def merge_speculative_decoding_config(
    enable_speculative_decoding: bool | None,
    speculative_decoding_config: interfaces.SpeculativeDecodingConfig | None,
) -> interfaces.SpeculativeDecodingConfig | None:
  """Merges the deprecated `enable_speculative_decoding` into the config.

  Args:
    enable_speculative_decoding: The deprecated flag. Used only when the config
      sets no mode: True is mode ON and False is mode OFF.
    speculative_decoding_config: The config; its mode takes precedence.

  Returns:
    The config to apply, or None to inherit (or use the runtime defaults).
  """
  if enable_speculative_decoding is None or (
      speculative_decoding_config is not None
      and speculative_decoding_config.mode is not None
  ):
    return speculative_decoding_config
  mode = (
      interfaces.SpeculativeDecodingMode.ON
      if enable_speculative_decoding
      else interfaces.SpeculativeDecodingMode.OFF
  )
  if speculative_decoding_config is None:
    return interfaces.SpeculativeDecodingConfig(mode=mode)
  return dataclasses.replace(speculative_decoding_config, mode=mode)


def create_c_speculative_decoding_config(
    lib, config: interfaces.SpeculativeDecodingConfig
) -> int:
  """Creates a C speculative decoding config holding the fields of `config`.

  Args:
    lib: The loaded LiteRT-LM C library.
    config: The speculative decoding config.

  Returns:
    A config pointer; the caller must delete it with
    `litert_lm_speculative_decoding_config_delete`.

  Raises:
    RuntimeError: If the config cannot be created or a field fails to apply.
      The config is freed in that case.
  """
  c_config = create_checked(lib, "litert_lm_speculative_decoding_config_create")
  try:
    if config.mode is not None:
      call_checked(
          lib,
          "litert_lm_speculative_decoding_config_set_mode",
          c_config,
          int(config.mode),
      )
    if config.lenience is not None:
      call_checked(
          lib,
          "litert_lm_speculative_decoding_config_set_lenience",
          c_config,
          config.lenience,
      )

  except BaseException:
    lib.litert_lm_speculative_decoding_config_delete(c_config)
    raise
  return c_config
