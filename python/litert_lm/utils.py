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
from ._ffi import LiteRtLmSamplerParams
from ._ffi import SamplerType
from ._ffi import TokenUnionType


def _sampler_config_to_params(
    config: interfaces.SamplerConfig | None,
) -> LiteRtLmSamplerParams:
  """Converts a SamplerConfig to a LiteRtLmSamplerParams ctypes structure."""
  params = LiteRtLmSamplerParams()
  if config is not None:
    params.type = SamplerType.TOP_P
    params.top_k = config.top_k if config.top_k is not None else 40
    params.top_p = config.top_p if config.top_p is not None else 0.95
    params.temperature = (
        config.temperature if config.temperature is not None else 1.0
    )
    params.seed = config.seed if config.seed is not None else 0
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
