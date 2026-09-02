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
"""Model capabilities extraction API.

This API allows querying model capabilities and defaults directly from a
compiled .litertlm file.

Example:

  import litert_lm

  capabilities = litert_lm.Capabilities("/path/to/model.litertlm")
  thinking = capabilities.supports_thinking()
  vision = capabilities.input_modalities.vision
  text = capabilities.input_modalities.text
  video = capabilities.input_modalities.video
  audio = capabilities.input_modalities.audio
  sampler_config = capabilities.default_sampler_params
  temperature = sampler_config.temperature
  top_k = sampler_config.top_k
  top_p = sampler_config.top_p
"""

from __future__ import annotations

import ctypes
import dataclasses
import os

from . import _ffi
from . import interfaces


@dataclasses.dataclass(frozen=True)
class SupportedModalities:
  """Modalities supported by the model."""

  text: bool
  vision: bool
  audio: bool
  video: bool


class Capabilities:
  """Exposes model capabilities directly from a LiteRT-LM file."""

  def __init__(self, model_path: str | os.PathLike[str]):
    """Loads a LiteRT-LM file and parses its metadata capabilities.

    Args:
      model_path: Path to the .litertlm file.

    Raises:
      FileNotFoundError: If the file does not exist.
      RuntimeError: If the capabilities could not be loaded.
    """
    model_path_str = os.fspath(model_path)
    if not os.path.exists(model_path_str):
      raise FileNotFoundError(f"Model file not found: {model_path_str}")

    self._lib = _ffi._get_lib()  # pylint: disable=protected-access
    self._handle = self._lib.litert_lm_loaded_file_create(model_path_str)

    if not self._handle:
      raise RuntimeError(
          f"Failed to load capabilities for model: {model_path_str}"
      )

  def close(self) -> None:
    """Closes the capabilities loader and releases C resources."""
    if hasattr(self, "_handle") and self._handle:
      self._lib.litert_lm_loaded_file_delete(self._handle)
      self._handle = None

  def __enter__(self) -> Capabilities:
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()

  def _check_closed(self) -> None:
    if not self._handle:
      raise RuntimeError("Capabilities object is closed")

  def has_speculative_decoding_support(self) -> bool:
    """Returns True if the model supports speculative decoding."""
    self._check_closed()
    return self._lib.litert_lm_loaded_file_has_speculative_decoding_support(
        self._handle
    )

  def supports_thinking(self) -> bool:
    """Returns True if the model supports thinking/reasoning steps."""
    self._check_closed()
    return self._lib.litert_lm_loaded_file_supports_thinking(self._handle)

  def supports_function_calling(self) -> bool:
    """Returns True if the model supports function calling."""
    self._check_closed()
    return self._lib.litert_lm_loaded_file_supports_function_calling(
        self._handle
    )

  @property
  def input_modalities(self) -> SupportedModalities:
    """Returns the input modalities supported by the model."""
    self._check_closed()
    return SupportedModalities(
        text=self._lib.litert_lm_loaded_file_supports_input_modality(
            self._handle, _ffi.LiteRtLmModality.TEXT
        ),
        vision=self._lib.litert_lm_loaded_file_supports_input_modality(
            self._handle, _ffi.LiteRtLmModality.VISION
        ),
        audio=self._lib.litert_lm_loaded_file_supports_input_modality(
            self._handle, _ffi.LiteRtLmModality.AUDIO
        ),
        video=self._lib.litert_lm_loaded_file_supports_input_modality(
            self._handle, _ffi.LiteRtLmModality.VIDEO
        ),
    )

  @property
  def default_sampler_params(self) -> interfaces.SamplerConfig:
    """Returns the default sampler parameters configured in the model."""
    self._check_closed()
    top_k = self._lib.litert_lm_loaded_file_sampler_top_k(self._handle)
    return interfaces.SamplerConfig(
        temperature=self._lib.litert_lm_loaded_file_sampler_temperature(
            self._handle
        ),
        top_k=top_k if top_k > 0 else None,
        top_p=self._lib.litert_lm_loaded_file_sampler_top_p(self._handle),
    )

  @property
  def max_vision_token_budget(self) -> int:
    """Returns maximum vision token budget, or -1 if not defined."""
    self._check_closed()
    return int(
        self._lib.litert_lm_loaded_file_max_vision_token_budget(self._handle)
    )

  @property
  def vision_signature_selection(self) -> list[int] | None:
    """Returns vision signature choices, or None if vision is unsupported."""
    self._check_closed()
    count = self._lib.litert_lm_loaded_file_vision_signature_selection(
        self._handle, None, 0
    )
    if count == -1:
      return None
    lengths = (ctypes.c_int32 * count)()
    self._lib.litert_lm_loaded_file_vision_signature_selection(
        self._handle, lengths, count
    )
    return list(lengths)

  @property
  def min_runtime_version(self) -> str | None:
    """Returns minimum runtime version required, or None if not defined."""
    self._check_closed()
    version_bytes = self._lib.litert_lm_loaded_file_min_runtime_version(
        self._handle
    )
    if version_bytes is None:
      return None
    return version_bytes.decode("utf-8")

  def supported_backends_for_modality(
      self, modality: _ffi.LiteRtLmModality
  ) -> set[str]:
    """Returns the set of supported backends for a given modality.

    Args:
      modality: The input modality to query backends for.

    Returns:
      A set of backend name strings (e.g. {"cpu", "gpu"}).
    """
    self._check_closed()
    mask = self._lib.litert_lm_loaded_file_modality_supported_backends(
        self._handle, int(modality)
    )
    backends = set()
    if mask & _ffi.LiteRtLmBackendMask.CPU:
      backends.add("cpu")
    if mask & _ffi.LiteRtLmBackendMask.GPU:
      backends.add("gpu")
    if mask & _ffi.LiteRtLmBackendMask.NPU:
      backends.add("npu")
    return backends

  def npu_brand_for_modality(
      self, modality: _ffi.LiteRtLmModality
  ) -> _ffi.LiteRtLmNpuBrand:
    """Returns the NPU brand of the model for a given modality.

    Args:
      modality: The input modality to query NPU brand for.

    Returns:
      The NpuBrand enum value for the modality.
    """
    self._check_closed()
    brand_val = self._lib.litert_lm_loaded_file_modality_npu_brand(
        self._handle, int(modality)
    )
    try:
      return _ffi.LiteRtLmNpuBrand(brand_val)
    except ValueError:
      return _ffi.LiteRtLmNpuBrand.UNKNOWN
