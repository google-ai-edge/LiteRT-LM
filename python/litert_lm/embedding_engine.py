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
"""EmbeddingEngine wrapper for LiteRT-LM."""

from __future__ import annotations

import ctypes
import dataclasses
import enum
from typing import Any, Sequence

from . import interfaces
from ._ffi import _get_lib
from ._ffi import InputDataType
from ._messages import AudioBytes
from ._messages import AudioFile
from ._messages import Content
from ._messages import ImageBytes
from ._messages import ImageFile
from ._messages import Text


@enum.unique
class InputOverflowStrategy(enum.IntEnum):
  """Strategy for handling inputs longer than maximum supported length."""

  # Chunk the input to the maximum supported length and average the embeddings.
  CHUNK_AND_AVERAGE = 0
  # Truncate the input to the maximum supported length.
  TRUNCATE = 1
  # Return an error if the input is longer than the maximum supported length.
  ERROR = 2


@dataclasses.dataclass(frozen=True, kw_only=True)
class EmbeddingOptions:
  """Options for embedding generation.

  Attributes:
    normalize: Whether to L2-normalize the output embedding vector. If None,
      uses the C++ engine default.
    insert_special_tokens: Whether to automatically insert special tokens (BOS,
      EOS, start/end of image, start/end of audio). If None, uses the C++ engine
      default.
    input_overflow_strategy: Strategy for handling inputs longer than the
      maximum supported signature length. If None, uses the C++ engine default.
    output_size: The output embedding size to truncate the embedding to. If
      None, uses the C++ engine default.
    vision_tokens_per_image: The number of vision soft tokens to generate per
      image. If None, uses the C++ engine default.
  """

  normalize: bool | None = None
  insert_special_tokens: bool | None = None
  input_overflow_strategy: InputOverflowStrategy | None = None
  output_size: int | None = None
  vision_tokens_per_image: int | None = None


@dataclasses.dataclass(frozen=True)
class EmbeddingResponse:
  """Represents the embedding result for an input item.

  Attributes:
    embedding: Dense float vector output representation.
  """

  embedding: list[float]


def _create_c_input_data(
    lib: Any, item: str | Content
) -> ctypes.c_void_p:
  """Creates a LiteRtLmInputData pointer from a string or Content."""
  if isinstance(item, str):
    data_bytes = item.encode("utf-8")
    ptr = lib.litert_lm_input_data_create(
        InputDataType.TEXT, data_bytes, len(data_bytes)
    )
    if not ptr:
      raise RuntimeError("Failed to create LiteRtLmInputData")
    return ptr
  elif isinstance(item, Text):
    data_bytes = item.text.encode("utf-8")
    ptr = lib.litert_lm_input_data_create(
        InputDataType.TEXT, data_bytes, len(data_bytes)
    )
    if not ptr:
      raise RuntimeError("Failed to create LiteRtLmInputData")
    return ptr
  elif isinstance(item, (ImageBytes, ImageFile)):
    if isinstance(item, ImageBytes):
      data_bytes = item.bytes
    else:
      with open(item.absolute_path, "rb") as f:
        data_bytes = f.read()
    img_ptr = lib.litert_lm_input_data_create(
        InputDataType.IMAGE, data_bytes, len(data_bytes)
    )
    if not img_ptr:
      raise RuntimeError("Failed to create LiteRtLmInputData for Image")
    return img_ptr
  elif isinstance(item, (AudioBytes, AudioFile)):
    if isinstance(item, AudioBytes):
      data_bytes = item.bytes
    else:
      with open(item.absolute_path, "rb") as f:
        data_bytes = f.read()
    audio_ptr = lib.litert_lm_input_data_create(
        InputDataType.AUDIO, data_bytes, len(data_bytes)
    )
    if not audio_ptr:
      raise RuntimeError("Failed to create LiteRtLmInputData for Audio")
    return audio_ptr
  else:
    raise TypeError(f"Unsupported content type for embedding: {type(item)}")


class EmbeddingEngine:
  """Manages the lifecycle of a LiteRT-LM Embedding Engine."""

  def __init__(
      self,
      model_path: str,
      backend: interfaces.Backend = interfaces.Backend.CPU(),
      vision_backend: interfaces.Backend | None = None,
      audio_backend: interfaces.Backend | None = None,
      cache_dir: str | None = None,
      max_input_length: int | None = None,
      vision_tokens_per_image: int | None = None,
      **kwargs: Any,
  ):
    del kwargs
    self._model_path = model_path
    self._backend = backend
    self._vision_backend = vision_backend
    self._audio_backend = audio_backend
    self._cache_dir = cache_dir
    self._max_input_length = max_input_length
    self._vision_tokens_per_image = vision_tokens_per_image

    self._lib = _get_lib()
    self._engine_ptr: ctypes.c_void_p | None = None

    settings = self._lib.litert_lm_embedding_engine_settings_create(
        self._model_path,
        self._backend.get_name(),
        (self._vision_backend.get_name() if self._vision_backend else None),
        (self._audio_backend.get_name() if self._audio_backend else None),
    )
    if not settings:
      raise RuntimeError("Failed to create LiteRtLmEmbeddingEngineSettings")

    try:
      if isinstance(self._audio_backend, interfaces.CPU):
        if self._audio_backend.thread_count is not None:
          self._lib.litert_lm_embedding_engine_settings_set_audio_num_threads(
              settings, self._audio_backend.thread_count
          )
      if self._cache_dir:
        self._lib.litert_lm_embedding_engine_settings_set_cache_dir(
            settings, self._cache_dir
        )
      if isinstance(self._backend, interfaces.NPU):
        if self._backend.litert_dispatch_lib_dir:
          self._lib.litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
              settings, self._backend.litert_dispatch_lib_dir
          )
      if isinstance(self._vision_backend, interfaces.NPU):
        if self._vision_backend.litert_dispatch_lib_dir:
          self._lib.litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
              settings, self._vision_backend.litert_dispatch_lib_dir
          )
      if isinstance(self._audio_backend, interfaces.NPU):
        if self._audio_backend.litert_dispatch_lib_dir:
          self._lib.litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
              settings, self._audio_backend.litert_dispatch_lib_dir
          )
      if self._max_input_length is not None:
        self._lib.litert_lm_embedding_engine_settings_set_max_input_length(
            settings, self._max_input_length
        )
      if self._vision_tokens_per_image is not None:
        self._lib.litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
            settings, self._vision_tokens_per_image
        )

      self._engine_ptr = self._lib.litert_lm_embedding_engine_create(settings)
      if not self._engine_ptr:
        raise RuntimeError("Failed to create LiteRtLmEmbeddingEngine")
    finally:
      self._lib.litert_lm_embedding_engine_settings_delete(settings)

  @property
  def model_path(self) -> str:
    return self._model_path

  @property
  def backend(self) -> interfaces.Backend:
    return self._backend

  @property
  def vision_backend(self) -> interfaces.Backend | None:
    return self._vision_backend

  @property
  def audio_backend(self) -> interfaces.Backend | None:
    return self._audio_backend

  @property
  def cache_dir(self) -> str | None:
    return self._cache_dir

  @property
  def max_input_length(self) -> int | None:
    """Returns the maximum input sequence length in tokens, or None if unset."""
    return self._max_input_length

  @property
  def vision_tokens_per_image(self) -> int | None:
    """Returns the desired number of vision tokens per image, or None if unset."""
    return self._vision_tokens_per_image

  def _check_alive(self) -> None:
    if self._engine_ptr is None:
      raise RuntimeError("EmbeddingEngine has already been closed.")

  def compute_embedding(
      self,
      contents: str | Content | Sequence[str | Content],
      options: EmbeddingOptions = EmbeddingOptions(),
  ) -> EmbeddingResponse:
    """Computes the embedding for the provided input contents.

    Args:
      contents: Input string, Content, or list/sequence of strings or Contents.
      options: EmbeddingOptions controlling generation (e.g. normalization).

    Returns:
      An EmbeddingResponse containing the embedding vector.
    """
    self._check_alive()

    if isinstance(contents, (str, Content)):
      items = [contents]
    else:
      items = list(contents)

    options_ptr = self._lib.litert_lm_embedding_options_create()
    try:
      if options.normalize is not None:
        self._lib.litert_lm_embedding_options_set_normalize(
            options_ptr, options.normalize
        )
      if options.insert_special_tokens is not None:
        self._lib.litert_lm_embedding_options_set_insert_special_tokens(
            options_ptr, options.insert_special_tokens
        )
      if options.input_overflow_strategy is not None:
        self._lib.litert_lm_embedding_options_set_input_overflow_strategy(
            options_ptr, int(options.input_overflow_strategy)
        )
      if options.output_size is not None:
        self._lib.litert_lm_embedding_options_set_output_size(
            options_ptr, options.output_size
        )
      if options.vision_tokens_per_image is not None:
        self._lib.litert_lm_embedding_options_set_vision_tokens_per_image(
            options_ptr, options.vision_tokens_per_image
        )

      created_ptrs = [_create_c_input_data(self._lib, item) for item in items]

      inputs_array = (ctypes.c_void_p * len(created_ptrs))(*created_ptrs)
      resp_ptr = self._lib.litert_lm_embedding_engine_compute_embedding(
          self._engine_ptr, inputs_array, len(created_ptrs), options_ptr
      )
      if not resp_ptr:
        raise RuntimeError("Failed to compute embedding.")

      try:
        size = self._lib.litert_lm_embedding_response_get_size(resp_ptr)
        vals_ptr = self._lib.litert_lm_embedding_response_get_values(resp_ptr)
        if not vals_ptr and size > 0:
          raise RuntimeError("Invalid embedding response values pointer.")
        embedding = [float(vals_ptr[i]) for i in range(size)]
        return EmbeddingResponse(embedding=embedding)
      finally:
        self._lib.litert_lm_embedding_response_delete(resp_ptr)
    finally:
      self._lib.litert_lm_embedding_options_delete(options_ptr)
      for ptr in created_ptrs:
        self._lib.litert_lm_input_data_delete(ptr)

  def compute_embedding_batch(
      self,
      contents_batch: Sequence[str | Content | Sequence[str | Content]],
      options: EmbeddingOptions = EmbeddingOptions(),
  ) -> list[EmbeddingResponse]:
    """Computes embeddings for a batch of requests.

    Args:
      contents_batch: Sequence of requests, where each request is a string,
        Content, or sequence of strings / Contents.
      options: EmbeddingOptions controlling generation (e.g. normalization).

    Returns:
      A list of EmbeddingResponse objects.
    """
    self._check_alive()

    batch_size = len(contents_batch)
    normalized_batch: list[list[str | Content]] = []
    for req in contents_batch:
      if isinstance(req, (str, Content)):
        normalized_batch.append([req])
      else:
        normalized_batch.append(list(req))

    all_created_ptrs: list[ctypes.c_void_p] = []
    options_ptr = self._lib.litert_lm_embedding_options_create()

    try:
      if options.normalize is not None:
        self._lib.litert_lm_embedding_options_set_normalize(
            options_ptr, options.normalize
        )
      if options.insert_special_tokens is not None:
        self._lib.litert_lm_embedding_options_set_insert_special_tokens(
            options_ptr, options.insert_special_tokens
        )
      if options.input_overflow_strategy is not None:
        self._lib.litert_lm_embedding_options_set_input_overflow_strategy(
            options_ptr, int(options.input_overflow_strategy)
        )
      if options.output_size is not None:
        self._lib.litert_lm_embedding_options_set_output_size(
            options_ptr, options.output_size
        )
      if options.vision_tokens_per_image is not None:
        self._lib.litert_lm_embedding_options_set_vision_tokens_per_image(
            options_ptr, options.vision_tokens_per_image
        )

      batch_inputs_arrays: list[Any] = []
      num_inputs_per_batch = (ctypes.c_size_t * batch_size)()

      for i, req in enumerate(normalized_batch):
        req_ptrs = [_create_c_input_data(self._lib, item) for item in req]
        all_created_ptrs.extend(req_ptrs)
        req_array = (ctypes.c_void_p * len(req_ptrs))(*req_ptrs)
        batch_inputs_arrays.append(req_array)
        num_inputs_per_batch[i] = len(req_ptrs)

      inputs_batch_array = (ctypes.POINTER(ctypes.c_void_p) * batch_size)(*[
          ctypes.cast(arr, ctypes.POINTER(ctypes.c_void_p))
          for arr in batch_inputs_arrays
      ])

      responses_ptr = (
          self._lib.litert_lm_embedding_engine_compute_embedding_batch(
              self._engine_ptr,
              inputs_batch_array,
              num_inputs_per_batch,
              batch_size,
              options_ptr,
          )
      )
      if not responses_ptr:
        raise RuntimeError("Failed to compute embedding batch.")

      try:
        num_responses = self._lib.litert_lm_embedding_responses_get_size(
            responses_ptr
        )
        results: list[EmbeddingResponse] = []
        for i in range(num_responses):
          resp_ptr = self._lib.litert_lm_embedding_responses_get_at(
              responses_ptr, i
          )
          if not resp_ptr:
            raise RuntimeError(
                f"Failed to get embedding response at index {i}."
            )
          size = self._lib.litert_lm_embedding_response_get_size(resp_ptr)
          vals_ptr = self._lib.litert_lm_embedding_response_get_values(resp_ptr)
          if not vals_ptr and size > 0:
            raise RuntimeError("Invalid embedding response values pointer.")
          results.append(
              EmbeddingResponse(
                  embedding=[float(vals_ptr[j]) for j in range(size)]
              )
          )
        return results
      finally:
        self._lib.litert_lm_embedding_responses_delete(responses_ptr)
    finally:
      self._lib.litert_lm_embedding_options_delete(options_ptr)
      for ptr in all_created_ptrs:
        self._lib.litert_lm_input_data_delete(ptr)

  def close(self) -> None:
    """Closes the engine and releases native resources."""
    if self._engine_ptr is not None:
      try:
        self._lib.litert_lm_embedding_engine_delete(self._engine_ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._engine_ptr = None

  def __enter__(self) -> EmbeddingEngine:
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()
