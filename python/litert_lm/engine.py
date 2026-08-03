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
"""Engine wrapper for LiteRT-LM."""

from __future__ import annotations

import collections.abc
import ctypes
import json
from typing import Any
import warnings

from . import interfaces
from . import tools as litert_tools
from ._ffi import _get_lib
from ._ffi import ActivationDataType
from ._messages import Message
from .conversation import Conversation
from .session import Session
from .utils import _parse_token_union
from .utils import _sampler_config_to_params


# TODO: b/482060476 - Drop support for passing Backend class in 0.13.0.
def _normalize_backend(backend: Any) -> Any:
  if isinstance(backend, type) and issubclass(backend, interfaces.Backend):
    warnings.warn(
        f"Passing Backend class {backend.__name__} is deprecated. "
        f"Please use an instance instead: {backend.__name__}()",
        category=DeprecationWarning,
        stacklevel=3,
    )
    return backend()
  return backend


class Engine(interfaces.AbstractEngine):
  """Engine wrapper for the LiteRT-LM C API."""

  def __init__(
      self,
      model_path: str,
      backend: (
          interfaces.Backend | type[interfaces.Backend]
      ) = interfaces.Backend.CPU(),
      max_num_tokens: int | None = None,
      max_num_images: int | None = None,
      cache_dir: str = "",
      vision_backend: (
          interfaces.Backend | type[interfaces.Backend] | None
      ) = None,
      audio_backend: (
          interfaces.Backend | type[interfaces.Backend] | None
      ) = None,
      lora_rank_config: interfaces.LoraRankConfig | None = None,
      activation_data_type: ActivationDataType | None = None,
      **kwargs,
  ):
    backend = _normalize_backend(backend)
    vision_backend = _normalize_backend(vision_backend)
    audio_backend = _normalize_backend(audio_backend)

    super().__init__(
        model_path=model_path,
        backend=backend,
        max_num_tokens=max_num_tokens,
        max_num_images=max_num_images,
        cache_dir=cache_dir,
        vision_backend=vision_backend,
        audio_backend=audio_backend,
        lora_rank_config=lora_rank_config,
        activation_data_type=activation_data_type,
        **kwargs,
    )

    self._lib = _get_lib()
    self._engine_ptr = None

    settings = self._lib.litert_lm_engine_settings_create(
        self.model_path,
        self.backend.get_name(),
        (self.vision_backend.get_name() if self.vision_backend else None),
        (self.audio_backend.get_name() if self.audio_backend else None),
    )

    if (
        isinstance(self.backend, interfaces.Backend.NPU)
        and self.backend.litert_dispatch_lib_dir
    ):
      self._lib.litert_lm_engine_settings_set_litert_dispatch_lib_dir(
          settings, self.backend.litert_dispatch_lib_dir
      )

    if not settings:
      raise RuntimeError(
          f"Failed to create engine settings for {self.model_path}. "
          "Verify the model path and backend."
      )

    if (
        isinstance(self.backend, interfaces.CPU)
        and self.backend.thread_count is not None
    ):
      self._lib.litert_lm_engine_settings_set_num_threads(
          settings, self.backend.thread_count
      )

    if (
        isinstance(self.audio_backend, interfaces.CPU)
        and self.audio_backend.thread_count is not None
    ):
      self._lib.litert_lm_engine_settings_set_audio_num_threads(
          settings, self.audio_backend.thread_count
      )

    if self.max_num_tokens is not None:
      self._lib.litert_lm_engine_settings_set_max_num_tokens(
          settings, self.max_num_tokens
      )
    if self.max_num_images is not None:
      self._lib.litert_lm_engine_settings_set_max_num_images(
          settings, self.max_num_images
      )
    if self.cache_dir:
      self._lib.litert_lm_engine_settings_set_cache_dir(
          settings, self.cache_dir
      )
    if self.enable_speculative_decoding is not None:
      self._lib.litert_lm_engine_settings_set_enable_speculative_decoding(
          settings, self.enable_speculative_decoding
      )
    if self.activation_data_type is not None:
      self._lib.litert_lm_engine_settings_set_activation_data_type(
          settings, self.activation_data_type.value
      )
    lora_rank = (
        self.lora_rank_config.lora_rank if self.lora_rank_config else None
    )
    audio_lora_rank = (
        self.lora_rank_config.audio_lora_rank
        if self.lora_rank_config
        else None
    )

    if lora_rank is not None:
      self._lib.litert_lm_engine_settings_set_lora_rank(settings, lora_rank)
      if lora_rank > 0:
        c_ranks = (ctypes.c_int * 1)(lora_rank)
        status = self._lib.litert_lm_engine_settings_set_supported_lora_ranks(
            settings, c_ranks, 1
        )
        if status != 0:
          raise RuntimeError("Failed to set supported LoRA ranks.")

    if audio_lora_rank is not None:
      self._lib.litert_lm_engine_settings_set_audio_lora_rank(
          settings, audio_lora_rank
      )
      if audio_lora_rank > 0:
        c_ranks = (ctypes.c_int * 1)(audio_lora_rank)
        status = (
            self._lib.litert_lm_engine_settings_set_supported_audio_lora_ranks(
                settings, c_ranks, 1
            )
        )
        if status != 0:
          raise RuntimeError("Failed to set supported audio LoRA ranks.")

    self._engine_ptr = self._lib.litert_lm_engine_create(settings)
    self._lib.litert_lm_engine_settings_delete(settings)

    if not self._engine_ptr:
      raise RuntimeError(
          f"Failed to create LiteRT-LM engine for {self.model_path}"
      )

  def close(self):
    if hasattr(self, "_engine_ptr") and self._engine_ptr and self._lib:
      try:
        self._lib.litert_lm_engine_delete(self._engine_ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._engine_ptr = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def create_conversation(
      self,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any] | Message]
          | None
      ) = None,
      tools: (
          collections.abc.Sequence[
              collections.abc.Callable[..., Any] | interfaces.Tool
          ]
          | None
      ) = None,
      tool_event_handler: interfaces.ToolEventHandler | None = None,
      automatic_tool_calling: bool = True,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      filter_channel_content_from_kv_cache: bool = False,
      sampler_config: interfaces.SamplerConfig | None = None,
      system_message: str | None = None,
      enable_constrained_decoding: bool = False,
      lora_config: interfaces.LoraConfig | None = None,
      max_output_tokens: int | None = None,
  ) -> Conversation:
    session_config = self._lib.litert_lm_session_config_create()
    if sampler_config:
      params = _sampler_config_to_params(self._lib, sampler_config)
      try:
        self._lib.litert_lm_session_config_set_sampler_params(
            session_config, params
        )
      finally:
        self._lib.litert_lm_sampler_params_delete(params)

    lora_path = lora_config.lora_path if lora_config else None
    audio_lora_path = lora_config.audio_lora_path if lora_config else None

    if lora_path:
      status = self._lib.litert_lm_session_config_set_lora_path(
          session_config, lora_path
      )
      if status != 0:
        raise RuntimeError(f"Failed to set LoRA path: {lora_path}")

    if audio_lora_path:
      status = self._lib.litert_lm_session_config_set_audio_lora_path(
          session_config, audio_lora_path
      )
      if status != 0:
        raise RuntimeError(f"Failed to set audio LoRA path: {audio_lora_path}")

    if max_output_tokens is not None:
      self._lib.litert_lm_session_config_set_max_output_tokens(
          session_config, int(max_output_tokens)
      )

    conv_config = self._lib.litert_lm_conversation_config_create()
    if not conv_config:
      raise RuntimeError("Failed to create conversation config")

    self._lib.litert_lm_conversation_config_set_session_config(
        conv_config, session_config
    )
    self._lib.litert_lm_session_config_delete(session_config)

    if system_message:
      self._lib.litert_lm_conversation_config_set_system_message(
          conv_config, system_message
      )

    if messages:
      serialized_messages = [
          m.to_json() if hasattr(m, "to_json") else m for m in messages
      ]
      self._lib.litert_lm_conversation_config_set_messages(
          conv_config, json.dumps(serialized_messages)
      )

    if extra_context:
      self._lib.litert_lm_conversation_config_set_extra_context(
          conv_config, json.dumps(extra_context)
      )

    tools_map = {}
    if tools:
      wrapped_tools = []
      for t in tools:
        if not isinstance(t, interfaces.Tool):
          t = litert_tools.tool_from_function(t)
        wrapped_tools.append(t)
        desc = t.get_tool_description()
        if "function" not in desc or "name" not in desc["function"]:
          raise ValueError(
              "interfaces.Tool description must contain ['function']['name']"
          )
        name = desc["function"]["name"]
        tools_map[name] = t

      tools_json = json.dumps([t.get_tool_description() for t in wrapped_tools])
      self._lib.litert_lm_conversation_config_set_tools(conv_config, tools_json)

    if enable_constrained_decoding:
      self._lib.litert_lm_conversation_config_set_enable_constrained_decoding(
          conv_config, True
      )

    if filter_channel_content_from_kv_cache:
      self._lib.litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
          conv_config, True
      )

    conv_ptr = self._lib.litert_lm_conversation_create(
        self._engine_ptr, conv_config
    )
    self._lib.litert_lm_conversation_config_delete(conv_config)

    if not conv_ptr:
      raise RuntimeError("Failed to create conversation")

    return Conversation(
        self._lib,
        conv_ptr,
        engine=self,
        messages=messages or [],
        tools=tools or [],
        tools_map=tools_map,
        tool_event_handler=tool_event_handler,
        automatic_tool_calling=automatic_tool_calling,
        extra_context=extra_context or {},
        sampler_config=sampler_config,
        lora_config=lora_config,
        max_output_tokens=max_output_tokens,
    )

  def create_session(
      self,
      *,
      apply_prompt_template: bool = True,
      sampler_config: interfaces.SamplerConfig | None = None,
      max_output_tokens: int | None = None,
      lora_config: interfaces.LoraConfig | None = None,
  ) -> Session:
    session_config = self._lib.litert_lm_session_config_create()
    if not session_config:
      raise RuntimeError("Failed to create session config")

    self._lib.litert_lm_session_config_set_apply_prompt_template(
        session_config, apply_prompt_template
    )

    if sampler_config:
      params = _sampler_config_to_params(self._lib, sampler_config)
      try:
        self._lib.litert_lm_session_config_set_sampler_params(
            session_config, params
        )
      finally:
        self._lib.litert_lm_sampler_params_delete(params)

    if max_output_tokens is not None:
      self._lib.litert_lm_session_config_set_max_output_tokens(
          session_config, int(max_output_tokens)
      )

    lora_path = lora_config.lora_path if lora_config else None
    audio_lora_path = lora_config.audio_lora_path if lora_config else None

    if lora_path:
      status = self._lib.litert_lm_session_config_set_lora_path(
          session_config, lora_path
      )
      if status != 0:
        raise RuntimeError(f"Failed to set LoRA path: {lora_path}")

    if audio_lora_path:
      status = self._lib.litert_lm_session_config_set_audio_lora_path(
          session_config, audio_lora_path
      )
      if status != 0:
        raise RuntimeError(f"Failed to set audio LoRA path: {audio_lora_path}")

    sess_ptr = self._lib.litert_lm_engine_create_session(
        self._engine_ptr, session_config
    )
    self._lib.litert_lm_session_config_delete(session_config)

    if not sess_ptr:
      raise RuntimeError("Failed to create session")
    return Session(self._lib, sess_ptr, engine=self)

  @property
  def bos_token_id(self) -> int | None:
    u_ptr = self._lib.litert_lm_engine_get_start_token(self._engine_ptr)
    val = _parse_token_union(self._lib, u_ptr)
    if isinstance(val, int):
      return val
    if isinstance(val, list) and val:
      return val[0]
    return None

  @property
  def eos_token_ids(self) -> list[list[int]]:
    unions_ptr = self._lib.litert_lm_engine_get_stop_tokens(self._engine_ptr)
    if not unions_ptr:
      return []
    try:
      num = self._lib.litert_lm_token_unions_get_num_tokens(unions_ptr)
      all_ids = []
      for i in range(num):
        u_ptr = self._lib.litert_lm_token_unions_get_token_at(unions_ptr, i)
        # _parse_token_union handles deleting the owned LiteRtLmTokenUnion
        # pointer.
        val = _parse_token_union(self._lib, u_ptr)
        if isinstance(val, int):
          all_ids.append([val])
        elif isinstance(val, list):
          all_ids.append(val)
      return all_ids
    finally:
      self._lib.litert_lm_token_unions_delete(unions_ptr)

  def tokenize(self, text: str) -> list[int]:
    res_ptr = self._lib.litert_lm_engine_tokenize(self._engine_ptr, text)
    if not res_ptr:
      raise RuntimeError("Tokenization failed")
    try:
      num = self._lib.litert_lm_tokenize_result_get_num_tokens(res_ptr)
      tokens = self._lib.litert_lm_tokenize_result_get_tokens(res_ptr)
      return [tokens[i] for i in range(num)]
    finally:
      self._lib.litert_lm_tokenize_result_delete(res_ptr)

  def detokenize(self, token_ids: list[int]) -> str:
    num_tokens = len(token_ids)
    c_ids = (ctypes.c_int * num_tokens)(*token_ids)

    res_ptr = self._lib.litert_lm_engine_detokenize(
        self._engine_ptr, c_ids, num_tokens
    )
    if not res_ptr:
      raise RuntimeError("Detokenization failed")

    try:
      resp_str = self._lib.litert_lm_detokenize_result_get_string(res_ptr)
      return resp_str.decode("utf-8") if resp_str else ""
    finally:
      self._lib.litert_lm_detokenize_result_delete(res_ptr)
