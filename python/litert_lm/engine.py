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

import collections.abc
import ctypes
import json
from typing import Any

from . import interfaces
from . import tools as litert_tools
from ._ffi import _get_lib
from ._ffi import TokenUnionType
from .conversation import Conversation
from .session import Session
from .utils import _parse_token_union
from .utils import _sampler_config_to_params


class Engine(interfaces.AbstractEngine):
  """Engine wrapper for the LiteRT-LM C API."""

  def __init__(
      self,
      model_path: str,
      backend: interfaces.Backend = interfaces.Backend.CPU,
      max_num_tokens: int | None = None,
      cache_dir: str = "",
      **kwargs,
  ):
    super().__init__(
        model_path=model_path,
        backend=backend,
        max_num_tokens=max_num_tokens,
        cache_dir=cache_dir,
        **kwargs,
    )
    self._lib = _get_lib()
    self._engine_ptr = None

    settings = self._lib.litert_lm_engine_settings_create(
        self.model_path,
        self.backend.name.lower(),
        (self.vision_backend.name.lower() if self.vision_backend else None),
        (self.audio_backend.name.lower() if self.audio_backend else None),
    )

    if not settings:
      raise RuntimeError(
          f"Failed to create engine settings for {self.model_path}. "
          "Verify the model path and backend."
      )

    if self.max_num_tokens is not None:
      self._lib.litert_lm_engine_settings_set_max_num_tokens(
          settings, self.max_num_tokens
      )
    if self.cache_dir:
      self._lib.litert_lm_engine_settings_set_cache_dir(
          settings, self.cache_dir
      )
    if self.enable_speculative_decoding is not None:
      self._lib.litert_lm_engine_settings_set_enable_speculative_decoding(
          settings, self.enable_speculative_decoding
      )

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
          collections.abc.Sequence[collections.abc.Mapping[str, Any]] | None
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
  ) -> Conversation:
    session_config = self._lib.litert_lm_session_config_create()
    if sampler_config:
      params = _sampler_config_to_params(sampler_config)
      self._lib.litert_lm_session_config_set_sampler_params(
          session_config, ctypes.byref(params)
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
      self._lib.litert_lm_conversation_config_set_messages(
          conv_config, json.dumps(messages)
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
    )

  def create_session(
      self,
      *,
      apply_prompt_template: bool = True,
      sampler_config: interfaces.SamplerConfig | None = None,
      max_output_tokens: int | None = None,
  ) -> Session:
    session_config = self._lib.litert_lm_session_config_create()
    if not session_config:
      raise RuntimeError("Failed to create session config")

    self._lib.litert_lm_session_config_set_apply_prompt_template(
        session_config, apply_prompt_template
    )

    if sampler_config:
      params = _sampler_config_to_params(sampler_config)
      self._lib.litert_lm_session_config_set_sampler_params(
          session_config, ctypes.byref(params)
      )

    if max_output_tokens is not None:
      self._lib.litert_lm_session_config_set_max_output_tokens(
          session_config, int(max_output_tokens)
      )

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
        # _parse_token_union handles deleting the owned LiteRtLmTokenUnion pointer.
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
