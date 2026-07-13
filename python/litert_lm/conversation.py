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
"""Conversation wrapper for LiteRT-LM."""

from __future__ import annotations

import collections.abc
import ctypes
import json
import logging
import queue
from typing import Any

from . import interfaces
from ._ffi import STREAM_CALLBACK_TYPE
from ._messages import Contents
from ._messages import Message
from ._messages import normalize_message
from .utils import thinking_config_to_params


class Conversation(interfaces.AbstractConversation):
  """Conversation wrapper for the LiteRT-LM C API."""

  def __init__(
      self,
      lib,
      conv_ptr,
      engine=None,
      messages=None,
      tools=None,
      tools_map=None,
      tool_event_handler=None,
      automatic_tool_calling=True,
      extra_context=None,
      thinking_config=None,
      sampler_config=None,
      lora_config=None,
      max_output_tokens=None,
  ):
    super().__init__(
        messages=messages,
        tools=tools,
        tool_event_handler=tool_event_handler,
        automatic_tool_calling=automatic_tool_calling,
        extra_context=extra_context,
        thinking_config=thinking_config,
        sampler_config=sampler_config,
        lora_config=lora_config,
        max_output_tokens=max_output_tokens,
    )
    self._lib = lib
    self._ptr = conv_ptr
    self._engine = engine  # Keep engine alive
    self._tools_map = tools_map or {}
    # Keep the active ctypes callback alive to prevent SIGSEGV if the C++ thread
    # calls it after the local variable is garbage collected during
    # cancellation.
    self._current_callback = None

  def close(self):
    if hasattr(self, "_ptr") and self._ptr and self._lib:
      try:
        self._lib.litert_lm_conversation_delete(self._ptr)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._ptr = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def _handle_tool_calls(
      self, response_dict: collections.abc.Mapping[str, Any]
  ) -> list[collections.abc.Mapping[str, Any]] | None:
    if "tool_calls" not in response_dict:
      return None

    tool_responses = []
    for tool_call in response_dict.get("tool_calls"):
      if "function" not in tool_call:
        raise ValueError("Missing 'function' in tool_call")
      function = tool_call.get("function")
      name = function.get("name", "")
      args = function.get("arguments", {})

      if self.tool_event_handler:
        if not self.tool_event_handler.approve_tool_call(tool_call):
          continue

      tool = self._tools_map.get(name)
      if not tool:
        result = f"Error: interfaces.Tool {name} not found"
      else:
        try:
          result = tool.execute(args)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.exception("interfaces.Tool execution failed: %s", name)
          result = f"Error: {str(e)}"

      if self.tool_event_handler:
        result = self.tool_event_handler.process_tool_response(result)

      tool_responses.append({
          "role": "tool",
          "content": [{
              "type": "tool_response",
              "name": name,
              "response": result,
          }],
      })

    return tool_responses

  def _create_optional_args(
      self,
      repetition_penalty_config: interfaces.RepetitionPenaltyConfig | None,
      no_repeat_ngram_config: interfaces.NoRepeatNgramConfig | None,
      max_output_tokens: int | None,
      thinking_config: interfaces.ThinkingConfig | None,
  ) -> ctypes.c_void_p | None:
    """Creates a C pointer for ConversationOptionalArgs if needed."""
    if (
        repetition_penalty_config is None
        and no_repeat_ngram_config is None
        and max_output_tokens is None
        and thinking_config is None
    ):
      return None
    ptr = self._lib.litert_lm_conversation_optional_args_create()
    try:
      if repetition_penalty_config is not None:
        rpp_ptr = self._lib.litert_lm_repetition_penalty_config_create()
        try:
          if repetition_penalty_config.repetition_penalty is not None:
            self._lib.litert_lm_repetition_penalty_config_set_repetition_penalty(
                rpp_ptr, repetition_penalty_config.repetition_penalty
            )
          if repetition_penalty_config.presence_penalty is not None:
            self._lib.litert_lm_repetition_penalty_config_set_presence_penalty(
                rpp_ptr, repetition_penalty_config.presence_penalty
            )
          if repetition_penalty_config.frequency_penalty is not None:
            self._lib.litert_lm_repetition_penalty_config_set_frequency_penalty(
                rpp_ptr, repetition_penalty_config.frequency_penalty
            )
          if repetition_penalty_config.window_size is not None:
            self._lib.litert_lm_repetition_penalty_config_set_window_size(
                rpp_ptr, repetition_penalty_config.window_size
            )
          self._lib.litert_lm_conversation_optional_args_set_repetition_penalty_config(
              ptr, rpp_ptr
          )
        finally:
          if rpp_ptr:
            self._lib.litert_lm_repetition_penalty_config_delete(rpp_ptr)
      if no_repeat_ngram_config is not None:
        nrn_ptr = self._lib.litert_lm_no_repeat_ngram_config_create()
        try:
          if no_repeat_ngram_config.no_repeat_ngram_size is not None:
            self._lib.litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
                nrn_ptr, no_repeat_ngram_config.no_repeat_ngram_size
            )
          if no_repeat_ngram_config.window_size is not None:
            self._lib.litert_lm_no_repeat_ngram_config_set_window_size(
                nrn_ptr, no_repeat_ngram_config.window_size
            )
          self._lib.litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
              ptr, nrn_ptr
          )
        finally:
          if nrn_ptr:
            self._lib.litert_lm_no_repeat_ngram_config_delete(nrn_ptr)
      if max_output_tokens is not None:
        self._lib.litert_lm_conversation_optional_args_set_max_output_tokens(
            ptr, max_output_tokens
        )
      if thinking_config is not None:
        tc_ptr = thinking_config_to_params(self._lib, thinking_config)
        try:
          self._lib.litert_lm_conversation_optional_args_set_thinking_config(
              ptr, tc_ptr
          )
        finally:
          if tc_ptr:
            self._lib.litert_lm_thinking_config_delete(tc_ptr)
      return ptr
    except Exception:
      self._lib.litert_lm_conversation_optional_args_delete(ptr)
      raise

  def _delete_optional_args(self, ptr: ctypes.c_void_p | None) -> None:
    """Deletes the ConversationOptionalArgs C pointer."""
    if ptr:
      self._lib.litert_lm_conversation_optional_args_delete(ptr)

  # TODO: b/482060476 - Change the return type to "Message".
  def send_message(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
      *,
      repetition_penalty_config: (
          interfaces.RepetitionPenaltyConfig | None
      ) = None,
      no_repeat_ngram_config: interfaces.NoRepeatNgramConfig | None = None,
      max_output_tokens: int | None = None,
      thinking_config: interfaces.ThinkingConfig | None = None,
  ) -> collections.abc.Mapping[str, Any]:
    """See base class."""
    if not self._ptr:
      raise RuntimeError("Conversation is closed.")
    current_message = normalize_message(message)

    while True:
      msg_json = json.dumps(current_message)
      ctx_json = json.dumps(getattr(self, "extra_context", {}))

      optional_args_ptr = self._create_optional_args(
          repetition_penalty_config,
          no_repeat_ngram_config,
          max_output_tokens,
          thinking_config,
      )
      try:
        resp_ptr = self._lib.litert_lm_conversation_send_message(
            self._ptr,
            msg_json,
            ctx_json,
            optional_args_ptr,
        )
        if not resp_ptr:
          raise RuntimeError("litert_lm_conversation_send_message failed")

        try:
          resp_str = self._lib.litert_lm_json_response_get_string(resp_ptr)
          response_dict = (
              json.loads(resp_str.decode("utf-8")) if resp_str else {}
          )
        finally:
          self._lib.litert_lm_json_response_delete(resp_ptr)
      finally:
        self._delete_optional_args(optional_args_ptr)

      if not self.automatic_tool_calling:
        return response_dict

      tool_responses = self._handle_tool_calls(response_dict)
      if not tool_responses:
        return response_dict

      current_message = tool_responses

  def send_message_async(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
      *,
      repetition_penalty_config: (
          interfaces.RepetitionPenaltyConfig | None
      ) = None,
      no_repeat_ngram_config: interfaces.NoRepeatNgramConfig | None = None,
      max_output_tokens: int | None = None,
      thinking_config: interfaces.ThinkingConfig | None = None,
  ) -> collections.abc.Iterator[collections.abc.Mapping[str, Any]]:
    """See base class."""
    if not self._ptr:
      raise RuntimeError("Conversation is closed.")
    current_message = normalize_message(message)

    while True:
      msg_json = json.dumps(current_message)
      ctx_json = json.dumps(getattr(self, "extra_context", {}))

      q = queue.Queue()

      def callback(unused_data, chunk_ptr):
        error_msg = self._lib.litert_lm_stream_chunk_get_error(chunk_ptr)
        if error_msg:
          q.put(RuntimeError(error_msg.decode("utf-8")))
        else:
          chunk = self._lib.litert_lm_stream_chunk_get_text(chunk_ptr)
          is_final = self._lib.litert_lm_stream_chunk_is_final(chunk_ptr)
          q.put((chunk.decode("utf-8") if chunk else "", is_final))

      c_callback = STREAM_CALLBACK_TYPE(callback)
      self._current_callback = c_callback

      optional_args_ptr = self._create_optional_args(
          repetition_penalty_config,
          no_repeat_ngram_config,
          max_output_tokens,
          thinking_config,
      )
      try:
        res = self._lib.litert_lm_conversation_send_message_stream(
            self._ptr,
            msg_json,
            ctx_json,
            optional_args_ptr,
            c_callback,
            None,
        )
      finally:
        self._delete_optional_args(optional_args_ptr)

      if res != 0:
        raise RuntimeError("litert_lm_conversation_send_message_stream failed")

      full_response_for_tools = None
      while True:
        item = q.get()
        if isinstance(item, Exception):
          err_msg = str(item)
          if (
              "CANCELLED" in err_msg
              or "Max number of tokens reached" in err_msg
          ):
            break
          raise item
        chunk_str, is_final = item
        if chunk_str:
          try:
            msg_dict = json.loads(chunk_str)
            if self.automatic_tool_calling:
              # If it's a tool call, we don't yield it yet.
              is_tool_call = "tool_calls" in msg_dict
              if not is_tool_call:
                contents = msg_dict.get("content", [])
                if not isinstance(contents, list):
                  contents = [contents]
                is_tool_call = any(
                    isinstance(c, dict) and c.get("type") == "tool_call"
                    for c in contents
                )

              if is_tool_call:
                full_response_for_tools = msg_dict
              else:
                yield msg_dict
            else:
              yield msg_dict
          except json.JSONDecodeError:
            yield {
                "role": "assistant",
                "content": [{"type": "text", "text": chunk_str}],
            }
        if is_final:
          break

      if not full_response_for_tools:
        break

      tool_responses = self._handle_tool_calls(full_response_for_tools)
      if not tool_responses:
        break
      current_message = tool_responses

  def render_message_to_string(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
  ) -> str:
    if not self._ptr:
      return ""
    msg_json = normalize_message(message)
    res_str = self._lib.litert_lm_conversation_render_message_to_string(
        self._ptr, json.dumps(msg_json)
    )
    return res_str.decode("utf-8") if res_str else ""

  def get_benchmark_info(self) -> interfaces.BenchmarkInfo:
    """See base class."""
    if not self._ptr:
      raise RuntimeError("Conversation is closed.")
    info_ptr = self._lib.litert_lm_conversation_get_benchmark_info(self._ptr)
    if not info_ptr:
      raise RuntimeError("Failed to get benchmark info.")
    try:
      return interfaces.create_benchmark_info(self._lib, info_ptr)
    finally:
      self._lib.litert_lm_benchmark_info_delete(info_ptr)

  def cancel_process(self) -> None:
    if self._ptr:
      self._lib.litert_lm_conversation_cancel_process(self._ptr)

  @property
  def token_count(self) -> int:
    """See base class."""
    if not self._ptr:
      raise RuntimeError("Conversation is closed.")
    res = self._lib.litert_lm_conversation_get_token_count(self._ptr)
    if res == -1:
      raise RuntimeError("Failed to get token count.")
    return res
