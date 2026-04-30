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

import collections.abc
import json
import logging
import queue
from typing import Any

from . import interfaces
from ._ffi import STREAM_CALLBACK_TYPE


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
      sampler_config=None,
  ):
    super().__init__(
        messages=messages,
        tools=tools,
        tool_event_handler=tool_event_handler,
        automatic_tool_calling=automatic_tool_calling,
        extra_context=extra_context,
        sampler_config=sampler_config,
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
    extracted_tool_calls = []

    # Support top-level tool_calls field
    if "tool_calls" in response_dict:
      tc_list = response_dict["tool_calls"]
      if isinstance(tc_list, list):
        for tc in tc_list:
          if isinstance(tc, dict):
            # If it's a direct tool call object (OpenAI-like)
            if tc.get("type") == "function" and "function" in tc:
              extracted_tool_calls.append(tc["function"])
            else:
              extracted_tool_calls.append(tc)

    # Support tool calls inside content list (multimodal format)
    contents = response_dict.get("content", [])
    if not isinstance(contents, list):
      contents = [contents]

    for content in contents:
      if isinstance(content, dict) and content.get("type") == "tool_call":
        extracted_tool_calls.append(content["tool_call"])

    if not extracted_tool_calls:
      return None

    tool_responses = []
    for tool_call in extracted_tool_calls:
      name = tool_call.get("name")
      args = tool_call.get("arguments", {})
      call_id = tool_call.get("id")

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

      tool_response = {
          "role": "tool",
          "content": [{"name": name, "response": result}],
      }
      if call_id:
        tool_response["tool_call_id"] = call_id
      if name:
        tool_response["name"] = name

      if self.tool_event_handler:
        tool_response = self.tool_event_handler.process_tool_response(
            tool_response
        )
      tool_responses.append(tool_response)

    return tool_responses

  def send_message(
      self, message: str | collections.abc.Mapping[str, Any]
  ) -> collections.abc.Mapping[str, Any]:
    current_message = (
        message
        if isinstance(message, dict)
        else {"role": "user", "content": message}
    )

    while True:
      msg_json = json.dumps(current_message)
      ctx_json = json.dumps(getattr(self, "extra_context", {}))

      resp_ptr = self._lib.litert_lm_conversation_send_message(
          self._ptr, msg_json, ctx_json
      )
      if not resp_ptr:
        raise RuntimeError("litert_lm_conversation_send_message failed")

      try:
        resp_str = self._lib.litert_lm_json_response_get_string(resp_ptr)
        response_dict = json.loads(resp_str.decode("utf-8")) if resp_str else {}
      finally:
        self._lib.litert_lm_json_response_delete(resp_ptr)

      if not self.automatic_tool_calling:
        return response_dict

      tool_responses = self._handle_tool_calls(response_dict)
      if not tool_responses:
        return response_dict

      current_message = tool_responses

  def send_message_async(
      self, message: str | collections.abc.Mapping[str, Any]
  ) -> collections.abc.Iterator[collections.abc.Mapping[str, Any]]:
    current_message = (
        message
        if isinstance(message, dict)
        else {"role": "user", "content": message}
    )

    while True:
      msg_json = json.dumps(current_message)
      ctx_json = json.dumps(getattr(self, "extra_context", {}))

      q = queue.Queue()

      def callback(unused_data, chunk, is_final, error_msg):
        if error_msg:
          q.put(RuntimeError(error_msg.decode("utf-8")))
        else:
          q.put((chunk.decode("utf-8") if chunk else "", is_final))

      c_callback = STREAM_CALLBACK_TYPE(callback)
      self._current_callback = c_callback
      res = self._lib.litert_lm_conversation_send_message_stream(
          self._ptr,
          msg_json,
          ctx_json,
          c_callback,
          None,
      )
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
      self, message: str | collections.abc.Mapping[str, Any]
  ) -> str:
    msg_json = (
        message
        if isinstance(message, dict)
        else {"role": "user", "content": message}
    )
    res_str = self._lib.litert_lm_conversation_render_message_to_string(
        self._ptr, json.dumps(msg_json)
    )
    return res_str.decode("utf-8") if res_str else ""

  def cancel_process(self) -> None:
    if self._ptr:
      self._lib.litert_lm_conversation_cancel_process(self._ptr)
