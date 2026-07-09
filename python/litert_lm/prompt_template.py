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
"""PromptTemplate wrapper for LiteRT-LM."""

import collections.abc
import ctypes
import json
from typing import Any
from typing_extensions import override
from . import interfaces


class PromptTemplate(interfaces.AbstractPromptTemplate):
  """PromptTemplate wrapper for the LiteRT-LM C API."""

  def __init__(self, lib: Any, ptr: ctypes.c_void_p):
    super().__init__()
    self._lib = lib
    self._ptr = ptr

  def __del__(self) -> None:
    self.close()

  def close(self) -> None:
    """Closes and releases the underlying C API prompt template resources."""
    if hasattr(self, "_ptr") and self._ptr:
      self._lib.litert_lm_prompt_template_delete(self._ptr)
      self._ptr = None

  @override
  def render(
      self,
      input_data: (
          collections.abc.Mapping[str, Any]
          | collections.abc.Sequence[collections.abc.Mapping[str, Any]]
          | None
      ) = None,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any]] | None
      ) = None,
      tools: collections.abc.Sequence[Any] | None = None,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      add_generation_prompt: bool = True,
  ) -> str:
    """Renders the prompt template given input messages and tools."""
    if not self._ptr:
      raise RuntimeError("PromptTemplate is closed or invalid.")

    payload: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
    }

    if isinstance(input_data, collections.abc.Mapping):
      if "add_generation_prompt" in input_data:
        payload["add_generation_prompt"] = input_data["add_generation_prompt"]
      if "messages" in input_data and messages is None:
        messages = input_data["messages"]
      if "tools" in input_data and tools is None:
        tools = input_data["tools"]
      if "extra_context" in input_data and extra_context is None:
        extra_context = input_data["extra_context"]
    elif isinstance(input_data, collections.abc.Sequence) and not isinstance(
        input_data, (str, bytes)
    ):
      if messages is None:
        messages = input_data

    if messages is not None:
      normalized_messages = []
      for msg in messages:
        if hasattr(msg, "__dict__") and not isinstance(msg, dict):
          try:
            normalized_messages.append(dict(msg))  # type: ignore
          except (TypeError, ValueError):
            normalized_messages.append(getattr(msg, "__dict__", {}))
        else:
          normalized_messages.append(msg)
      payload["messages"] = normalized_messages

    if tools is not None:
      normalized_tools = []
      for t in tools:
        if hasattr(t, "get_tool_description"):
          normalized_tools.append(t.get_tool_description())
        elif isinstance(t, collections.abc.Mapping):
          normalized_tools.append(t)
        elif callable(t):
          normalized_tools.append(getattr(t, "description", str(t)))
        else:
          normalized_tools.append(t)
      payload["tools"] = normalized_tools

    if extra_context is not None:
      payload["extra_context"] = dict(extra_context)

    input_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    res_ptr = self._lib.litert_lm_prompt_template_render(self._ptr, input_json)
    if not res_ptr:
      raise RuntimeError("Failed to render prompt template.")
    raw_bytes = getattr(res_ptr, "value", res_ptr)
    if raw_bytes is None:
      raise RuntimeError("Failed to render prompt template (null output).")
    return raw_bytes.decode("utf-8")
