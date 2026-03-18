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

"""Interfaces for LiteRT LM engines and conversations."""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import enum
import pathlib
from typing import Any


class Backend(enum.Enum):
  """Hardware backends for LiteRT-LM."""

  UNSPECIFIED = 0
  CPU = 3
  GPU = 4
  NPU = 6


class ToolEventHandler(abc.ABC):
  """Handler for tool call and tool response events."""

  @abc.abstractmethod
  def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
    """Handles a tool call.

    Args:
        tool_call: The tool call JSON, including the tool name and args.

    Returns:
        True to allow the tool call, False to disallow.
    """

  @abc.abstractmethod
  def process_tool_response(
      self, tool_response: dict[str, Any]
  ) -> dict[str, Any]:
    """Handles a tool response.

    This allows the user to clean up or modify the response before it is sent
    to the model (e.g., stripping away sensitive content).

    Args:
        tool_response: The tool response.

    Returns:
        The tool response that will be sent to the model.
    """


@dataclasses.dataclass(kw_only=True)
class AbstractEngine(abc.ABC):
  """Abstract base class for LiteRT-LM engines.

  Attributes:
      model_path: Path to the model file.
      backend: The hardware backend used for inference.
      max_num_tokens: Maximum number of tokens for the KV cache.
      cache_dir: Directory for caching compiled model artifacts.
  """

  model_path: str
  backend: Backend
  max_num_tokens: int = 4096
  cache_dir: str = ""

  def __enter__(self) -> AbstractEngine:
    """Initializes the engine resources."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the engine resources."""
    del exc_type, exc_val, exc_tb

  @abc.abstractmethod
  def create_conversation(
      self,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any]] | None
      ) = None,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any]] | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
  ) -> AbstractConversation:
    """Creates a new conversation for this engine.

    Args:
        messages: A sequence of messages for the conversation preface. Each
          message is a mapping that should contain 'role' and 'content' keys.
        tools: A list of Python functions to be used as tools.
        tool_event_handler: A handler for tool call and tool response events.
    """


class AbstractConversation(abc.ABC):
  """Abstract base class for managing LiteRT-LM conversations.

  Attributes:
      messages: A sequence of messages for the conversation preface.
      tools: A list of Python functions to be used as tools.
      tool_event_handler: A handler for tool call and tool response events.
  """

  def __init__(
      self,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any]] | None
      ) = None,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any]] | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
  ):
    """Initializes the instance.

    Args:
        messages: A sequence of messages for the conversation preface. Each
          message is a mapping that should contain 'role' and 'content' keys.
        tools: A list of Python functions to be used as tools.
        tool_event_handler: A handler for tool call and tool response events.
    """
    self.messages = messages or []
    self.tools = tools or []
    self.tool_event_handler = tool_event_handler

  def __enter__(self) -> AbstractConversation:
    """Initializes the conversation."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the conversation."""
    del exc_type, exc_val, exc_tb

  @abc.abstractmethod
  def send_message(
      self, message: str | collections.abc.Mapping[str, Any]
  ) -> collections.abc.Mapping[str, Any]:
    """Sends a message and returns the response.

    Args:
        message: The input message to send to the model. Example: "Hello" or
          {"role": "user", "content": "Hello"}.

    Returns:
        A dictionary containing the model's response. The structure is:
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    """

  @abc.abstractmethod
  def send_message_async(
      self, message: str | collections.abc.Mapping[str, Any]
  ) -> collections.abc.Iterator[collections.abc.Mapping[str, Any]]:
    """Sends a message and streams the response.

    Args:
        message: The input message to send to the model. Example: "Hello" or
          {"role": "user", "content": "Hello"}.

    Returns:
        An iterator yielding dictionaries containing chunks of the model's
        response.
    """

  def cancel_process(self) -> None:
    """Cancels the current inference process."""


@dataclasses.dataclass
class BenchmarkInfo(abc.ABC):
  """Results from a benchmark run.

  Attributes:
      init_time_in_second: The time in seconds to initialize the engine and the
        conversation.
      time_to_first_token_in_second: The time in seconds to the first token.
      last_prefill_token_count: The number of tokens in the last prefill.
      last_prefill_tokens_per_second: The number of tokens processed per second
        in the last prefill.
      last_decode_token_count: The number of tokens in the last decode.
      last_decode_tokens_per_second: The number of tokens processed per second
        in the last decode.
  """

  init_time_in_second: float
  time_to_first_token_in_second: float
  last_prefill_token_count: int
  last_prefill_tokens_per_second: float
  last_decode_token_count: int
  last_decode_tokens_per_second: float


@dataclasses.dataclass
class AbstractBenchmark(abc.ABC):
  """Abstract base class for LiteRT-LM benchmarks.

  Attributes:
      model_path: Path to the model file.
      backend: The hardware backend used for inference.
      prefill_tokens: Number of tokens for the prefill phase.
      decode_tokens: Number of tokens for the decode phase.
      cache_dir: Directory for caching compiled model artifacts.
  """

  model_path: str
  backend: Backend
  prefill_tokens: int = 256
  decode_tokens: int = 256
  cache_dir: str = ""

  @abc.abstractmethod
  def run(self) -> BenchmarkInfo:
    """Runs the benchmark and returns the result."""
