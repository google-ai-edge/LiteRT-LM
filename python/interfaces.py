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


@dataclasses.dataclass
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
  max_num_tokens: int = 512
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
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any]] | None
      ) = None,
  ) -> AbstractConversation:
    """Creates a new conversation for this engine."""

  @abc.abstractmethod
  def create_session(self) -> AbstractSession:
    """Creates a new session for this engine.

    Returns:
        A new session instance for low-level interaction with the model.
    """


class AbstractConversation(abc.ABC):
  """Abstract base class for managing LiteRT-LM conversations."""

  def __init__(
      self,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any]] | None
      ) = None,
  ):
    """Initializes the instance.

    Args:
        tools: A list of Python functions to be used as tools.
    """
    self.tools = tools or []

  def __enter__(self) -> AbstractConversation:
    """Initializes the conversation."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the conversation."""
    del exc_type, exc_val, exc_tb

  @abc.abstractmethod
  def send_message(self, message: str | dict[str, Any]) -> dict[str, Any]:
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
      self, message: str | dict[str, Any]
  ) -> collections.abc.Iterator[dict[str, Any]]:
    """Sends a message and streams the response.

    Args:
        message: The input message to send to the model. Example: "Hello" or
          {"role": "user", "content": "Hello"}.

    Returns:
        An iterator yielding dictionaries containing chunks of the model's
        response.
    """


@dataclasses.dataclass
class Responses:
  """A container to host the model responses.

  Note: This class is only used in the Session API.

  Attributes:
      texts: The generated text(s) from the model.
      scores: The scores associated with the generated text(s).
      token_lengths: The number of tokens in each generated text.
  """

  texts: list[str]
  scores: list[float]
  token_lengths: list[int] | None = None


class AbstractSession(abc.ABC):
  """Abstract base class for managing LiteRT-LM sessions."""

  def __init__(self):
    """Initializes the instance."""

  def __enter__(self) -> AbstractSession:
    """Initializes the session."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the session."""
    del exc_type, exc_val, exc_tb

  @abc.abstractmethod
  def run_prefill(self, contents: list[str]) -> None:
    """Runs the prefill stage of the session.

    Args:
        contents: A list of input strings to prefill the model with. Note that
          the user can break down their prompt/query into multiple chunks and
          call this function multiple times.
    """

  @abc.abstractmethod
  def run_decode(self) -> Responses:
    """Runs the decode stage of the session.

    Returns:
        The generated response from the model based on the input prompt/query
        added after using run_prefill.
    """

  @abc.abstractmethod
  def run_text_scoring(
      self, target_text: list[str], store_token_lengths: bool = False
  ) -> Responses:
    """Runs the scoring stage of the session.

    Args:
        target_text: A list of target strings to score.
        store_token_lengths: Whether to store the token lengths of the target
          texts in the result.

    Returns:
        Responses: The log likelihood scores of the target text given the
        existing session state.
    """
