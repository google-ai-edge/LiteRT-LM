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
from typing import Any


class Backend(enum.Enum):
  """Hardware backends for LiteRT-LM."""

  UNSPECIFIED = 0
  CPU = 3
  GPU = 4


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


class Tool(abc.ABC):
  """A tool that can be executed."""

  @abc.abstractmethod
  def get_tool_description(self) -> dict[str, Any]:
    """Returns a JSON representing the tool in OpenAPI schema."""

  @abc.abstractmethod
  def execute(self, param: collections.abc.Mapping[str, Any]) -> Any:
    """Executes the underlying function and returns the result.

    Args:
        param: A dictionary containing the parameters for the tool.

    Returns:
        The result of the tool execution.
    """


@dataclasses.dataclass
class SamplerConfig:
  """Configuration for the sampling process.

  Attributes:
      top_k: The number of top logits used during sampling.
      top_p: The cumulative probability threshold for nucleus sampling.
      temperature: The temperature to use for sampling.
      seed: The seed to use for randomization. Defaults to None.
  """

  top_k: int | None = None
  top_p: float | None = None
  temperature: float | None = None
  seed: int | None = None

  def __post_init__(self):
    if self.top_k is not None and self.top_k <= 0:
      raise ValueError(f"top_k should be positive, but got {self.top_k}.")
    if self.top_p is not None and not (0 <= self.top_p <= 1):
      raise ValueError(
          f"top_p should between 0 and 1 inclusively, but got {self.top_p}."
      )
    if self.temperature is not None and self.temperature < 0:
      raise ValueError(
          f"temperature should be non-negative, but got {self.temperature}."
      )


@dataclasses.dataclass(kw_only=True)
class AbstractEngine(abc.ABC):
  """Abstract base class for LiteRT-LM engines.

  Attributes:
      model_path: Path to the model file.
      backend: The hardware backend used for inference.
      max_num_tokens: Maximum number of tokens for the KV cache. If None, use
        the engine/model's default.
      cache_dir: Directory for caching compiled model artifacts.
      vision_backend: The hardware backend used for vision encoding.
      audio_backend: The hardware backend used for audio encoding.
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default. If True, enable speculative decoding; an
        error will be thrown if the model does not support it. If False, disable
        it.
      bos_token_id: The BOS token id for the model if one is configured.
      eos_token_ids: Stop token sequences configured for the model.
  """

  model_path: str
  backend: Backend
  max_num_tokens: int | None = None
  cache_dir: str = ""
  vision_backend: Backend | None = None
  audio_backend: Backend | None = None
  enable_speculative_decoding: bool | None = None

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
          collections.abc.Sequence[collections.abc.Callable[..., Any] | Tool]
          | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
      automatic_tool_calling: bool = True,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      filter_channel_content_from_kv_cache: bool = False,
      sampler_config: SamplerConfig | None = None,
  ) -> AbstractConversation:
    """Creates a new conversation for this engine.

    Args:
        messages: A sequence of messages for the conversation preface. Each
          message is a mapping that should contain 'role' and 'content' keys.
        tools: A list of Python functions or Tool instances to be used as tools.
        tool_event_handler: A handler for tool call and tool response events.
        automatic_tool_calling: Whether to automatically call tools. If False,
          tool calls will be returned to the user to execute.
        extra_context: Extra context for the conversation.
        filter_channel_content_from_kv_cache: Whether to filter channel content
          from the KV cache. This is useful when the model responds with
          "channel" content, e.g. thinking/reasoning tokens, that should not be
          persisted in the KV cache.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.
    """

  @abc.abstractmethod
  def create_session(
      self,
      *,
      apply_prompt_template: bool = True,
      sampler_config: SamplerConfig | None = None,
  ) -> AbstractSession:
    """Creates a new session for this engine.

    Args:
        apply_prompt_template: Whether to apply the basic prompt templates in
          the session.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.

    Returns:
        A new session instance for low-level interaction with the model.
    """

  @property
  @abc.abstractmethod
  def bos_token_id(self) -> int | None:
    """Returns the configured BOS token id for the model, if any."""

  @property
  @abc.abstractmethod
  def eos_token_ids(self) -> list[list[int]]:
    """Returns the configured EOS/stop token sequences for the model."""

  @abc.abstractmethod
  def tokenize(self, text: str) -> list[int]:
    """Tokenizes text using the engine's tokenizer."""

  @abc.abstractmethod
  def detokenize(self, token_ids: list[int]) -> str:
    """Decodes token ids using the engine's tokenizer."""


class AbstractConversation(abc.ABC):
  """Abstract base class for managing LiteRT-LM conversations.

  Attributes:
      messages: A sequence of messages for the conversation preface.
      tools: A list of Python functions or Tool instances to be used as tools.
      tool_event_handler: A handler for tool call and tool response events.
      automatic_tool_calling: Whether to automatically call tools.
      extra_context: Extra context for the chat template.
      sampler_config: Configuration for the sampling process.
  """

  def __init__(
      self,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any]] | None
      ) = None,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any] | Tool]
          | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
      automatic_tool_calling: bool = True,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      sampler_config: SamplerConfig | None = None,
  ):
    """Initializes the instance.

    Args:
        messages: A sequence of messages for the conversation preface. Each
          message is a mapping that should contain 'role' and 'content' keys.
        tools: A list of Python functions or Tool instances to be used as tools.
        tool_event_handler: A handler for tool call and tool response events.
        automatic_tool_calling: Whether to automatically call tools. If False,
          tool calls will be returned to the user to execute.
        extra_context: Extra context for the chat template.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.
    """
    self.messages = messages or []
    self.tools = tools or []
    self.tool_event_handler = tool_event_handler
    self.automatic_tool_calling = automatic_tool_calling
    self.extra_context = extra_context or {}
    self.sampler_config = sampler_config

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
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default. If True, enable speculative decoding; an
        error will be thrown if the model does not support it. If False, disable
        it.
      bos_token_id: The BOS token id for the model if one is configured.
      eos_token_ids: Stop token sequences configured for the model.
  """

  model_path: str
  backend: Backend
  prefill_tokens: int = 256
  decode_tokens: int = 256
  cache_dir: str = ""
  enable_speculative_decoding: bool | None = None

  @abc.abstractmethod
  def run(self) -> BenchmarkInfo:
    """Runs the benchmark and returns the result."""


@dataclasses.dataclass
class Responses:
  """A container to host the model responses.

  This class is only used in the Session API. "Batch size" is the number of
  parallel response processed in decode. Most models have batch size equals 1.

  Attributes:
      texts: The generated text(s) from the model. The list length is equal to
        the batch size in "run_decode".  This field is only used in
        "run_decode". "run_text_scoring".
      scores: The scores associated with the generated text(s). The list length
        is equal to length of the "target_text" in "run_text_scoring" or the
        batch size in "run_decode".
      token_lengths: The number of tokens in each generated text. The list
        length is equal to length of the "target_text" in "run_text_scoring".
        This field is only used in `run_text_scoring` when `store_token_lengths`
        is True.
  """

  texts: list[str] = dataclasses.field(default_factory=list)
  scores: list[float] = dataclasses.field(default_factory=list)
  token_lengths: list[int] = dataclasses.field(default_factory=list)


# TODO(b/482060476): Add clone() API once switching to advanced engine.
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

    TODO(b/482060476): Support multi-modality in contents.

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
  def run_decode_async(self) -> collections.abc.Iterator[Responses]:
    """Runs the decode stage of the session asynchronously.

    Returns:
        An iterator yielding chunks of the generated response (Responses).
    """

  @abc.abstractmethod
  def run_text_scoring(
      self, target_text: list[str], store_token_lengths: bool = False
  ) -> Responses:
    """Runs the scoring stage of the session.

    Args:
        target_text: A list of target strings to score.
        store_token_lengths: Whether to store the token lengths of the target
          texts in the result. If True, the token lengths will be included in
          the return value: `Responses`. Otherwise, it will be None.

    Returns:
        Responses: The log likelihood scores of the target text given the
        existing session state.
    """

  @abc.abstractmethod
  def cancel_process(self) -> None:
    """Cancels the ongoing inference process."""
