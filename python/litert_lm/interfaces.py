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
from importlib import resources
import logging
import os
import sys
from typing import Any

from ._ffi import ActivationDataType
from ._messages import Contents
from ._messages import Message


class Backend(abc.ABC):
  """Hardware backends for LiteRT-LM.

  This is the abstract base class for all hardware backends used by LiteRT-LM.
  Use the subclasses (CPU, GPU, NPU) to specify the backend and its options.
  """

  def get_name(self) -> str:
    """Returns the string representation of the backend (e.g., 'cpu', 'gpu', 'npu')."""
    return type(self).__name__.lower()

  def __eq__(self, other: Any) -> bool:
    if type(self) is not type(other):
      return NotImplemented
    return True


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPU(Backend):
  """CPU hardware backend for LiteRT-LM.

  Attributes:
    thread_count: The number of threads to use for CPU backend.
  """

  thread_count: int | None = None


class GPU(Backend):
  """GPU hardware backend for LiteRT-LM."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class NPU(Backend):
  """NPU hardware backend for LiteRT-LM.

  Attributes:
    litert_dispatch_lib_dir: The directory containing LiteRT dispatch libs.
  """

  litert_dispatch_lib_dir: str | None = None

  def __post_init__(self):
    """Initializes the NPU backend.

    Raises:
      RuntimeError: If the NPU backend is not supported on the current platform,
        if the 'openvino' package fails to import, or if no NPU device is
        detected.
    """
    if self.litert_dispatch_lib_dir == "":  # pylint: disable=g-explicit-bool-comparison
      logging.warning(
          "NPU backend is initialized with an empty litert_dispatch_lib_dir."
          " This means the model will be simulated on the CPU, where the model"
          " is expected to NOT be AOT compiled."
      )
      object.__setattr__(self, "litert_dispatch_lib_dir", "")
      return
    elif self.litert_dispatch_lib_dir is None:
      if sys.platform != "win32":
        raise RuntimeError(
            "NPU is supported only for Intel OpenVINO on Windows. Current"
            f" platform is '{sys.platform}'."
        )

      try:
        import openvino as ov  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
      except ImportError as e:
        raise RuntimeError(
            "NPU is supported only for Intel OpenVINO on Windows. Failed to"
            " import the 'openvino' package. Please ensure 'openvino' is"
            " installed."
        ) from e

      available_devices = ov.Core().available_devices
      if "NPU" not in available_devices:
        raise RuntimeError(
            "NPU is supported only for Intel OpenVINO on Windows. No NPU"
            " device detected by OpenVINO (available devices:"
            f" {available_devices})."
        )

      litert_dispatch_lib_dir = str(
          resources.files(__package__) / "vendors/intel_openvino/dispatch/"
      )
      object.__setattr__(
          self, "litert_dispatch_lib_dir", litert_dispatch_lib_dir
      )

      # openvino package place the NPU libs in "libs".
      # Includes to PATH so Windows can load it.
      libs_dir = os.path.join(os.path.dirname(ov.__file__), "libs")
      os.environ["PATH"] = os.environ["PATH"] + ";" + libs_dir

    if not self.litert_dispatch_lib_dir:
      raise RuntimeError(
          "NPU backend could not be initialized because an invalid or"
          f" empty litert_dispatch_lib_dir ({self.litert_dispatch_lib_dir!r})"
          " was provided."
      )

  def __eq__(self, other: Any) -> bool:
    if type(self) is not type(other):
      return NotImplemented
    return self.litert_dispatch_lib_dir == other.litert_dispatch_lib_dir


Backend.CPU = CPU
Backend.GPU = GPU
Backend.NPU = NPU


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
class ThinkingConfig:
  """Configuration for thinking/reasoning generation.

  Attributes:
      enable_thinking: Whether thinking is enabled.
      thinking_token_budget: Budget for token-by-token reasoning generation.
        Defaults to -1 (infinite budget).
  """

  enable_thinking: bool = True
  thinking_token_budget: int = -1


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


@dataclasses.dataclass
class RepetitionPenaltyConfig:
  """Configuration for penalizing repetitive tokens during generation.

  Attributes:
      repetition_penalty: A multiplicative penalty for any token already
        generated (e.g., 1.0 = no penalty, 1.2 = moderate penalty). Positive
        logits are divided by this penalty, and negative logits are multiplied.
      presence_penalty: A scalar subtracted from a logit if a token has appeared
        at least once.
      frequency_penalty: A scalar subtracted from a logit, scaled linearly by
        the number of times that token has previously appeared.
      window_size: The maximum number of recent tokens to consider. A value of 0
        means track all infinite history.
  """

  repetition_penalty: float | None = None
  presence_penalty: float | None = None
  frequency_penalty: float | None = None
  window_size: int | None = None

  def __post_init__(self):
    if self.repetition_penalty is not None and self.repetition_penalty < 1.0:
      raise ValueError(
          "repetition_penalty should be >= 1.0, but got"
          f" {self.repetition_penalty}."
      )
    if self.window_size is not None and self.window_size < 0:
      raise ValueError(
          f"window_size should be >= 0, but got {self.window_size}."
      )


@dataclasses.dataclass
class NoRepeatNgramConfig:
  """Configuration for banning repetitive ngrams during generation.

  Attributes:
      no_repeat_ngram_size: The size of ngrams to ban. If set to an integer
        greater than 0, all ngrams of that size can only occur once.
      window_size: The maximum number of recent tokens to consider for banning.
        A value of 0 means track all infinite history.
  """

  no_repeat_ngram_size: int | None = None
  window_size: int | None = None

  def __post_init__(self):
    if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size < 0:
      raise ValueError(
          "no_repeat_ngram_size should be >= 0, but got"
          f" {self.no_repeat_ngram_size}."
      )
    if self.window_size is not None and self.window_size < 0:
      raise ValueError(
          f"window_size should be >= 0, but got {self.window_size}."
      )


@dataclasses.dataclass
class LoraRankConfig:
  """Configuration for LoRA ranks.

  Attributes:
      lora_rank: The rank of the text LoRA weights. If 0 or None, LoRA is
        disabled.
      audio_lora_rank: The rank of the audio LoRA weights. If 0 or None, audio
        LoRA is disabled.
  """

  lora_rank: int | None = None
  audio_lora_rank: int | None = None


@dataclasses.dataclass
class LoraConfig:
  """Configuration for LoRA weights.

  Attributes:
      lora_path: Path to the text LoRA weights file.
      audio_lora_path: Path to the audio LoRA weights file.
  """

  lora_path: str | None = None
  audio_lora_path: str | None = None


@dataclasses.dataclass(kw_only=True)
class AbstractEngine(abc.ABC):
  """Abstract base class for LiteRT-LM engines.

  Attributes:
      model_path: Path to the model file.
      backend: The hardware backend used for inference.
      max_num_tokens: Maximum number of tokens for the KV cache. If None, use
        the engine/model's default.
      max_num_images: Maximum number of images that can be processed in a single
        inference call.
      cache_dir: Directory for caching compiled model artifacts.
      vision_backend: The hardware backend used for vision encoding.
      audio_backend: The hardware backend used for audio encoding.
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default. If True, enable speculative decoding; an
        error will be thrown if the model does not support it. If False, disable
        it.
      lora_rank_config: Configuration for LoRA ranks.
      bos_token_id: The BOS token id for the model if one is configured.
      eos_token_ids: Stop token sequences configured for the model.
      activation_data_type: The activation data type used for model execution.
  """

  model_path: str
  backend: Backend
  max_num_tokens: int | None = None
  max_num_images: int | None = None
  cache_dir: str = ""
  vision_backend: Backend | None = None
  audio_backend: Backend | None = None
  enable_speculative_decoding: bool | None = None
  lora_rank_config: LoraRankConfig | None = None
  activation_data_type: ActivationDataType | None = None

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
          collections.abc.Sequence[collections.abc.Mapping[str, Any] | Message]
          | None
      ) = None,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any] | Tool]
          | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
      automatic_tool_calling: bool = True,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      filter_channel_content_from_kv_cache: bool = False,
      thinking_config: ThinkingConfig | None = None,
      sampler_config: SamplerConfig | None = None,
      lora_config: LoraConfig | None = None,
      max_output_tokens: int | None = None,
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
        thinking_config: Configuration for thinking/reasoning generation.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.
        lora_config: Configuration for LoRA adapters.
        max_output_tokens: The maximum number of output tokens.
    """

  @abc.abstractmethod
  def create_session(
      self,
      *,
      apply_prompt_template: bool = True,
      sampler_config: SamplerConfig | None = None,
      lora_config: LoraConfig | None = None,
  ) -> AbstractSession:
    """Creates a new session for this engine.

    Args:
        apply_prompt_template: Whether to apply the basic prompt templates in
          the session.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.
        lora_config: Configuration for LoRA adapters.

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
      thinking_config: Configuration for thinking/reasoning generation.
      sampler_config: Configuration for the sampling process.
      lora_config: Configuration for LoRA adapters.
      max_output_tokens: The maximum number of output tokens.
  """

  def __init__(
      self,
      *,
      messages: (
          collections.abc.Sequence[collections.abc.Mapping[str, Any] | Message]
          | None
      ) = None,
      tools: (
          collections.abc.Sequence[collections.abc.Callable[..., Any] | Tool]
          | None
      ) = None,
      tool_event_handler: ToolEventHandler | None = None,
      automatic_tool_calling: bool = True,
      extra_context: collections.abc.Mapping[str, Any] | None = None,
      thinking_config: ThinkingConfig | None = None,
      sampler_config: SamplerConfig | None = None,
      lora_config: LoraConfig | None = None,
      max_output_tokens: int | None = None,
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
        thinking_config: Configuration for thinking/reasoning generation.
        sampler_config: Configuration for the sampling process. If None, then
          uses the engine's default values.
        lora_config: Configuration for LoRA adapters.
        max_output_tokens: The maximum number of output tokens.
    """
    self.messages = messages or []
    self.tools = tools or []
    self.tool_event_handler = tool_event_handler
    self.automatic_tool_calling = automatic_tool_calling
    self.extra_context = extra_context or {}
    self.thinking_config = thinking_config
    self.sampler_config = sampler_config
    self.lora_config = lora_config
    self.max_output_tokens = max_output_tokens

  def __enter__(self) -> AbstractConversation:
    """Initializes the conversation."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the conversation."""
    del exc_type, exc_val, exc_tb

  @abc.abstractmethod
  def send_message(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
      *,
      repetition_penalty_config: RepetitionPenaltyConfig | None = None,
      no_repeat_ngram_config: NoRepeatNgramConfig | None = None,
      max_output_tokens: int | None = None,
      thinking_config: ThinkingConfig | None = None,
  ) -> collections.abc.Mapping[str, Any]:
    """Sends a message and returns the response.

    Args:
        message: The input message to send. Supported types are: `str` (for most
          simple text input, automatically wrapped as a user message),
          `Contents` (for multi-modality interleaving, automatically wrapped as
          a user message), `Message` (full message object, useful when automatic
          tool calling is disabled and a tool response is required), or
          `collections.abc.Mapping` (super flexible raw dictionary format).
        repetition_penalty_config: Configuration for penalizing repetitive
          tokens.
        no_repeat_ngram_config: Configuration for banning repetitive ngrams.
        max_output_tokens: The maximum number of output tokens.
        thinking_config: Configuration for thinking/reasoning generation.

    Returns:
        A dictionary containing the model's response. The structure is:
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    """

  @abc.abstractmethod
  def send_message_async(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
      *,
      repetition_penalty_config: RepetitionPenaltyConfig | None = None,
      no_repeat_ngram_config: NoRepeatNgramConfig | None = None,
      max_output_tokens: int | None = None,
      thinking_config: ThinkingConfig | None = None,
  ) -> collections.abc.Iterator[collections.abc.Mapping[str, Any]]:
    """Sends a message and streams the response.

    Args:
        message: The input message to send. Supported types are: `str` (for most
          simple text input, automatically wrapped as a user message),
          `Contents` (for multi-modality interleaving, automatically wrapped as
          a user message), `Message` (full message object, useful when automatic
          tool calling is disabled and a tool response is required), or
          `collections.abc.Mapping` (super flexible raw dictionary format).
        repetition_penalty_config: Configuration for penalizing repetitive
          tokens.
        no_repeat_ngram_config: Configuration for banning repetitive ngrams.
        max_output_tokens: The maximum number of output tokens.
        thinking_config: Configuration for thinking/reasoning generation.

    Returns:
        An iterator yielding dictionaries containing chunks of the model's
        response.
    """

  @abc.abstractmethod
  def render_message_to_string(
      self,
      message: str | Contents | Message | collections.abc.Mapping[str, Any],
  ) -> str:
    """Renders a message into a string according to the template.

    Args:
        message: The input message to render. Supported types are: `str` (for
          most simple text input, automatically wrapped as a user message),
          `Contents` (for multi-modality interleaving, automatically wrapped as
          a user message), `Message` (full message object, useful when automatic
          tool calling is disabled and a tool response is required), or
          `collections.abc.Mapping` (super flexible raw dictionary format).

    Returns:
        The rendered string.
    """

  @property
  @abc.abstractmethod
  def token_count(self) -> int:
    """The number of tokens in the KV Cache (prefill + decode)."""

  @abc.abstractmethod
  def get_benchmark_info(self) -> BenchmarkInfo:
    """Returns the benchmark info of the conversation."""

  def cancel_process(self) -> None:
    """Cancels the current inference process."""


@dataclasses.dataclass
class BenchmarkInfo:
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


def create_benchmark_info(lib: Any, info_ptr: Any) -> BenchmarkInfo:
  """Creates a BenchmarkInfo object from a C API pointer."""
  num_prefill_turns = lib.litert_lm_benchmark_info_get_num_prefill_turns(
      info_ptr
  )
  if num_prefill_turns > 0:
    last_prefill_count = (
        lib.litert_lm_benchmark_info_get_prefill_token_count_at(
            info_ptr, num_prefill_turns - 1
        )
    )
    last_prefill_tps = (
        lib.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
            info_ptr, num_prefill_turns - 1
        )
    )
  else:
    last_prefill_count = 0
    last_prefill_tps = 0.0

  num_decode_turns = lib.litert_lm_benchmark_info_get_num_decode_turns(info_ptr)
  if num_decode_turns > 0:
    last_decode_count = lib.litert_lm_benchmark_info_get_decode_token_count_at(
        info_ptr, num_decode_turns - 1
    )
    last_decode_tps = lib.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
        info_ptr, num_decode_turns - 1
    )
  else:
    last_decode_count = 0
    last_decode_tps = 0.0

  return BenchmarkInfo(
      init_time_in_second=lib.litert_lm_benchmark_info_get_total_init_time_in_second(
          info_ptr
      ),
      time_to_first_token_in_second=lib.litert_lm_benchmark_info_get_time_to_first_token(
          info_ptr
      ),
      last_prefill_token_count=last_prefill_count,
      last_prefill_tokens_per_second=last_prefill_tps,
      last_decode_token_count=last_decode_count,
      last_decode_tokens_per_second=last_decode_tps,
  )


@dataclasses.dataclass
class AbstractBenchmark(abc.ABC):
  """Abstract base class for LiteRT-LM benchmarks.

  Attributes:
      model_path: Path to the model file.
      backend: The hardware backend used for inference.
      prefill_tokens: Number of tokens for the prefill phase.
      decode_tokens: Number of tokens for the decode phase.
      max_num_tokens: Maximum number of tokens for the KV cache. If None, use
        the engine/model's default.
      cache_dir: Directory for caching compiled model artifacts.
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default. If True, enable speculative decoding; an
        error will be thrown if the model does not support it. If False, disable
        it.
      bos_token_id: The BOS token id for the model if one is configured.
      eos_token_ids: Stop token sequences configured for the model.
      prompt: The custom prompt string to tokenize and run. If the tokenized
        prompt is shorter than `prefill_tokens`, the remaining tokens are padded
        with zero. If it is longer, the prompt is truncated to `prefill_tokens`.
      activation_data_type: The activation data type used for model execution.
  """

  model_path: str
  backend: Backend
  prefill_tokens: int = 256
  decode_tokens: int = 256
  max_num_tokens: int | None = None
  cache_dir: str = ""
  enable_speculative_decoding: bool | None = None
  prompt: str = "How are you"
  activation_data_type: ActivationDataType | None = None

  @abc.abstractmethod
  def run(self) -> BenchmarkInfo:
    """Runs the benchmark and returns the result."""


@dataclasses.dataclass
class Responses:
  """A container to host the model responses.

  This class is only used in the Session API. "Batch size" is the number of
  parallel response processed in decode. Most models have batch size equals 1.

  Attributes:
      texts: The generated text(s) from the model in "run_decode", or the target
        text(s) in "run_text_scoring". The list length is equal to the batch
        size in "run_decode" or the length of "target_text" in
        "run_text_scoring".
      scores: The scores associated with the generated text(s). The list length
        is equal to length of the "target_text" in "run_text_scoring" or the
        batch size in "run_decode".
      token_lengths: The number of tokens in each generated text. The list
        length is equal to length of the "target_text" in "run_text_scoring".
        This field is only used in `run_text_scoring` when `store_token_lengths`
        is True.
      token_scores: The log likelihood scores of the target text given the
        existing session state. The list length is equal to length of the
        "target_text" in "run_text_scoring". The inner list contains the log
        likelihood score for each token in the corresponding "target_text"
        element. This field is only used in `run_text_scoring`.
  """

  texts: list[str] = dataclasses.field(default_factory=list)
  scores: list[float] = dataclasses.field(default_factory=list)
  token_lengths: list[int] = dataclasses.field(default_factory=list)
  token_scores: list[list[float]] = dataclasses.field(default_factory=list)


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
  def get_benchmark_info(self) -> BenchmarkInfo:
    """Returns the benchmark info of the session."""

  @abc.abstractmethod
  def cancel_process(self) -> None:
    """Cancels the ongoing inference process."""
