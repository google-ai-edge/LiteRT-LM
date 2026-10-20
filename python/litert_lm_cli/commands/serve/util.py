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

"""Shared core utilities for managing LiteRT-LM serving lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import http.server
import socket
from typing import Any

import click

import litert_lm
from litert_lm_builder import litertlm_builder
from litert_lm_cli import common
from litert_lm_cli import model


class LiteRTLMServer(http.server.HTTPServer):
  """Custom HTTP server tracking persistent LiteRT-LM engine lifecycles.

  Attributes:
    litert_lm_engine: The LiteRT-LM engine instance, or None if not initialized.
    model_id: The identifier of the model currently loaded in the engine, or
      None.
    backend: The hardware backend used by the current engine, or None.
    max_num_tokens: The maximum number of tokens configured for the current
      engine, or None.
    vision_backend: The hardware backend used for vision encoding, or None.
    audio_backend: The hardware backend used for audio encoding, or None.
    activation_data_type: The activation data type used for model execution, or
      None.
    litert_lm_conversation: The currently cached conversation instance, or None.
    conversation_cache_key: The cache key tuple for the cached conversation, or
      None.
    litert_lm_embedding_engine: The LiteRT-LM embedding engine instance, or None
      if not initialized.
    embedding_model_id: The identifier of the embedding model currently loaded,
      or None.
    embedding_backend: The hardware backend used by the embedding engine, or
      None.
    embedding_vision_backend: The hardware backend used for embedding vision
      encoding, or None.
    embedding_audio_backend: The hardware backend used for embedding audio
      encoding, or None.
    allowed_origins: Allowed CORS origins.
    address_family: Socket address family (e.g. AF_INET or AF_INET6).
  """

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
      allowed_origins: tuple[str, ...] = (),
  ):
    host, _ = server_address
    if ":" in host:
      self.address_family = socket.AF_INET6
    super().__init__(server_address, RequestHandlerClass)
    self.allowed_origins = allowed_origins
    self.litert_lm_engine: litert_lm.Engine | None = None
    self.model_id: str | None = None
    self.backend: litert_lm.Backend | None = None
    self.max_num_tokens: int | None = None
    self.vision_backend: litert_lm.Backend | None = None
    self.audio_backend: litert_lm.Backend | None = None
    self.activation_data_type: litert_lm.ActivationDataType | None = None
    self.litert_lm_conversation: litert_lm.AbstractConversation | None = None
    self.conversation_cache_key: tuple[Any, ...] | None = None
    self.litert_lm_embedding_engine: litert_lm.EmbeddingEngine | None = None
    self.embedding_model_id: str | None = None
    self.embedding_backend: litert_lm.Backend | None = None
    self.embedding_vision_backend: litert_lm.Backend | None = None
    self.embedding_audio_backend: litert_lm.Backend | None = None

  def close_conversation(self) -> None:
    """Closes and resets the currently cached conversation, if any."""
    if self.litert_lm_conversation is not None:
      self.litert_lm_conversation.__exit__(None, None, None)
      self.litert_lm_conversation = None
    self.conversation_cache_key = None


class CORSRequestHandler(http.server.BaseHTTPRequestHandler):
  """Base HTTP request handler that adds CORS headers to all responses."""

  def end_headers(self) -> None:
    origin = self.headers.get("Origin")
    allowed_origins = getattr(self.server, "allowed_origins", ())

    has_cors = False
    if "*" in allowed_origins:
      self.send_header("Access-Control-Allow-Origin", "*")
      has_cors = True
    elif origin and origin in allowed_origins:
      self.send_header("Access-Control-Allow-Origin", origin)
      self.send_header("Vary", "Origin")
      has_cors = True

    if has_cors:
      self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
      self.send_header(
          "Access-Control-Allow-Headers",
          "Content-Type, Authorization, X-Requested-With",
      )
    super().end_headers()

  def do_OPTIONS(self) -> None:  # pylint: disable=invalid-name
    self.send_response(200)
    self.end_headers()


def get_or_initialize_server_engine(
    server: LiteRTLMServer,
    *,
    model_id: str,
    backend: str | None = None,
    max_num_tokens: int | None = None,
) -> litert_lm.Engine:
  """Retrieves the persistent server engine or initializes it on first request.

  Lifetime Management:
  The LiteRT-LM Engine is a globally scoped persistent resource attached
  directly to explicit runtime properties on the custom server context object.
  - Initialization: Invokes `__enter__` dynamically upon the arrival of the
    first incoming inference request.
  - Termination: The running server's parent execution process is responsible
    for explicitly invoking `__exit__` on `server.litert_lm_engine` during outer
    context teardown loops (e.g., in `run_server` finally blocks).

  Args:
    server: The active custom LiteRTLMServer instance object.
    model_id: The requested model identifier string.
    backend: Optional requested backend override (e.g. 'cpu', 'gpu', 'npu').
    max_num_tokens: Optional requested max_num_tokens override.

  Returns:
    The shared LiteRT-LM Engine context object.

  Raises:
    FileNotFoundError: If the model package path does not exist.
  """
  m = model.Model.from_model_reference(model_id)

  if not m.exists():
    raise FileNotFoundError(f"Model {model_id} not found")

  resolved_backend = model.parse_backend(backend, model_obj=m)
  vision_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          litertlm_builder.TfLiteModelType.VISION_ENCODER.value,
      },
      label="vision",
  )
  audio_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
      },
      label="audio",
  )
  resolved_max_num_tokens = model.resolve_config_option(
      max_num_tokens, m, "max_num_tokens"
  )
  cache = model.resolve_config_option(None, m, "cache")
  cache_dir_val = common.cache_dir_value_from_cache_mode(cache)
  speculative_decoding = model.resolve_config_option(
      None, m, "speculative_decoding"
  )
  activation_data_type_str = model.resolve_config_option(
      None, m, "activation_data_type"
  )
  activation_data_type = (
      litert_lm.ActivationDataType.from_str(activation_data_type_str)
      if activation_data_type_str
      else None
  )

  if server.litert_lm_engine is not None:
    if (
        server.model_id == model_id
        and server.backend == resolved_backend
        and server.max_num_tokens == resolved_max_num_tokens
        and server.vision_backend == vision_backend
        and server.audio_backend == audio_backend
        and server.activation_data_type == activation_data_type
    ):
      return server.litert_lm_engine

    click.echo(
        click.style(
            f"Re-initializing engine (model: {model_id}, backend:"
            f" {resolved_backend}, max_num_tokens: {resolved_max_num_tokens})",
            fg="yellow",
        )
    )
    # TODO: b/513076049 - Support multiple concurrent engines instead of
    # re-initializing (which is disruptive to other clients).
    server.close_conversation()
    server.litert_lm_engine.__exit__(None, None, None)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None
    server.vision_backend = None
    server.audio_backend = None
    server.activation_data_type = None

  click.echo(
      click.style(f"Initializing engine for model: {m.model_path}", fg="cyan")
  )
  engine = litert_lm.Engine(
      m.model_path,
      backend=resolved_backend,  # pyrefly: ignore[bad-argument-type]
      max_num_tokens=resolved_max_num_tokens,
      vision_backend=vision_backend,
      audio_backend=audio_backend,
      cache_dir=cache_dir_val,
      enable_speculative_decoding=speculative_decoding,
      activation_data_type=activation_data_type,
      enable_benchmark=True,
      use_ringbuffers_local_attention=True,
  )
  engine.__enter__()
  server.litert_lm_engine = engine
  server.model_id = model_id
  server.backend = resolved_backend
  server.max_num_tokens = resolved_max_num_tokens
  server.vision_backend = vision_backend
  server.audio_backend = audio_backend
  server.activation_data_type = activation_data_type
  return engine


def get_or_initialize_server_embedding_engine(
    server: LiteRTLMServer,
    *,
    model_id: str,
    backend: str | None = None,
) -> litert_lm.EmbeddingEngine:
  """Retrieves the persistent server embedding engine or initializes it on first request.

  Args:
    server: The active custom LiteRTLMServer instance object.
    model_id: The requested model identifier string.
    backend: Optional requested backend override (e.g. 'cpu', 'gpu', 'npu').

  Returns:
    The shared LiteRT-LM EmbeddingEngine context object.

  Raises:
    FileNotFoundError: If the model package path does not exist.
  """
  m = model.Model.from_model_reference(model_id)

  if not m.exists():
    raise FileNotFoundError(f"Model {model_id} not found")

  resolved_backend = model.parse_backend(backend, model_obj=m)
  vision_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          litertlm_builder.TfLiteModelType.VISION_ENCODER.value,
      },
      label="vision",
  )
  audio_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
      },
      label="audio",
  )
  cache = model.resolve_config_option(None, m, "cache")
  cache_dir_val = common.cache_dir_value_from_cache_mode(cache)

  if server.litert_lm_embedding_engine is not None:
    if (
        server.embedding_model_id == model_id
        and server.embedding_backend == resolved_backend
        and server.embedding_vision_backend == vision_backend
        and server.embedding_audio_backend == audio_backend
    ):
      return server.litert_lm_embedding_engine

    click.echo(
        click.style(
            f"Re-initializing embedding engine (model: {model_id}, backend:"
            f" {resolved_backend})",
            fg="yellow",
        )
    )
    server.litert_lm_embedding_engine.close()
    server.litert_lm_embedding_engine = None
    server.embedding_model_id = None
    server.embedding_backend = None
    server.embedding_vision_backend = None
    server.embedding_audio_backend = None

  click.echo(
      click.style(
          f"Initializing embedding engine for model: {m.model_path}", fg="cyan"
      )
  )
  engine = litert_lm.EmbeddingEngine(
      m.model_path,
      backend=resolved_backend,  # pyrefly: ignore[bad-argument-type]
      vision_backend=vision_backend,
      audio_backend=audio_backend,
      cache_dir=cache_dir_val,
  )
  server.litert_lm_embedding_engine = engine
  server.embedding_model_id = model_id
  server.embedding_backend = resolved_backend
  server.embedding_vision_backend = vision_backend
  server.embedding_audio_backend = audio_backend
  return engine


@contextlib.contextmanager
def get_or_create_server_conversation(
    server: Any,
    engine: litert_lm.Engine,
    *,
    context_messages: list[dict[str, Any]],
    prompt: str | dict[str, Any],
    tools: list[Any] | None,
    sampler_config: litert_lm.SamplerConfig | None,
    thinking_config: litert_lm.ThinkingConfig | None,
    constrained_decoding_config: litert_lm.ConstrainedDecodingConfig | None,
) -> Iterator[tuple[Any, Callable[[dict[str, Any] | None], None]]]:
  """Yields a cached or newly created Conversation and a save callback.

  On each turn, the caller splits the incoming request's messages into
  `context_messages` (`translated_messages[:-1]`, representing all prior turns)
  and `prompt` (`translated_messages[-1]`, the new user or tool message for the
  current turn). If `context_messages` and the decoding configuration match the
  `conversation_cache_key` saved at the end of the previous turn
  (`[*prev_context_messages, prev_prompt, prev_assistant]`), the existing
  conversation is reused.

  Args:
    server: The HTTP server instance (cached when it is a `LiteRTLMServer`).
    engine: The active LiteRT-LM Engine instance.
    context_messages: Prior conversation messages excluding the current turn's
      prompt (`translated_messages[:-1]`).
    prompt: The current turn's user or tool message (`translated_messages[-1]`).
    tools: Optional list of tool definitions (`_ProxyTool` instances).
    sampler_config: Optional sampler configuration.
    thinking_config: Optional thinking configuration.
    constrained_decoding_config: Optional constrained decoding configuration.

  Yields:
    A tuple `(conv, save_assistant)` where `conv` is the LiteRT-LM Conversation
    and `save_assistant` records the generated assistant message so the cache
    key can be advanced to `[*context_messages, prompt, assistant_msg]`.
  """
  cache_server = server if isinstance(server, LiteRTLMServer) else None
  cache_key = (
      context_messages,
      tools,
      sampler_config,
      thinking_config,
      constrained_decoding_config,
  )
  if (
      cache_server is not None
      and cache_server.litert_lm_conversation is not None
      and cache_server.conversation_cache_key == cache_key
  ):
    click.echo(
        click.style(
            "Conversation cache hit (reusing conversation with"
            f" {len(context_messages)} context messages)",
            fg="green",
        )
    )
    conv = cache_server.litert_lm_conversation
  else:
    click.echo(
        click.style(
            "Conversation cache miss (initializing new conversation with"
            f" {len(context_messages)} context messages)",
            fg="cyan",
        )
    )
    if cache_server is not None:
      cache_server.close_conversation()
    conv = engine.create_conversation(
        messages=context_messages,
        tools=tools,
        automatic_tool_calling=False,
        sampler_config=sampler_config,
        thinking_config=thinking_config,
        constrained_decoding_config=constrained_decoding_config,
    ).__enter__()
    if cache_server is not None:
      cache_server.litert_lm_conversation = conv

  saved_assistant: list[dict[str, Any] | None] = [None]

  def save_assistant(assistant_msg: dict[str, Any] | None) -> None:
    saved_assistant[0] = assistant_msg

  try:
    yield conv, save_assistant
  except Exception:
    if cache_server is not None:
      cache_server.close_conversation()
    else:
      conv.__exit__(None, None, None)
    raise
  else:
    if cache_server is not None:
      if saved_assistant[0] is not None:
        cache_server.conversation_cache_key = (
            [*context_messages, prompt, saved_assistant[0]],
            tools,
            sampler_config,
            thinking_config,
            constrained_decoding_config,
        )
      else:
        cache_server.close_conversation()
    else:
      conv.__exit__(None, None, None)
