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

import http.server
import socket

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
    self.litert_lm_embedding_engine: litert_lm.EmbeddingEngine | None = None
    self.embedding_model_id: str | None = None
    self.embedding_backend: litert_lm.Backend | None = None
    self.embedding_vision_backend: litert_lm.Backend | None = None
    self.embedding_audio_backend: litert_lm.Backend | None = None


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
