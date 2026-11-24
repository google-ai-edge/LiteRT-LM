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
import socketserver
import threading
import time
from typing import Any

import click

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import model
try:
  # PyPI: litert-lm-builder bundles the protos under its own package.
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
except ImportError:
  # Bazel: the protos are importable from the workspace root.
  from runtime.proto import llm_metadata_pb2


# Errors raised when the client hung up before the reply was written (for
# example a health check that timed out while a long generation held the
# engine). Nothing can be sent back, so they are not server errors.
CLIENT_GONE_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)

# Engine construction maps the whole model and allocates the KV cache. Workers
# load one at a time so start-up and reloads never need twice the memory peak.
_ENGINE_INIT_LOCK = threading.Lock()


class EngineSlot:
  """One extra LLM engine (with its own cached conversation) of a worker pool.

  Worker 0 is the `LiteRTLMServer` itself, so a server started with one worker
  behaves exactly as before. Slots carry the same engine attributes as the
  server, which lets `get_or_initialize_server_engine` and
  `get_or_create_server_conversation` work on either.
  """

  def __init__(self, index: int):
    self.index = index
    self.busy = False
    self.last_used = time.monotonic()
    self.litert_lm_engine: litert_lm.Engine | None = None
    self.model_id: str | None = None
    self.backend: litert_lm.Backend | None = None
    self.max_num_tokens: int | None = None
    self.vision_backend: litert_lm.Backend | None = None
    self.audio_backend: litert_lm.Backend | None = None
    self.activation_data_type: litert_lm.ActivationDataType | None = None
    self.engine_cache_key: tuple[str, str | None, int | None] | None = None
    self.litert_lm_conversation: litert_lm.AbstractConversation | None = None
    self.conversation_cache_key: tuple[Any, ...] | None = None

  def close_conversation(self) -> None:
    """Closes and resets the currently cached conversation, if any."""
    if self.litert_lm_conversation is not None:
      self.litert_lm_conversation.__exit__(None, None, None)
      self.litert_lm_conversation = None
    self.conversation_cache_key = None


def unload_engine(slot: Any) -> str | None:
  """Closes a worker's conversation and engine, freeing their memory.

  Args:
    slot: The `LiteRTLMServer` (worker 0) or an `EngineSlot`.

  Returns:
    The model id that was unloaded, or None when no engine was loaded.
  """
  engine = slot.litert_lm_engine
  if engine is None:
    return None
  model_id = slot.model_id
  slot.close_conversation()
  slot.litert_lm_engine = None
  slot.model_id = None
  slot.backend = None
  slot.max_num_tokens = None
  slot.vision_backend = None
  slot.audio_backend = None
  slot.activation_data_type = None
  slot.engine_cache_key = None
  engine.__exit__(None, None, None)
  return model_id


class LiteRTLMServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  """Custom HTTP server tracking persistent LiteRT-LM engine lifecycles.

  Every request runs on its own thread, so `GET /v1/models` and `GET /health`
  answer at once even while a long generation is running. Generations run on a
  pool of `workers` engines: each request holds one worker for its whole
  duration and waits when all of them are busy. Worker 0 is the server object
  itself (the attributes below); further workers are `EngineSlot`s.

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
    engine_cache_key: The cache key tuple (model_id, backend, max_num_tokens)
      for the currently loaded engine, or None.
    litert_lm_conversation: The currently cached conversation instance, or None.
    conversation_cache_key: The cache key tuple for the cached conversation, or
      None.
    litert_lm_embedding_engine: The LiteRT-LM embedding engine instance, or None
      if not initialized.
    embedding_model_id: The identifier of the embedding model currently loaded,
      or None.
    embedding_backend: The hardware backend used by the embedding engine, or
      None.
    embedding_max_num_tokens: The maximum input token length configured for the
      embedding engine, or None.
    embedding_vision_backend: The hardware backend used for embedding vision
      encoding, or None.
    embedding_audio_backend: The hardware backend used for embedding audio
      encoding, or None.
    embedding_engine_cache_key: The cache key tuple (model_id, backend,
      max_num_tokens) for the currently loaded embedding engine, or None.
    allowed_origins: Allowed CORS origins.
    address_family: Socket address family (e.g. AF_INET or AF_INET6).
    slots: The worker pool; `slots[0]` is this server.
    idle_unload_seconds: Engines that served no request for this long are
      unloaded (0 keeps them loaded).
    verbose: Whether health-check requests are written to the access log.
  """

  daemon_threads = True
  block_on_close = False
  # socketserver's default backlog of 5 refuses bursts of connections (for
  # example health checks arriving while requests queue for a worker).
  request_queue_size = 128

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
      allowed_origins: tuple[str, ...] = (),
      *,
      workers: int = 1,
      idle_unload_seconds: float = 0,
      verbose: bool = False,
  ):
    host, _ = server_address
    if ":" in host:
      self.address_family = socket.AF_INET6
    super().__init__(server_address, RequestHandlerClass)
    self.allowed_origins = allowed_origins
    self.index = 0
    self.busy = False
    self.last_used = time.monotonic()
    self.slots: list[Any] = [self]
    self.slots.extend(EngineSlot(i) for i in range(1, max(1, workers)))
    self.waiting = 0
    self._slot_cond = threading.Condition()
    self.embedding_lock = threading.Lock()
    self.embedding_last_used = time.monotonic()
    self.idle_unload_seconds = max(0.0, float(idle_unload_seconds))
    self.verbose = verbose
    self._stop_event = threading.Event()
    self._background: list[threading.Thread] = []
    self.litert_lm_engine: litert_lm.Engine | None = None
    self.model_id: str | None = None
    self.backend: litert_lm.Backend | None = None
    self.max_num_tokens: int | None = None
    self.vision_backend: litert_lm.Backend | None = None
    self.audio_backend: litert_lm.Backend | None = None
    self.activation_data_type: litert_lm.ActivationDataType | None = None
    self.engine_cache_key: tuple[str, str | None, int | None] | None = None
    self.litert_lm_conversation: litert_lm.AbstractConversation | None = None
    self.conversation_cache_key: tuple[Any, ...] | None = None
    self.litert_lm_embedding_engine: litert_lm.EmbeddingEngine | None = None
    self.embedding_model_id: str | None = None
    self.embedding_backend: litert_lm.Backend | None = None
    self.embedding_max_num_tokens: int | None = None
    self.embedding_vision_backend: litert_lm.Backend | None = None
    self.embedding_audio_backend: litert_lm.Backend | None = None
    self.embedding_engine_cache_key: (
        tuple[str, str | None, int | None] | None
    ) = None

  def close_conversation(self) -> None:
    """Closes and resets the currently cached conversation, if any."""
    if self.litert_lm_conversation is not None:
      self.litert_lm_conversation.__exit__(None, None, None)
      self.litert_lm_conversation = None
    self.conversation_cache_key = None

  def acquire_slot(
      self,
      *,
      model_id: str | None = None,
      context_messages: list[Any] | None = None,
  ) -> Any:
    """Waits for a free worker and reserves it for the calling request.

    Among free workers it prefers, in order: the one whose cached
    conversation continues `context_messages` (no prefill of the history), one
    with `model_id` loaded and no cached conversation, one with `model_id`
    loaded whose cached conversation must be dropped (least recently used
    first), an empty one, and finally one holding another model. Otherwise
    ties go to the most recently used worker, so light traffic keeps reusing
    one engine and the others can be unloaded when idle.

    Args:
      model_id: The model the request asks for, if known.
      context_messages: The request's prior messages, if known.

    Returns:
      The reserved worker (this server or an `EngineSlot`). Hand it back with
      `release_slot`.
    """

    def score(slot: Any) -> tuple[int, float]:
      if slot.litert_lm_engine is None:
        return 1, slot.last_used
      if model_id is not None and slot.model_id != model_id:
        return 0, slot.last_used
      if slot.litert_lm_conversation is None:
        return 3, slot.last_used
      if (
          context_messages is not None
          and slot.conversation_cache_key is not None
          and slot.conversation_cache_key[0] == context_messages
      ):
        return 4, slot.last_used
      return 2, -slot.last_used

    with self._slot_cond:
      self.waiting += 1
      try:
        while True:
          free = [s for s in self.slots if not s.busy]
          if free:
            break
          self._slot_cond.wait()
      finally:
        self.waiting -= 1
      slot = max(free, key=score)
      slot.busy = True
      return slot

  def release_slot(self, slot: Any) -> None:
    """Returns a worker reserved by `acquire_slot` to the pool."""
    with self._slot_cond:
      slot.busy = False
      slot.last_used = time.monotonic()
      self._slot_cond.notify()

  def unload_idle_engines(self, now: float | None = None) -> int:
    """Unloads engines that have served no request for `idle_unload_seconds`.

    The process keeps running and keeps answering health checks; the next
    request loads the engine again.

    Args:
      now: The `time.monotonic()` reading to compare against (for tests).

    Returns:
      The number of engines unloaded.
    """
    if self.idle_unload_seconds <= 0:
      return 0
    now = time.monotonic() if now is None else now
    with self._slot_cond:
      idle = [
          s
          for s in self.slots
          if not s.busy
          and s.litert_lm_engine is not None
          and now - s.last_used >= self.idle_unload_seconds
      ]
      for s in idle:
        s.busy = True  # Keeps requests off the worker while it unloads.
    unloaded = 0
    for s in idle:
      idle_minutes = (now - s.last_used) / 60
      try:
        model_id = unload_engine(s)
      finally:
        with self._slot_cond:
          s.busy = False
          self._slot_cond.notify()
      if model_id is not None:
        unloaded += 1
        click.echo(
            click.style(
                f"Unloaded engine (model: {model_id}, worker {s.index}) after"
                f" {idle_minutes:.0f} min without requests; the next request"
                " loads it again.",
                fg="cyan",
            )
        )
    if (
        self.litert_lm_embedding_engine is not None
        and now - self.embedding_last_used >= self.idle_unload_seconds
        and self.embedding_lock.acquire(blocking=False)
    ):
      try:
        emb_engine = self.litert_lm_embedding_engine
        if emb_engine is not None:
          model_id = self.embedding_model_id
          self.litert_lm_embedding_engine = None
          self.embedding_model_id = None
          self.embedding_backend = None
          self.embedding_max_num_tokens = None
          self.embedding_vision_backend = None
          self.embedding_audio_backend = None
          self.embedding_engine_cache_key = None
          emb_engine.close()
          unloaded += 1
          click.echo(
              click.style(
                  f"Unloaded embedding engine (model: {model_id}) after"
                  " a period without requests.",
                  fg="cyan",
              )
          )
      finally:
        self.embedding_lock.release()
    return unloaded

  def preload(
      self,
      model_id: str,
      *,
      backend: str | None = None,
      max_num_tokens: int | None = None,
  ) -> None:
    """Loads `model_id` on every worker, one after the other.

    Requests that arrive meanwhile take a worker that is already loaded or
    wait for one. A failure is reported and leaves lazy loading in place.

    Args:
      model_id: The model identifier.
      backend: Optional backend override (as in the request's model field).
      max_num_tokens: Optional max_num_tokens override.
    """
    for index in range(len(self.slots)):
      if self._stop_event.is_set():
        return
      slot = self.slots[index]
      with self._slot_cond:
        while slot.busy:
          self._slot_cond.wait()
        slot.busy = True
      try:
        get_or_initialize_server_engine(
            slot,
            model_id=model_id,
            backend=backend,
            max_num_tokens=max_num_tokens,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(
            click.style(
                f"Preloading {model_id!r} on worker {index} failed: {e!r}",
                fg="red",
            )
        )
        return
      finally:
        self.release_slot(slot)

  def start_background_tasks(
      self,
      preload: tuple[str, str | None, int | None] | None = None,
  ) -> None:
    """Starts the idle-unload watcher and the optional preload.

    Args:
      preload: `(model_id, backend, max_num_tokens)` to load on every worker
        at start-up, or None to load lazily on the first request.
    """
    if self.idle_unload_seconds > 0:
      interval = min(30.0, max(1.0, self.idle_unload_seconds / 4))

      def watch() -> None:
        while not self._stop_event.wait(interval):
          try:
            self.unload_idle_engines()
          except Exception as e:  # pylint: disable=broad-exception-caught
            click.echo(click.style(f"Idle unload failed: {e!r}", fg="red"))

      self._background.append(
          threading.Thread(target=watch, name="idle-unload", daemon=True)
      )
    if preload is not None:
      model_id, backend, max_num_tokens = preload
      self._background.append(
          threading.Thread(
              target=lambda: self.preload(
                  model_id, backend=backend, max_num_tokens=max_num_tokens
              ),
              name="preload",
              daemon=True,
          )
      )
    for thread in self._background:
      thread.start()

  def close_all_engines(self) -> None:
    """Stops background tasks and closes every engine and conversation."""
    self._stop_event.set()
    for slot in self.slots:
      unload_engine(slot)
    emb_engine = self.litert_lm_embedding_engine
    if emb_engine is not None:
      self.litert_lm_embedding_engine = None
      emb_engine.close()

  def status(self) -> dict[str, Any]:
    """Returns the worker pool state for `GET /health`."""
    now = time.monotonic()
    with self._slot_cond:
      workers = [
          {
              "worker": s.index,
              "busy": s.busy,
              "model": (
                  s.model_id if s.litert_lm_engine is not None else None
              ),
              "idle_seconds": round(now - s.last_used, 1),
          }
          for s in self.slots
      ]
      waiting = self.waiting
    return {
        "status": "ok",
        "workers": workers,
        "busy": sum(1 for w in workers if w["busy"]),
        "waiting": waiting,
        "idle_unload_seconds": self.idle_unload_seconds,
    }


def conversation_owner(handler: Any) -> Any:
  """Returns the worker whose conversation cache a request should use."""
  slot = getattr(handler, "engine_slot", None)
  if isinstance(slot, (LiteRTLMServer, EngineSlot)):
    return slot
  return getattr(handler, "server", None)


class CORSRequestHandler(http.server.BaseHTTPRequestHandler):
  """Base HTTP request handler that adds CORS headers to all responses."""

  # GET paths polled by health checks; successful ones are left out of the
  # access log unless the server runs with --verbose.
  QUIET_PATHS = ("/v1/models", "/health")

  # The worker reserved for the request being handled, if any.
  engine_slot: Any = None

  def acquire_engine_slot(
      self,
      *,
      model_id: str | None = None,
      context_messages: list[Any] | None = None,
  ) -> Any:
    """Reserves a worker for this request (once) and returns it."""
    if self.engine_slot is None:
      server = self.server
      if isinstance(server, LiteRTLMServer):
        self.engine_slot = server.acquire_slot(
            model_id=model_id, context_messages=context_messages
        )
      else:
        self.engine_slot = server
    return self.engine_slot

  def release_engine_slot(self) -> None:
    """Hands the request's worker back to the pool."""
    slot, self.engine_slot = self.engine_slot, None
    if slot is not None and isinstance(self.server, LiteRTLMServer):
      self.server.release_slot(slot)

  def handle_one_request(self) -> None:
    try:
      super().handle_one_request()
    finally:
      self.release_engine_slot()

  def handle(self) -> None:
    try:
      super().handle()
    except CLIENT_GONE_ERRORS:
      # The client closed the connection; there is no one to answer.
      self.close_connection = True

  def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
    if (
        self.command == "GET"
        and isinstance(code, int)
        and 200 <= code < 300
        and self.path.split("?", 1)[0] in self.QUIET_PATHS
        and not getattr(self.server, "verbose", False)
    ):
      return
    super().log_request(code, size)

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

  def send_error(
      self,
      code: int,
      message: str | None = None,
      explain: str | None = None,
  ) -> None:
    """Sends an HTTP error response with a sanitized reason phrase.

    Sanitizes `message` into a single-line Latin-1 string suitable for HTTP
    reason phrases (RFC 9112) while preserving multiline or Unicode error
    details in `explain` when `explain` is not explicitly provided.

    Args:
      code: The HTTP status code.
      message: Optional short error message. If None, the standard HTTP reason
        phrase for `code` is used.
      explain: Optional detailed explanation. If None and `message` is provided,
        defaults to the original unsanitized `message`.
    """
    if message is not None:
      if explain is None:
        explain = message
      # HTTP status reason-phrases must be single-line Latin-1 (RFC 9112),
      # whereas C++ runtime errors may contain newlines and Unicode (e.g. '└').
      message = (
          " ".join(message.splitlines())
          .encode("latin-1", errors="replace")
          .decode("latin-1")
      )
    super().send_error(code, message, explain)


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
  cache_key = (model_id, backend, max_num_tokens)
  if (
      server.litert_lm_engine is not None
      and server.engine_cache_key == cache_key
  ):
    return server.litert_lm_engine

  m = model.Model.from_model_reference(model_id)

  if not m.exists():
    raise FileNotFoundError(f"Model {model_id} not found")

  resolved_backend = model.parse_backend(backend, model_obj=m)
  resolved_max_num_tokens = model.resolve_config_option(
      max_num_tokens, m, "max_num_tokens"
  )

  if server.litert_lm_engine is not None:
    if (
        server.model_id == model_id
        and server.backend == resolved_backend
        and server.max_num_tokens == resolved_max_num_tokens
    ):
      server.engine_cache_key = cache_key
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
    server.engine_cache_key = None

  vision_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ENCODER,
      },
      label="vision",
  )
  audio_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          llm_metadata_pb2.LlmMetadata.TF_LITE_AUDIO_ENCODER_HW,
      },
      label="audio",
  )
  cache = model.resolve_config_option(None, m, "cache")
  cache_dir_val = common.cache_dir_value_from_cache_mode(cache)
  speculative_decoding = common.normalize_speculative_decoding(
      model.resolve_config_option(None, m, "speculative_decoding")
  )
  activation_data_type_str = model.resolve_config_option(
      None, m, "activation_data_type"
  )
  activation_data_type = (
      litert_lm.ActivationDataType.from_str(activation_data_type_str)
      if activation_data_type_str
      else None
  )
  head_chunked_attention_mib = model.resolve_config_option(
      None, m, "head_chunked_attention_mib"
  )
  kv_cache_type = model.resolve_kv_cache_type(None, m, resolved_backend)

  worker = getattr(server, "index", 0)
  click.echo(
      click.style(
          f"Initializing engine for model: {m.model_path}"
          + (f" (worker {worker})" if worker else ""),
          fg="cyan",
      )
  )
  with _ENGINE_INIT_LOCK:
    engine = litert_lm.Engine(
        m.model_path,
        backend=resolved_backend,  # pyrefly: ignore[bad-argument-type]
        max_num_tokens=resolved_max_num_tokens,
        vision_backend=vision_backend,
        audio_backend=audio_backend,
        cache_dir=cache_dir_val,
        speculative_decoding_config=model.speculative_decoding_config_from_mode(
            speculative_decoding
        ),
        activation_data_type=activation_data_type,
        enable_benchmark=True,
        use_ringbuffers_local_attention=True,
        head_chunked_attention_mib=head_chunked_attention_mib,
        kv_cache_type=kv_cache_type,
    )
    engine.__enter__()
  server.litert_lm_engine = engine
  server.model_id = model_id
  server.backend = resolved_backend
  server.max_num_tokens = resolved_max_num_tokens
  server.vision_backend = vision_backend
  server.audio_backend = audio_backend
  server.activation_data_type = activation_data_type
  server.engine_cache_key = cache_key
  return engine


def get_or_initialize_server_embedding_engine(
    server: LiteRTLMServer,
    *,
    model_id: str,
    backend: str | None = None,
    max_num_tokens: int | None = None,
) -> litert_lm.EmbeddingEngine:
  """Retrieves or initializes the persistent server embedding engine.

  Args:
    server: The active custom LiteRTLMServer instance object.
    model_id: The requested model identifier string.
    backend: Optional requested backend override (e.g. 'cpu', 'gpu', 'npu').
    max_num_tokens: Optional requested max_num_tokens override, propagated to
      EmbeddingEngine as max_input_length.

  Returns:
    The shared LiteRT-LM EmbeddingEngine context object.

  Raises:
    FileNotFoundError: If the model package path does not exist.
  """
  cache_key = (model_id, backend, max_num_tokens)
  if (
      server.litert_lm_embedding_engine is not None
      and server.embedding_engine_cache_key == cache_key
  ):
    return server.litert_lm_embedding_engine

  m = model.Model.from_model_reference(model_id)

  if not m.exists():
    raise FileNotFoundError(f"Model {model_id} not found")

  resolved_backend = model.parse_backend(backend, model_obj=m)
  resolved_max_num_tokens = model.resolve_config_option(
      max_num_tokens, m, "max_num_tokens"
  )

  if server.litert_lm_embedding_engine is not None:
    if (
        server.embedding_model_id == model_id
        and server.embedding_backend == resolved_backend
        and server.embedding_max_num_tokens == resolved_max_num_tokens
    ):
      server.embedding_engine_cache_key = cache_key
      return server.litert_lm_embedding_engine

    click.echo(
        click.style(
            f"Re-initializing embedding engine (model: {model_id}, backend:"
            f" {resolved_backend}, max_num_tokens:"
            f" {resolved_max_num_tokens})",
            fg="yellow",
        )
    )
    server.litert_lm_embedding_engine.close()
    server.litert_lm_embedding_engine = None
    server.embedding_model_id = None
    server.embedding_backend = None
    server.embedding_max_num_tokens = None
    server.embedding_vision_backend = None
    server.embedding_audio_backend = None
    server.embedding_engine_cache_key = None

  vision_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ENCODER,
      },
      label="vision",
  )
  audio_backend = model.parse_backend(
      None,
      model_obj=m,
      target_model_types={
          llm_metadata_pb2.LlmMetadata.TF_LITE_AUDIO_ENCODER_HW,
      },
      label="audio",
  )
  cache = model.resolve_config_option(None, m, "cache")
  cache_dir_val = common.cache_dir_value_from_cache_mode(cache)

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
      max_input_length=resolved_max_num_tokens,
  )
  server.litert_lm_embedding_engine = engine
  server.embedding_model_id = model_id
  server.embedding_backend = resolved_backend
  server.embedding_max_num_tokens = resolved_max_num_tokens
  server.embedding_vision_backend = vision_backend
  server.embedding_audio_backend = audio_backend
  server.embedding_engine_cache_key = cache_key
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
  cache_server = (
      server if isinstance(server, (LiteRTLMServer, EngineSlot)) else None
  )
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
