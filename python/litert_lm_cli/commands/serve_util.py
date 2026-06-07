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

import dataclasses
import http.server
import io
import socket

import click

import litert_lm
from litert_lm_builder import litertlm_peek
from litert_lm_cli import model


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineKey:
  """Cache key for identifying unique engine instances.

  Attributes:
    model_id: The identifier of the model.
    backend: The hardware backend to use.
    max_num_tokens: The maximum number of tokens.
    vision_backend: The hardware backend for vision encoding.
    audio_backend: The hardware backend for audio encoding.
    enable_speculative_decoding: Whether speculative decoding is enabled.
  """
  model_id: str
  backend: str | None = None
  max_num_tokens: int | None = None
  vision_backend: str | None = None
  audio_backend: str | None = None
  enable_speculative_decoding: bool | None = None


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
    engines: A dictionary mapping EngineKey to loaded Engine instances.
    engine_lru: A list of EngineKey tracking the LRU order of active engines.
    max_pool_size: The maximum number of engines to keep in the cache.
    address_family: The socket family to use (e.g. AF_INET6 for IPv6).
    enable_speculative_decoding: Whether speculative decoding is enabled, or
      None.
  """

  engines: dict[EngineKey, litert_lm.Engine]
  engine_lru: list[EngineKey]
  max_pool_size: int
  litert_lm_engine: litert_lm.Engine | None
  model_id: str | None
  backend: litert_lm.Backend | None
  max_num_tokens: int | None
  vision_backend: litert_lm.Backend | None
  audio_backend: litert_lm.Backend | None
  enable_speculative_decoding: bool | None

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
      max_pool_size: int = 3,
  ):
    """Initializes the instance.

    Args:
      server_address: A tuple of (host, port) to listen on.
      RequestHandlerClass: The HTTP handler class to use.
      max_pool_size: The maximum number of engines to keep in the cache.
    """
    host, _ = server_address
    if ":" in host:
      self.address_family = socket.AF_INET6
    super().__init__(server_address, RequestHandlerClass)
    self.engines: dict[EngineKey, litert_lm.Engine] = {}
    self.engine_lru: list[EngineKey] = []
    self.max_pool_size = max_pool_size
    self.litert_lm_engine: litert_lm.Engine | None = None
    self.model_id: str | None = None
    self.backend: litert_lm.Backend | None = None
    self.max_num_tokens: int | None = None
    self.vision_backend: litert_lm.Backend | None = None
    self.audio_backend: litert_lm.Backend | None = None
    self.enable_speculative_decoding: bool | None = None

  def server_close(self) -> None:
    """Closes the server and releases all loaded engines."""
    try:
      super().server_close()
    finally:
      for key, engine in self.engines.items():
        try:
          engine.__exit__(None, None, None)
        except Exception as e:  # pylint: disable=broad-exception-caught
          click.echo(
              click.style(
                  f"Warning: Failed to close engine {key.model_id} during"
                  f" shutdown: {e!r}",
                  fg="yellow",
              ),
              err=True,
          )
      self.engines.clear()
      self.engine_lru.clear()


def _is_gpu_only_model(model_path: str) -> bool:
  """Returns True if the model is GPU-only."""
  try:
    with io.StringIO() as dummy_out:
      metadata = litertlm_peek.read_litertlm_header(model_path, dummy_out)
  except Exception as e:  # pylint: disable=broad-exception-caught
    click.echo(
        click.style(f"Failed to inspect model metadata: {e!r}", fg="yellow")
    )
    return False

  section_metadata = metadata.SectionMetadata()
  if not section_metadata:
    return False
  for i in range(section_metadata.ObjectsLength()):
    section = section_metadata.Objects(i)
    if not section or section.ItemsLength() == 0:
      continue
    for j in range(section.ItemsLength()):
      item_dict = litertlm_peek.kvp_to_dict(section.Items(j))
      if item_dict.get("key") != "backend_constraint":
        continue
      val = item_dict.get("value")
      if isinstance(val, str) and val.lower() == "gpu_artisan":
        return True

  return False


def _select_backend(model_path: str) -> litert_lm.Backend:
  """Inspects .litertlm file metadata to select the execution backend.

  Args:
    model_path: The absolute filesystem path to the .litertlm model bundle.

  Returns:
    Backend.GPU() if the model metadata specifies 'gpu_artisan' as the backend
    constraint, otherwise Backend.CPU().
  """
  if _is_gpu_only_model(model_path):
    return litert_lm.Backend.GPU()
  return litert_lm.Backend.CPU()


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Represents a parsed model specification.

  Attributes:
    model_id: The identifier for the model.
    backend: The hardware backend to use, or None for auto-selection.
    max_num_tokens: The maximum number of tokens, or None for model default.
  """

  model_id: str
  backend: litert_lm.Backend | None = None
  max_num_tokens: int | None = None


def parse_model_spec(model_spec: str) -> ModelSpec:
  """Parses model spec in format 'model_id[,backend][,max_tokens]'."""
  parts = model_spec.split(",")
  if not parts or not parts[0]:
    raise ValueError("Empty model spec")

  # Trailing comma is invalid (e.g., "model,").
  if len(parts) > 1 and not parts[-1]:
    raise ValueError(f"Trailing comma in model spec: {model_spec}")

  model_id = parts[0]
  backend = None
  max_tokens = None

  if len(parts) > 1 and parts[1]:
    backend_str = parts[1]
    backend_lower = backend_str.lower()
    if backend_lower == "gpu":
      backend = litert_lm.Backend.GPU()
    elif backend_lower == "npu":
      backend = litert_lm.Backend.NPU()
    elif backend_lower == "cpu":
      backend = litert_lm.Backend.CPU()
    else:
      raise ValueError(
          f"Unavailable backend {backend_str!r}, available backends are 'cpu'"
          " and 'gpu'"
      )

  if len(parts) > 2 and parts[2]:
    try:
      max_tokens = int(parts[2])
    except ValueError as e:
      raise ValueError(f"Invalid max_tokens: {parts[2]}") from e

  if len(parts) > 3:
    raise ValueError(f"Too many parameters in model spec: {model_spec}")

  # TODO: b/514897675 - Add a cap on max_tokens to prevent OOM.
  return ModelSpec(
      model_id=model_id, backend=backend, max_num_tokens=max_tokens
  )


def get_or_initialize_server_engine(
    server: LiteRTLMServer,
    *,
    model_id: str,
    backend: litert_lm.Backend | None = None,
    max_num_tokens: int | None = None,
    vision_backend: litert_lm.Backend | None = None,
    audio_backend: litert_lm.Backend | None = None,
) -> litert_lm.Engine:
  """Retrieves the persistent server engine or initializes it on first request.

  Lifetime Management:
    The LiteRT-LM Engine is a globally scoped persistent resource.
    - Initialization: Invokes `__enter__` dynamically upon the arrival of the
      first incoming inference request.
    - Eviction: When the cache pool exceeds `max_pool_size`, the least recently
      used engine is evicted and its `__exit__` is called to release resources.
    - Termination: The running server's parent execution process is responsible
      for explicitly invoking `__exit__` on all engines in the cache pool during
      outer context teardown loops (e.g., in `run_server` finally blocks).

  Args:
    server: The active custom LiteRTLMServer instance object.
    model_id: The requested model identifier string.
    backend: The hardware backend to use. If None, it will be auto-selected.
    max_num_tokens: The maximum number of tokens. If None, uses model default.
    vision_backend: The hardware backend to use for vision encoding.
    audio_backend: The hardware backend to use for audio encoding.

  Returns:
    The shared LiteRT-LM Engine context object.

  Raises:
    FileNotFoundError: If the model package path does not exist.
  """
  resolved_max_num_tokens = (
      server.max_num_tokens if max_num_tokens is None else max_num_tokens
  )

  m = model.Model.from_model_id(model_id)

  resolved_backend = (
      backend if backend is not None else _select_backend(m.model_path)
  )

  enable_speculative_decoding = server.enable_speculative_decoding

  engine_key = EngineKey(
      model_id=model_id,
      backend=(
          resolved_backend.get_name()
          if resolved_backend is not None
          else None
      ),
      max_num_tokens=resolved_max_num_tokens,
      vision_backend=(
          vision_backend.get_name() if vision_backend is not None else None
      ),
      audio_backend=(
          audio_backend.get_name() if audio_backend is not None else None
      ),
      enable_speculative_decoding=enable_speculative_decoding,
  )

  engine = server.engines.get(engine_key)
  if engine is not None:
    server.engine_lru.remove(engine_key)
    server.engine_lru.append(engine_key)
  else:
    if len(server.engines) >= server.max_pool_size:
      evicted_key = server.engine_lru.pop(0)
      evicted_engine = server.engines.pop(evicted_key)
      click.echo(
          click.style(
              f"Evicting engine {evicted_key.model_id} from cache pool to free"
              " system memory.",
              fg="yellow",
          )
      )
      try:
        evicted_engine.__exit__(None, None, None)
      except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(
            click.style(
                f"Warning: Failed to close evicted engine"
                f" {evicted_key.model_id}: {e!r}",
                fg="yellow",
            ),
            err=True,
        )

    click.echo(
        click.style(f"Initializing engine for model: {m.model_path}", fg="cyan")
    )

    try:
      engine = litert_lm.Engine(
          m.model_path,
          backend=resolved_backend,
          max_num_tokens=resolved_max_num_tokens,
          vision_backend=vision_backend,
          audio_backend=audio_backend,
          enable_speculative_decoding=enable_speculative_decoding,
      )
      engine.__enter__()
    except RuntimeError as e:
      # We check if the model exists to raise a descriptive FileNotFoundError.
      # Otherwise, litert_lm.Engine would raise a generic RuntimeError.
      if not m.exists():
        raise FileNotFoundError(f"Model {model_id} not found") from e
      raise

    server.engines[engine_key] = engine
    server.engine_lru.append(engine_key)

  # Keep fallback attributes for backward compatibility.
  server.litert_lm_engine = engine
  server.model_id = model_id
  server.backend = resolved_backend
  server.max_num_tokens = resolved_max_num_tokens
  server.vision_backend = vision_backend
  server.audio_backend = audio_backend
  return engine
