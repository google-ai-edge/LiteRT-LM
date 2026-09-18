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

"""OpenAI /v1/embeddings endpoint handler for LiteRT-LM."""

from __future__ import annotations

import base64
import math
import struct
from typing import Any
import urllib.request

import litert_lm
from litert_lm_cli.commands.serve import openai_common


def embedding_to_base64(embedding: list[float]) -> str:
  """Converts a float embedding vector to IEEE 754 float32 base64 string."""
  raw_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
  return base64.b64encode(raw_bytes).decode("utf-8")


def l2_normalize(vec: list[float]) -> list[float]:
  """Applies L2 normalization to a float vector (after dimension slicing)."""
  norm = math.sqrt(sum(x * x for x in vec))
  if norm > 0:
    return [x / norm for x in vec]
  return vec


def _parse_embedding_content_part(part: Any) -> str | litert_lm.Content:
  """Parses a content part (str, int, dict, or Content) into a LiteRT-LM input item."""
  if isinstance(part, str):
    return part
  if isinstance(part, int):
    return str(part)
  if isinstance(part, litert_lm.Content):
    return part
  if isinstance(part, dict):
    part_type = part.get("type")
    if part_type == "text":
      return part.get("text", "")
    if part_type == "image_url":
      image_url = part.get("image_url", {})
      url = image_url.get("url", "")
      if url.startswith("data:"):
        header, data = url.split(",", 1)
        if "base64" in header:
          return litert_lm.Content.ImageBytes(base64.b64decode(data))
        raise ValueError(
            "Unsupported data URL format (only base64 is supported)"
        )
      if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=10) as response:
          return litert_lm.Content.ImageBytes(response.read())
      path = url
      if path.startswith("file://"):
        path = path[7:]
      return litert_lm.Content.ImageFile(path)
    if part_type == "input_audio":
      input_audio = part.get("input_audio", {})
      data = input_audio.get("data", "")
      return litert_lm.Content.AudioBytes(base64.b64decode(data))
    if part_type == "image":
      if "blob" in part:
        return litert_lm.Content.ImageBytes(base64.b64decode(part["blob"]))
      if "path" in part:
        return litert_lm.Content.ImageFile(part["path"])
    if part_type == "audio":
      if "blob" in part:
        return litert_lm.Content.AudioBytes(base64.b64decode(part["blob"]))
      if "path" in part:
        return litert_lm.Content.AudioFile(part["path"])
    raise ValueError(f"Unsupported content part type: {part_type!r}")
  raise ValueError(f"Unsupported input element type: {type(part).__name__}")


def normalize_embedding_input(
    input_data: Any,
) -> list[str | litert_lm.Content | list[str | litert_lm.Content]]:
  """Normalizes OpenAI embedding 'input' parameter into a batch of items."""
  if not input_data:
    raise ValueError("input cannot be empty")

  if isinstance(input_data, (str, dict, litert_lm.Content)):
    return [_parse_embedding_content_part(input_data)]

  if isinstance(input_data, list):
    if all(isinstance(x, int) and not isinstance(x, bool) for x in input_data):
      return [[_parse_embedding_content_part(x) for x in input_data]]

    if all(isinstance(x, str) for x in input_data):
      return [x for x in input_data]

    if all(isinstance(x, dict) for x in input_data):
      if all("type" in x for x in input_data):
        return [[_parse_embedding_content_part(x) for x in input_data]]
      return [_parse_embedding_content_part(x) for x in input_data]

    if all(isinstance(x, list) for x in input_data):
      return [
          [_parse_embedding_content_part(part) for part in item]
          for item in input_data
      ]

    return [_parse_embedding_content_part(x) for x in input_data]

  raise ValueError(f"Unsupported input type: {type(input_data).__name__}")


def handle_post_embeddings(handler: Any) -> None:
  """Handles POST requests to embeddings endpoints (/v1/embeddings)."""
  body = handler.parse_request_body()
  if body is None:
    return

  raw_model_str = body.get("model")
  if not raw_model_str:
    handler.send_error(400, "Missing model")
    return

  try:
    model_id, backend, _ = openai_common.parse_model_parameter(raw_model_str)
  except ValueError as e:
    handler.send_error(400, f"Invalid model parameter: {e}")
    return

  input_data = body.get("input")
  try:
    contents_batch = normalize_embedding_input(input_data)
  except (ValueError, RuntimeError) as e:
    handler.send_error(400, f"Invalid input parameter: {e}")
    return

  encoding_format = body.get("encoding_format", "float")
  if encoding_format not in ("float", "base64"):
    handler.send_error(
        400,
        f"Invalid encoding_format: {encoding_format!r}. Supported formats:"
        " 'float', 'base64'.",
    )
    return

  dimensions = body.get("dimensions")
  if dimensions is not None:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
      handler.send_error(400, "dimensions must be an integer")
      return
    if dimensions <= 0:
      handler.send_error(400, "dimensions must be a positive integer")
      return

  engine = handler.get_embedding_engine(model_id, backend=backend)
  if engine is None:
    return

  options = litert_lm.EmbeddingOptions(
      normalize=body.get("normalize"),
      insert_special_tokens=body.get("insert_special_tokens"),
  )

  try:
    responses = engine.compute_embedding_batch(contents_batch, options=options)
    data_list = []
    for i, resp in enumerate(responses):
      vec = resp.embedding
      if dimensions is not None:
        if dimensions > len(vec):
          handler.send_error(
              400,
              f"dimensions must be between 1 and {len(vec)}, got {dimensions}",
          )
          return
        vec = l2_normalize(vec[:dimensions])

      if encoding_format == "base64":
        formatted_embedding = embedding_to_base64(vec)
      else:
        formatted_embedding = vec

      data_list.append({
          "object": "embedding",
          "index": i,
          "embedding": formatted_embedding,
      })

    # TODO: b/562598581 - Populate prompt_tokens accurately once EmbeddingEngine
    # exposes token count metadata from tokenization.
    resp_body = {
        "object": "list",
        "data": data_list,
        "model": raw_model_str,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }

    handler.headers_sent = True
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (openai_common.dump_json(resp_body, indent=2) + "\n").encode("utf-8")
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    handler.handle_inference_error(e, raw_model_str, input_data)
