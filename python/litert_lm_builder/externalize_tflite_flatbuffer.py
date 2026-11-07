# Copyright 2025 Google LLC.
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
"""Externalize supported TFLite weight buffers into one weight blob.

This FlatBuffer-level tool moves only large constants that LiteRT-LM's scoped
external-weight path can consume: weights at input 1 of FullyConnected, Conv2D,
DepthwiseConv2D with depth multiplier 1, and EmbeddingLookup. Other constants
remain embedded so they do not become delegate partition inputs.

After externalization, a model can additionally be made to reference payloads
stored in *another* model's weight blob instead of its own copy (see
`collect_external_weight_digests` and `redirect_external_weights`). This removes
weights that are byte-identical across models, e.g. a token embedding table that
an embedder model looks up and a decoder model reuses as its tied output
projection.

This is a copy of `litert/tools/externalize_tflite_flatbuffer.py` from LiteRT
that is bundled with the LiteRT-LM builder so that `litert-lm-builder` and
`litert-lm pack` can externalize weights without depending on LiteRT's tools.
Unlike the LiteRT copy, it reports its progress with `logging` instead of
printing to stdout.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import hashlib
import logging
import mmap
import pathlib
import shutil

import flatbuffers

from litert_lm_builder import tflite_schema_py_generated as schema

np = flatbuffers.compat.import_numpy()

_LITERT_HOST_MEMORY_BUFFER_ALIGNMENT = 64
_TFLITE_BUFFER_ALIGNMENT = 16
# If the buffers left inline would exceed this size, the buffers that were
# appended in the input model are appended to the output model again instead
# of being packed into the FlatBuffer. This keeps the FlatBuffer below its
# 2 GiB limit and avoids the slow element-wise Python packing of large
# payloads.
_MAX_INLINE_BUFFER_BYTES = 1 << 30


def _enum_value(enum_obj, name: str) -> int:
  enum_cls = getattr(enum_obj, enum_obj.__name__, enum_obj)
  return int(getattr(enum_cls, name))


def _builtin_code(operator_code) -> int:
  builtin = int(operator_code.builtinCode)
  if builtin == 0 and int(operator_code.deprecatedBuiltinCode) != 0:
    return int(operator_code.deprecatedBuiltinCode)
  return builtin


def _tensor_element_count(tensor) -> int:
  if tensor.shape is None:
    return 0
  shape = [int(dimension) for dimension in tensor.shape]
  if not shape or any(dimension < 0 for dimension in shape):
    return 0
  element_count = 1
  for dimension in shape:
    element_count *= dimension
  return element_count


def _buffer_bytes(buffer) -> memoryview[int] | None:
  if buffer.data is None:
    return None
  try:
    data = memoryview(buffer.data)
  except TypeError:
    data = memoryview(bytes(buffer.data))
  if data.nbytes == 0:
    return None
  if data.format != "B" or data.ndim != 1:
    data = data.cast("B")
  return data


def _materialize_appended_buffer_data(model, input_bytes: bytes) -> set[int]:
  """Moves TFLite appended buffer payloads into the in-memory object model.

  The payloads are zero-copy uint8 numpy views of `input_bytes`. The generated
  `BufferT.Pack` only uses its fast `CreateNumpyVector` path when the data is
  exactly an `np.ndarray`; any other sequence is serialized one byte at a time.
  Both paths produce the same bytes. When numpy is not importable (e.g. macOS
  builds, where the BUILD `select` drops it), the payloads are `memoryview`
  slices instead, and packing falls back to the slower per-byte path.

  Args:
    model: The unpacked `ModelT` whose buffers may reference appended data.
    input_bytes: The serialized model file, including appended buffer data.

  Returns:
    Indices of buffers whose non-empty payloads were appended in the input file.
  """
  input_size = len(input_bytes)
  appended_buffers: set[int] = set()
  for buffer_index, buffer in enumerate(model.buffers or []):
    offset = int(buffer.offset)
    size = int(buffer.size)
    if offset == 0 and size == 0:
      continue
    if size == 0:
      # Zero-length appended payload. TFLite's reader treats `offset > 1` with
      # `size == 0` as a non-null, zero-byte constant, which is what an empty
      # inline vector encodes; offsets 0 and 1 are "no appended data".
      if buffer.data is None and offset > 1:
        buffer.data = np.empty(0, dtype=np.uint8) if np is not None else b""
      buffer.offset = 0
      buffer.size = 0
      continue
    if offset <= 1 or size < 0:
      raise ValueError(
          f"TFLite buffer {buffer_index} has invalid external range: "
          f"offset={offset}, size={size}"
      )
    if _buffer_bytes(buffer) is not None:
      raise ValueError(
          f"TFLite buffer {buffer_index} has both inline and appended data"
      )
    end = offset + size
    if end > input_size:
      raise ValueError(
          f"TFLite buffer {buffer_index} range exceeds the model file: "
          f"offset={offset}, size={size}, file_size={input_size}"
      )
    if np is not None:
      buffer.data = np.frombuffer(
          input_bytes, dtype=np.uint8, count=size, offset=offset
      )
    else:
      buffer.data = memoryview(input_bytes)[offset:end]
    buffer.offset = 0
    buffer.size = 0
    appended_buffers.add(buffer_index)
  return appended_buffers


def _bias_tensor_indices(model) -> set[tuple[int, int]]:
  """Returns `(subgraph_index, tensor_index)` pairs of bias tensors."""
  bias_ops = {
      _enum_value(schema.BuiltinOperator, "FULLY_CONNECTED"),
      _enum_value(schema.BuiltinOperator, "CONV_2D"),
      _enum_value(schema.BuiltinOperator, "DEPTHWISE_CONV_2D"),
  }
  bias_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for op in subgraph.operators or []:
      if op.inputs is None or len(op.inputs) <= 2:
        continue
      opcode = model.operatorCodes[int(op.opcodeIndex)]
      if _builtin_code(opcode) in bias_ops:
        bias_idx = int(op.inputs[2])
        if bias_idx >= 0:
          bias_tensors.add((sg_idx, bias_idx))
  return bias_tensors


def _shareable_weight_tensor_indices(model) -> set[tuple[int, int]]:
  """Returns `(subgraph_index, tensor_index)` pairs of externalizable weights."""
  depthwise_conv = _enum_value(schema.BuiltinOperator, "DEPTHWISE_CONV_2D")
  weight_ops = {
      _enum_value(schema.BuiltinOperator, "EMBEDDING_LOOKUP"),
      _enum_value(schema.BuiltinOperator, "FULLY_CONNECTED"),
      _enum_value(schema.BuiltinOperator, "CONV_2D"),
      depthwise_conv,
  }
  weight_tensors: set[tuple[int, int]] = set()
  inline_weight_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for op in subgraph.operators or []:
      if op.inputs is None or len(op.inputs) <= 1:
        continue
      opcode = model.operatorCodes[int(op.opcodeIndex)]
      builtin_code = _builtin_code(opcode)
      if builtin_code in weight_ops:
        weight_idx = int(op.inputs[1])
        if weight_idx >= 0:
          weight_tensors.add((sg_idx, weight_idx))
          if builtin_code == depthwise_conv and getattr(
              op.builtinOptions, "depthMultiplier", None
          ) != 1:
            # The GPU parser transposes these weights on the CPU. Keep them
            # inline even if another supported operator shares the tensor.
            inline_weight_tensors.add((sg_idx, weight_idx))
  return weight_tensors - inline_weight_tensors


def _subgraph_input_indices(model) -> set[tuple[int, int]]:
  input_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    if subgraph.inputs is None:
      continue
    for tensor_idx in subgraph.inputs:
      if int(tensor_idx) >= 0:
        input_tensors.add((sg_idx, int(tensor_idx)))
  return input_tensors


def _referenced_buffers(model, externalized: set[tuple[int, int]]) -> set[int]:
  referenced: set[int] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for tensor_idx, tensor in enumerate(subgraph.tensors or []):
      if (sg_idx, tensor_idx) not in externalized:
        referenced.add(int(tensor.buffer))
  return referenced


def _ensure_group(model, group_name: str) -> int:
  """Returns the index of `group_name` in `model.externalBufferGroups`."""
  if model.externalBufferGroups is None:
    model.externalBufferGroups = []
  if not model.externalBufferGroups:
    placeholder = schema.ExternalBufferGroupT()
    placeholder.name = ""
    model.externalBufferGroups.append(placeholder)
  for idx, group in enumerate(model.externalBufferGroups):
    name = (
        group.name.decode("utf-8")
        if isinstance(group.name, bytes)
        else group.name
    )
    if name == group_name:
      return idx
  group = schema.ExternalBufferGroupT()
  group.name = group_name
  model.externalBufferGroups.append(group)
  return len(model.externalBufferGroups) - 1


def _make_external_buffer_id(model) -> int:
  # Match LiteRT flatbuffer_export.cc: external buffer IDs have the MSB set so
  # they cannot be confused with ordinary TFLite buffer indices.
  return (1 << 31) | len(model.externalBuffers or [])


def _write_model(
    model, appended_buffers: set[int], output_model: pathlib.Path
) -> None:
  """Serializes `model` to `output_model`.

  Buffers that were appended in the input file are appended again (instead of
  being packed into the FlatBuffer) when the inline payload would otherwise
  approach the FlatBuffer size limit.

  Args:
    model: The `ModelT` to serialize. Appended buffers must already be
      materialized (see `_materialize_appended_buffer_data`).
    appended_buffers: Indices of buffers whose payloads were appended in the
      input file.
    output_model: Destination `.tflite` path.
  """
  remaining_inline_bytes = sum(
      len(b.data) for b in model.buffers or [] if b.data is not None
  )
  reappend_payloads: list[tuple[int, memoryview[int]]] = []
  if remaining_inline_bytes >= _MAX_INLINE_BUFFER_BYTES:
    for buf_idx in sorted(appended_buffers):
      buf = model.buffers[buf_idx]
      data = _buffer_bytes(buf)
      if data is not None:
        reappend_payloads.append((buf_idx, data))
        buf.data = None
        buf.offset = 1
        buf.size = len(data)

  def _pack_flatbuffer() -> bytes:
    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())

  packed_fb = _pack_flatbuffer()
  if reappend_payloads:
    cursor = len(packed_fb)
    for buf_idx, data in reappend_payloads:
      cursor = (
          (cursor + _TFLITE_BUFFER_ALIGNMENT - 1)
          // _TFLITE_BUFFER_ALIGNMENT
          * _TFLITE_BUFFER_ALIGNMENT
      )
      model.buffers[buf_idx].offset = cursor
      cursor += len(data)
    packed_fb = _pack_flatbuffer()
    with output_model.open("wb") as out_f:
      out_f.write(packed_fb)
      for buf_idx, data in reappend_payloads:
        pad = model.buffers[buf_idx].offset - out_f.tell()
        if pad > 0:
          out_f.write(b"\0" * pad)
        out_f.write(data)
  else:
    output_model.write_bytes(packed_fb)


def externalize(
    input_model: pathlib.Path,
    output_dir: pathlib.Path,
    group_name: str,
    num_elements_threshold: int,
    existing_weights: pathlib.Path | None = None,
) -> None:
  """Externalizes eligible weight buffers of `input_model` into `output_dir`."""
  output_dir.mkdir(parents=True, exist_ok=True)
  output_model = output_dir / "model.tflite"
  output_weights = output_dir / group_name

  input_bytes = input_model.read_bytes()
  model = schema.ModelT.InitFromPackedBuf(input_bytes, 0)
  appended_buffers = _materialize_appended_buffer_data(model, input_bytes)
  if model.externalBuffers is None:
    model.externalBuffers = []

  group_idx = None
  bias_tensors = _bias_tensor_indices(model)
  shareable_weight_tensors = _shareable_weight_tensor_indices(model)
  input_tensors = _subgraph_input_indices(model)
  externalized_tensors: set[tuple[int, int]] = set()
  candidate_buffers: set[int] = set()
  bytes_cache: dict[tuple[int, bytes], tuple[int, int]] = {}
  total_externalized_bytes = 0

  with output_weights.open("wb") as weights_file:
    if existing_weights is not None:
      with existing_weights.open("rb") as existing_weights_file:
        shutil.copyfileobj(existing_weights_file, weights_file)

    for sg_idx, subgraph in enumerate(model.subgraphs or []):
      for tensor_idx, tensor in enumerate(subgraph.tensors or []):
        tensor_key = (sg_idx, tensor_idx)
        if tensor_key not in shareable_weight_tensors:
          continue
        if tensor_key in input_tensors or tensor_key in bias_tensors:
          continue
        if int(tensor.externalBuffer) != 0:
          continue
        if tensor.isVariable:
          continue
        if _tensor_element_count(tensor) <= num_elements_threshold:
          continue

        buffer_idx = int(tensor.buffer)
        if buffer_idx <= 0 or buffer_idx >= len(model.buffers or []):
          continue
        data = _buffer_bytes(model.buffers[buffer_idx])
        if data is None:
          continue

        digest = hashlib.sha256(data).digest()
        cache_key = (len(data), digest)
        if cache_key in bytes_cache:
          offset, length = bytes_cache[cache_key]
        else:
          padding = (
              -weights_file.tell()
          ) % _LITERT_HOST_MEMORY_BUFFER_ALIGNMENT
          if padding:
            weights_file.write(b"\0" * padding)
          offset = weights_file.tell()
          length = len(data)
          weights_file.write(data)
          bytes_cache[cache_key] = (offset, length)
          total_externalized_bytes += length

        if group_idx is None:
          group_idx = _ensure_group(model, group_name)
        external_buffer_id = _make_external_buffer_id(model)
        external_buffer = schema.ExternalBufferT()
        external_buffer.id = external_buffer_id
        external_buffer.group = group_idx
        external_buffer.offset = offset
        external_buffer.length = length
        external_buffer.packing = "unpacked"
        model.externalBuffers.append(external_buffer)

        tensor.externalBuffer = external_buffer_id
        candidate_buffers.add(buffer_idx)
        tensor.buffer = 0
        externalized_tensors.add(tensor_key)

  still_referenced = _referenced_buffers(model, externalized_tensors)
  cleared_buffers = 0
  for buffer_idx in candidate_buffers:
    if buffer_idx not in still_referenced and 0 <= buffer_idx < len(
        model.buffers
    ):
      model.buffers[buffer_idx].data = None
      model.buffers[buffer_idx].offset = 0
      model.buffers[buffer_idx].size = 0
      cleared_buffers += 1

  if not externalized_tensors:
    # Nothing changed, so keep the input model (including any appended buffer
    # data) byte-for-byte instead of repacking it.
    output_model.write_bytes(input_bytes)
  else:
    _write_model(model, appended_buffers, output_model)

  logging.info(
      "Externalized %s: wrote %s and %s (%s bytes, %s newly externalized);"
      " externalized tensors: %s, cleared buffers: %s, model size: %.2fMB ->"
      " %.2fMB",
      input_model,
      output_model,
      output_weights,
      f"{output_weights.stat().st_size:,}",
      f"{total_externalized_bytes:,}",
      f"{len(externalized_tensors):,}",
      f"{cleared_buffers:,}",
      input_model.stat().st_size / 1024 / 1024,
      output_model.stat().st_size / 1024 / 1024,
  )


# Identity of a weight payload: (byte length, sha256 digest).
WeightDigest = tuple[int, bytes]


def _weight_digest(data) -> WeightDigest:
  return (len(data), hashlib.sha256(data).digest())


def _find_group(model, group_name: str) -> int | None:
  for idx, group in enumerate(model.externalBufferGroups or []):
    name = (
        group.name.decode("utf-8")
        if isinstance(group.name, bytes)
        else group.name
    )
    if name == group_name:
      return idx
  return None


@contextlib.contextmanager
def _map_weights(weights: pathlib.Path) -> Iterator[mmap.mmap | bytes]:
  """Yields a read-only view of `weights` (empty bytes when the file is empty)."""
  if weights.stat().st_size == 0:
    yield b""
    return
  with weights.open("rb") as weights_file:
    with mmap.mmap(weights_file.fileno(), 0, access=mmap.ACCESS_READ) as blob:
      yield blob


def collect_external_weight_digests(
    input_model: pathlib.Path, weights: pathlib.Path, group_name: str
) -> dict[WeightDigest, tuple[int, int]]:
  """Returns where each external payload of `input_model` lives in `weights`.

  Args:
    input_model: An externalized `.tflite` file.
    weights: The weight blob backing `group_name` (e.g. the output of
      `externalize`).
    group_name: External buffer group whose payloads are inspected.

  Returns:
    A mapping from payload digest to its `(offset, length)` inside `weights`.
    Empty when the model has no external buffers in `group_name`.

  Raises:
    ValueError: If an external buffer range exceeds `weights`.
  """
  model = schema.ModelT.InitFromPackedBuf(input_model.read_bytes(), 0)
  group_idx = _find_group(model, group_name)
  if group_idx is None:
    return {}
  placements: dict[WeightDigest, tuple[int, int]] = {}
  with _map_weights(weights) as blob:
    for external_buffer in model.externalBuffers or []:
      if int(external_buffer.group) != group_idx:
        continue
      offset = int(external_buffer.offset)
      length = int(external_buffer.length)
      if offset + length > len(blob):
        raise ValueError(
            f"External buffer {external_buffer.id} [{offset},"
            f" {offset + length}) exceeds weight file {weights}"
            f" ({len(blob)} bytes)"
        )
      digest = _weight_digest(blob[offset : offset + length])
      placements.setdefault(digest, (offset, length))
  return placements


@dataclasses.dataclass(frozen=True)
class RedirectResult:
  """Summary of one `redirect_external_weights` call.

  Attributes:
    redirected_buffers: Number of external buffers now pointing at another
      group.
    removed_bytes: Bytes removed from the model's own weight blob.
  """

  redirected_buffers: int
  removed_bytes: int


def redirect_external_weights(
    input_model: pathlib.Path,
    input_weights: pathlib.Path,
    output_dir: pathlib.Path,
    group_name: str,
    redirects: Mapping[WeightDigest, tuple[str, int, int]],
) -> RedirectResult:
  """Points external payloads at copies stored in other weight groups.

  Every external buffer of `input_model` in `group_name` whose payload digest
  is a key of `redirects` is rewritten to reference `(target_group, offset,
  length)` instead. The model's own blob is then compacted so that it only
  keeps payloads that are still referenced (64-byte aligned, as written by
  `externalize`).

  Args:
    input_model: An externalized `.tflite` file.
    input_weights: The weight blob backing `group_name` for `input_model`.
    output_dir: Directory receiving `model.tflite` and the compacted blob named
      `group_name`.
    group_name: The model's own external buffer group.
    redirects: Maps a payload digest to the `(group, offset, length)` where an
      identical payload is stored. `group` must differ from `group_name`.

  Returns:
    A `RedirectResult`. When nothing matches, both outputs are byte-identical
    copies of the inputs.

  Raises:
    ValueError: If a redirect target group equals `group_name` or has a
      mismatched payload length.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  output_model = output_dir / "model.tflite"
  output_weights = output_dir / group_name

  input_bytes = input_model.read_bytes()
  model = schema.ModelT.InitFromPackedBuf(input_bytes, 0)
  group_idx = _find_group(model, group_name)
  if group_idx is None or not redirects:
    shutil.copyfile(input_model, output_model)
    shutil.copyfile(input_weights, output_weights)
    return RedirectResult(redirected_buffers=0, removed_bytes=0)
  appended_buffers = _materialize_appended_buffer_data(model, input_bytes)

  with _map_weights(input_weights) as blob:
    kept = []
    redirected = 0
    for external_buffer in model.externalBuffers or []:
      if int(external_buffer.group) != group_idx:
        continue
      offset = int(external_buffer.offset)
      length = int(external_buffer.length)
      target = redirects.get(_weight_digest(blob[offset : offset + length]))
      if target is None:
        kept.append(external_buffer)
        continue
      target_group, target_offset, target_length = target
      if target_group == group_name:
        raise ValueError(
            f"Redirect target group must differ from {group_name!r}"
        )
      if target_length != length:
        raise ValueError(
            f"Redirect length {target_length} != payload length {length}"
        )
      external_buffer.group = _ensure_group(model, target_group)
      external_buffer.offset = target_offset
      redirected += 1

    if not redirected:
      shutil.copyfile(input_model, output_model)
      shutil.copyfile(input_weights, output_weights)
      return RedirectResult(redirected_buffers=0, removed_bytes=0)

    new_offsets: dict[tuple[int, int], int] = {}
    with output_weights.open("wb") as weights_file:
      for external_buffer in sorted(kept, key=lambda b: int(b.offset)):
        key = (int(external_buffer.offset), int(external_buffer.length))
        if key not in new_offsets:
          padding = (
              -weights_file.tell()
          ) % _LITERT_HOST_MEMORY_BUFFER_ALIGNMENT
          if padding:
            weights_file.write(b"\0" * padding)
          new_offsets[key] = weights_file.tell()
          weights_file.write(blob[key[0] : key[0] + key[1]])
        external_buffer.offset = new_offsets[key]

  _write_model(model, appended_buffers, output_model)
  removed_bytes = input_weights.stat().st_size - output_weights.stat().st_size
  logging.info(
      "Redirected %d external buffer(s) of %s; weight file %s: %s bytes (%s"
      " removed)",
      redirected,
      input_model,
      output_weights,
      f"{output_weights.stat().st_size:,}",
      f"{removed_bytes:,}",
  )
  return RedirectResult(
      redirected_buffers=redirected, removed_bytes=removed_bytes
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_model", type=pathlib.Path, required=True)
  parser.add_argument("--output_dir", type=pathlib.Path, required=True)
  parser.add_argument("--group_name", default="tflite_weights")
  parser.add_argument("--num_elements_threshold", type=int, default=256)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  externalize(
      input_model=args.input_model,
      output_dir=args.output_dir,
      group_name=args.group_name,
      num_elements_threshold=args.num_elements_threshold,
  )


if __name__ == "__main__":
  main()
