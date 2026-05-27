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

"""Utilities for compiling PEFT audio LoRA weights for LiteRT-LM.

LiteRT-LM loads LoRA weights through `runtime/util/lora_data.cc`. The expected
artifact is a small TFLite flatbuffer whose tensors are named exactly like the
LoRA input tensors in the target graph. Tensor payloads are stored as external
buffer ranges inside the same file via the TFLite Buffer `offset` and `size`
fields.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import flatbuffers


_LORA_RANK_METADATA_NAME = "lora_rank"
_TFLITE_FILE_IDENTIFIER = b"TFL3"
_DEFAULT_PAYLOAD_ALIGNMENT = 64 * 1024
_MMAP_SAFE_ALIGNMENT = 64 * 1024
_TFLITE_MODEL_VERSION = 3
_TENSOR_TYPE_FLOAT32 = 0
_TENSOR_TYPE_FLOAT16 = 1
_TENSOR_TYPE_INT32 = 2
_TENSOR_TYPE_UINT8 = 3
_TENSOR_TYPE_INT16 = 7
_TENSOR_TYPE_INT8 = 9
_TENSOR_TYPE_UINT32 = 15
_TENSOR_TYPE_UINT16 = 16
_TENSOR_TYPE_BFLOAT16 = 18

# Minimal TFLite schema field offsets used by the sidecar compiler.
#
# LiteRT-LM already depends on TensorFlow Lite FlatBuffers at runtime, but this
# Python builder intentionally avoids a PyPI `tflite` dependency. Keep this
# reader/writer private and narrow: it only touches Model/SubGraph/Tensor/Buffer
# fields needed for LoRA sidecars and target-graph input validation. Runtime
# compatibility is covered by runtime/util/lora_data_test.cc.
_MODEL_VERSION = 4
_MODEL_OPERATOR_CODES = 6
_MODEL_SUBGRAPHS = 8
_MODEL_DESCRIPTION = 10
_MODEL_BUFFERS = 12
_MODEL_METADATA = 16
_SUBGRAPH_TENSORS = 4
_SUBGRAPH_INPUTS = 6
_SUBGRAPH_OUTPUTS = 8
_SUBGRAPH_OPERATORS = 10
_SUBGRAPH_NAME = 12
_TENSOR_SHAPE = 4
_TENSOR_TYPE = 6
_TENSOR_BUFFER = 8
_TENSOR_NAME = 10
_BUFFER_OFFSET = 6
_BUFFER_SIZE = 8
_METADATA_NAME = 4
_METADATA_BUFFER = 6
_OPERATOR_INPUTS = 6

_LITERT_LORA_INPUT_RE = re.compile(
    r"^(?:(?:(query|key|value|post)_w_prime_(left|right))|"
    r"(?:lora_atten_(q|k|v|o)_(a|b)_prime_weight)|"
    r"(?:lora_audio_attn_(q|k|v|o)_(a|b)_weight)|"
    r"(?:lora_audio_"
    r"(ff1_l1|ff1_l2|ff2_l1|ff2_l2|lconv_start|lconv_end|output_proj)"
    r"_(a|b)_weight))_(\d+)$"
)

_PEFT_AUDIO_ATTENTION_RE = re.compile(
    r"(?:^|.*\.)audio_tower\.layers\.(?P<layer>\d+)\."
    r"(?:(?:self_)?attn|attention)\."
    r"(?P<proj>q_proj|k_proj|v_proj|o_proj|post)"
    r"(?:\.linear)?\."
    r"lora_(?P<side>A|B)\.weight$"
)

_PEFT_AUDIO_TOWER_RE = re.compile(
    r"(?:^|.*\.)audio_tower\.layers\.(?P<layer>\d+)\."
    r"(?P<module>"
    r"feed_forward1\.ffw_layer_1\.linear|"
    r"feed_forward1\.ffw_layer_2\.linear|"
    r"feed_forward2\.ffw_layer_1\.linear|"
    r"feed_forward2\.ffw_layer_2\.linear|"
    r"lconv1d\.linear_start\.linear|"
    r"lconv1d\.linear_end\.linear)\."
    r"lora_(?P<side>A|B)\.weight$"
)

_PEFT_AUDIO_OUTPUT_RE = re.compile(
    r"(?:^|.*\.)audio_tower\.output_proj\.lora_(?P<side>A|B)\.weight$"
)

_PROJ_TO_LITERT = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "post": "o",
}

_AUDIO_MODULE_TO_LITERT = {
    "feed_forward1.ffw_layer_1.linear": "ff1_l1",
    "feed_forward1.ffw_layer_2.linear": "ff1_l2",
    "feed_forward2.ffw_layer_1.linear": "ff2_l1",
    "feed_forward2.ffw_layer_2.linear": "ff2_l2",
    "lconv1d.linear_start.linear": "lconv_start",
    "lconv1d.linear_end.linear": "lconv_end",
}

_LORA_PROJ_ORDER = {
    "q": 0,
    "query": 0,
    "k": 1,
    "key": 1,
    "v": 2,
    "value": 2,
    "o": 3,
    "post": 3,
    "ff1_l1": 4,
    "ff1_l2": 5,
    "lconv_start": 6,
    "lconv_end": 7,
    "ff2_l1": 8,
    "ff2_l2": 9,
    "output_proj": 10,
}
_LORA_SIDE_ORDER = {"a": 0, "left": 0, "b": 1, "right": 1}


@dataclasses.dataclass(frozen=True)
class LoraTensor:
  """A tensor payload to place in a LiteRT-LM LoRA TFLite sidecar."""

  name: str
  data: bytes
  shape: tuple[int, ...]
  tensor_type: int = _TENSOR_TYPE_FLOAT16


@dataclasses.dataclass(frozen=True)
class TargetTensorSpec:
  """LoRA input tensor contract read from a target TFLite graph."""

  name: str
  shape: tuple[int, ...]
  tensor_type: int
  byte_size: int
  is_consumed: bool = True


class _RawSafeTensorArray:
  """Small safetensors array wrapper used when NumPy is unavailable.

  The public converter is often run from a Bazel py_binary where optional NumPy
  native extensions may be absent. Audio LoRA conversion only needs a narrow
  subset of array behavior: shape inspection, optional 2-D transpose, scalar
  multiplication, dtype conversion to the target TFLite input type, and C-order
  byte emission.
  """

  def __init__(
      self,
      raw_data: bytes,
      shape: Sequence[int],
      source_dtype: str,
      target_dtype: str | None = None,
      scale: float = 1.0,
      transposed: bool = False,
  ):
    self._raw_data = raw_data
    self._source_dtype = source_dtype
    self._target_dtype = target_dtype or source_dtype
    self._scale = scale
    self._transposed = transposed
    self._source_shape = tuple(int(dim) for dim in shape)
    self.shape = (
        (self._source_shape[1], self._source_shape[0])
        if transposed and len(self._source_shape) == 2
        else self._source_shape
    )
    self.ndim = len(self.shape)

  @property
  def T(self):
    if self.ndim != 2:
      return self
    return _RawSafeTensorArray(
        raw_data=self._raw_data,
        shape=self._source_shape,
        source_dtype=self._source_dtype,
        target_dtype=self._target_dtype,
        scale=self._scale,
        transposed=not self._transposed,
    )

  @property
  def nbytes(self) -> int:
    count = 1
    for dim in self.shape:
      count *= dim
    return count * _safe_tensor_dtype_nbytes(self._target_dtype)

  def __mul__(self, scale: float):
    return _RawSafeTensorArray(
        raw_data=self._raw_data,
        shape=self._source_shape,
        source_dtype=self._source_dtype,
        target_dtype=self._target_dtype,
        scale=self._scale * float(scale),
        transposed=self._transposed,
    )

  def astype(self, dtype: Any):
    return _RawSafeTensorArray(
        raw_data=self._raw_data,
        shape=self._source_shape,
        source_dtype=self._source_dtype,
        target_dtype=_normalize_float_dtype_name(dtype),
        scale=self._scale,
        transposed=self._transposed,
    )

  def tobytes(self, order: str = "C") -> bytes:
    if order != "C":
      raise ValueError("only C-order safetensors emission is supported")
    if (
        not self._transposed
        and self._scale == 1.0
        and self._source_dtype == self._target_dtype
    ):
      return self._raw_data

    target_dtype = _normalize_float_dtype_name(self._target_dtype)
    if target_dtype not in ("F16", "F32"):
      raise ValueError(f"unsupported target dtype without NumPy: {target_dtype}")

    values = self._iter_values()
    if target_dtype == "F16":
      output = bytearray(self.nbytes)
      for index, value in enumerate(values):
        struct.pack_into("<e", output, index * 2, value)
      return bytes(output)

    output = bytearray(self.nbytes)
    for index, value in enumerate(values):
      struct.pack_into("<f", output, index * 4, value)
    return bytes(output)

  def _iter_values(self):
    if self._transposed and len(self._source_shape) == 2:
      rows, cols = self._source_shape
      for col in range(cols):
        for row in range(rows):
          yield self._read_value(row * cols + col) * self._scale
      return
    for index in range(self._element_count()):
      yield self._read_value(index) * self._scale

  def _element_count(self) -> int:
    count = 1
    for dim in self._source_shape:
      count *= dim
    return count

  def _read_value(self, index: int) -> float:
    if self._source_dtype == "F32":
      return struct.unpack_from("<f", self._raw_data, index * 4)[0]
    if self._source_dtype == "F16":
      return struct.unpack_from("<e", self._raw_data, index * 2)[0]
    if self._source_dtype == "BF16":
      raw = struct.unpack_from("<H", self._raw_data, index * 2)[0]
      return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    raise ValueError(f"unsupported safetensors dtype without NumPy: {self._source_dtype}")


@dataclasses.dataclass(frozen=True)
class CompileReport:
  """Summary of a PEFT-to-LiteRT-LM Audio LoRA conversion."""

  lora_rank: int
  lora_alpha: float
  scale: float
  tensors_written: tuple[str, ...]
  target_inputs: tuple[str, ...]
  missing_target_inputs: tuple[str, ...]
  unmatched_peft_keys: tuple[str, ...]
  unused_peft_keys: tuple[str, ...]
  unconsumed_target_inputs: tuple[str, ...]

  def to_json_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def is_litert_lora_input_name(name: str) -> bool:
  return _LITERT_LORA_INPUT_RE.fullmatch(name) is not None


def _litert_lora_sort_key(name: str) -> tuple[int, int, int, str]:
  match = _LITERT_LORA_INPUT_RE.fullmatch(name)
  if match is None:
    return (2**31 - 1, 2**31 - 1, 2**31 - 1, name)
  (
      legacy_proj,
      legacy_side,
      atten_proj,
      atten_side,
      audio_atten_proj,
      audio_atten_side,
      audio_module,
      audio_side,
      layer,
  ) = match.groups()
  proj = atten_proj or audio_atten_proj or legacy_proj or audio_module
  side = atten_side or audio_atten_side or legacy_side or audio_side
  return (
      int(layer),
      _LORA_PROJ_ORDER.get(proj, 2**31 - 1),
      _LORA_SIDE_ORDER.get(side, 2**31 - 1),
      name,
  )


def peft_key_to_litert_lora_name(key: str) -> str | None:
  """Maps common PEFT audio LoRA keys to LiteRT-LM audio input names."""

  match = _PEFT_AUDIO_ATTENTION_RE.fullmatch(key)
  if match is not None:
    proj = _PROJ_TO_LITERT[match.group("proj")]
    side = match.group("side").lower()
    layer = match.group("layer")
    return f"lora_audio_attn_{proj}_{side}_weight_{layer}"

  match = _PEFT_AUDIO_TOWER_RE.fullmatch(key)
  if match is not None:
    module = _AUDIO_MODULE_TO_LITERT[match.group("module")]
    side = match.group("side").lower()
    layer = match.group("layer")
    return f"lora_audio_{module}_{side}_weight_{layer}"

  match = _PEFT_AUDIO_OUTPUT_RE.fullmatch(key)
  if match is not None:
    side = match.group("side").lower()
    return f"lora_audio_output_proj_{side}_weight_0"

  return None


def _align_up(value: int, alignment: int) -> int:
  if alignment <= 0:
    raise ValueError("alignment must be positive")
  return ((value + alignment - 1) // alignment) * alignment


def _read_u16(data: bytes, offset: int) -> int:
  return struct.unpack_from("<H", data, offset)[0]


def _read_i32(data: bytes, offset: int) -> int:
  return struct.unpack_from("<i", data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
  return struct.unpack_from("<I", data, offset)[0]


def _read_u64(data: bytes, offset: int) -> int:
  return struct.unpack_from("<Q", data, offset)[0]


def _read_i8(data: bytes, offset: int) -> int:
  return struct.unpack_from("<b", data, offset)[0]


def _table_field_pos(data: bytes, table_pos: int, vtable_offset: int) -> int | None:
  vtable_pos = table_pos - _read_i32(data, table_pos)
  if vtable_offset >= _read_u16(data, vtable_pos):
    return None
  field_offset = _read_u16(data, vtable_pos + vtable_offset)
  if field_offset == 0:
    return None
  return table_pos + field_offset


def _table_u32(data: bytes, table_pos: int, vtable_offset: int, default: int = 0):
  field_pos = _table_field_pos(data, table_pos, vtable_offset)
  return default if field_pos is None else _read_u32(data, field_pos)


def _table_u64(data: bytes, table_pos: int, vtable_offset: int, default: int = 0):
  field_pos = _table_field_pos(data, table_pos, vtable_offset)
  return default if field_pos is None else _read_u64(data, field_pos)


def _table_i8(data: bytes, table_pos: int, vtable_offset: int, default: int = 0):
  field_pos = _table_field_pos(data, table_pos, vtable_offset)
  return default if field_pos is None else _read_i8(data, field_pos)


def _indirect(data: bytes, offset_pos: int) -> int:
  return offset_pos + _read_u32(data, offset_pos)


def _table_vector(data: bytes, table_pos: int, vtable_offset: int) -> int | None:
  field_pos = _table_field_pos(data, table_pos, vtable_offset)
  return None if field_pos is None else _indirect(data, field_pos)


def _vector_length(data: bytes, vector_pos: int | None) -> int:
  return 0 if vector_pos is None else _read_u32(data, vector_pos)


def _vector_i32(data: bytes, vector_pos: int | None) -> tuple[int, ...]:
  if vector_pos is None:
    return ()
  return tuple(
      _read_i32(data, vector_pos + 4 + 4 * i)
      for i in range(_vector_length(data, vector_pos))
  )


def _vector_tables(data: bytes, vector_pos: int | None) -> tuple[int, ...]:
  if vector_pos is None:
    return ()
  tables = []
  for i in range(_vector_length(data, vector_pos)):
    element_pos = vector_pos + 4 + 4 * i
    tables.append(_indirect(data, element_pos))
  return tuple(tables)


def _table_string(data: bytes, table_pos: int, vtable_offset: int) -> str:
  field_pos = _table_field_pos(data, table_pos, vtable_offset)
  if field_pos is None:
    return ""
  string_pos = _indirect(data, field_pos)
  size = _read_u32(data, string_pos)
  return data[string_pos + 4 : string_pos + 4 + size].decode("utf-8")


class _TfliteBuffer:
  def __init__(self, data: bytes, table_pos: int):
    self._data = data
    self._table_pos = table_pos

  def offset(self) -> int:
    return _table_u64(self._data, self._table_pos, _BUFFER_OFFSET)

  def size(self) -> int:
    return _table_u64(self._data, self._table_pos, _BUFFER_SIZE)


class _TfliteTensor:
  def __init__(self, data: bytes, table_pos: int):
    self._data = data
    self._table_pos = table_pos

  def name(self) -> str:
    return _table_string(self._data, self._table_pos, _TENSOR_NAME)

  def shape(self) -> tuple[int, ...]:
    return _vector_i32(
        self._data, _table_vector(self._data, self._table_pos, _TENSOR_SHAPE)
    )

  def tensor_type(self) -> int:
    return _table_i8(self._data, self._table_pos, _TENSOR_TYPE)

  def buffer_index(self) -> int:
    return _table_u32(self._data, self._table_pos, _TENSOR_BUFFER)


class _TfliteSubgraph:
  def __init__(self, data: bytes, table_pos: int):
    self._data = data
    self._table_pos = table_pos

  def inputs(self) -> tuple[int, ...]:
    return _vector_i32(
        self._data, _table_vector(self._data, self._table_pos, _SUBGRAPH_INPUTS)
    )

  def tensor(self, index: int) -> _TfliteTensor:
    tensors = _vector_tables(
        self._data, _table_vector(self._data, self._table_pos, _SUBGRAPH_TENSORS)
    )
    return _TfliteTensor(self._data, tensors[index])

  def tensors_length(self) -> int:
    return _vector_length(
        self._data, _table_vector(self._data, self._table_pos, _SUBGRAPH_TENSORS)
    )

  def operator_inputs(self) -> tuple[int, ...]:
    inputs = []
    operators = _vector_tables(
        self._data,
        _table_vector(self._data, self._table_pos, _SUBGRAPH_OPERATORS),
    )
    for operator in operators:
      inputs.extend(
          _vector_i32(self._data, _table_vector(self._data, operator, _OPERATOR_INPUTS))
      )
    return tuple(inputs)


class _TfliteModelReader:
  def __init__(self, data: bytes):
    if len(data) < 8 or data[4:8] != _TFLITE_FILE_IDENTIFIER:
      raise ValueError("input is not a TFLite flatbuffer")
    self._data = data
    self._root = _read_u32(data, 0)

  def version(self) -> int:
    return _table_u32(self._data, self._root, _MODEL_VERSION)

  def subgraph(self, index: int) -> _TfliteSubgraph:
    subgraphs = _vector_tables(
        self._data, _table_vector(self._data, self._root, _MODEL_SUBGRAPHS)
    )
    return _TfliteSubgraph(self._data, subgraphs[index])

  def subgraphs_length(self) -> int:
    return _vector_length(
        self._data, _table_vector(self._data, self._root, _MODEL_SUBGRAPHS)
    )

  def buffer(self, index: int) -> _TfliteBuffer:
    buffers = _vector_tables(
        self._data, _table_vector(self._data, self._root, _MODEL_BUFFERS)
    )
    return _TfliteBuffer(self._data, buffers[index])

  def buffers_length(self) -> int:
    return _vector_length(
        self._data, _table_vector(self._data, self._root, _MODEL_BUFFERS)
    )

  def metadata_buffer(self, name: str) -> int | None:
    metadata_tables = _vector_tables(
        self._data, _table_vector(self._data, self._root, _MODEL_METADATA)
    )
    for metadata in metadata_tables:
      if _table_string(self._data, metadata, _METADATA_NAME) == name:
        return _table_u32(self._data, metadata, _METADATA_BUFFER)
    return None


def _create_offsets_vector(builder, offsets: Sequence[int], start_fn) -> int:
  del start_fn
  builder.StartVector(4, len(offsets), 4)
  for offset in reversed(offsets):
    builder.PrependUOffsetTRelative(offset)
  return builder.EndVector()


def _create_i32_vector(builder, values: Sequence[int], start_fn) -> int:
  del start_fn
  builder.StartVector(4, len(values), 4)
  for value in reversed(values):
    builder.PrependInt32(value)
  return builder.EndVector()


def _finish_object(builder) -> int:
  return builder.EndObject()


def _build_lora_model_header(
    tensors: Sequence[LoraTensor],
    lora_rank: int,
    payload_offsets: Sequence[int],
) -> bytes:
  builder = flatbuffers.Builder(4096)
  tensor_names = [builder.CreateString(tensor.name) for tensor in tensors]
  description = builder.CreateString("LiteRT-LM LoRA sidecar")
  metadata_name = builder.CreateString(_LORA_RANK_METADATA_NAME)
  subgraph_name = builder.CreateString("lora")

  shape_offsets = [
      _create_i32_vector(builder, tensor.shape, None)
      for tensor in tensors
  ]

  tensor_offsets = []
  for i, (tensor, shape_offset) in enumerate(zip(tensors, shape_offsets)):
    builder.StartObject(11)
    builder.PrependUOffsetTRelativeSlot(0, shape_offset, 0)
    builder.PrependInt8Slot(1, tensor.tensor_type, 0)
    builder.PrependUint32Slot(2, i + 1, 0)
    builder.PrependUOffsetTRelativeSlot(3, tensor_names[i], 0)
    tensor_offsets.append(_finish_object(builder))

  tensors_vector = _create_offsets_vector(
      builder, tensor_offsets, None
  )
  inputs_vector = _create_i32_vector(builder, (), None)
  outputs_vector = _create_i32_vector(builder, (), None)
  operators_vector = _create_offsets_vector(builder, (), None)

  builder.StartObject(6)
  builder.PrependUOffsetTRelativeSlot(0, tensors_vector, 0)
  builder.PrependUOffsetTRelativeSlot(1, inputs_vector, 0)
  builder.PrependUOffsetTRelativeSlot(2, outputs_vector, 0)
  builder.PrependUOffsetTRelativeSlot(3, operators_vector, 0)
  builder.PrependUOffsetTRelativeSlot(4, subgraph_name, 0)
  subgraph = _finish_object(builder)

  buffer_offsets = []
  builder.StartObject(3)
  buffer_offsets.append(_finish_object(builder))
  for tensor, payload_offset in zip(tensors, payload_offsets):
    builder.StartObject(3)
    builder.PrependUint64Slot(1, payload_offset, 0)
    builder.PrependUint64Slot(2, len(tensor.data), 0)
    buffer_offsets.append(_finish_object(builder))

  while len(buffer_offsets) <= lora_rank:
    builder.StartObject(3)
    buffer_offsets.append(_finish_object(builder))

  buffers_vector = _create_offsets_vector(
      builder, buffer_offsets, None
  )
  subgraphs_vector = _create_offsets_vector(
      builder, [subgraph], None
  )
  opcodes_vector = _create_offsets_vector(builder, (), None)

  builder.StartObject(2)
  builder.PrependUOffsetTRelativeSlot(0, metadata_name, 0)
  # LiteRT-LM's LoraData currently reads the LoRA rank from Metadata.buffer().
  builder.PrependUint32Slot(1, lora_rank, 0)
  metadata = _finish_object(builder)
  metadata_vector = _create_offsets_vector(builder, [metadata], None)

  builder.StartObject(10)
  builder.PrependUint32Slot(0, _TFLITE_MODEL_VERSION, 0)
  builder.PrependUOffsetTRelativeSlot(1, opcodes_vector, 0)
  builder.PrependUOffsetTRelativeSlot(2, subgraphs_vector, 0)
  builder.PrependUOffsetTRelativeSlot(3, description, 0)
  builder.PrependUOffsetTRelativeSlot(4, buffers_vector, 0)
  builder.PrependUOffsetTRelativeSlot(6, metadata_vector, 0)
  root = _finish_object(builder)
  builder.Finish(root, file_identifier=_TFLITE_FILE_IDENTIFIER)
  return bytes(builder.Output())


def build_lora_tflite_bytes(
    tensors: Sequence[LoraTensor],
    lora_rank: int,
    payload_alignment: int = _DEFAULT_PAYLOAD_ALIGNMENT,
) -> bytes:
  """Builds a LiteRT-LM LoRA TFLite sidecar."""

  if not tensors:
    raise ValueError("at least one LoRA tensor is required")
  if lora_rank <= 0:
    raise ValueError("lora_rank must be positive")
  for tensor in tensors:
    if not is_litert_lora_input_name(tensor.name):
      raise ValueError(f"unsupported LiteRT-LM LoRA tensor name: {tensor.name}")

  payload_base = payload_alignment
  while True:
    offsets = []
    cursor = payload_base
    for tensor in tensors:
      offsets.append(cursor)
      cursor += len(tensor.data)
    header = _build_lora_model_header(tensors, lora_rank, offsets)
    if len(header) <= payload_base:
      break
    payload_base = _align_up(len(header), payload_alignment)

  output = bytearray(_align_up(cursor, max(payload_alignment, _MMAP_SAFE_ALIGNMENT)))
  output[: len(header)] = header
  for tensor, offset in zip(tensors, offsets):
    output[offset : offset + len(tensor.data)] = tensor.data
  return bytes(output)


def write_lora_tflite(
    tensors: Sequence[LoraTensor],
    lora_rank: int,
    output_path: os.PathLike[str] | str,
    payload_alignment: int = _DEFAULT_PAYLOAD_ALIGNMENT,
) -> None:
  pathlib.Path(output_path).write_bytes(
      build_lora_tflite_bytes(tensors, lora_rank, payload_alignment)
  )


def _float_dtype_for_tensor_type(tensor_type: int) -> str:
  if tensor_type == _TENSOR_TYPE_FLOAT16:
    return "F16"
  if tensor_type == _TENSOR_TYPE_FLOAT32:
    return "F32"
  raise ValueError(f"unsupported LoRA target tensor type: {tensor_type}")


def _tensor_type_nbytes(tensor_type: int) -> int:
  if tensor_type in (
      _TENSOR_TYPE_FLOAT16,
      _TENSOR_TYPE_BFLOAT16,
      _TENSOR_TYPE_INT16,
      _TENSOR_TYPE_UINT16,
  ):
    return 2
  if tensor_type in (
      _TENSOR_TYPE_FLOAT32,
      _TENSOR_TYPE_INT32,
      _TENSOR_TYPE_UINT32,
  ):
    return 4
  if tensor_type in (_TENSOR_TYPE_INT8, _TENSOR_TYPE_UINT8):
    return 1
  raise ValueError(f"unsupported LoRA input tensor type: {tensor_type}")


def _normalize_float_dtype_name(dtype: Any) -> str:
  dtype_name = getattr(dtype, "__name__", str(dtype)).lower()
  if dtype_name in ("f16", "float16", "<class 'numpy.float16'>"):
    return "F16"
  if dtype_name in ("f32", "float32", "<class 'numpy.float32'>"):
    return "F32"
  if dtype_name in ("bf16", "bfloat16"):
    return "BF16"
  return str(dtype).upper()


def _safe_tensor_dtype_nbytes(dtype: str) -> int:
  dtype = _normalize_float_dtype_name(dtype)
  if dtype in ("F16", "BF16"):
    return 2
  if dtype == "F32":
    return 4
  raise ValueError(f"unsupported safetensors dtype without NumPy: {dtype}")


def _load_safetensors_without_numpy(file_path: pathlib.Path) -> dict[str, Any]:
  """Loads a safetensors file as raw tensor wrappers without importing NumPy."""

  data = file_path.read_bytes()
  if len(data) < 8:
    raise ValueError(f"{file_path} is too small to be a safetensors file")
  header_size = struct.unpack_from("<Q", data, 0)[0]
  header_start = 8
  header_end = header_start + header_size
  if header_end > len(data):
    raise ValueError(f"{file_path} has an invalid safetensors header size")
  header = json.loads(data[header_start:header_end])
  tensors = {}
  for name, spec in header.items():
    if name == "__metadata__":
      continue
    dtype = _normalize_float_dtype_name(spec["dtype"])
    shape = tuple(int(dim) for dim in spec["shape"])
    begin, end = (int(offset) for offset in spec["data_offsets"])
    byte_count = end - begin
    expected = 1
    for dim in shape:
      expected *= dim
    expected *= _safe_tensor_dtype_nbytes(dtype)
    if byte_count != expected:
      raise ValueError(
          f"{file_path}:{name} byte count mismatch: {byte_count} vs {expected}"
      )
    tensors[name] = _RawSafeTensorArray(
        raw_data=data[header_end + begin : header_end + end],
        shape=shape,
        source_dtype=dtype,
    )
  return tensors


def extract_lora_input_specs(
    target_tflite_path: os.PathLike[str] | str,
    signature_subgraph_index: int = 0,
) -> dict[str, TargetTensorSpec]:
  """Extracts LiteRT-LM LoRA input specs from a target TFLite graph."""

  data = pathlib.Path(target_tflite_path).read_bytes()
  model = _TfliteModelReader(data)
  subgraph = model.subgraph(signature_subgraph_index)
  input_tensor_indices = subgraph.inputs()
  consumed_tensor_indices = set(subgraph.operator_inputs())

  specs = {}
  for tensor_index in input_tensor_indices:
    tensor_index = int(tensor_index)
    tensor = subgraph.tensor(tensor_index)
    name = tensor.name()
    if not is_litert_lora_input_name(name):
      continue
    shape = tensor.shape()
    tensor_type = tensor.tensor_type()
    nbytes = _tensor_type_nbytes(tensor_type)
    num_elements = 1
    for dim in shape:
      num_elements *= dim
    specs[name] = TargetTensorSpec(
        name=name,
        shape=shape,
        tensor_type=tensor_type,
        byte_size=num_elements * nbytes,
        is_consumed=tensor_index in consumed_tensor_indices,
    )
  return specs


def _load_safetensors(peft_lora_dir: os.PathLike[str] | str):
  peft_path = pathlib.Path(peft_lora_dir)
  files = sorted(peft_path.glob("*.safetensors"))
  if not files:
    raise FileNotFoundError(f"no safetensors files found in {peft_path}")
  tensors = {}
  for file_path in files:
    tensors.update(_load_safetensors_without_numpy(file_path))
  return tensors


def _read_adapter_config(peft_lora_dir: os.PathLike[str] | str) -> dict[str, Any]:
  config_path = pathlib.Path(peft_lora_dir) / "adapter_config.json"
  if not config_path.exists():
    return {}
  return json.loads(config_path.read_text(encoding="utf-8"))


def _infer_rank_from_peft_tensors(peft_tensors: Mapping[str, Any]) -> int:
  ranks = set()
  for key, value in peft_tensors.items():
    if peft_key_to_litert_lora_name(key) is None:
      continue
    if key.endswith(".lora_A.weight"):
      ranks.add(int(value.shape[0]))
    elif key.endswith(".lora_B.weight"):
      ranks.add(int(value.shape[-1]))
  if len(ranks) != 1:
    raise ValueError(f"unable to infer a unique LoRA rank from tensors: {ranks}")
  return ranks.pop()


def _orient_to_target_shape(
    value: Any,
    target_shape: tuple[int, ...] | None,
    tensor_name: str,
) -> Any:
  if target_shape is None or len(target_shape) != 2:
    return value
  if tuple(value.shape) == target_shape:
    return value
  if value.ndim == 2 and tuple(value.T.shape) == target_shape:
    return value.T
  raise ValueError(
      f"{tensor_name} shape {tuple(value.shape)} does not match target "
      f"shape {target_shape}, with or without transpose"
  )


def _as_array(value: Any) -> Any:
  if hasattr(value, "shape") and hasattr(value, "astype"):
    return value
  raise ImportError(
      "PEFT tensors must provide shape/astype/tobytes. Safetensors files are "
      "loaded through the built-in raw safetensors reader."
  )


def _as_contiguous_typed_array(value: Any, dtype: str) -> Any:
  return value.astype(dtype)


def compile_peft_audio_lora(
    peft_lora_dir: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    target_tflite_path: os.PathLike[str] | str | None = None,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    payload_alignment: int = _DEFAULT_PAYLOAD_ALIGNMENT,
    peft_key_regex: str | None = None,
    fail_on_missing_target_inputs: bool = True,
    fail_on_unmatched_peft_keys: bool = True,
    fail_on_unused_peft_keys: bool = True,
    fail_on_unconsumed_target_inputs: bool = True,
) -> CompileReport:
  """Compiles a PEFT audio LoRA directory into a LiteRT-LM sidecar."""

  peft_tensors = _load_safetensors(peft_lora_dir)
  key_filter = re.compile(peft_key_regex) if peft_key_regex else None
  if key_filter is None:
    selected_peft_tensors = peft_tensors
  else:
    selected_peft_tensors = {
        key: value for key, value in peft_tensors.items() if key_filter.search(key)
    }

  config = _read_adapter_config(peft_lora_dir)
  rank = lora_rank or int(
      config.get("r") or _infer_rank_from_peft_tensors(selected_peft_tensors)
  )
  alpha = float(
      lora_alpha if lora_alpha is not None else config.get("lora_alpha", rank)
  )
  scale = alpha / rank
  target_specs = (
      extract_lora_input_specs(target_tflite_path) if target_tflite_path else {}
  )
  if target_tflite_path and not target_specs:
    raise ValueError(
        "target_tflite has no LiteRT-LM LoRA input tensors. The target audio "
        "graph must expose official LoRA input names such as "
        "lora_audio_attn_q_a_weight_0."
    )
  unconsumed = tuple(
      sorted(
          (name for name, spec in target_specs.items() if not spec.is_consumed),
          key=_litert_lora_sort_key,
      )
  )

  lora_arrays: dict[str, Any] = {}
  lora_sources: dict[str, str] = {}
  unmatched = []
  for key, value in selected_peft_tensors.items():
    target_name = peft_key_to_litert_lora_name(key)
    if target_name is None:
      unmatched.append(key)
      continue
    array = _as_array(value)
    if key.endswith(".lora_B.weight"):
      array = array * scale
    spec = target_specs.get(target_name)
    array = _orient_to_target_shape(
        array, spec.shape if spec is not None else None, target_name
    )
    if target_name in lora_arrays:
      raise ValueError(
          f"multiple PEFT tensors map to {target_name}: "
          f"{lora_sources[target_name]} and {key}. Use --peft_key_regex "
          "to select one component."
      )
    target_type = spec.tensor_type if spec is not None else _TENSOR_TYPE_FLOAT16
    target_dtype = _float_dtype_for_tensor_type(target_type)
    lora_arrays[target_name] = _as_contiguous_typed_array(array, target_dtype)
    lora_sources[target_name] = key

  if target_specs:
    missing = tuple(
        name for name in sorted(target_specs, key=_litert_lora_sort_key)
        if name not in lora_arrays
    )
    names_to_write = [
        name for name in sorted(target_specs, key=_litert_lora_sort_key)
        if name in lora_arrays
    ]
    unused = tuple(
        lora_sources[name]
        for name in sorted(lora_arrays, key=_litert_lora_sort_key)
        if name not in target_specs
    )
  else:
    missing = ()
    unused = ()
    names_to_write = sorted(lora_arrays, key=_litert_lora_sort_key)

  if fail_on_missing_target_inputs and missing:
    raise ValueError(
        "target graph has LoRA inputs with no matching PEFT tensor: "
        f"{', '.join(missing)}"
    )
  if fail_on_unmatched_peft_keys and unmatched:
    raise ValueError(
        "selected PEFT keys do not map to the LiteRT-LM LoRA ABI: "
        f"{', '.join(sorted(unmatched))}"
    )
  if fail_on_unused_peft_keys and unused:
    raise ValueError(
        "selected PEFT keys map to LoRA tensors absent from the target graph: "
        f"{', '.join(sorted(unused))}"
    )
  if fail_on_unconsumed_target_inputs and unconsumed:
    preview = ", ".join(unconsumed[:8])
    if len(unconsumed) > 8:
      preview += ", ..."
    raise ValueError(
        "target graph has LoRA inputs that are not consumed by any op: "
        f"{preview}"
    )

  tensors = []
  for name in names_to_write:
    array = lora_arrays[name]
    tensor_type = (
        target_specs[name].tensor_type
        if name in target_specs
        else _TENSOR_TYPE_FLOAT16
    )
    tensor = LoraTensor(
        name=name,
        data=array.tobytes(order="C"),
        shape=tuple(int(dim) for dim in array.shape) or (array.nbytes,),
        tensor_type=tensor_type,
    )
    if name in target_specs and len(tensor.data) != target_specs[name].byte_size:
      raise ValueError(
          f"{name} byte size mismatch: generated {len(tensor.data)} vs "
          f"target {target_specs[name].byte_size}"
      )
    tensors.append(tensor)

  write_lora_tflite(tensors, rank, output_path, payload_alignment)
  return CompileReport(
      lora_rank=rank,
      lora_alpha=alpha,
      scale=scale,
      tensors_written=tuple(tensor.name for tensor in tensors),
      target_inputs=tuple(sorted(target_specs, key=_litert_lora_sort_key)),
      missing_target_inputs=missing,
      unmatched_peft_keys=tuple(sorted(unmatched)),
      unused_peft_keys=tuple(sorted(unused)),
      unconsumed_target_inputs=unconsumed,
  )
