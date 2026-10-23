# Copyright 2025 The ODML Authors.
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


"""Builder class for LiteRT-LM files.

Example usage:
```
builder = litertlm_builder.LitertLmFileBuilder()
builder.add_system_metadata(
    litertlm_builder.Metadata(
        key="Authors",
        value="The ODML Authors",
        dtype=litertlm_builder.DType.STRING,
    )
)
builder.add_tflite_model(
    model_path,
    llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE,
)
builder.add_sentencepiece_tokenizer(tokenizer_path)
builder.add_llm_metadata(llm_metadata_path)
with litertlm_core.open_file(output_path, "wb") as f:
  builder.build(f)
```

Note: The build method uses `seek` to write the header and sections. The io
interface used must support `seek`.
"""

import contextlib
import dataclasses
import datetime
import enum
import io
import logging
import os
import pathlib
import shutil
from typing import Any, BinaryIO, Callable, Optional, TypeVar, cast
import uuid
import zlib

import flatbuffers
from google.protobuf import message
from google.protobuf import text_format
# TODO(b/514828153): Migrate to standard library tomllib when Python 3.10
# support is dropped.
import tomli as tomllib

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder import litertlm_peek
from runtime.proto import any2any_metadata_pb2
from runtime.proto import embedding_metadata_pb2
from runtime.proto import executor_metadata_pb2
from runtime.proto import llm_metadata_pb2
from runtime.proto import speech2text_metadata_pb2
from runtime.proto import text2image_metadata_pb2
from runtime.proto import text2speech_metadata_pb2


@enum.unique
class DType(enum.Enum):
  """DType enum.

  This enum maps to the data types defined in the LiteRT-LM flatbuffers schema.
  """

  INT8 = "int8"
  INT16 = "int16"
  INT32 = "int32"
  INT64 = "int64"
  UINT8 = "uint8"
  UINT16 = "uint16"
  UINT32 = "uint32"
  UINT64 = "uint64"
  FLOAT32 = "float32"
  DOUBLE = "double"
  BOOL = "bool"
  STRING = "string"


@dataclasses.dataclass
class Metadata:
  """Metadata class."""

  key: str
  value: Any
  dtype: DType

  @classmethod
  def from_key_value_pair(cls, kvp: schema.KeyValuePairT) -> "Metadata":
    """Creates a `Metadata` object from a `KeyValuePairT`."""
    assert kvp.key is not None
    if isinstance(key := kvp.key, bytes):
      key = key.decode()
    assert kvp.value is not None
    value = kvp.value.value
    match kvp.valueType:
      case schema.VData.UInt8:
        dtype = DType.UINT8
      case schema.VData.Int8:
        dtype = DType.INT8
      case schema.VData.UInt16:
        dtype = DType.UINT16
      case schema.VData.Int16:
        dtype = DType.INT16
      case schema.VData.UInt32:
        dtype = DType.UINT32
      case schema.VData.Int32:
        dtype = DType.INT32
      case schema.VData.UInt64:
        dtype = DType.UINT64
      case schema.VData.Int64:
        dtype = DType.INT64
      case schema.VData.Float32:
        dtype = DType.FLOAT32
      case schema.VData.Double:
        dtype = DType.DOUBLE
      case schema.VData.Bool:
        dtype = DType.BOOL
      case schema.VData.StringValue:
        dtype = DType.STRING
      case _:
        raise ValueError(f"Unsupported value type: {kvp.valueType}")
    return cls(key=key, value=value, dtype=dtype)

  def to_key_value_pair(self) -> schema.KeyValuePairT:
    """Converts the Metadata object to a `KeyValuePairT`."""
    match self.dtype:
      case DType.UINT8:
        value = schema.UInt8T(self.value)
        value_type = schema.VData.UInt8
      case DType.INT8:
        value = schema.Int8T(self.value)
        value_type = schema.VData.Int8
      case DType.UINT16:
        value = schema.UInt16T(self.value)
        value_type = schema.VData.UInt16
      case DType.INT16:
        value = schema.Int16T(self.value)
        value_type = schema.VData.Int16
      case DType.UINT32:
        value = schema.UInt32T(self.value)
        value_type = schema.VData.UInt32
      case DType.INT32:
        value = schema.Int32T(self.value)
        value_type = schema.VData.Int32
      case DType.FLOAT32:
        value = schema.Float32T(self.value)
        value_type = schema.VData.Float32
      case DType.BOOL:
        value = schema.BoolT(self.value)
        value_type = schema.VData.Bool
      case DType.STRING:
        value = schema.StringValueT(self.value)
        value_type = schema.VData.StringValue
      case DType.UINT64:
        value = schema.UInt64T(self.value)
        value_type = schema.VData.UInt64
      case DType.INT64:
        value = schema.Int64T(self.value)
        value_type = schema.VData.Int64
      case DType.DOUBLE:
        value = schema.DoubleT(self.value)
        value_type = schema.VData.Double
      case _:
        raise ValueError(f"Unsupported dtype: {self.dtype}")
    return schema.KeyValuePairT(key=self.key, value=value, valueType=value_type)


def populate_system_metadata(
    system_metadata: list[Metadata],
) -> list[Metadata]:
  """Populates system metadata with default UUID and creation timestamp.

  Args:
    system_metadata: The list of system metadata.

  Returns:
    The updated list of system metadata.
  """
  system_metadata = [
      m for m in system_metadata if m.key not in ("uuid", "creation_timestamp")
  ]
  system_metadata.append(
      Metadata(
          key="uuid",
          value=str(uuid.uuid4()),
          dtype=DType.STRING,
      )
  )
  system_metadata.append(
      Metadata(
          key="creation_timestamp",
          value=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          dtype=DType.STRING,
      )
  )
  return system_metadata


@dataclasses.dataclass(frozen=True)
class ExternalizationSummary:
  """Summary of an opt-in LiteRT-LM weight externalization pass."""

  models_inspected: int
  models_with_external_weights: int
  newly_externalized_bytes: int


@enum.unique
class Backend(str, enum.Enum):
  """Backend enum."""

  CPU = "cpu"
  GPU = "gpu"
  NPU = "npu"
  GPU_ARTISAN = "gpu_artisan"


@dataclasses.dataclass
class _SectionObject:
  # Metadata for the section.
  metadata: list[Metadata]
  # The data type of the section.
  data_type: schema.AnySectionDataType | int
  # The data writer for the section. This should write the data to stream.
  data_writer: Callable[[BinaryIO], None]
  # Source path for sections that can be transformed during packaging.
  source_path: str | None = None


@dataclasses.dataclass(frozen=True)
class _PendingModelType:
  """A proto `TfLiteModelType` enum value whose capability is not known yet.

  Proto `TfLiteModelType` enums are scoped to a capability metadata proto (e.g.
  `LlmMetadata.TF_LITE_PREFILL_DECODE` vs.
  `EmbeddingMetadata.TF_LITE_TEXT_ENCODER`), so an integer model type can only
  be resolved to its wire string once the capability metadata is known. A
  section added before any capability metadata stores this placeholder as its
  `model_type` metadata value until `LitertLmFileBuilder.build` resolves it.

  Attributes:
    value: The capability-scoped proto `TfLiteModelType` enum value.
  """

  value: int


def _get_raw_model_type(
    section: _SectionObject,
) -> str | _PendingModelType | None:
  """Returns the section's model_type, which may still be pending."""
  values = [item.value for item in section.metadata if item.key == "model_type"]
  if not values:
    return None
  if (
      len(values) != 1
      or not isinstance(values[0], (str, _PendingModelType))
      or not values[0]
  ):
    raise ValueError("Section must have exactly one string model_type")
  return values[0]


def _get_model_type(section: _SectionObject) -> str | None:
  """Returns the section's resolved model_type wire string, if any."""
  model_type = _get_raw_model_type(section)
  if isinstance(model_type, _PendingModelType):
    raise ValueError(
        f"Section has an unresolved integer model_type {model_type.value}."
    )
  return model_type


# The capability-scoped proto `TfLiteModelType` enums, keyed by the capability
# names tracked in `LitertLmFileBuilder._capability_types`. These protos are the
# single source of truth for TFLite model type wire strings.
_CAPABILITY_TFLITE_MODEL_TYPE_ENUMS = {
    "llm": llm_metadata_pb2.LlmMetadata.TfLiteModelType,
    "embedding": embedding_metadata_pb2.EmbeddingMetadata.TfLiteModelType,
    "speech2text": speech2text_metadata_pb2.Speech2TextMetadata.TfLiteModelType,
    "text2speech": text2speech_metadata_pb2.Text2SpeechMetadata.TfLiteModelType,
    "text2image": text2image_metadata_pb2.Text2ImageMetadata.TfLiteModelType,
    "any2any": any2any_metadata_pb2.Any2AnyMetadata.TfLiteModelType,
}

_TFLITE_MODEL_TYPE_PREFIX = "tf_lite_"

_ALL_CAPABILITY_TFLITE_MODEL_TYPES: set[str] = set().union(*(
    enum_type.keys()
    for enum_type in _CAPABILITY_TFLITE_MODEL_TYPE_ENUMS.values()
)) - {"TF_LITE_MODEL_TYPE_UNSPECIFIED"}


def _to_tf_free_model_type(wire_str: str) -> str:
  """Strips the `tf_lite_` prefix from a lowercase TFLite model type string."""
  return wire_str.removeprefix(_TFLITE_MODEL_TYPE_PREFIX)


# Every valid (non-UNSPECIFIED) proto `TfLiteModelType` enum value across the
# capability metadata protos.
_ALL_CAPABILITY_TFLITE_MODEL_TYPE_VALUES: set[int] = set().union(*(
    enum_type.values()
    for enum_type in _CAPABILITY_TFLITE_MODEL_TYPE_ENUMS.values()
)) - {0}


def _is_proto_enum_model_type(model_type: Any) -> bool:
  """Returns whether `model_type` is a proto `TfLiteModelType` enum value."""
  return isinstance(model_type, int) and not isinstance(model_type, bool)


def get_all_tflite_model_types() -> list[str]:
  """Returns all supported TF-free TFLite model type strings across capabilities."""
  return sorted({
      _to_tf_free_model_type(name.lower())
      for name in _ALL_CAPABILITY_TFLITE_MODEL_TYPES
  })


def _resolve_model_type_str(model_type: str) -> str:
  """Resolves a TF-free or `tf_lite_`-prefixed string to its wire string.

  Args:
    model_type: The TF-free model type (e.g. "prefill_decode") or the prefixed
      wire string (e.g. "tf_lite_prefill_decode"), case-insensitive.

  Returns:
    The lowercase wire string, e.g. "tf_lite_prefill_decode".

  Raises:
    ValueError: If the model type is not defined by any capability proto.
  """
  model_type_lower = model_type.lower()
  if model_type_lower.startswith(_TFLITE_MODEL_TYPE_PREFIX):
    # For handling old models. All new models should use the TF-free format.
    logging.warning("Input '%s' already starts with 'tf_lite_'.", model_type)
    wire_str = model_type_lower
  else:
    wire_str = _TFLITE_MODEL_TYPE_PREFIX + model_type_lower
  if wire_str.upper() not in _ALL_CAPABILITY_TFLITE_MODEL_TYPES:
    raise ValueError(f"Unknown TfLiteModelType: {model_type}")
  return wire_str


def _resolve_model_type_wire_string(
    model_type: int | str,
    capability_types: set[str] | None = None,
) -> str:
  """Resolves a model_type to its lowercase wire string.

  Args:
    model_type: A TF-free or `tf_lite_`-prefixed string, or a capability-scoped
      proto `TfLiteModelType` enum value.
    capability_types: The capability metadata types added to the builder. Used
      to pick the proto enum when `model_type` is an integer.

  Returns:
    The lowercase wire string, e.g. "tf_lite_prefill_decode".

  Raises:
    ValueError: If the model type cannot be resolved.
  """
  if isinstance(model_type, int) and not isinstance(model_type, bool):
    if model_type == 0:
      raise ValueError(
          "TF_LITE_MODEL_TYPE_UNSPECIFIED (0) is not a valid model_type."
      )
    if not capability_types:
      raise ValueError(
          "Capability metadata must be added before resolving an integer"
          " model_type, or pass a string model_type."
      )
    if len(capability_types) > 1:
      raise ValueError(
          "Ambiguous integer model_type when multiple capability metadata"
          f" sections ({sorted(capability_types)}) are present; pass a string"
          " model_type."
      )
    capability_type = next(iter(capability_types))
    enum_type = _CAPABILITY_TFLITE_MODEL_TYPE_ENUMS.get(capability_type)
    if enum_type is None:
      raise ValueError(f"Unknown capability_type: {capability_type!r}")
    return enum_type.Name(model_type).lower()
  if isinstance(model_type, str):
    return _resolve_model_type_str(model_type)
  raise ValueError(f"Unsupported model_type type: {type(model_type)}")


_TOKENIZER_SECTION_TYPES = frozenset({
    schema.AnySectionDataType.SP_Tokenizer,
    schema.AnySectionDataType.HF_Tokenizer_Zlib,
})


def _check_and_record_tokenizer(
    model_type_key: str | _PendingModelType | None,
    tokenizers_by_model_type: set[str | _PendingModelType | None],
) -> None:
  """Checks that a tokenizer is unique for its model type and records it.

  A tokenizer without a model type is the default tokenizer and conflicts with
  a `tf_lite_prefill_decode` tokenizer. A pending model type only conflicts with
  the same pending value here; it is re-validated once resolved at build time.

  Args:
    model_type_key: The tokenizer's model type wire string, pending model type,
      or None for the default tokenizer.
    tokenizers_by_model_type: The model types of the tokenizers added so far.
      `model_type_key` is added to it.

  Raises:
    ValueError: If a tokenizer was already added for the model type.
  """
  prefill_decode = llm_metadata_pb2.LlmMetadata.TfLiteModelType.Name(
      llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE
  ).lower()
  if model_type_key is None or model_type_key == prefill_decode:
    if (
        None in tokenizers_by_model_type
        or prefill_decode in tokenizers_by_model_type
    ):
      raise ValueError("Tokenizer already added.")
  elif model_type_key in tokenizers_by_model_type:
    raise ValueError(
        f"Tokenizer already added for model_type: {model_type_key}."
    )
  tokenizers_by_model_type.add(model_type_key)


def _get_active_model_type_message(
    msg: llm_metadata_pb2.LlmMetadata,
) -> message.Message:
  """Returns the active sub-message inside llm_model_type, or generic_model."""
  field = msg.llm_model_type.WhichOneof("model_type")
  if field is not None:
    return getattr(msg.llm_model_type, field)
  return msg.llm_model_type.generic_model


def _has_max_num_patches(msg: llm_metadata_pb2.LlmMetadata) -> bool:
  """Returns True if max_num_patches > 0 is set on the active model type."""
  sub_msg = _get_active_model_type_message(msg)
  field_desc = sub_msg.DESCRIPTOR.fields_by_name.get("max_num_patches")
  if field_desc is None:
    return False
  if field_desc.has_presence and not sub_msg.HasField("max_num_patches"):
    return False
  return getattr(sub_msg, "max_num_patches", 0) > 0


def _has_pooling_kernel_size(msg: llm_metadata_pb2.LlmMetadata) -> bool:
  """Returns True if pooling_kernel_size > 0 is set on the active model type."""
  sub_msg = _get_active_model_type_message(msg)
  field_desc = sub_msg.DESCRIPTOR.fields_by_name.get("pooling_kernel_size")
  if field_desc is None:
    return False
  if field_desc.has_presence and not sub_msg.HasField("pooling_kernel_size"):
    return False
  return getattr(sub_msg, "pooling_kernel_size", 0) > 0


def _set_vision_patch_metadata(
    msg: llm_metadata_pb2.LlmMetadata,
    max_num_patches: int | None,
    pooling_kernel_size: int | None,
) -> None:
  """Sets max_num_patches and pooling_kernel_size on the active model type."""
  if max_num_patches is None and pooling_kernel_size is None:
    return
  sub_msg = _get_active_model_type_message(msg)
  if (
      max_num_patches is not None
      and "max_num_patches" in sub_msg.DESCRIPTOR.fields_by_name
  ):
    setattr(sub_msg, "max_num_patches", max_num_patches)
  if (
      pooling_kernel_size is not None
      and "pooling_kernel_size" in sub_msg.DESCRIPTOR.fields_by_name
  ):
    setattr(sub_msg, "pooling_kernel_size", pooling_kernel_size)


LitertLmFileBuilderT = TypeVar(
    "LitertLmFileBuilderT", bound="LitertLmFileBuilder"
)


class LitertLmFileBuilder:
  """LitertLmFileBuilder class.

  This is the primary entry point for building a LiteRT-LM file. It provides
  methods to add system metadata, sections, and llm metadata to the file.

  Example usage:
  ```
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="The ODML Authors",
            dtype=litertlm_builder.DType.STRING,
        )
    )
    builder.add_tflite_model(
        model_path,
        llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE,
    )
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    builder.add_llm_metadata(llm_metadata_path)
    with litertlm_core.open_file(output_path, "wb") as f:
      builder.build(f)
  ```
  """

  def __init__(self):
    self._system_metadata: list[Metadata] = []
    self._sections: list[_SectionObject] = []
    self._has_llm_metadata = False
    self._llm_metadata: llm_metadata_pb2.LlmMetadata | None = None
    self._has_executor_metadata = False
    self._has_embedding_metadata = False
    self._embedding_metadata: (
        embedding_metadata_pb2.EmbeddingMetadata | None
    ) = None
    self._has_speech2text_metadata = False
    self._has_text2speech_metadata = False
    self._has_text2image_metadata = False
    self._has_any2any_metadata = False
    self._capability_types: set[str] = set()
    self._tokenizers_by_model_type: set[str | _PendingModelType | None] = set()

  def _check_no_capability_metadata_added(self, new_capability: str) -> None:
    """Validates capability compatibility and resolves pending model types."""
    new_set = self._capability_types | {new_capability}
    if len(new_set) > 1 and not (new_set <= {"llm", "embedding"}):
      existing = ", ".join(sorted(self._capability_types))
      raise ValueError(
          "A .litertlm file can contain only one top-level capability metadata"
          f" section. Already has '{existing}', cannot add"
          f" '{new_capability}'."
      )
    self._capability_types.add(new_capability)
    if len(self._capability_types) == 1:
      try:
        self._resolve_pending_model_types()
      except ValueError:
        self._capability_types.discard(new_capability)
        raise

  def _model_type_key(self, model_type: int | str) -> str | _PendingModelType:
    """Resolves a model_type argument to its wire string, or defers it.

    Strings are always resolved immediately. A
    proto `TfLiteModelType` enum value is resolved immediately when capability
    metadata has already been added; otherwise its resolution is deferred to
    `build`, so model sections can be added before the capability metadata.

    Args:
      model_type: The model type argument passed to a builder method.

    Returns:
      The lowercase wire string (e.g. "tf_lite_prefill_decode"), or a
      `_PendingModelType` if the integer cannot be resolved yet.

    Raises:
      ValueError: If the model type is invalid.
    """
    if _is_proto_enum_model_type(model_type) and not self._capability_types:
      model_type = cast(int, model_type)
      if model_type == 0:
        raise ValueError(
            "TF_LITE_MODEL_TYPE_UNSPECIFIED (0) is not a valid model_type."
        )
      if model_type not in _ALL_CAPABILITY_TFLITE_MODEL_TYPE_VALUES:
        raise ValueError(f"Unknown TfLiteModelType enum value: {model_type}")
      return _PendingModelType(model_type)
    return _resolve_model_type_wire_string(model_type, self._capability_types)

  def _try_resolve_model_type(
      self, model_type: str | _PendingModelType | None
  ) -> str | _PendingModelType | None:
    """Resolves a pending model type if exactly one capability is known."""
    if (
        isinstance(model_type, _PendingModelType)
        and len(self._capability_types) == 1
    ):
      return _resolve_model_type_wire_string(
          model_type.value, self._capability_types
      )
    return model_type

  def _section_model_type(
      self, section: _SectionObject
  ) -> str | _PendingModelType | None:
    """Returns the section's model type, resolving it on demand if possible."""
    return self._try_resolve_model_type(_get_raw_model_type(section))

  def _resolve_pending_model_types(self) -> None:
    """Resolves every pending integer model_type to its wire string.

    Called by `build` once all capability metadata has been added. Tokenizer
    uniqueness is re-validated when any tokenizer had a pending model type.

    Raises:
      ValueError: If an integer model_type still cannot be resolved, e.g. no or
        multiple capability metadata sections were added, or if resolving
        reveals a duplicate tokenizer.
    """
    has_pending_tokenizer = False
    for section in self._sections:
      for i, item in enumerate(section.metadata):
        if item.key != "model_type" or not isinstance(
            item.value, _PendingModelType
        ):
          continue
        try:
          wire_str = _resolve_model_type_wire_string(
              item.value.value, self._capability_types
          )
        except ValueError as e:
          raise ValueError(
              f"Cannot resolve integer model_type {item.value.value} at build"
              f" time: {e}"
          ) from e
        section.metadata[i] = dataclasses.replace(item, value=wire_str)
        if section.data_type in _TOKENIZER_SECTION_TYPES:
          has_pending_tokenizer = True

    if has_pending_tokenizer:
      tokenizers_by_model_type: set[str | _PendingModelType | None] = set()
      for section in self._sections:
        if section.data_type not in _TOKENIZER_SECTION_TYPES:
          continue
        model_type = _get_model_type(section)
        _check_and_record_tokenizer(
            _resolve_model_type_wire_string(model_type, self._capability_types)
            if model_type is not None
            else None,
            tokenizers_by_model_type,
        )
      self._tokenizers_by_model_type = tokenizers_by_model_type

  @property
  def _has_tokenizer(self) -> bool:
    return bool(self._tokenizers_by_model_type)

  def _has_section_with_llm_model_type(
      self, *model_types: llm_metadata_pb2.LlmMetadata.TfLiteModelType
  ) -> bool:
    """Returns True if any section has one of the given LLM model types."""
    wire_strings = {
        llm_metadata_pb2.LlmMetadata.TfLiteModelType.Name(m).lower()
        for m in model_types
    }
    return any(
        self._section_model_type(s) in wire_strings for s in self._sections
    )

  @property
  def is_llm_model(self) -> bool:
    """Returns True if the builder contains an LLM model or LLM metadata."""
    if self._has_llm_metadata:
      return True
    return self._has_section_with_llm_model_type(
        llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE,
        llm_metadata_pb2.LlmMetadata.TF_LITE_ARTISAN_TEXT_DECODER,
    )

  _VISION_TRANSFORMER_MODEL_TYPES = frozenset({"gemma4", "lfm2"})

  _EMBEDDING_VISION_TRANSFORMER_MODEL_TYPES = frozenset({
      "embedding_gemma_v2",
  })

  @property
  def is_vision_model(self) -> bool:
    """Returns True if the builder contains a vision adapter/encoder model."""
    return self._has_section_with_llm_model_type(
        llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ADAPTER,
        llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ENCODER,
        llm_metadata_pb2.LlmMetadata.TF_LITE_END_OF_VISION,
    )

  @property
  def is_llm_vision_transformer_model(self) -> bool:
    """Returns True if the LLM model is a vision transformer (patch-based) model."""
    if not self.is_vision_model or self._llm_metadata is None:
      return False
    active_type = self._llm_metadata.llm_model_type.WhichOneof("model_type")
    if active_type in self._VISION_TRANSFORMER_MODEL_TYPES:
      return True
    if active_type == "generic_model" or active_type is None:
      if self._llm_metadata.llm_model_type.HasField("generic_model"):
        gm = self._llm_metadata.llm_model_type.generic_model
        return (
            gm.HasField("max_num_patches")
            or gm.HasField("pooling_kernel_size")
            or gm.HasField("patch_width")
            or gm.HasField("patch_height")
        )
    return False

  @property
  def is_embedding_vision_transformer_model(self) -> bool:
    """Returns True if the embedding model is a vision transformer model."""
    if not self.is_vision_model or self._embedding_metadata is None:
      return False
    active_type = self._embedding_metadata.embedding_model_type.WhichOneof(
        "model_type"
    )
    if active_type in self._EMBEDDING_VISION_TRANSFORMER_MODEL_TYPES:
      eg = self._embedding_metadata.embedding_model_type.embedding_gemma_v2
      has_vision_fields = (
          eg.HasField("start_of_image_token")
          or eg.HasField("end_of_image_token")
          or eg.patch_width > 0
          or eg.patch_height > 0
          or eg.max_num_patches > 0
          or eg.pooling_kernel_size > 0
      )
      if has_vision_fields:
        return True
      if not self.is_llm_model:
        return True
    return False

  @property
  def is_vision_transformer_model(self) -> bool:
    """Returns True if the model is a vision transformer (patch-based) model."""
    return (
        self.is_llm_vision_transformer_model
        or self.is_embedding_vision_transformer_model
    )

  def validate_metadata(self) -> None:
    """Validates mandatory LLM and Vision fields before conversion/packaging.

    Raises:
      ValueError: If mandatory fields for LLM (`supports_thinking`,
        `supports_function_calling`) or Vision Transformer (`max_num_patches`,
        `pooling_kernel_size`) models are missing.
    """
    if self.is_llm_model:
      if self._llm_metadata is None:
        raise ValueError(
            "LLM model conversion requires `LlmMetadata` to be added."
        )
      if not self._llm_metadata.HasField("supports_thinking"):
        raise ValueError(
            "LLM model conversion error: `supports_thinking` is mandatory."
        )
      if not self._llm_metadata.HasField("supports_function_calling"):
        raise ValueError(
            "LLM model conversion error: `supports_function_calling` is"
            " mandatory."
        )
      if self.is_llm_vision_transformer_model:
        if not _has_max_num_patches(self._llm_metadata):
          raise ValueError(
              "Vision model conversion error: `max_num_patches` is mandatory"
              " when vision transformer model is present."
          )
        if not _has_pooling_kernel_size(self._llm_metadata):
          raise ValueError(
              "Vision model conversion error: `pooling_kernel_size` is"
              " mandatory when vision transformer model is present."
          )

    if (
        self._has_embedding_metadata
        and self.is_embedding_vision_transformer_model
    ):
      if self._embedding_metadata is not None:
        if self._embedding_metadata.embedding_model_type.HasField(
            "embedding_gemma_v2"
        ):
          eg = self._embedding_metadata.embedding_model_type.embedding_gemma_v2
          if eg.max_num_patches <= 0:
            raise ValueError(
                "Vision model conversion error: `max_num_patches` is mandatory"
                " when vision transformer model is present."
            )
          if eg.pooling_kernel_size <= 0:
            raise ValueError(
                "Vision model conversion error: `pooling_kernel_size` is"
                " mandatory when vision transformer model is present."
            )

  @classmethod
  def from_toml_str(
      cls,
      toml_str: str,
      parent_dir: str | None = None,
      jinja_prompt_template_path: str | None = None,
  ) -> LitertLmFileBuilderT:
    """Initializes a LitertLmFileBuilder from a loaded TOML string.

    Args:
      toml_str: The TOML string to parse.
      parent_dir: The parent directory of the TOML file. If provided, it will be
        used to resolve the relative paths in the TOML file.
      jinja_prompt_template_path: Optional path to a Jinja template file to
        overwrite jinja_prompt_template.

    Returns:
      The LitertLmFileBuilder object.

    Raises:
      ValueError: If the TOML string is invalid.
    """
    builder = cls()
    toml_data = tomllib.loads(toml_str)

    for key in toml_data.keys():
      if key not in ["section", "system_metadata"]:
        raise ValueError(f"Unexpected key: {key}")

    if "system_metadata" in toml_data:
      assert (
          "entries" in toml_data["system_metadata"]
      ), "System metadata does not have entries."
      for entry in toml_data["system_metadata"]["entries"]:
        builder.add_system_metadata(
            Metadata(
                key=entry["key"],
                value=entry["value"],
                dtype=DType(str(entry["value_type"]).lower()),
            )
        )

    if "section" in toml_data:
      for section in toml_data["section"]:
        assert "section_type" in section, "Section does not have section_type."
        assert "data_path" in section, "Section does not have data_path."

        additional_metadata = None
        if "additional_metadata" in section and section["additional_metadata"]:
          additional_metadata = []
          for m in section["additional_metadata"]:
            additional_metadata.append(
                Metadata(
                    key=m["key"],
                    value=m["value"],
                    dtype=DType(str(m["value_type"]).lower()),
                )
            )

        section_type = section["section_type"]

        if section_type == "LlmMetadata":
          builder.add_llm_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
              jinja_prompt_template_path=jinja_prompt_template_path,
              min_runtime_version=section.get("min_runtime_version", None),
              supports_thinking=section.get("supports_thinking", None),
              supports_function_calling=section.get(
                  "supports_function_calling", None
              ),
              max_num_patches=section.get("max_num_patches", None),
              pooling_kernel_size=section.get("pooling_kernel_size", None),
          )
        elif section_type == "ExecutorMetadata":
          builder.add_executor_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "EmbeddingMetadata":
          builder.add_embedding_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "Speech2TextMetadata":
          builder.add_speech2text_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "Text2SpeechMetadata":
          builder.add_text2speech_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "Text2ImageMetadata":
          builder.add_text2image_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "Any2AnyMetadata":
          builder.add_any2any_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section_type == "TFLiteModel":
          if "model_type" not in section:
            raise ValueError("TFLiteModel section does not have model_type.")
          builder.add_tflite_model(
              _resolve_path(section["data_path"], parent_dir),
              section["model_type"],
              backend_constraint=section.get("backend_constraint", None),
              prefer_activation_type=section.get(
                  "prefer_activation_type", None
              ),
              additional_metadata=additional_metadata,
          )
        elif section_type == "TFLiteWeights":
          if "model_type" not in section:
            raise ValueError("TFLiteWeights section does not have model_type.")
          builder.add_tflite_weights(
              _resolve_path(section["data_path"], parent_dir),
              section["model_type"],
              additional_metadata=additional_metadata,
          )
        elif section_type == "SP_Tokenizer":
          model_type = section.get("model_type", None)
          builder.add_sentencepiece_tokenizer(
              _resolve_path(section["data_path"], parent_dir),
              model_type=model_type,
              additional_metadata=additional_metadata,
          )
        elif section_type == "HF_Tokenizer":
          model_type = section.get("model_type", None)
          builder.add_hf_tokenizer(
              _resolve_path(section["data_path"], parent_dir),
              model_type=model_type,
              additional_metadata=additional_metadata,
          )
        elif section_type == "GenericBinaryData":
          builder.add_generic_binary_data(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        else:
          raise ValueError(f"Unexpected section type: {section_type}")

    if jinja_prompt_template_path is not None and not builder._has_llm_metadata:
      raise ValueError(
          "Cannot apply chat template: TOML configuration does not contain an"
          " LlmMetadata section."
      )

    return builder  # pyrefly: ignore[bad-return]

  @classmethod
  def from_toml_file(
      cls, toml_path: str, jinja_prompt_template_path: str | None = None
  ) -> LitertLmFileBuilderT:
    """Initializes a LitertLmFileBuilder from a TOML file."""
    with litertlm_core.open_file(toml_path, "r") as f:
      parent_path = pathlib.Path(toml_path).parent.as_posix()
      return cls.from_toml_str(
          f.read(),
          parent_path,
          jinja_prompt_template_path=jinja_prompt_template_path,
      )

  @classmethod
  def unpack(
      cls,
      litertlm_path: str,
      output_dir: str,
      jinja_prompt_template_path: str | None = None,
  ) -> LitertLmFileBuilderT:
    """Unpacks a LiteRT-LM file into output_dir and returns a LitertLmFileBuilder initialized from the unpacked model.toml.

    Args:
      litertlm_path: The path to the LiteRT-LM file to unpack.
      output_dir: The directory where unpacked files and model.toml will be
        saved.
      jinja_prompt_template_path: Optional path where jinja_prompt_template will
        be unpacked.

    Returns:
      The LitertLmFileBuilder object initialized from the unpacked model.toml.
    """
    toml_path = unpack(
        litertlm_path,
        output_dir,
        jinja_prompt_template_path=jinja_prompt_template_path,
    )
    return cls.from_toml_file(toml_path)

  def add_system_metadata(
      self,
      metadata: Metadata,
  ) -> LitertLmFileBuilderT:
    """Adds system level metadata to the litertlm file."""
    for existing_metadata in self._system_metadata:
      if existing_metadata.key == metadata.key:
        raise ValueError(
            f"System metadata already exists for key: {metadata.key}"
        )
    self._system_metadata.append(metadata)
    return self  # pyrefly: ignore[bad-return]

  def add_llm_metadata(
      self,
      llm_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
      jinja_prompt_template_path: Optional[str] = None,
      min_runtime_version: Optional[str] = None,
      supports_thinking: Optional[bool] = None,
      supports_function_calling: Optional[bool] = None,
      max_num_patches: Optional[int] = None,
      pooling_kernel_size: Optional[int] = None,
  ) -> LitertLmFileBuilderT:
    """Adds llm metadata to the litertlm file.

    Args:
      llm_metadata_path: The path to the llm metadata file. Can be binary or
        textproto format.
      additional_metadata: Additional metadata to add to the llm metadata.
      jinja_prompt_template_path: Optional path to a Jinja file to overwrite
        jinja_prompt_template.
      min_runtime_version: The minimum LiteRT-LM runtime version required.
      supports_thinking: Whether the model supports thinking/reasoning.
      supports_function_calling: Whether the model supports function calling.
      max_num_patches: Maximum number of vision patches (for vision models).
      pooling_kernel_size: Spatial pooling kernel size (for vision models).

    Returns:
      The currentLitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the llm metadata file or jinja template file is not
        found.
    """
    assert not self._has_llm_metadata, "Llm metadata already added."
    self._check_no_capability_metadata_added("llm")
    self._has_llm_metadata = True
    if not litertlm_core.path_exists(llm_metadata_path):
      raise FileNotFoundError(
          f"Llm metadata file not found: {llm_metadata_path}"
      )

    if jinja_prompt_template_path and not litertlm_core.path_exists(
        jinja_prompt_template_path
    ):
      raise FileNotFoundError(
          f"Jinja template file not found: {jinja_prompt_template_path}"
      )

    msg = llm_metadata_pb2.LlmMetadata()
    if _is_binary_proto(llm_metadata_path):
      with litertlm_core.open_file(llm_metadata_path, "rb") as f:
        msg.ParseFromString(f.read())
    else:
      with litertlm_core.open_file(llm_metadata_path, "r") as f:
        text_format.Parse(f.read(), msg)

    if jinja_prompt_template_path:
      with litertlm_core.open_file(jinja_prompt_template_path, "r") as f_jinja:
        msg.jinja_prompt_template = f_jinja.read()
    if min_runtime_version:
      msg.min_runtime_version = min_runtime_version
    if supports_thinking is not None:
      msg.supports_thinking = supports_thinking
    if supports_function_calling is not None:
      msg.supports_function_calling = supports_function_calling
    _set_vision_patch_metadata(msg, max_num_patches, pooling_kernel_size)
    self._llm_metadata = msg

    has_overrides = (
        jinja_prompt_template_path
        or min_runtime_version
        or supports_thinking is not None
        or supports_function_calling is not None
        or max_num_patches is not None
        or pooling_kernel_size is not None
    )

    if _is_binary_proto(llm_metadata_path) and not has_overrides:

      def data_writer(stream: BinaryIO):
        with litertlm_core.open_file(llm_metadata_path, "rb") as f:
          _copy_file_to_stream(f, stream)

    else:

      def data_writer(stream: BinaryIO):
        assert self._llm_metadata is not None
        stream.write(self._llm_metadata.SerializeToString())

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.LlmMetadataProto,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def _add_proto_metadata_section(
      self,
      metadata_path: str,
      proto_cls: Any,
      data_type: int,
      label: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Validates, parses, and appends a protobuf metadata section."""
    if not litertlm_core.path_exists(metadata_path):
      raise FileNotFoundError(
          f"{label} metadata file not found: {metadata_path}"
      )

    if _is_binary_proto(metadata_path, proto_cls):

      def data_writer(stream: BinaryIO):
        with litertlm_core.open_file(metadata_path, "rb") as f:
          _copy_file_to_stream(f, stream)

    else:

      def data_writer(stream: BinaryIO):
        with litertlm_core.open_file(metadata_path, "r") as f:
          data = text_format.Parse(f.read(), proto_cls()).SerializeToString()
          stream.write(data)

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=data_type,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def add_executor_metadata(
      self,
      executor_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds executor metadata to the litertlm file.

    Args:
      executor_metadata_path: The path to the executor metadata file. Can be
        binary or textproto format.
      additional_metadata: Additional metadata to add to the executor metadata.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the executor metadata file is not found.
    """
    assert not self._has_executor_metadata, "Executor metadata already added."
    self._has_executor_metadata = True
    return self._add_proto_metadata_section(
        executor_metadata_path,
        executor_metadata_pb2.ExecutorMetadata,
        schema.AnySectionDataType.ExecutorMetadataProto,
        "Executor",
        additional_metadata,
    )

  def add_embedding_metadata(
      self,
      embedding_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds embedding metadata to the litertlm file.

    Args:
      embedding_metadata_path: The path to the embedding metadata file. Can be
        binary or textproto format.
      additional_metadata: Additional metadata to add to the embedding metadata.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the embedding metadata file is not found.
    """
    assert not self._has_embedding_metadata, "Embedding metadata already added."
    self._check_no_capability_metadata_added("embedding")
    self._has_embedding_metadata = True
    if not litertlm_core.path_exists(embedding_metadata_path):
      raise FileNotFoundError(
          f"Embedding metadata file not found: {embedding_metadata_path}"
      )

    msg = embedding_metadata_pb2.EmbeddingMetadata()
    if _is_binary_proto(
        embedding_metadata_path,
        embedding_metadata_pb2.EmbeddingMetadata,
    ):
      with litertlm_core.open_file(embedding_metadata_path, "rb") as f:
        msg.ParseFromString(f.read())
    else:
      with litertlm_core.open_file(embedding_metadata_path, "r") as f:
        text_format.Parse(f.read(), msg)
    self._embedding_metadata = msg

    def data_writer(stream: BinaryIO):
      stream.write(msg.SerializeToString())

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.EmbeddingMetadataProto,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def add_speech2text_metadata(
      self,
      speech2text_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds speech-to-text metadata to the litertlm file."""
    assert (
        not self._has_speech2text_metadata
    ), "Speech2Text metadata already added."
    self._check_no_capability_metadata_added("speech2text")
    self._has_speech2text_metadata = True
    return self._add_proto_metadata_section(
        speech2text_metadata_path,
        speech2text_metadata_pb2.Speech2TextMetadata,
        schema.AnySectionDataType.Speech2TextMetadataProto,
        "Speech2Text",
        additional_metadata,
    )

  def add_text2speech_metadata(
      self,
      text2speech_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds text-to-speech metadata to the litertlm file."""
    assert (
        not self._has_text2speech_metadata
    ), "Text2Speech metadata already added."
    self._check_no_capability_metadata_added("text2speech")
    self._has_text2speech_metadata = True
    return self._add_proto_metadata_section(
        text2speech_metadata_path,
        text2speech_metadata_pb2.Text2SpeechMetadata,
        schema.AnySectionDataType.Text2SpeechMetadataProto,
        "Text2Speech",
        additional_metadata,
    )

  def add_text2image_metadata(
      self,
      text2image_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds text-to-image metadata to the litertlm file."""
    assert (
        not self._has_text2image_metadata
    ), "Text2Image metadata already added."
    self._check_no_capability_metadata_added("text2image")
    self._has_text2image_metadata = True
    return self._add_proto_metadata_section(
        text2image_metadata_path,
        text2image_metadata_pb2.Text2ImageMetadata,
        schema.AnySectionDataType.Text2ImageMetadataProto,
        "Text2Image",
        additional_metadata,
    )

  def add_any2any_metadata(
      self,
      any2any_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds any-to-any metadata to the litertlm file."""
    assert not self._has_any2any_metadata, "Any2Any metadata already added."
    self._check_no_capability_metadata_added("any2any")
    self._has_any2any_metadata = True
    return self._add_proto_metadata_section(
        any2any_metadata_path,
        any2any_metadata_pb2.Any2AnyMetadata,
        schema.AnySectionDataType.Any2AnyMetadataProto,
        "Any2Any",
        additional_metadata,
    )

  def add_tflite_model(
      self,
      tflite_model_path: str,
      model_type: int | str,
      backend_constraint: Optional[str] = None,
      prefer_activation_type: Optional[str] = None,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a tflite model to the litertlm file.

    Args:
      tflite_model_path: The path to the tflite model file.
      model_type: The type of the tflite model: a TF-free string (e.g.
        "prefill_decode"), a lowercase wire string (e.g.
        "tf_lite_prefill_decode"), or a capability-scoped proto enum value (e.g.
        `llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE`). A proto enum
        value is resolved against the capability metadata; if none has been
        added yet, it is resolved when `build` is called.
      backend_constraint: The backend constraint for the tflite model.
      prefer_activation_type: The preferred activation type for the tflite
        model. - fp16/float16 for float16 activation. - fp32/float32 for float32
        activation. - fp32_fp16 for mixed activation.
      additional_metadata: Additional metadata to add to the tflite model.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the tflite model file is not found.
      ValueError: If the model type metadata is overridden or backend_constraint
      is invalid.
    """
    if not litertlm_core.path_exists(tflite_model_path):
      raise FileNotFoundError(
          f"Tflite model file not found: {tflite_model_path}"
      )
    model_type_key = self._model_type_key(model_type)
    metadata = [
        Metadata(key="model_type", value=model_type_key, dtype=DType.STRING)
    ]
    if backend_constraint:
      _validate_backend_constraints(backend_constraint)
      metadata.append(
          Metadata(
              key="backend_constraint",
              value=backend_constraint.lower(),
              dtype=DType.STRING,
          )
      )
    if prefer_activation_type:
      print(f"Adding prefer_activation_type: {prefer_activation_type}")
      metadata.append(
          Metadata(
              key="prefer_activation_type",
              value=prefer_activation_type.lower(),
              dtype=DType.STRING,
          )
      )
    if additional_metadata:
      for metadata_item in additional_metadata:
        if metadata_item.key == "model_type":
          raise ValueError("Model type metadata cannot be overridden.")
        if metadata_item.key == "backend_constraint":
          raise ValueError("Backend constraint metadata cannot be overridden.")
      metadata.extend(additional_metadata)

    def data_writer(stream: BinaryIO):
      with litertlm_core.open_file(tflite_model_path, "rb") as f:
        _copy_file_to_stream(f, stream)

    section_object = _SectionObject(
        metadata=metadata,
        data_type=schema.AnySectionDataType.TFLiteModel,
        data_writer=data_writer,
        source_path=tflite_model_path,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def add_tflite_weights(
      self,
      tflite_weights_path: str,
      model_type: int | str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds tflite weights to the litertlm file.

    Args:
      tflite_weights_path: The path to the tflite weights file.
      model_type: The type of the tflite model these weights correspond to.
        Accepts the same values as `add_tflite_model`.
      additional_metadata: Additional metadata to add to the tflite weights.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the tflite weights file is not found.
      ValueError: If the model type metadata is overridden.
    """
    if not litertlm_core.path_exists(tflite_weights_path):
      raise FileNotFoundError(
          f"Tflite weights file not found: {tflite_weights_path}"
      )
    model_type_key = self._model_type_key(model_type)
    metadata = [
        Metadata(key="model_type", value=model_type_key, dtype=DType.STRING)
    ]
    if additional_metadata is not None:
      for metadata_item in additional_metadata:
        if metadata_item.key == "model_type":
          raise ValueError("Model type metadata cannot be overridden.")
      metadata.extend(additional_metadata)

    def data_writer(stream: BinaryIO):
      with litertlm_core.open_file(tflite_weights_path, "rb") as f:
        _copy_file_to_stream(f, stream)

    section_object = _SectionObject(
        metadata=metadata,
        data_type=schema.AnySectionDataType.TFLiteWeights,
        data_writer=data_writer,
        source_path=tflite_weights_path,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def _prepare_tokenizer_metadata(
      self,
      model_type: Optional[int | str],
      additional_metadata: Optional[list[Metadata]],
  ) -> list[Metadata]:
    """Validates tokenizer uniqueness and prepares its metadata."""
    model_type_key = (
        self._model_type_key(model_type) if model_type is not None else None
    )

    metadata: list[Metadata] = []
    if model_type_key is not None:
      metadata.append(
          Metadata(key="model_type", value=model_type_key, dtype=DType.STRING)
      )

    if additional_metadata:
      for metadata_item in additional_metadata:
        if metadata_item.key == "model_type":
          if model_type_key is not None:
            raise ValueError("Model type metadata cannot be overridden.")
          model_type_key = self._model_type_key(metadata_item.value)
          if _is_proto_enum_model_type(metadata_item.value):
            # Integers are stored as their (possibly pending) wire string.
            metadata_item = dataclasses.replace(
                metadata_item, value=model_type_key
            )
        metadata.append(metadata_item)

    _check_and_record_tokenizer(model_type_key, self._tokenizers_by_model_type)
    return metadata

  def add_sentencepiece_tokenizer(
      self,
      sp_tokenizer_path: str,
      model_type: Optional[int | str] = None,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a sentencepiece tokenizer to the litertlm file.

    Args:
      sp_tokenizer_path: The path to the sentencepiece tokenizer file.
      model_type: The model type this tokenizer corresponds to (e.g.
        "prefill_decode"). Accepts the same values as `add_tflite_model`.
      additional_metadata: Additional metadata to add to the sentencepiece
        tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the sentencepiece tokenizer file is not found.
      ValueError: If model_type metadata is overridden.
    """
    if not litertlm_core.path_exists(sp_tokenizer_path):
      raise FileNotFoundError(
          f"Sentencepiece tokenizer file not found: {sp_tokenizer_path}"
      )
    metadata = self._prepare_tokenizer_metadata(model_type, additional_metadata)

    def data_writer(stream: BinaryIO):
      with litertlm_core.open_file(sp_tokenizer_path, "rb") as f:
        _copy_file_to_stream(f, stream)

    section_object = _SectionObject(
        metadata=metadata,
        data_type=schema.AnySectionDataType.SP_Tokenizer,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def add_hf_tokenizer(
      self,
      hf_tokenizer_path: str,
      model_type: Optional[int | str] = None,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a hf tokenizer to the litertlm file.

    Args:
      hf_tokenizer_path: The path to the hf tokenizer `tokenizer.json` file.
      model_type: The model type this tokenizer corresponds to (e.g.
        "prefill_decode"). Accepts the same values as `add_tflite_model`.
      additional_metadata: Additional metadata to add to the hf tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the hf tokenizer file is not found.
      ValueError: If model_type metadata is overridden.
    """
    if not litertlm_core.path_exists(hf_tokenizer_path):
      raise FileNotFoundError(
          f"HF tokenizer file not found: {hf_tokenizer_path}"
      )
    metadata = self._prepare_tokenizer_metadata(model_type, additional_metadata)

    def write_and_compress(stream: BinaryIO):
      with litertlm_core.open_file(hf_tokenizer_path, "rb") as f:
        content = f.read()
        if hf_tokenizer_path.endswith(".zlib"):
          stream.write(content)
        else:
          assert hf_tokenizer_path.endswith(
              ".json"
          ), "HF tokenizer file must be either .json or .zlib format."
          uncompressed_size = len(content)
          compressed_content = zlib.compress(content)
          stream.write(uncompressed_size.to_bytes(8, "little"))
          stream.write(compressed_content)

    section_object = _SectionObject(
        metadata=metadata,
        data_type=schema.AnySectionDataType.HF_Tokenizer_Zlib,
        data_writer=write_and_compress,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def add_generic_binary_data(
      self,
      generic_binary_data_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds generic binary data to the litertlm file."""
    if not litertlm_core.path_exists(generic_binary_data_path):
      raise FileNotFoundError(
          f"Generic binary data file not found: {generic_binary_data_path}"
      )

    def data_writer(stream: BinaryIO):
      with litertlm_core.open_file(generic_binary_data_path, "rb") as f:
        _copy_file_to_stream(f, stream)

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.GenericBinaryData,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self  # pyrefly: ignore[bad-return]

  def set_backend_constraint(
      self,
      model_type: int | str,
      backend_constraint: str | None,
  ) -> LitertLmFileBuilderT:
    """Sets or clears the backend_constraint metadata on matching sections.

    Useful after post-processing an unpacked package (e.g. NPU compilation)
    where a section previously constrained to CPU can now run elsewhere.

    Args:
      model_type: Target section model type, e.g. a TF-free string such as
        'audio_encoder_hw', a lowercase wire string such as
        'tf_lite_audio_encoder_hw', or a capability-scoped proto enum value.
      backend_constraint: New constraint string (e.g. 'npu', 'cpu, gpu'), or
        None to remove the constraint entirely.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      KeyError: If no section with the given model_type exists.
      ValueError: If backend_constraint is not a valid backend string, or if an
        integer model_type cannot be matched against a string one because no
        capability metadata has been added yet.
    """
    target_model_type = self._model_type_key(model_type)
    if backend_constraint is not None:
      _validate_backend_constraints(backend_constraint)

    matched = False
    has_unresolved_pending = False
    for section in self._sections:
      section_model_type = self._section_model_type(section)
      if section_model_type is None:
        continue
      if isinstance(section_model_type, _PendingModelType) != isinstance(
          target_model_type, _PendingModelType
      ):
        has_unresolved_pending = True
        continue
      if section_model_type == target_model_type:
        matched = True
        section.metadata = [
            m for m in section.metadata if m.key != "backend_constraint"
        ]
        if backend_constraint is not None:
          section.metadata.append(
              Metadata(
                  key="backend_constraint",
                  value=backend_constraint.lower(),
                  dtype=DType.STRING,
              )
          )
    if not matched:
      if has_unresolved_pending:
        raise ValueError(
            "Cannot match an integer model_type against a string model_type"
            " before capability metadata is added; add the capability"
            " metadata first."
        )
      raise KeyError(f"No section found with model_type {target_model_type}")
    return self  # pyrefly: ignore[bad-return]

  def remove_backend_constraint(
      self, model_type: int | str
  ) -> LitertLmFileBuilderT:
    """Removes the backend_constraint metadata from matching sections."""
    return self.set_backend_constraint(model_type, None)

  def build(
      self,
      stream: BinaryIO,
      *,
      validate_metadata: bool = False,
  ) -> ExternalizationSummary | None:
    """Builds the litertlm into the given stream.

    Integer (proto enum) model types of sections added before any capability
    metadata are resolved to their wire strings here.

    Args:
      stream: The binary output stream to write the LiteRT-LM file to.
      validate_metadata: Whether to validate the metadata before building.
      externalize_weights: Whether to externalize large tensor weights into a
        separate weights section.
      externalization_work_dir: Optional directory path for temporary weight
        externalization files.
      num_elements_threshold: Minimum number of tensor elements for a weight to
        be externalized.
      weight_group_name: Name of the externalized weights group section.

    Returns:
      An `ExternalizationSummary` when `externalize_weights` is True, or None
      otherwise.
    """
    self._resolve_pending_model_types()
    if validate_metadata:
      self.validate_metadata()
    self._build_sections(stream, self._sections)
    return None

  def _build_sections(
      self, stream: BinaryIO, sections: list[_SectionObject]
  ) -> None:
    """Packs metadata and section data and writes the LiteRT-LM file.

    Args:
      stream: The binary output stream to write the LiteRT-LM file to.
      sections: The section objects to serialize into the file.
    """
    # Add UUID if not already present, but always generate a new timestamp.
    self._system_metadata = populate_system_metadata(self._system_metadata)

    # Populate a SystemMetadataT object from `self._system_metadata`.
    system_metadata = schema.SystemMetadataT(
        entries=[m.to_key_value_pair() for m in self._system_metadata]
    )

    # Populate a SectionMetadataT object from `self._sections`.
    section_metadata = schema.SectionMetadataT(
        objects=[
            schema.SectionObjectT(
                items=[m.to_key_value_pair() for m in s.metadata],
                dataType=s.data_type,
                beginOffset=1,  # Use a non-zero (default value) placeholder.
                endOffset=1,  # Use a non-zero (default value) placeholder
            )
            for s in sections
        ]
    )

    # Populate and pack the `LiteRTLMMetaDataT` to get its size.
    litertlm_metadata = schema.LiteRTLMMetaDataT(
        systemMetadata=system_metadata, sectionMetadata=section_metadata
    )
    metadata_builder = flatbuffers.Builder(litertlm_core.BLOCK_SIZE)
    metadata_builder.Finish(litertlm_metadata.Pack(metadata_builder))
    packed_metadata_size = metadata_builder.Offset()

    # Write the section data and populate the section offsets.
    offset = _round_up_to_block_size(
        litertlm_core.HEADER_BEGIN_BYTE_OFFSET + packed_metadata_size
    )
    for section, section_fb in zip(sections, section_metadata.objects):  # pyrefly: ignore[bad-argument-type]
      stream.seek(offset)
      section_fb.beginOffset = offset
      section.data_writer(stream)
      offset = stream.tell()
      section_fb.endOffset = offset
      offset = _round_up_to_block_size(offset)

    # Go back and write the header and updated metadata at the start of the
    # output file.
    metadata_builder.Clear()
    metadata_builder.Finish(litertlm_metadata.Pack(metadata_builder))
    assert packed_metadata_size == metadata_builder.Offset()
    stream.seek(0)
    stream.write(litertlm_core.HEADER_MAGIC_BYTES)
    stream.write(litertlm_core.LITERTLM_MAJOR_VERSION.to_bytes(4, "little"))
    stream.write(litertlm_core.LITERTLM_MINOR_VERSION.to_bytes(4, "little"))
    stream.write(litertlm_core.LITERTLM_PATCH_VERSION.to_bytes(4, "little"))
    _write_padding(stream, litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    stream.write(
        (
            litertlm_core.HEADER_BEGIN_BYTE_OFFSET + packed_metadata_size
        ).to_bytes(8, "little")
    )
    stream.write(metadata_builder.Output())


def _round_up_to_block_size(offset: int) -> int:
  """Rounds `offset` up to the next multiple of `litertlm_core.BLOCK_SIZE`."""
  return (offset + litertlm_core.BLOCK_SIZE - 1) & ~(
      litertlm_core.BLOCK_SIZE - 1
  )


def _copy_file_to_stream(f_src: Any, f_dst: BinaryIO, buffer_size=1024 * 1024):
  """Copies data from f_src to f_dst efficiently."""
  # Try to use os.sendfile (zero-copy) if available.
  if hasattr(os, "sendfile"):
    try:
      # Flush the destination stream to ensure all buffered data is written
      # before using os.sendfile, which operates directly on the file
      # descriptor.
      f_dst.flush()

      in_fd, out_fd = f_src.fileno(), f_dst.fileno()
      num_bytes = os.fstat(in_fd).st_size
      offset = 0
      while num_bytes > 0 and (
          bytes_sent := os.sendfile(
              out_fd, in_fd, offset=offset, count=num_bytes
          )
      ):
        offset += bytes_sent
        num_bytes -= bytes_sent
    except OSError:
      pass
    else:
      if num_bytes == 0:
        return

  # If the above did not work, then just copy the file in chunks to avoid
  # flooding the memory memory when reading/writing large files.
  shutil.copyfileobj(f_src, f_dst, length=buffer_size)


def _validate_backend_constraints(backend_constraint: str) -> None:
  """Validates the backend constraint string."""
  backends = [b.strip().lower() for b in backend_constraint.split(",")]
  valid_backends = set(Backend)
  for backend in backends:
    if backend not in valid_backends:
      raise ValueError(
          f"Invalid backend constraint: {backend}. Must be one of"
          f" {list(valid_backends)}"
      )


def _is_binary_proto(
    filepath: str,
    message_type: Any = llm_metadata_pb2.LlmMetadata,
) -> bool:  # pyrefly: ignore[bad-return]
  """Checks if a file is a binary protobuf or a textproto.

  Args:
      filepath (str): The path to the file.
      message_type: The protobuf message class to try parsing with.

  Returns:
      bool: True if the file is a binary protobuf, False if it's a textproto.
  """
  assert litertlm_core.path_exists(filepath), f"File {filepath} does not exist."

  name = message_type().__class__.__name__
  try:
    with litertlm_core.open_file(filepath, "rb") as f:
      content = f.read()
      msg = message_type()
      msg.ParseFromString(content)
      if msg.IsInitialized():
        return True
  except message.DecodeError:
    # This is expected if the file is in text format. We'll just pass and try
    # the next format.
    pass

  try:
    with litertlm_core.open_file(filepath, "r") as f:
      content = f.read()
      msg = text_format.Parse(content, message_type())
      if msg.IsInitialized():
        return False
  except (text_format.ParseError, UnicodeDecodeError) as e:
    raise ValueError(
        f"Failed to parse {name} from {filepath}. Exception: {e}"
    ) from e


def _write_padding(stream: BinaryIO, block_size: int) -> None:
  """Writes zero padding to align to the next block size."""
  current_pos = stream.tell()
  padding_needed = (block_size - (current_pos % block_size)) % block_size
  if padding_needed > 0:
    stream.write(b"\0" * padding_needed)


def _resolve_path(path: str, parent_dir: str | None) -> str:
  """Resolve the path and check if it exists."""
  is_abs = os.path.isabs(path)
  if not is_abs and not parent_dir:
    raise ValueError("Parent directory is required for relative path.")

  abs_path = path if is_abs else os.path.join(parent_dir, path)  # pyrefly: ignore[no-matching-overload]
  if not litertlm_core.path_exists(abs_path):
    raise FileNotFoundError(f"File {abs_path} does not exist.")
  return abs_path


def unpack(
    litertlm_path: str,
    output_dir: str,
    jinja_prompt_template_path: str | None = None,
) -> str:
  """Unpacks a LiteRT-LM file into the specified directory.

  Args:
    litertlm_path: The path to the LiteRT-LM file to unpack.
    output_dir: The directory where the unpacked files and model.toml will be
      saved.
    jinja_prompt_template_path: Optional path where jinja_prompt_template will
      be saved.

  Returns:
    The path to the generated model.toml file.
  """
  litertlm_peek.peek_litertlm_file(
      litertlm_path,
      dump_files_dir=output_dir,
      output_stream=io.StringIO(),
      jinja_prompt_template_path=jinja_prompt_template_path,
  )
  return os.path.join(output_dir, "model.toml")


unpack_litertlm_file = unpack


def pack_with_summary(
    toml_path: str,
    output_path: str,
    jinja_prompt_template_path: str | None = None,
) -> tuple[str, ExternalizationSummary | None]:
  """Packs TOML and returns the output path and externalization summary."""
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  builder = LitertLmFileBuilder.from_toml_file(
      toml_path, jinja_prompt_template_path=jinja_prompt_template_path
  )
  with litertlm_core.open_file(output_path, "wb") as f:
    summary = builder.build(
        cast(BinaryIO, f),
    )
  return output_path, summary


def pack(
    toml_path: str,
    output_path: str,
    jinja_prompt_template_path: str | None = None,
) -> str:
  """Packs a TOML configuration and its referenced files into a LiteRT-LM file.

  Args:
    toml_path: The path to the input TOML configuration file (e.g., model.toml).
    output_path: The path where the packed LiteRT-LM file will be saved.
    jinja_prompt_template_path: Optional path to a Jinja file to overwrite
      jinja_prompt_template.

  Returns:
    The path to the generated LiteRT-LM file.
  """
  packed_path, _ = pack_with_summary(
      toml_path,
      output_path,
      jinja_prompt_template_path,
  )
  return packed_path


pack_litertlm_file = pack
