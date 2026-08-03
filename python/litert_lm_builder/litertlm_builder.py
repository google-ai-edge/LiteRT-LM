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
    litertlm_builder.TfLiteModelType.PREFILL_DECODE,
)
builder.add_sentencepiece_tokenizer(tokenizer_path)
builder.add_llm_metadata(llm_metadata_path)
with litertlm_core.open_file(output_path, "wb") as f:
  builder.build(f)
```

Note: The build method uses `seek` to write the header and sections. The io
interface used must support `seek`.
"""

import dataclasses
import datetime
import enum
import os
import pathlib
import shutil
from typing import Any, BinaryIO, Callable, Optional, TypeVar
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
from runtime.proto import llm_metadata_pb2


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


@enum.unique
class TfLiteModelType(enum.Enum):
  """TfLiteModelType enum.

  This enum maps to the model types defined in the LiteRT-LM flatbuffers schema.
  """

  PREFILL_DECODE = "tf_lite_prefill_decode"

  EMBEDDER = "tf_lite_embedder"
  PER_LAYER_EMBEDDER = "tf_lite_per_layer_embedder"

  AUX = "tf_lite_aux"

  AUDIO_FRONTEND = "tf_lite_audio_frontend"
  AUDIO_ENCODER_HW = "tf_lite_audio_encoder_hw"
  AUDIO_ADAPTER = "tf_lite_audio_adapter"
  END_OF_AUDIO = "tf_lite_end_of_audio"

  VISION_ENCODER = "tf_lite_vision_encoder"
  VISION_ADAPTER = "tf_lite_vision_adapter"
  END_OF_VISION = "tf_lite_end_of_vision"
  ARTISAN_TEXT_DECODER = "tf_lite_artisan_text_decoder"
  MTP_DRAFTER = "tf_lite_mtp_drafter"
  MTP_AUX = "tf_lite_mtp_aux"

  @classmethod
  def get_enum_from_tf_free_value(cls, tf_free_value: str) -> "TfLiteModelType":
    """A helper method to get the enum value from the TF-free value."""
    value = "tf_lite_" + tf_free_value.lower()
    return cls(value)


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
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
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
    self._has_tokenizer = False

  @classmethod
  def from_toml_str(
      cls, toml_str: str, parent_dir: str | None = None
  ) -> LitertLmFileBuilderT:
    """Initializes a LitertLmFileBuilder from a loaded TOML string.

    Args:
      toml_str: The TOML string to parse.
      parent_dir: The parent directory of the TOML file. If provided, it will be
        used to resolve the relative paths in the TOML file.

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

        if section["section_type"] == "LlmMetadata":
          builder.add_llm_metadata(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section["section_type"] == "TFLiteModel":
          if "model_type" not in section:
            raise ValueError("TFLiteModel section does not have model_type.")
          model_type = TfLiteModelType.get_enum_from_tf_free_value(
              section["model_type"]
          )
          builder.add_tflite_model(
              _resolve_path(section["data_path"], parent_dir),
              model_type,
              backend_constraint=section.get("backend_constraint", None),
              prefer_activation_type=section.get(
                  "prefer_activation_type", None
              ),
              additional_metadata=additional_metadata,
          )
        elif section["section_type"] == "TFLiteWeights":
          if "model_type" not in section:
            raise ValueError("TFLiteWeights section does not have model_type.")
          model_type = TfLiteModelType.get_enum_from_tf_free_value(
              section["model_type"]
          )
          builder.add_tflite_weights(
              _resolve_path(section["data_path"], parent_dir),
              model_type,
              additional_metadata=additional_metadata,
          )
        elif section["section_type"] == "SP_Tokenizer":
          builder.add_sentencepiece_tokenizer(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section["section_type"] == "HF_Tokenizer":
          builder.add_hf_tokenizer(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        elif section["section_type"] == "GenericBinaryData":
          builder.add_generic_binary_data(
              _resolve_path(section["data_path"], parent_dir),
              additional_metadata=additional_metadata,
          )
        else:
          raise ValueError(
              f"Unexpected section type: {section['section_type']}"
          )

    return builder

  @classmethod
  def from_toml_file(cls, toml_path: str) -> LitertLmFileBuilderT:
    """Initializes a LitertLmFileBuilder from a TOML file."""
    with litertlm_core.open_file(toml_path, "r") as f:
      parent_path = pathlib.Path(toml_path).parent.as_posix()
      return cls.from_toml_str(f.read(), parent_path)

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
    return self

  def add_llm_metadata(
      self,
      llm_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds llm metadata to the litertlm file.

    Args:
      llm_metadata_path: The path to the llm metadata file. Can be binary or
        textproto format.
      additional_metadata: Additional metadata to add to the llm metadata.

    Returns:
      The currentLitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the llm metadata file is not found.
    """
    assert not self._has_llm_metadata, "Llm metadata already added."
    self._has_llm_metadata = True
    if not litertlm_core.path_exists(llm_metadata_path):
      raise FileNotFoundError(
          f"Llm metadata file not found: {llm_metadata_path}"
      )

    if _is_binary_proto(llm_metadata_path):

      def data_writer(stream: BinaryIO):
        with litertlm_core.open_file(llm_metadata_path, "rb") as f:
          _copy_file_to_stream(f, stream)

    else:

      def data_writer(stream: BinaryIO):
        with litertlm_core.open_file(llm_metadata_path, "r") as f:
          data = text_format.Parse(
              f.read(), llm_metadata_pb2.LlmMetadata()
          ).SerializeToString()
          stream.write(data)

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.LlmMetadataProto,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self

  def add_tflite_model(
      self,
      tflite_model_path: str,
      model_type: TfLiteModelType,
      backend_constraint: Optional[str] = None,
      prefer_activation_type: Optional[str] = None,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a tflite model to the litertlm file.

    Args:
      tflite_model_path: The path to the tflite model file.
      model_type: The type of the tflite model.
      backend_constraint: The backend constraint for the tflite model.
      prefer_activation_type: The preferred activation type for the tflite
        model.
        - fp16/float16 for float16 activation.
        - fp32/float32 for float32 activation.
        - fp32_fp16 for mixed activation.
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
    metadata = [
        Metadata(key="model_type", value=model_type.value, dtype=DType.STRING)
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
    )
    self._sections.append(section_object)
    return self

  def add_tflite_weights(
      self,
      tflite_weights_path: str,
      model_type: TfLiteModelType,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds tflite weights to the litertlm file.

    Args:
      tflite_weights_path: The path to the tflite weights file.
      model_type: The type of the tflite model these weights correspond to.
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
    metadata = [
        Metadata(key="model_type", value=model_type.value, dtype=DType.STRING)
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
    )
    self._sections.append(section_object)
    return self

  def add_sentencepiece_tokenizer(
      self,
      sp_tokenizer_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a sentencepiece tokenizer to the litertlm file.

    Args:
      sp_tokenizer_path: The path to the sentencepiece tokenizer file.
      additional_metadata: Additional metadata to add to the sentencepiece
        tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the sentencepiece tokenizer file is not found.
    """
    assert not self._has_tokenizer, "Tokenizer already added."
    self._has_tokenizer = True
    if not litertlm_core.path_exists(sp_tokenizer_path):
      raise FileNotFoundError(
          f"Sentencepiece tokenizer file not found: {sp_tokenizer_path}"
      )

    def data_writer(stream: BinaryIO):
      with litertlm_core.open_file(sp_tokenizer_path, "rb") as f:
        _copy_file_to_stream(f, stream)

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.SP_Tokenizer,
        data_writer=data_writer,
    )
    self._sections.append(section_object)
    return self

  def add_hf_tokenizer(
      self,
      hf_tokenizer_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a hf tokenizer to the litertlm file.

    Args:
      hf_tokenizer_path: The path to the hf tokenizer `tokenizer.json` file.
      additional_metadata: Additional metadata to add to the hf tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the hf tokenizer file is not found.
    """
    assert not self._has_tokenizer, "Tokenizer already added."
    self._has_tokenizer = True
    if not litertlm_core.path_exists(hf_tokenizer_path):
      raise FileNotFoundError(
          f"HF tokenizer file not found: {hf_tokenizer_path}"
      )

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
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.HF_Tokenizer_Zlib,
        data_writer=write_and_compress,
    )
    self._sections.append(section_object)
    return self

  def add_generic_binary_data(
      self,
      generic_binary_data_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds generic binary data to the litertlm file.

    Args:
      generic_binary_data_path: The path to the generic binary data file.
      additional_metadata: Additional metadata to add to the sentencepiece
        tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the generic binary data file is not found.
    """
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
    return self

  def build(self, stream: BinaryIO) -> None:
    """Builds the litertlm into the given stream."""
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
            for s in self._sections
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
    for section, section_fb in zip(self._sections, section_metadata.objects):
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


def _copy_file_to_stream(
    f_src: Any, f_dst: BinaryIO, buffer_size=1024 * 1024
):
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


def _is_binary_proto(filepath: str) -> bool:
  """Checks if a file is a binary protobuf or a textproto version of LlmMetadata.

  Args:
      filepath (str): The path to the file.

  Returns:
      bool: True if the file is a binary protobuf, False if it's a textproto.
      TextProto.
  """
  assert litertlm_core.path_exists(filepath), f"File {filepath} does not exist."

  try:
    with litertlm_core.open_file(filepath, "rb") as f:
      content = f.read()
      msg = llm_metadata_pb2.LlmMetadata()
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
      msg = text_format.Parse(content, llm_metadata_pb2.LlmMetadata())
      if msg.IsInitialized():
        return False
  except (text_format.ParseError, UnicodeDecodeError) as e:
    raise ValueError(
        f"Failed to parse LlmMetadata from {filepath}. Exception: {e}"
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

  abs_path = path if is_abs else os.path.join(parent_dir, path)
  if not litertlm_core.path_exists(abs_path):
    raise FileNotFoundError(f"File {abs_path} does not exist.")
  return abs_path
