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

import io
import os
import pathlib
import zlib
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
from litert_lm.runtime.proto import llm_metadata_pb2
from litert_lm.schema.py import litertlm_builder
from litert_lm.schema.py import litertlm_core
from litert_lm.schema.py import litertlm_peek

_TOML_TEMPLATE = """
# A template for testing the TOML parser.

[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The ODML Authors" }
]

[[section]]
# Section 0: LlmMetadataProto
section_type = "LlmMetadata"
data_path = "{LLM_METADATA_PATH}"

[[section]]
# Section 1: SP_Tokenizer
section_type = "SP_Tokenizer"
data_path = "{SP_TOKENIZER_PATH}"

[[section]]
# Section 2: TFLiteModel (Embedder)
section_type = "TFLiteModel"
model_type = "EMBEDDER"
data_path = "{EMBEDDER_PATH}"

[[section]]
# Section 3: TFLiteModel (Prefill/Decode)
section_type = "TFLiteModel"
model_type = "PREFILL_DECODE"
data_path = "{PREFILL_DECODE_PATH}"
additional_metadata = [
  { key = "License", value_type = "String", value = "Example" }
]

[[section]]
# Section 4: GenericBinaryData
section_type = "GenericBinaryData"
data_path = "{GENERIC_BINARY_PATH}"
"""


class LitertlmBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def _create_dummy_file(self, filename: str, content: bytes) -> str:
    filepath = os.path.join(self.temp_dir, filename)
    with litertlm_core.open_file(filepath, "wb") as f:
      f.write(content)
    return filepath

  def _add_system_metadata(self, builder: litertlm_builder.LitertLmFileBuilder):
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="sys_test_k",
            value="sys_test_v",
            dtype=litertlm_builder.DType.STRING,
        )
    )

  def _build_and_read_litertlm(
      self, builder: litertlm_builder.LitertLmFileBuilder
  ) -> str:
    path = os.path.join(self.temp_dir, "litertlm.litertlm")
    with litertlm_core.open_file(path, "wb") as f:
      builder.build(f)
    stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(path, self.temp_dir, stream)
    return stream.getvalue()

  def test_add_system_metadata(self):
    """Tests that system metadata is added correctly."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Key: sys_test_k, Value (String): sys_test_v", ss)
    self.assertIn("Sections (0)", ss)

  def test_auto_generated_metadata(self):
    """Tests that uuid and creation_timestamp are automatically added."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Key: uuid, Value (String):", ss)
    self.assertIn("Key: creation_timestamp, Value (String):", ss)

  def test_override_existing_timestamp(self):
    """Tests that existing creation_timestamp is overridden."""
    builder = litertlm_builder.LitertLmFileBuilder()
    custom_time = "2020-01-01T00:00:00Z"
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="creation_timestamp",
            value=custom_time,
            dtype=litertlm_builder.DType.STRING,
        )
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertNotIn(
        f"Key: creation_timestamp, Value (String): {custom_time}", ss
    )
    self.assertIn("Key: creation_timestamp, Value (String):", ss)

  def test_preserve_existing_uuid(self):
    """Tests that existing uuid is not overridden."""
    builder = litertlm_builder.LitertLmFileBuilder()
    custom_uuid = "my-custom-uuid-123"
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="uuid",
            value=custom_uuid,
            dtype=litertlm_builder.DType.STRING,
        )
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn(f"Key: uuid, Value (String): {custom_uuid}", ss)

  def test_add_system_metadata_duplicate_key(self):
    """Tests that adding system metadata with a duplicate key raises a ValueError."""
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="sys_key1",
            value="sys_val1",
            dtype=litertlm_builder.DType.STRING,
        )
    )
    with self.assertRaises(ValueError):
      builder.add_system_metadata(
          litertlm_builder.Metadata(
              key="sys_key1",
              value="sys_val2",
              dtype=litertlm_builder.DType.STRING,
          )
      )

  def test_add_llm_metadata_binary(self):
    """Tests that LLM metadata can be added from a binary proto file."""
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    bin_proto = llm_metadata.SerializeToString()
    metadata_path = self._create_dummy_file("llm.pb", bin_proto)

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_llm_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Sections (1)", ss)

  def test_add_llm_metadata_text(self):
    """Tests that LLM metadata can be added from a text proto file."""
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    text_proto = text_format.MessageToString(llm_metadata)
    metadata_path = self._create_dummy_file(
        "llm.textproto", text_proto.encode("utf-8")
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_llm_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Sections (1)", ss)

  def test_add_llm_metadata_not_found(self):
    """Tests that adding a non-existent LLM metadata file raises a FileNotFoundError."""
    builder = litertlm_builder.LitertLmFileBuilder()
    with self.assertRaises(FileNotFoundError):
      builder.add_llm_metadata("nonexistent.pb")

  def test_add_llm_metadata_already_added(self):
    builder = litertlm_builder.LitertLmFileBuilder()
    metadata_path = self._create_dummy_file("llm.pb", b"")
    builder.add_llm_metadata(metadata_path)
    with self.assertRaises(AssertionError):
      builder.add_llm_metadata(metadata_path)

  @parameterized.named_parameters(
      ("prefill_decode", litertlm_builder.TfLiteModelType.PREFILL_DECODE),
      ("mtp_drafter", litertlm_builder.TfLiteModelType.MTP_DRAFTER),
  )
  def test_add_tflite_model(self, model_type: litertlm_builder.TfLiteModelType):
    """Tests that a TFLite model can be added correctly."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        tflite_path,
        model_type,
        additional_metadata=[
            litertlm_builder.Metadata(
                key="test_key",
                value="test_value",
                dtype=litertlm_builder.DType.STRING,
            )
        ],
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn(f"Key: model_type, Value (String): {model_type.value}", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

  def test_add_tflite_model_with_backend_constraint(self):
    """Tests that a TFLite model with backend constraint added correctly."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        tflite_path,
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
        backend_constraint="gpu",
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: backend_constraint, Value (String): gpu", ss)

  def test_add_tflite_model_with_multiple_backend_constraint(self):
    """Tests that a TFLite model with backend constraint added correctly."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        tflite_path,
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
        backend_constraint="cpu, GPU",
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: backend_constraint, Value (String): cpu, gpu", ss)

  def test_add_tflite_model_with_invalid_backend_constraint(self):
    """Tests that a TFLite model with backend constraint added correctly."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)

    with self.assertRaisesRegex(ValueError, "Invalid backend constraint"):
      builder.add_tflite_model(
          tflite_path,
          litertlm_builder.TfLiteModelType.PREFILL_DECODE,
          backend_constraint="foo, bar",
      )

  def test_add_tflite_model_override_type(self):
    """Tests that overriding the model type in additional metadata raises a ValueError."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )
    additional_metadata = [
        litertlm_builder.Metadata(
            key="model_type", value="bad", dtype=litertlm_builder.DType.STRING
        )
    ]
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    with self.assertRaises(ValueError):
      builder.add_tflite_model(
          tflite_path,
          litertlm_builder.TfLiteModelType.EMBEDDER,
          additional_metadata=additional_metadata,
      )

  def test_add_tflite_weights(self):
    """Tests that a TFLite weights file can be added correctly."""
    tflite_weights_path = self._create_dummy_file(
        "model.weights", b"dummy tflite weights content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_weights(
        tflite_weights_path,
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
        additional_metadata=[
            litertlm_builder.Metadata(
                key="test_key",
                value="test_value",
                dtype=litertlm_builder.DType.STRING,
            )
        ],
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteWeights", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

  def test_add_sentencepiece_tokenizer(self):
    """Tests that a SentencePiece tokenizer can be added correctly."""
    sp_path = self._create_dummy_file("sp.model", b"dummy sp content")
    additional_metadata = [
        litertlm_builder.Metadata(
            key="test_key",
            value="test_value",
            dtype=litertlm_builder.DType.STRING,
        )
    ]

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_sentencepiece_tokenizer(
        sp_path, additional_metadata=additional_metadata
    )
    ss = self._build_and_read_litertlm(builder)
    print(ss)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

  def test_add_generic_binary_data(self):
    """Tests that generic binary data can be added correctly."""
    binary_content = b"dummy binary content"
    binary_path = self._create_dummy_file("data.bin", binary_content)
    additional_metadata = [
        litertlm_builder.Metadata(
            key="test_key",
            value="test_value",
            dtype=litertlm_builder.DType.STRING,
        )
    ]
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_generic_binary_data(
        binary_path, additional_metadata=additional_metadata
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    GenericBinaryData", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

    # Verify content
    with litertlm_core.open_file(
        os.path.join(self.temp_dir, "litertlm.litertlm"), "rb"
    ) as f:
      f.seek(litertlm_core.BLOCK_SIZE)
      read_content = f.read(len(binary_content))
      self.assertEqual(read_content, binary_content)

  def test_add_hf_tokenizer(self):
    """Tests that a HuggingFace tokenizer can be added correctly."""
    hf_content = b'{"version": "1.0"}'
    hf_path = self._create_dummy_file("tokenizer.json", hf_content)
    additional_metadata = [
        litertlm_builder.Metadata(
            key="test_key",
            value="test_value",
            dtype=litertlm_builder.DType.STRING,
        )
    ]
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_hf_tokenizer(hf_path, additional_metadata=additional_metadata)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    HF_Tokenizer_Zlib", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

    # Verify content compression
    with litertlm_core.open_file(
        os.path.join(self.temp_dir, "litertlm.litertlm"), "rb"
    ) as f:
      f.seek(litertlm_core.BLOCK_SIZE)
      # Read uncompressed size (8 bytes)
      uncompressed_size = int.from_bytes(f.read(8), "little")
      self.assertLen(hf_content, uncompressed_size)
      # Read remaining data (compressed)
      compressed_data = f.read()
      # Decompress and verify. zlib.decompress will stop at end of stream,
      # ignoring padding
      decompressed = zlib.decompress(compressed_data)
      self.assertEqual(decompressed, hf_content)

  def test_add_hf_tokenizer_zlib(self):
    """Tests that a zipped HuggingFace tokenizer is handled correctly."""
    zlib_content = b"dummy zlib content"
    hf_path = self._create_dummy_file("tokenizer.zlib", zlib_content)
    additional_metadata = [
        litertlm_builder.Metadata(
            key="test_key",
            value="test_value",
            dtype=litertlm_builder.DType.STRING,
        )
    ]
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_hf_tokenizer(hf_path, additional_metadata=additional_metadata)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    HF_Tokenizer_Zlib", ss)
    self.assertIn("Key: test_key, Value (String): test_value", ss)

    # Verify content is raw (not re-compressed and no size prefix)
    with litertlm_core.open_file(
        os.path.join(self.temp_dir, "litertlm.litertlm"), "rb"
    ) as f:
      f.seek(litertlm_core.BLOCK_SIZE)
      # Should match exact content immediately
      read_content = f.read(len(zlib_content))
      self.assertEqual(read_content, zlib_content)

  def test_add_tokenizer_already_added(self):
    """Tests that adding a tokenizer more than once raises an AssertionError."""
    sp_path = self._create_dummy_file("sp.model", b"")

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_sentencepiece_tokenizer(sp_path)

    with self.assertRaises(AssertionError):
      builder.add_hf_tokenizer(self._create_dummy_file("tokenizer.json", b""))
    with self.assertRaises(AssertionError):
      builder.add_sentencepiece_tokenizer(
          self._create_dummy_file("tokenizer.json", b"")
      )

  def test_end_to_end(self):
    """Tests a more complex end-to-end scenario with multiple sections."""
    sp_path = self._create_dummy_file("sp.model", b"dummy sp content")
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    bin_proto = llm_metadata.SerializeToString()
    metadata_path = self._create_dummy_file("llm.pb", bin_proto)

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_sentencepiece_tokenizer(sp_path)
    builder.add_tflite_model(
        tflite_path, model_type=litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        tflite_path, model_type=litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_llm_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (4)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)

  @parameterized.named_parameters(
      ("relative_path", True),
      ("absolute_path", False),
  )
  def test_from_toml(self, use_relative_path: bool):
    """Tests that a LitertLmFileBuilder can be initialized from a TOML file."""
    sp_filename = "sp.model"
    tflite_filename = "model.tflite"
    metadata_filename = "llm.pb"

    sp_path_abs = self._create_dummy_file(sp_filename, b"dummy sp content")
    tflite_path_abs = self._create_dummy_file(
        tflite_filename, b"dummy tflite content"
    )
    metadata_path_abs = self._create_dummy_file(
        metadata_filename,
        llm_metadata_pb2.LlmMetadata(max_num_tokens=123).SerializeToString(),
    )
    generic_binary_filename = "data.bin"
    generic_binary_path_abs = self._create_dummy_file(
        generic_binary_filename, b"dummy binary content"
    )

    if use_relative_path:
      sp_path = sp_filename
      tflite_path = tflite_filename
      metadata_path = metadata_filename
      generic_binary_path = generic_binary_filename
    else:
      sp_path = pathlib.Path(sp_path_abs).as_posix()
      tflite_path = pathlib.Path(tflite_path_abs).as_posix()
      metadata_path = pathlib.Path(metadata_path_abs).as_posix()
      generic_binary_path = pathlib.Path(generic_binary_path_abs).as_posix()

    toml_path = self._create_dummy_file(
        "test.toml",
        _TOML_TEMPLATE.replace("{LLM_METADATA_PATH}", metadata_path)
        .replace("{SP_TOKENIZER_PATH}", sp_path)
        .replace("{EMBEDDER_PATH}", tflite_path)
        .replace("{PREFILL_DECODE_PATH}", tflite_path)
        .replace("{GENERIC_BINARY_PATH}", generic_binary_path)
        .encode("utf-8"),
    )
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (5)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Data Type:    GenericBinaryData", ss)


if __name__ == "__main__":
  absltest.main()
