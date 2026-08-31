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
from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder import litertlm_peek
from runtime.proto import embedding_metadata_pb2
from runtime.proto import executor_metadata_pb2
from runtime.proto import llm_metadata_pb2

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

[[section]]
# Section 5: ExecutorMetadata
section_type = "ExecutorMetadata"
data_path = "{EXECUTOR_METADATA_PATH}"
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

  def test_override_existing_uuid(self):
    """Tests that existing uuid is overridden."""
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
    self.assertNotIn(f"Key: uuid, Value (String): {custom_uuid}", ss)
    self.assertIn("Key: uuid, Value (String):", ss)

  def test_populate_system_metadata(self):
    """Tests populate_system_metadata helper function."""
    metadata = []
    updated = litertlm_builder.populate_system_metadata(metadata)

    keys = {m.key for m in updated}
    self.assertIn("uuid", keys)
    self.assertIn("creation_timestamp", keys)

    custom_uuid = "my-custom-uuid"
    metadata = [
        litertlm_builder.Metadata(
            key="uuid",
            value=custom_uuid,
            dtype=litertlm_builder.DType.STRING,
        )
    ]
    updated = litertlm_builder.populate_system_metadata(metadata)
    uuid_val = next(m.value for m in updated if m.key == "uuid")
    self.assertNotEqual(uuid_val, custom_uuid)

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

  def test_add_executor_metadata_binary(self):
    """Tests that executor metadata can be added from a binary proto file."""
    executor_metadata = executor_metadata_pb2.ExecutorMetadata(
        llm_executor_metadata=executor_metadata_pb2.LlmExecutorMetadata(
            max_history_size=5
        )
    )
    bin_proto = executor_metadata.SerializeToString()
    metadata_path = self._create_dummy_file("executor.pb", bin_proto)

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_executor_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("max_history_size: 5", ss)
    self.assertIn("Sections (1)", ss)

  def test_add_executor_metadata_text(self):
    """Tests that executor metadata can be added from a text proto file."""
    executor_metadata = executor_metadata_pb2.ExecutorMetadata(
        llm_executor_metadata=executor_metadata_pb2.LlmExecutorMetadata(
            max_history_size=5
        )
    )
    text_proto = text_format.MessageToString(executor_metadata)
    metadata_path = self._create_dummy_file(
        "executor.textproto", text_proto.encode("utf-8")
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_executor_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("max_history_size: 5", ss)
    self.assertIn("Sections (1)", ss)

  def test_add_executor_metadata_not_found(self):
    """Tests that adding a non-existent executor metadata file raises a FileNotFoundError."""
    builder = litertlm_builder.LitertLmFileBuilder()
    with self.assertRaises(FileNotFoundError):
      builder.add_executor_metadata("nonexistent.pb")

  def test_add_executor_metadata_already_added(self):
    builder = litertlm_builder.LitertLmFileBuilder()
    metadata_path = self._create_dummy_file("executor.pb", b"")
    builder.add_executor_metadata(metadata_path)
    with self.assertRaises(AssertionError):
      builder.add_executor_metadata(metadata_path)

  def test_add_embedding_metadata_binary(self):
    """Tests that Embedding metadata can be added from a binary proto file."""
    embedding_metadata = embedding_metadata_pb2.EmbeddingMetadata()
    embedding_metadata.embedding_model_type.embedding_gemma_v2.patch_width = 16
    bin_proto = embedding_metadata.SerializeToString()
    metadata_path = self._create_dummy_file("embedding.pb", bin_proto)

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_embedding_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("patch_width: 16", ss)
    self.assertIn("Sections (1)", ss)

  def test_add_embedding_metadata_text(self):
    """Tests that Embedding metadata can be added from a text proto file."""
    embedding_metadata = embedding_metadata_pb2.EmbeddingMetadata()
    embedding_metadata.embedding_model_type.embedding_gemma_v2.patch_width = 16
    text_proto = text_format.MessageToString(embedding_metadata)
    metadata_path = self._create_dummy_file(
        "embedding.textproto", text_proto.encode("utf-8")
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_embedding_metadata(metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("patch_width: 16", ss)
    self.assertIn("Sections (1)", ss)

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

  def test_add_tflite_model_with_prefer_activation_type(self):
    """Tests that a TFLite model with prefer_activation_type added correctly."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        tflite_path,
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
        prefer_activation_type="fp16",
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: prefer_activation_type, Value (String): fp16", ss)

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

    executor_metadata = executor_metadata_pb2.ExecutorMetadata(
        llm_executor_metadata=executor_metadata_pb2.LlmExecutorMetadata(
            max_history_size=5
        )
    )
    executor_bin_proto = executor_metadata.SerializeToString()
    executor_metadata_path = self._create_dummy_file(
        "executor.pb", executor_bin_proto
    )

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
    builder.add_executor_metadata(executor_metadata_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (5)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Data Type:    ExecutorMetadataProto", ss)
    self.assertIn("max_history_size: 5", ss)

  @parameterized.named_parameters(
      ("relative_path", True),
      ("absolute_path", False),
  )
  def test_from_toml(self, use_relative_path: bool):
    """Tests that a LitertLmFileBuilder can be initialized from a TOML file."""
    sp_filename = "sp.model"
    tflite_filename = "model.tflite"
    metadata_filename = "llm.pb"
    executor_filename = "executor.pb"

    sp_path_abs = self._create_dummy_file(sp_filename, b"dummy sp content")
    tflite_path_abs = self._create_dummy_file(
        tflite_filename, b"dummy tflite content"
    )
    metadata_path_abs = self._create_dummy_file(
        metadata_filename,
        llm_metadata_pb2.LlmMetadata(max_num_tokens=123).SerializeToString(),
    )
    executor_path_abs = self._create_dummy_file(
        executor_filename,
        executor_metadata_pb2.ExecutorMetadata(
            llm_executor_metadata=executor_metadata_pb2.LlmExecutorMetadata(
                max_history_size=5
            )
        ).SerializeToString(),
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
      executor_path = executor_filename
    else:
      sp_path = pathlib.Path(sp_path_abs).as_posix()
      tflite_path = pathlib.Path(tflite_path_abs).as_posix()
      metadata_path = pathlib.Path(metadata_path_abs).as_posix()
      generic_binary_path = pathlib.Path(generic_binary_path_abs).as_posix()
      executor_path = pathlib.Path(executor_path_abs).as_posix()

    toml_path = self._create_dummy_file(
        "test.toml",
        _TOML_TEMPLATE.replace("{LLM_METADATA_PATH}", metadata_path)
        .replace("{SP_TOKENIZER_PATH}", sp_path)
        .replace("{EMBEDDER_PATH}", tflite_path)
        .replace("{PREFILL_DECODE_PATH}", tflite_path)
        .replace("{GENERIC_BINARY_PATH}", generic_binary_path)
        .replace("{EXECUTOR_METADATA_PATH}", executor_path)
        .encode("utf-8"),
    )
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (6)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Data Type:    GenericBinaryData", ss)
    self.assertIn("Data Type:    ExecutorMetadataProto", ss)
    self.assertIn("max_history_size: 5", ss)

  def test_from_toml_with_prefer_activation_type(self):
    """Tests that a LitertLmFileBuilder can be initialized with prefer_activation_type from TOML."""
    tflite_filename = "model.tflite"
    tflite_path_abs = self._create_dummy_file(
        tflite_filename, b"dummy tflite content"
    )
    toml_str = f"""
    [[section]]
    section_type = "TFLiteModel"
    model_type = "PREFILL_DECODE"
    data_path = "{pathlib.Path(tflite_path_abs).as_posix()}"
    prefer_activation_type = "int8"
    """
    toml_path = self._create_dummy_file("test.toml", toml_str.encode("utf-8"))
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: prefer_activation_type, Value (String): int8", ss)

  def test_unpack(self):
    """Tests unpacking a litertlm file using standalone unpack and classmethod unpack."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    tflite_path = self._create_dummy_file("model.tflite", b"dummy content")
    builder.add_tflite_model(
        tflite_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    litertlm_path = os.path.join(self.temp_dir, "test.litertlm")
    with litertlm_core.open_file(litertlm_path, "wb") as f:
      builder.build(f)

    unpack_dir = os.path.join(self.temp_dir, "unpacked")
    toml_path = litertlm_builder.unpack(litertlm_path, unpack_dir)
    self.assertTrue(os.path.exists(toml_path))

    cls_unpack_dir = os.path.join(self.temp_dir, "unpacked_cls")
    rebuilt_builder = litertlm_builder.LitertLmFileBuilder.unpack(
        litertlm_path, cls_unpack_dir
    )
    self.assertIsNotNone(rebuilt_builder)
    self.assertLen(rebuilt_builder._sections, 1)
    self.assertEqual(
        rebuilt_builder._sections[0].data_type,
        schema.AnySectionDataType.TFLiteModel,
    )
    rebuild_path = os.path.join(self.temp_dir, "rebuilt.litertlm")
    with litertlm_core.open_file(rebuild_path, "wb") as f:
      rebuilt_builder.build(f)
    self.assertTrue(os.path.exists(rebuild_path))

  def test_pack(self):
    """Tests packing a litertlm file from a TOML configuration using pack."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    tflite_path = self._create_dummy_file("model.tflite", b"dummy content")
    builder.add_tflite_model(
        tflite_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    litertlm_path = os.path.join(self.temp_dir, "test_orig.litertlm")
    with litertlm_core.open_file(litertlm_path, "wb") as f:
      builder.build(f)

    unpack_dir = os.path.join(self.temp_dir, "unpacked_for_pack")
    toml_path = litertlm_builder.unpack(litertlm_path, unpack_dir)

    packed_litertlm_path = os.path.join(self.temp_dir, "packed.litertlm")
    res_path = litertlm_builder.pack(toml_path, packed_litertlm_path)
    self.assertEqual(res_path, packed_litertlm_path)
    self.assertTrue(os.path.exists(packed_litertlm_path))

  def test_pack_and_unpack_with_jinja_path(self):
    """Tests packing with jinja_prompt_template_path overwrites jinja_prompt_template and unpacking with jinja_prompt_template_path extracts it."""
    llm_metadata_content = 'jinja_prompt_template: "original template"\n'
    meta_path = self._create_dummy_file(
        "metadata.pbtext", llm_metadata_content.encode()
    )
    jinja_input_path = self._create_dummy_file(
        "input.jinja", b"overwritten jinja template"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    tflite_path = self._create_dummy_file("model.tflite", b"dummy content")
    builder.add_tflite_model(
        tflite_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_llm_metadata(meta_path)
    orig_litertlm_path = os.path.join(self.temp_dir, "orig.litertlm")
    with litertlm_core.open_file(orig_litertlm_path, "wb") as f:
      builder.build(f)

    unpack_dir = os.path.join(self.temp_dir, "unpacked_for_jinja")
    toml_path = litertlm_builder.unpack(orig_litertlm_path, unpack_dir)

    packed_path = os.path.join(self.temp_dir, "packed_custom_jinja.litertlm")
    litertlm_builder.pack(
        toml_path, packed_path, jinja_prompt_template_path=jinja_input_path
    )

    extracted_jinja_path = os.path.join(self.temp_dir, "extracted.jinja")
    unpack_dir_2 = os.path.join(self.temp_dir, "unpacked_custom_jinja")
    litertlm_builder.unpack(
        packed_path,
        unpack_dir_2,
        jinja_prompt_template_path=extracted_jinja_path,
    )
    self.assertTrue(os.path.exists(extracted_jinja_path))
    with open(extracted_jinja_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "overwritten jinja template")

  def test_unpack_raises_error_when_jinja_template_missing(self):
    """Tests unpacking with jinja_prompt_template_path raises error when missing."""
    llm_metadata_content = "max_num_tokens: 100\n"
    meta_path = self._create_dummy_file(
        "metadata_no_jinja.pbtext", llm_metadata_content.encode()
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    tflite_path = self._create_dummy_file(
        "model_no_jinja.tflite", b"dummy content"
    )
    builder.add_tflite_model(
        tflite_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_llm_metadata(meta_path)
    model_path = os.path.join(self.temp_dir, "no_jinja.litertlm")
    with litertlm_core.open_file(model_path, "wb") as f:
      builder.build(f)

    unpack_dir = os.path.join(self.temp_dir, "unpacked_no_jinja")
    target_jinja = os.path.join(self.temp_dir, "should_fail.jinja")
    with self.assertRaises(ValueError):
      litertlm_builder.unpack(
          model_path, unpack_dir, jinja_prompt_template_path=target_jinja
      )

  def test_pack_invalid_toml_does_not_truncate_output_file(self):
    """Tests that packing with an invalid TOML does not truncate existing output file."""
    output_path = os.path.join(self.temp_dir, "existing_model.litertlm")
    original_content = b"original model bytes 12345"
    with open(output_path, "wb") as f:
      f.write(original_content)

    invalid_toml_path = os.path.join(self.temp_dir, "broken.toml")
    with open(invalid_toml_path, "w") as f:
      f.write("[invalid toml syntax <<<")

    with self.assertRaises(Exception):
      litertlm_builder.pack(invalid_toml_path, output_path)

    with open(output_path, "rb") as f:
      self.assertEqual(f.read(), original_content)

  def test_pack_raises_error_when_llm_metadata_missing_for_jinja(self):
    """Tests packing with jinja_prompt_template_path raises error when TOML lacks LlmMetadata."""
    tflite_path = self._create_dummy_file("model_for_error.tflite", b"dummy")
    tflite_name = os.path.basename(tflite_path)
    toml_content = f"""
[[section]]
section_type = "TFLiteModel"
model_type = "prefill_decode"
data_path = "{tflite_name}"
"""
    toml_path = self._create_dummy_file(
        "no_llm_meta.toml", toml_content.encode()
    )
    jinja_path = self._create_dummy_file("temp.jinja", b"template")
    output_path = os.path.join(self.temp_dir, "out.litertlm")
    with self.assertRaises(ValueError):
      litertlm_builder.pack(
          toml_path, output_path, jinja_prompt_template_path=jinja_path
      )

  @parameterized.named_parameters(
      (
          "prefill_decode_lowercase",
          "prefill_decode",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE,
      ),
      (
          "prefill_decode_uppercase",
          "PREFILL_DECODE",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE,
      ),
      (
          "mtp_drafter",
          "mtp_drafter",
          litertlm_builder.TfLiteModelType.MTP_DRAFTER,
      ),
      (
          "vision_encoder",
          "vision_encoder",
          litertlm_builder.TfLiteModelType.VISION_ENCODER,
      ),
  )
  def test_get_enum_from_tf_free_value_without_prefix(
      self, input_val: str, expected_enum: litertlm_builder.TfLiteModelType
  ):
    self.assertEqual(
        litertlm_builder.TfLiteModelType.get_enum_from_tf_free_value(input_val),
        expected_enum,
    )

  @parameterized.named_parameters(
      (
          "prefill_decode_with_prefix",
          "tf_lite_prefill_decode",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE,
      ),
      (
          "prefill_decode_with_prefix_upper",
          "TF_LITE_PREFILL_DECODE",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE,
      ),
      (
          "embedder_with_prefix",
          "tf_lite_embedder",
          litertlm_builder.TfLiteModelType.EMBEDDER,
      ),
      (
          "mtp_drafter_with_prefix",
          "tf_lite_mtp_drafter",
          litertlm_builder.TfLiteModelType.MTP_DRAFTER,
      ),
  )
  def test_get_enum_from_tf_free_value_with_prefix(
      self, input_val: str, expected_enum: litertlm_builder.TfLiteModelType
  ):
    with self.assertLogs(level="WARNING") as cm:
      result = litertlm_builder.TfLiteModelType.get_enum_from_tf_free_value(
          input_val
      )
    self.assertEqual(result, expected_enum)
    self.assertTrue(
        any(
            f"Input '{input_val}' already starts with 'tf_lite_'." in output
            for output in cm.output
        )
    )

  def test_get_enum_from_tf_free_value_invalid_raises_error(self):
    for invalid_val in (
        "nonexistent_model_type",
        "tf_lite_nonexistent_model_type",
    ):
      with self.subTest(invalid_val=invalid_val):
        with self.assertRaises(ValueError):
          litertlm_builder.TfLiteModelType.get_enum_from_tf_free_value(
              invalid_val
          )


if __name__ == "__main__":
  absltest.main()
