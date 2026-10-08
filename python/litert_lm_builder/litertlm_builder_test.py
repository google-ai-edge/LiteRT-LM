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
from runtime.proto import asr_metadata_pb2
from runtime.proto import embedding_metadata_pb2
from runtime.proto import executor_metadata_pb2
from runtime.proto import image_gen_metadata_pb2
from runtime.proto import llm_metadata_pb2
from runtime.proto import tts_metadata_pb2

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

  def test_set_and_remove_backend_constraint(self):
    """Tests mutating and clearing backend_constraint on an existing section."""
    tflite_path = self._create_dummy_file(
        "model.tflite", b"dummy tflite content"
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        tflite_path,
        litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW,
        backend_constraint="cpu",
    )

    builder.set_backend_constraint("audio_encoder_hw", "npu")
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Key: backend_constraint, Value (String): npu", ss)

    builder.remove_backend_constraint(
        litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertNotIn("Key: backend_constraint", ss)

    with self.assertRaises(KeyError):
      builder.remove_backend_constraint("vision_encoder")

    with self.assertRaisesRegex(ValueError, "Invalid backend constraint"):
      builder.set_backend_constraint("audio_encoder_hw", "invalid_backend")

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
    """Tests that adding a tokenizer more than once raises a ValueError."""
    sp_path = self._create_dummy_file("sp.model", b"")

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_sentencepiece_tokenizer(sp_path)

    with self.assertRaises(ValueError):
      builder.add_hf_tokenizer(self._create_dummy_file("tokenizer.json", b""))
    with self.assertRaises(ValueError):
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

  def test_add_llm_metadata_with_min_runtime_version(self):
    """Tests that min_runtime_version is added to llm metadata."""
    meta_path = self._create_dummy_file(
        "metadata.pb", llm_metadata_pb2.LlmMetadata().SerializeToString()
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_llm_metadata(meta_path, min_runtime_version="0.12.3")
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("min_runtime_version: \"0.12.3\"", ss)

  def test_add_llm_metadata_textproto_with_min_runtime_version(self):
    """Tests that min_runtime_version is added to textproto llm metadata."""
    meta_path = self._create_dummy_file("metadata.pbtext", b"")
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_llm_metadata(meta_path, min_runtime_version="0.12.3")
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("min_runtime_version: \"0.12.3\"", ss)

  def test_from_toml_file_with_min_runtime_version(self):
    """Tests that min_runtime_version specified in TOML section is applied."""
    metadata_path = self._create_dummy_file("metadata.pbtext", b"")
    metadata_filename = os.path.basename(metadata_path)
    toml_content = f"""
[[section]]
section_type = "LlmMetadata"
data_path = "{metadata_filename}"
min_runtime_version = "0.12.3"
"""
    toml_path = self._create_dummy_file(
        "min_version.toml", toml_content.encode()
    )
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("min_runtime_version: \"0.12.3\"", ss)

  def test_add_llm_metadata_with_thinking_and_function_calling(self):
    """Tests supports_thinking and supports_function_calling in llm metadata."""
    meta_path = self._create_dummy_file(
        "metadata.pb", llm_metadata_pb2.LlmMetadata().SerializeToString()
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=False,
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("supports_thinking: true", ss)
    self.assertIn("supports_function_calling: false", ss)

  def test_from_toml_file_with_vision_patch_metadata(self):
    """Tests max_num_patches and pooling_kernel_size from TOML."""
    metadata_path = self._create_dummy_file("metadata.pbtext", b"")
    metadata_filename = os.path.basename(metadata_path)
    toml_content = f"""
[[section]]
section_type = "LlmMetadata"
data_path = "{metadata_filename}"
supports_thinking = true
supports_function_calling = true
max_num_patches = 4
pooling_kernel_size = 2
"""
    toml_path = self._create_dummy_file(
        "capabilities_vision.toml", toml_content.encode()
    )
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("supports_thinking: true", ss)
    self.assertIn("supports_function_calling: true", ss)
    self.assertIn("max_num_patches: 4", ss)
    self.assertIn("pooling_kernel_size: 2", ss)

  def test_is_llm_model(self):
    """Tests is_llm_model property detection."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self.assertFalse(builder.is_llm_model)

    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    self.assertTrue(builder.is_llm_model)

  def test_validate_metadata_llm_missing_fields(self):
    """Tests validate_metadata raises ValueError when LLM fields are missing."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_path = self._create_dummy_file(
        "metadata.pb", llm_metadata_pb2.LlmMetadata().SerializeToString()
    )

    # Missing both supports_thinking and supports_function_calling
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_llm_metadata(meta_path)
    with self.assertRaisesRegex(ValueError, "supports_thinking"):
      builder.validate_metadata()

    # Missing supports_function_calling
    builder2 = litertlm_builder.LitertLmFileBuilder()
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder2.add_llm_metadata(meta_path, supports_thinking=True)
    with self.assertRaisesRegex(ValueError, "supports_function_calling"):
      builder2.validate_metadata()

    # Valid LLM metadata passes validation
    builder3 = litertlm_builder.LitertLmFileBuilder()
    builder3.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder3.add_llm_metadata(
        meta_path,
        supports_thinking=False,
        supports_function_calling=True,
    )
    builder3.validate_metadata()

  def test_validate_metadata_requires_llm_metadata(self):
    """Tests validate_metadata raises when an LLM model has no LlmMetadata."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    with self.assertRaisesRegex(ValueError, "LlmMetadata"):
      builder.validate_metadata()

  def test_validate_metadata_skipped_for_non_llm_model(self):
    """Tests validate_metadata is a no-op when no LLM model is present."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    self.assertFalse(builder.is_llm_model)
    builder.validate_metadata()

  def test_is_vision_model(self):
    """Tests is_vision_model property detection."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self.assertFalse(builder.is_vision_model)

    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    self.assertTrue(builder.is_vision_model)

  def test_is_vision_transformer_model(self):
    """Tests is_vision_transformer_model property detection."""
    builder = litertlm_builder.LitertLmFileBuilder()
    self.assertFalse(builder.is_vision_transformer_model)

    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )

    # Gemma3N is not a vision transformer
    meta_gemma3n = llm_metadata_pb2.LlmMetadata()
    meta_gemma3n.llm_model_type.gemma3n.SetInParent()
    path_gemma3n = self._create_dummy_file(
        "meta_gemma3n.pb", meta_gemma3n.SerializeToString()
    )
    builder.add_llm_metadata(path_gemma3n)
    self.assertFalse(builder.is_vision_transformer_model)

    # Gemma4 is a vision transformer
    builder2 = litertlm_builder.LitertLmFileBuilder()
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    meta_gemma4 = llm_metadata_pb2.LlmMetadata()
    meta_gemma4.llm_model_type.gemma4.SetInParent()
    path_gemma4 = self._create_dummy_file(
        "meta_gemma4.pb", meta_gemma4.SerializeToString()
    )
    builder2.add_llm_metadata(path_gemma4)
    self.assertTrue(builder2.is_vision_transformer_model)

  def test_validate_metadata_vision_missing_fields(self):
    """Tests validate_metadata raises for missing Vision Transformer fields."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta = llm_metadata_pb2.LlmMetadata()
    meta.llm_model_type.gemma4.SetInParent()
    meta_path = self._create_dummy_file(
        "metadata_gemma4.pb", meta.SerializeToString()
    )

    # Missing max_num_patches and pooling_kernel_size
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
    )
    with self.assertRaisesRegex(ValueError, "max_num_patches"):
      builder.validate_metadata()

    # Missing pooling_kernel_size
    builder2 = litertlm_builder.LitertLmFileBuilder()
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder2.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
        max_num_patches=4,
    )
    with self.assertRaisesRegex(ValueError, "pooling_kernel_size"):
      builder2.validate_metadata()

    # Valid vision transformer metadata passes validation
    builder3 = litertlm_builder.LitertLmFileBuilder()
    builder3.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder3.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder3.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
        max_num_patches=4,
        pooling_kernel_size=2,
    )
    builder3.validate_metadata()

  def test_validate_metadata_vision_non_transformer_passes(self):
    """Tests non-vision-transformer models (e.g. Gemma3N) pass validation."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta = llm_metadata_pb2.LlmMetadata()
    meta.llm_model_type.gemma3n.image_tensor_height = 768
    meta.llm_model_type.gemma3n.image_tensor_width = 768
    meta_path = self._create_dummy_file(
        "metadata_gemma3n.pb", meta.SerializeToString()
    )

    # Missing supports_thinking raises
    builder_missing_thinking = litertlm_builder.LitertLmFileBuilder()
    builder_missing_thinking.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder_missing_thinking.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder_missing_thinking.add_llm_metadata(meta_path)
    with self.assertRaisesRegex(ValueError, "supports_thinking"):
      builder_missing_thinking.validate_metadata()

    # Valid Gemma3N vision model passes without patch parameters
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
    )
    self.assertTrue(builder.is_llm_model)
    self.assertTrue(builder.is_vision_model)
    self.assertFalse(builder.is_vision_transformer_model)
    builder.validate_metadata()

    path = os.path.join(self.temp_dir, "gemma3n.litertlm")
    with litertlm_core.open_file(path, "wb") as f:
      builder.build(f, validate_metadata=True)
    stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(path, self.temp_dir, stream)
    ss = stream.getvalue()
    self.assertIn("image_tensor_height: 768", ss)
    self.assertIn("supports_thinking: true", ss)

  def test_validate_metadata_vision_generic_model(self):
    """Tests validation for GenericModel with and without patch parameters."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta = llm_metadata_pb2.LlmMetadata()
    meta.llm_model_type.generic_model.image_tensor_height = 256
    meta.llm_model_type.generic_model.image_tensor_width = 256
    meta_path = self._create_dummy_file(
        "metadata_generic.pb", meta.SerializeToString()
    )

    # GenericModel without patchify passes
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
    )
    builder.validate_metadata()

    # GenericModel with max_num_patches but missing pooling_kernel_size raises
    builder2 = litertlm_builder.LitertLmFileBuilder()
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder2.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder2.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
        max_num_patches=4,
    )
    with self.assertRaisesRegex(ValueError, "pooling_kernel_size"):
      builder2.validate_metadata()

    # GenericModel with both passes
    builder3 = litertlm_builder.LitertLmFileBuilder()
    builder3.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder3.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ADAPTER
    )
    builder3.add_llm_metadata(
        meta_path,
        supports_thinking=True,
        supports_function_calling=True,
        max_num_patches=4,
        pooling_kernel_size=2,
    )
    builder3.validate_metadata()

  def test_build_with_validate_metadata(self):
    """Tests that builder.build(stream, validate_metadata=True) triggers validation."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    # validate_metadata=False (default) does not raise
    with io.BytesIO() as buf:
      builder.build(buf)
    # validate_metadata=True triggers validation and raises ValueError
    with io.BytesIO() as buf:
      with self.assertRaisesRegex(ValueError, "LlmMetadata"):
        builder.build(buf, validate_metadata=True)

  def test_add_multiple_tokenizers(self):
    """Tests that multiple tokenizers with different model types can be added."""
    sp_prefill = self._create_dummy_file(
        "sp_prefill.model", b"dummy prefill sp"
    )
    sp_embedder = self._create_dummy_file(
        "sp_embedder.model", b"dummy embedder sp"
    )

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_sentencepiece_tokenizer(
        sp_prefill,
        model_type=litertlm_builder.TfLiteModelType.PREFILL_DECODE,
    )
    builder.add_sentencepiece_tokenizer(
        sp_embedder,
        model_type=litertlm_builder.TfLiteModelType.EMBEDDER,
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (2)", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)

    # Adding a duplicate model_type should fail
    with self.assertRaises(ValueError):
      builder.add_sentencepiece_tokenizer(
          sp_embedder,
          model_type=litertlm_builder.TfLiteModelType.EMBEDDER,
      )

    # Adding duplicate via additional_metadata without tf_lite_ prefix fails
    with self.assertRaises(ValueError):
      builder.add_sentencepiece_tokenizer(
          sp_embedder,
          additional_metadata=[
              litertlm_builder.Metadata(
                  key="model_type",
                  value="embedder",
                  dtype=litertlm_builder.DType.STRING,
              )
          ],
      )

  def _create_llm_metadata_file(
      self,
      filename: str,
      model_type: str,
      supports_thinking: bool = True,
      supports_function_calling: bool = True,
      max_num_patches: int | None = None,
      pooling_kernel_size: int | None = None,
  ) -> str:
    meta = llm_metadata_pb2.LlmMetadata()
    meta.supports_thinking = supports_thinking
    meta.supports_function_calling = supports_function_calling
    sub_msg = getattr(meta.llm_model_type, model_type)
    sub_msg.SetInParent()
    if max_num_patches is not None:
      sub_msg.max_num_patches = max_num_patches
    if pooling_kernel_size is not None:
      sub_msg.pooling_kernel_size = pooling_kernel_size
    return self._create_dummy_file(filename, meta.SerializeToString())

  def _create_embedding_metadata_file(
      self,
      filename: str,
      is_vision: bool = False,
      max_num_patches: int | None = None,
      pooling_kernel_size: int | None = None,
  ) -> str:
    meta = embedding_metadata_pb2.EmbeddingMetadata()
    eg = meta.embedding_model_type.embedding_gemma_v2
    eg.SetInParent()
    if is_vision:
      eg.start_of_image_token.token_str = "<|image>"
      eg.end_of_image_token.token_str = "<image|>"
      eg.patch_width = 16
      eg.patch_height = 16
      if max_num_patches is not None:
        eg.max_num_patches = max_num_patches
      if pooling_kernel_size is not None:
        eg.pooling_kernel_size = pooling_kernel_size
    return self._create_dummy_file(filename, meta.SerializeToString())

  def test_validate_metadata_embedding_vit_valid(self):
    """Tests valid embedding ViT model passes validation."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_path = self._create_embedding_metadata_file(
        "emb_valid.pb",
        is_vision=True,
        max_num_patches=1260,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_embedding_metadata(meta_path)
    self.assertTrue(builder.is_embedding_vision_transformer_model)
    self.assertTrue(builder.is_vision_transformer_model)
    builder.validate_metadata()

  def test_validate_metadata_embedding_vit_missing_patches(self):
    """Tests embedding ViT missing max_num_patches raises ValueError."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_path = self._create_embedding_metadata_file(
        "emb_no_patches.pb",
        is_vision=True,
        max_num_patches=None,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_embedding_metadata(meta_path)
    with self.assertRaisesRegex(ValueError, "max_num_patches"):
      builder.validate_metadata()

  def test_validate_metadata_embedding_vit_missing_pooling_kernel_size(self):
    """Tests embedding ViT missing pooling_kernel_size raises ValueError."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_path = self._create_embedding_metadata_file(
        "emb_no_kernel.pb",
        is_vision=True,
        max_num_patches=1260,
        pooling_kernel_size=None,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_embedding_metadata(meta_path)
    with self.assertRaisesRegex(ValueError, "pooling_kernel_size"):
      builder.validate_metadata()

  def test_validate_metadata_embedding_vit_empty_fields_with_encoder(self):
    """Tests embedding with vision encoder but unpopulated vision fields raises."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_path = self._create_embedding_metadata_file(
        "emb_empty_vision.pb", is_vision=False
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_embedding_metadata(meta_path)
    with self.assertRaisesRegex(ValueError, "max_num_patches"):
      builder.validate_metadata()

  def test_validate_metadata_hybrid_non_vit_llm_and_vit_embedding(self):
    """Tests non-ViT LLM (Gemma3) + ViT embedding passes without conflict."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    meta_llm = llm_metadata_pb2.LlmMetadata(
        supports_thinking=True,
        supports_function_calling=True,
    )
    meta_llm.llm_model_type.gemma3.image_tensor_height = 768
    meta_llm.llm_model_type.gemma3.image_tensor_width = 768
    llm_path = self._create_dummy_file(
        "llm_gemma3.pb", meta_llm.SerializeToString()
    )
    emb_path = self._create_embedding_metadata_file(
        "emb_vit.pb",
        is_vision=True,
        max_num_patches=1260,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)
    self.assertFalse(builder.is_llm_vision_transformer_model)
    self.assertTrue(builder.is_embedding_vision_transformer_model)
    self.assertTrue(builder.is_vision_transformer_model)
    builder.validate_metadata()

  def test_validate_metadata_hybrid_vit_llm_and_text_embedding(self):
    """Tests ViT LLM (Gemma4) + pure text embedding passes without conflict."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    llm_path = self._create_llm_metadata_file(
        "llm_gemma4.pb", "gemma4", max_num_patches=4, pooling_kernel_size=2
    )
    emb_path = self._create_embedding_metadata_file(
        "emb_text.pb", is_vision=False
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)
    self.assertTrue(builder.is_llm_vision_transformer_model)
    self.assertFalse(builder.is_embedding_vision_transformer_model)
    self.assertTrue(builder.is_vision_transformer_model)
    builder.validate_metadata()

  def test_validate_metadata_hybrid_vit_llm_and_vit_embedding_both_valid(self):
    """Tests both ViT LLM and ViT embedding valid passes."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    llm_path = self._create_llm_metadata_file(
        "llm_gemma4.pb", "gemma4", max_num_patches=4, pooling_kernel_size=2
    )
    emb_path = self._create_embedding_metadata_file(
        "emb_vit.pb",
        is_vision=True,
        max_num_patches=1260,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)
    self.assertTrue(builder.is_llm_vision_transformer_model)
    self.assertTrue(builder.is_embedding_vision_transformer_model)
    builder.validate_metadata()

  def test_validate_metadata_hybrid_vit_llm_and_vit_embedding_emb_invalid(self):
    """Tests hybrid catches missing patch parameters in embedding."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    llm_path = self._create_llm_metadata_file(
        "llm_gemma4.pb", "gemma4", max_num_patches=4, pooling_kernel_size=2
    )
    emb_path = self._create_embedding_metadata_file(
        "emb_vit_bad.pb",
        is_vision=True,
        max_num_patches=None,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)
    with self.assertRaisesRegex(ValueError, "max_num_patches"):
      builder.validate_metadata()

  def test_validate_metadata_hybrid_vit_llm_and_vit_embedding_llm_invalid(self):
    """Tests hybrid catches missing patch parameters in LLM."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy")
    llm_path = self._create_llm_metadata_file(
        "llm_gemma4_bad.pb",
        "gemma4",
        max_num_patches=None,
        pooling_kernel_size=2,
    )
    emb_path = self._create_embedding_metadata_file(
        "emb_vit.pb",
        is_vision=True,
        max_num_patches=1260,
        pooling_kernel_size=3,
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.VISION_ENCODER
    )
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)
    with self.assertRaisesRegex(ValueError, "max_num_patches"):
      builder.validate_metadata()

  def test_hybrid_llm_and_embedding_roundtrip(self):
    """Tests build, unpack, and rebuild of hybrid container."""
    dummy_tflite = self._create_dummy_file("model.tflite", b"dummy tflite")
    dummy_sp = self._create_dummy_file("sp.model", b"dummy sp")
    llm_path = self._create_llm_metadata_file(
        "llm.pb",
        "gemma3n",
        supports_thinking=True,
        supports_function_calling=True,
    )
    emb_path = self._create_embedding_metadata_file("emb.pb", is_vision=False)

    builder = litertlm_builder.LitertLmFileBuilder()
    self._add_system_metadata(builder)
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.EMBEDDER
    )
    builder.add_tflite_model(
        dummy_tflite, litertlm_builder.TfLiteModelType.TEXT_ENCODER
    )
    builder.add_sentencepiece_tokenizer(dummy_sp)
    builder.add_llm_metadata(llm_path)
    builder.add_embedding_metadata(emb_path)

    out_file = os.path.join(self.temp_dir, "hybrid.litertlm")
    with open(out_file, "wb") as f:
      builder.build(f, validate_metadata=True)

    # Unpack and verify
    unpack_dir = os.path.join(self.temp_dir, "unpacked_hybrid")
    unpacked_builder = litertlm_builder.LitertLmFileBuilder.unpack(
        out_file, unpack_dir
    )
    self.assertTrue(unpacked_builder._has_llm_metadata)
    self.assertTrue(unpacked_builder._has_embedding_metadata)

    # Verify model.toml has both sections
    with open(os.path.join(unpack_dir, "model.toml"), "r") as f:
      toml_text = f.read()
    self.assertIn('section_type = "LlmMetadata"', toml_text)
    self.assertIn('section_type = "EmbeddingMetadata"', toml_text)

    # Rebuild from unpacked
    rebuild_file = os.path.join(self.temp_dir, "rebuilt_hybrid.litertlm")
    with open(rebuild_file, "wb") as f:
      unpacked_builder.build(f, validate_metadata=True)
    self.assertGreater(os.path.getsize(rebuild_file), 0)

  def test_from_toml_with_asr_metadata(self):
    """Tests that TOML with AsrMetadata and capability model types builds properly."""
    asr_meta = asr_metadata_pb2.AsrMetadata()
    asr_meta_path = self._create_dummy_file(
        "asr_meta.pb", asr_meta.SerializeToString()
    )
    enc_dec_path = self._create_dummy_file(
        "enc_dec.tflite", b"dummy enc_dec tflite"
    )

    toml_content = f"""
[system_metadata]
entries = [
  {{ key = "author", value_type = "String", value = "ODML" }}
]

[[section]]
section_type = "AsrMetadata"
data_path = "{asr_meta_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "ENCODER_DECODER"
data_path = "{enc_dec_path}"
"""
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_str(toml_content)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (2)", ss)
    self.assertIn("Data Type:    AsrMetadataProto", ss)
    self.assertIn(
        "Key: model_type, Value (String): tf_lite_encoder_decoder", ss
    )

  def test_from_toml_with_tts_metadata(self):
    """Tests that TOML with TtsMetadata and capability model types builds properly."""
    tts_meta = tts_metadata_pb2.TtsMetadata(
        output_sample_rate=24000, supported_languages=["en-US"]
    )
    tts_meta_path = self._create_dummy_file(
        "tts_meta.textproto",
        text_format.MessageToString(tts_meta).encode("utf-8"),
    )
    acoustic_path = self._create_dummy_file(
        "acoustic.tflite", b"dummy acoustic tflite"
    )
    vocoder_path = self._create_dummy_file(
        "vocoder.tflite", b"dummy vocoder tflite"
    )

    toml_content = f"""
[system_metadata]
entries = [
  {{ key = "author", value_type = "String", value = "ODML" }}
]

[[section]]
section_type = "TtsMetadata"
data_path = "{tts_meta_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "ACOUSTIC"
data_path = "{acoustic_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "TF_LITE_VOCODER"
data_path = "{vocoder_path}"
"""
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_str(toml_content)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (3)", ss)
    self.assertIn("Data Type:    TtsMetadataProto", ss)
    self.assertIn("output_sample_rate: 24000", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_acoustic", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_vocoder", ss)

  def test_from_toml_with_image_gen_metadata(self):
    """Tests that TOML with ImageGenMetadata and capability model types builds properly."""
    image_gen_meta_content = """
    image_gen_model_type {
      bonsai_flux2 {
        flux2_params {
          img_size: 256
          default_steps: 4
          packed_ch: 128
          seq_len: 128
        }
      }
    }
    """
    image_gen_meta_path = self._create_dummy_file(
        "image_gen_meta.textproto", image_gen_meta_content.encode()
    )
    text_enc_path = self._create_dummy_file(
        "textenc.tflite", b"dummy textenc tflite"
    )
    dit_path = self._create_dummy_file("dit.tflite", b"dummy dit tflite")
    vae_path = self._create_dummy_file("vae.tflite", b"dummy vae tflite")

    toml_content = f"""
[system_metadata]
entries = [
  {{ key = "author", value_type = "String", value = "ODML" }}
]

[[section]]
section_type = "ImageGenMetadata"
data_path = "{image_gen_meta_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "TF_LITE_TEXT_ENCODER"
data_path = "{text_enc_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "IMAGE_DENOISER"
data_path = "{dit_path}"

[[section]]
section_type = "TFLiteModel"
model_type = "IMAGE_DECODER"
data_path = "{vae_path}"
"""
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_str(toml_content)
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Sections (4)", ss)
    self.assertIn("Data Type:    ImageGenMetadataProto", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_text_encoder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_image_denoiser", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_image_decoder", ss)

  def test_capability_tflite_model_type_int_enum_resolution(self):
    """Tests resolving capability proto enum ints for ASR, TTS, and ImageGen."""
    tflite_path = self._create_dummy_file("m.tflite", b"dummy")
    asr_path = self._create_dummy_file(
        "asr.pb", asr_metadata_pb2.AsrMetadata().SerializeToString()
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_asr_metadata(asr_path)
    builder.add_tflite_model(
        tflite_path, asr_metadata_pb2.AsrMetadata.TF_LITE_AUDIO_ENCODER
    )
    ss = self._build_and_read_litertlm(builder)
    self.assertIn("Key: model_type, Value (String): tf_lite_audio_encoder", ss)

    tts_path = self._create_dummy_file(
        "tts.pb", tts_metadata_pb2.TtsMetadata().SerializeToString()
    )
    builder_tts = litertlm_builder.LitertLmFileBuilder()
    builder_tts.add_tts_metadata(tts_path)
    builder_tts.add_tflite_model(
        tflite_path, tts_metadata_pb2.TtsMetadata.TF_LITE_ACOUSTIC
    )
    ss_tts = self._build_and_read_litertlm(builder_tts)
    self.assertIn("Key: model_type, Value (String): tf_lite_acoustic", ss_tts)

    img_path = self._create_dummy_file(
        "img.pb", image_gen_metadata_pb2.ImageGenMetadata().SerializeToString()
    )
    builder_img = litertlm_builder.LitertLmFileBuilder()
    builder_img.add_image_gen_metadata(img_path)
    builder_img.add_tflite_model(
        tflite_path,
        image_gen_metadata_pb2.ImageGenMetadata.TF_LITE_DIFFUSION_TRANSFORMER_INITIAL,
    )
    ss_img = self._build_and_read_litertlm(builder_img)
    self.assertIn(
        "Key: model_type, Value (String):"
        " tf_lite_diffusion_transformer_initial",
        ss_img,
    )
    self.assertNotIn(
        "unspecified", litertlm_builder.get_all_tflite_model_types()
    )

    builder_no_cap = litertlm_builder.LitertLmFileBuilder()
    with self.assertRaisesRegex(
        ValueError, "Capability metadata must be added before resolving"
    ):
      builder_no_cap.add_tflite_model(
          tflite_path, asr_metadata_pb2.AsrMetadata.TF_LITE_AUDIO_ENCODER
      )

    with self.assertRaisesRegex(ValueError, "TF_LITE_MODEL_TYPE_UNSPECIFIED"):
      builder_img.add_tflite_model(
          tflite_path,
          image_gen_metadata_pb2.ImageGenMetadata.TF_LITE_MODEL_TYPE_UNSPECIFIED,
      )

  def test_multiple_capability_metadata_raises_error(self):
    """Tests that adding conflicting capability metadata raises ValueError."""
    llm_meta_path = self._create_dummy_file(
        "llm.textproto", b"max_num_tokens: 10\n"
    )
    asr_meta_path = self._create_dummy_file("asr.textproto", b"")
    tts_meta_path = self._create_dummy_file(
        "tts.textproto", b"output_sample_rate: 24000\n"
    )
    image_gen_meta_path = self._create_dummy_file(
        "image_gen.textproto", b"image_gen_model_type { bonsai_flux2 {} }\n"
    )
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_llm_metadata(llm_meta_path)
    with self.assertRaisesRegex(
        ValueError, "can contain only one top-level capability metadata section"
    ):
      builder.add_asr_metadata(asr_meta_path)
    with self.assertRaisesRegex(
        ValueError, "can contain only one top-level capability metadata section"
    ):
      builder.add_tts_metadata(tts_meta_path)
    with self.assertRaisesRegex(
        ValueError, "can contain only one top-level capability metadata section"
    ):
      builder.add_image_gen_metadata(image_gen_meta_path)


if __name__ == "__main__":
  absltest.main()
