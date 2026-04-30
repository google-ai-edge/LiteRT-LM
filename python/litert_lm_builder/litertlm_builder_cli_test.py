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
import subprocess

from absl.testing import absltest

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_peek
from litert_lm.runtime.proto import llm_metadata_pb2

from python import runfiles

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
# Section 1: HF_Tokenizer
section_type = "HF_Tokenizer"
data_path = "{HF_TOKENIZER_PATH}"

[[section]]
# Section 2: TFLiteModel (Embedder)
section_type = "TFLiteModel"
model_type = "EMBEDDER"
data_path = "{EMBEDDER_PATH}"

[[section]]
# Section 3: TFLiteModel (Prefill/Decode)
section_type = "TFLiteModel"
model_type = "PREFILL_DECODE"
backend_constraint = "GPU"
data_path = "{PREFILL_DECODE_PATH}"
additional_metadata = [
  { key = "License", value_type = "String", value = "Example" }
]
"""


class LiteRTLMBuilderCLITest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def _create_placeholder_file(self, filename: str, content: bytes) -> str:
    filepath = os.path.join(self.temp_dir, filename)
    with litertlm_core.open_file(filepath, "wb") as f:
      f.write(content)
    return filepath

  def _get_command_path(self) -> str:
    """Returns the path to the command binary."""
    r = runfiles.Create()
    return r.Rlocation(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "litertlm_builder_cli"
        )
    )

  def _run_command(self, *args) -> str:
    """Runs the command with the given arguments."""
    output_path = os.path.join(self.temp_dir, "litertlm.litertlm")
    command = [
        self._get_command_path(),
        *args,
        "output",
        "--path",
        output_path,
    ]
    try:
      subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
      print("command stdout:\n", e.stdout.decode("utf-8"))
      print("command stderr:\n", e.stderr.decode("utf-8"))
      raise e
    return output_path

  def _peek_litertlm_file(self, path: str) -> str:
    """Peeks the litertlm file and returns the string representation."""
    stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(path, self.temp_dir, stream)
    return stream.getvalue()

  def test_system_metadata(self):
    """Tests that system metadata can be added correctly."""
    args = ["system_metadata", "--str", "key1", "value1"]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Key: key1, Value (String): value1", ss)
    self.assertIn("Sections (0)", ss)

  def test_llm_metadata(self):
    """Tests that LLM metadata can be added from a binary proto file."""
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    bin_proto = llm_metadata.SerializeToString()
    metadata_path = self._create_placeholder_file("llm.pb", bin_proto)
    args = [
        "system_metadata",
        "--int",
        "my_key",
        "23",
        "llm_metadata",
        "--path",
        metadata_path,
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("max_num_tokens: 123", ss)
    self.assertIn("Sections (1)", ss)

  def test_tflite_model(self):
    """Tests that a TFLite model can be added correctly."""
    tflite_path = self._create_placeholder_file(
        "model.tflite", b"dummy tflite content"
    )
    args = [
        "system_metadata",
        "--int",
        "my_key",
        "23",
        "tflite_model",
        "--path",
        tflite_path,
        "--model_type",
        "prefill_decode",
        "--str_metadata",
        "model_version",
        "1.0.1",
        "--backend_constraint",
        "CPU",
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: model_version, Value (String): 1.0.1", ss)
    self.assertIn("Key: backend_constraint, Value (String): cpu", ss)

  def test_tflite_weights(self):
    """Tests that TFLite weights can be added correctly via CLI."""
    tflite_path = self._create_placeholder_file(
        "model.weights", b"dummy tflite weights content"
    )
    args = [
        "system_metadata",
        "--int",
        "my_key",
        "23",
        "tflite_weights",
        "--path",
        tflite_path,
        "--model_type",
        "prefill_decode",
        "--str_metadata",
        "weights_version",
        "1.0.1",
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    TFLiteWeights", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: weights_version, Value (String): 1.0.1", ss)

  def test_sp_tokenizer(self):
    """Tests that a SentencePiece tokenizer can be added correctly."""
    sp_path = self._create_placeholder_file("sp.model", b"dummy sp content")
    args = [
        "system_metadata",
        "--int",
        "my_key",
        "23",
        "sp_tokenizer",
        "--path",
        sp_path,
        "--str_metadata",
        "tokenizer_version",
        "1.0.1",
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Key: tokenizer_version, Value (String): 1.0.1", ss)

  def test_hf_tokenizer(self):
    """Tests that a HuggingFace tokenizer can be added correctly."""
    hf_path = self._create_placeholder_file(
        "tokenizer.json", b'{"version": "1.0"}'
    )
    args = [
        "system_metadata",
        "--int",
        "my_key",
        "23",
        "hf_tokenizer",
        "--path",
        hf_path,
        "--str_metadata",
        "tokenizer_version",
        "1.0.1",
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (1)", ss)
    self.assertIn("Data Type:    HF_Tokenizer_Zlib", ss)
    self.assertIn("Key: tokenizer_version, Value (String): 1.0.1", ss)

  def test_end_to_end(self):
    """Tests a more complex end-to-end scenario with multiple sections."""
    sp_path = self._create_placeholder_file("sp.model", b"dummy sp content")
    tflite_path = self._create_placeholder_file(
        "model.tflite", b"dummy tflite content"
    )
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    bin_proto = llm_metadata.SerializeToString()
    metadata_path = self._create_placeholder_file("llm.pb", bin_proto)

    args = [
        "system_metadata",
        "--str",
        "Authors",
        "ODML team",
        "sp_tokenizer",
        "--path",
        sp_path,
        "tflite_model",
        "--path",
        tflite_path,
        "--model_type",
        "embedder",
        "tflite_model",
        "--path",
        tflite_path,
        "--model_type",
        "prefill_decode",
        "--backend_constraint",
        "GPU",
        "llm_metadata",
        "--path",
        metadata_path,
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (4)", ss)
    self.assertIn("Data Type:    SP_Tokenizer", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: backend_constraint, Value (String): gpu", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)

  def test_toml_file(self):
    """Tests that a TOML file can be added correctly."""
    hf_path = pathlib.Path(
        self._create_placeholder_file("tokenizer.json", b'{"version": "1.0"}')
    ).as_posix()
    tflite_path = pathlib.Path(
        self._create_placeholder_file("model.tflite", b"dummy tflite content")
    ).as_posix()
    llm_metadata = llm_metadata_pb2.LlmMetadata(max_num_tokens=123)
    bin_proto = llm_metadata.SerializeToString()
    metadata_path = pathlib.Path(
        self._create_placeholder_file("llm.pb", bin_proto)
    ).as_posix()
    toml_path = self._create_placeholder_file(
        "test.toml",
        _TOML_TEMPLATE.replace("{LLM_METADATA_PATH}", metadata_path)
        .replace("{HF_TOKENIZER_PATH}", hf_path)
        .replace("{EMBEDDER_PATH}", tflite_path)
        .replace("{PREFILL_DECODE_PATH}", tflite_path)
        .encode("utf-8"),
    )
    args = [
        "toml",
        "--path",
        toml_path,
    ]
    output_path = self._run_command(*args)
    self.assertTrue(os.path.exists(output_path))
    ss = self._peek_litertlm_file(output_path)
    self.assertIn("Sections (4)", ss)
    self.assertIn("Data Type:    HF_Tokenizer_Zlib", ss)
    self.assertIn("Data Type:    TFLiteModel", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_embedder", ss)
    self.assertIn("Key: model_type, Value (String): tf_lite_prefill_decode", ss)
    self.assertIn("Key: backend_constraint, Value (String): gpu", ss)
    self.assertIn("Data Type:    LlmMetadataProto", ss)
    self.assertIn("max_num_tokens: 123", ss)

  def test_toml_cannot_be_used_with_other_args(self):
    """Tests that a TOML file cannot be used with other args."""
    tflite_path = self._create_placeholder_file(
        "model.tflite", b"dummy tflite content"
    )
    toml_path = self._create_placeholder_file(
        "test.toml",
        _TOML_TEMPLATE.replace("{PREFILL_DECODE_PATH}", tflite_path).encode(
            "utf-8"
        ),
    )
    args = [
        "toml",
        "--path",
        toml_path,
        "system_metadata",
        "--str",
        "key1",
        "value1",
    ]
    with self.assertRaises(subprocess.CalledProcessError):
      self._run_command(*args)

  def test_help_root(self):
    """Tests that the help command prints the correct output."""
    command = [self._get_command_path(), "--help"]
    output = subprocess.run(command, check=True, capture_output=True)
    self.assertEqual(output.returncode, 0)
    stdout = output.stdout.decode("utf-8")
    self.assertIn(
        "Build a LiteRT-LM file from input files and metadata", stdout
    )

  def test_help_subcommand(self):
    """Tests that the help command prints the correct output for subcommand."""
    command = [self._get_command_path(), "system_metadata", "--help"]
    output = subprocess.run(command, check=True, capture_output=True)
    self.assertEqual(output.returncode, 0)
    stdout = output.stdout.decode("utf-8")
    self.assertIn(
        "Add one or more system metadata key-value pairs to the LiteRT-LM"
        " file.",
        stdout,
    )


if __name__ == "__main__":
  absltest.main()
