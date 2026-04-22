# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import shutil
import tempfile
import textwrap
from unittest import mock

from absl.testing import absltest

from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_peek


class LitertlmPeekPyTest(absltest.TestCase):

  def test_process_litertlm_file(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)
    self.assertIn("LlmMetadataProto", stdout)
    self.assertIn("<<<<<<<< start of LlmMetadata", stdout)

  def test_process_litertlm_file_hf_tokenizer(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_hf_tokenizer.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("HF_Tokenizer_Zlib", stdout)

  def test_process_litertlm_file_tokenizer_tflite(self):
    """Tests the process_litertlm_file function directly."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tokenizer_tflite.litertlm",
    )

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly.
    litertlm_peek.peek_litertlm_file(test_data_path, None, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)

  def test_process_litertlm_file_with_dump_files(self):
    """Tests the process_litertlm_file function with dump_files_dir."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm",
    )

    # Create a temporary directory for dumping files.
    dump_dir = self.create_tempdir().full_path

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly with dump_files_dir.
    litertlm_peek.peek_litertlm_file(test_data_path, dump_dir, output_stream)

    # Get the output and perform assertions.
    stdout = output_stream.getvalue()
    self.assertNotEmpty(stdout)

    # Assert that the output contains expected strings.
    self.assertIn("LiteRT-LM Version:", stdout)
    self.assertIn("System Metadata", stdout)
    self.assertIn("Sections", stdout)
    self.assertIn("SP_Tokenizer", stdout)
    self.assertIn("TFLiteModel", stdout)
    self.assertIn("LlmMetadataProto", stdout)
    self.assertIn("<<<<<<<< start of LlmMetadata", stdout)

    # Check if files were dumped in the specified directory.
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, "Section0_SP_Tokenizer.spiece"))
    )
    tflite_model_filename = "Section1_TFLiteModel_tf_lite_prefill_decode.tflite"
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, tflite_model_filename))
    )
    self.assertTrue(
        os.path.exists(os.path.join(dump_dir, "LlmMetadataProto.pbtext"))
    )
    toml_path = os.path.join(dump_dir, "model.toml")
    self.assertTrue(os.path.exists(toml_path))

    with open(os.path.join(dump_dir, "model.toml"), "r") as f:
      toml_content = f.read()
      self.assertEqual(textwrap.dedent("""\
          [system_metadata]
          entries = [
            { key = "arch", value_type = "String", value = "all" },
            { key = "version", value_type = "String", value = "0.1" },
          ]

          [[section]]
          additional_metadata = [
            { key = "vocab_size", value_type = "Int32", value = 10000 },
            { key = "algorithm", value_type = "String", value = "bpe" },
          ]
          section_type = "SP_Tokenizer"
          data_path = "Section0_SP_Tokenizer.spiece"

          [[section]]
          additional_metadata = [
            { key = "quantized", value_type = "Bool", value = true },
            { key = "model_size", value_type = "Int32", value = 1234567 },
          ]
          model_type = "prefill_decode"
          section_type = "TFLiteModel"
          data_path = "Section1_TFLiteModel_tf_lite_prefill_decode.tflite"

          [[section]]
          additional_metadata = [
            { key = "model", value_type = "String", value = "gemma3" },
          ]
          section_type = "LlmMetadata"
          data_path = "LlmMetadataProto.pbtext"

          [[section]]
          section_type = "GenericBinaryData"
          data_path = "Section3_GenericBinaryData.bin"
          """), toml_content)

  def test_dumped_litertlm_file_can_be_rebuilt(self):
    """Tests the process_litertlm_file function with dump_files_dir."""
    test_data_path = os.path.join(
        os.environ.get("TEST_SRCDIR", ""),
        "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm",
    )

    # Create a temporary directory for dumping files.
    dump_dir = self.create_tempdir().full_path

    # Use an in-memory stream to capture the output.
    output_stream = io.StringIO()

    # Call the function directly with dump_files_dir.
    litertlm_peek.peek_litertlm_file(test_data_path, dump_dir, output_stream)

    toml_path = os.path.join(dump_dir, "model.toml")
    self.assertTrue(os.path.exists(toml_path))

    # Check that the model can be rebuilt from the dumped files.
    rebuilt_model = self.create_tempfile("rebuilt.litertlm")
    with open(rebuilt_model.full_path, "wb") as f:
      builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
      builder.build(f)
      # Ideally we'd compare the contents of the rebuilt model with the
      # original, but the contents may differ over time as the .litertlm
      # format evolves. Rather than requiring the test to be updated every time
      # the format changes, we only check that rebuilding the model runs
      # successfully.


class LitertlmPeekUtilTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def _create_mock_section(self, items):
    mock_section = mock.Mock()
    mock_section.ItemsLength.return_value = len(items)
    mock_section.Items.side_effect = items
    mock_section.BeginOffset.return_value = 0
    mock_section.EndOffset.return_value = 10  # Dummy size
    return mock_section

  def test_get_tflite_weight_filename_with_type(self):
    with mock.patch.object(
        litertlm_peek, "_get_model_type", return_value="decoder"
    ):
      filename = litertlm_peek._get_tflite_weight_filename(mock.Mock(), 0)
      self.assertEqual(filename, "Section0_TFLiteWeights_decoder.weight")

  def test_get_tflite_weight_filename_without_type(self):
    with mock.patch.object(litertlm_peek, "_get_model_type", return_value=None):
      filename = litertlm_peek._get_tflite_weight_filename(mock.Mock(), 1)
      self.assertEqual(filename, "Section1_TFLiteWeights.weight")

  def test_dump_tflite_weight(self):
    mock_section = self._create_mock_section([])
    mock_stream = io.BytesIO(b"0123456789")
    mock_output = io.StringIO()
    file_content = b"0123456789"
    with mock.patch.object(
        litertlm_peek,
        "_get_tflite_weight_filename",
        return_value="model.weight",
    ) as mock_get_name:
      litertlm_peek._dump_tflite_weight(
          mock_stream, mock_section, 0, self.temp_dir, mock_output
      )
      mock_get_name.assert_called_once()
      file_path = os.path.join(self.temp_dir, "model.weight")
      self.assertTrue(os.path.exists(file_path))
      with open(file_path, "rb") as f:
        self.assertEqual(f.read(), file_content)
      self.assertIn("model.weight dumped to", mock_output.getvalue())


if __name__ == "__main__":
  absltest.main()
