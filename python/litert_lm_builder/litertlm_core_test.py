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

"""Tests for litertlm_core library.

This file contains unit tests for the functions and utilities provided by the
litertlm_core library. These tests cover the following aspects:

- Conversion of AnySectionDataType enum values to strings.
- Extraction of file extensions from filenames.
- Determination of section types and names based on filenames.
"""

from absl.testing import absltest
from absl.testing import parameterized
from litert_lm_builder import litertlm_core
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema


class LitertlmCoreTest(parameterized.TestCase):

  @parameterized.parameters(
      (schema.AnySectionDataType.NONE, "NONE"),
      (schema.AnySectionDataType.GenericBinaryData, "GenericBinaryData"),
      (schema.AnySectionDataType.TFLiteModel, "TFLiteModel"),
      (schema.AnySectionDataType.SP_Tokenizer, "SP_Tokenizer"),
      (schema.AnySectionDataType.LlmMetadataProto, "LlmMetadataProto"),
      (schema.AnySectionDataType.HF_Tokenizer_Zlib, "HF_Tokenizer_Zlib"),
  )
  def test_any_section_data_type_to_string(self, data_type, expected_str):
    """Tests the conversion of AnySectionDataType enum values to strings.

    Args:
      data_type: The AnySectionDataType enum value to convert.
      expected_str: The expected string representation of the enum value.
    """
    self.assertEqual(
        litertlm_core.any_section_data_type_to_string(data_type), expected_str
    )

  def test_any_section_data_type_to_string_unknown(self):
    """Tests the conversion of an unknown AnySectionDataType enum value.

    This test ensures that an appropriate error is raised when an
    unrecognized enum value is provided.
    """
    with self.assertRaises(ValueError):
      litertlm_core.any_section_data_type_to_string(999)


if __name__ == "__main__":
  absltest.main()
