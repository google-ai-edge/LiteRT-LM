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

import pathlib
from absl.testing import absltest
from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_test_util


class LitertlmTestUtilTest(absltest.TestCase):

  def test_overwrite_uuid_in_file(self):
    file_path = pathlib.Path(self.create_tempfile("test.litertlm").full_path)
    builder = litertlm_builder.LitertLmFileBuilder()
    # We need to add at least one section or metadata to make it valid
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="test_key",
            value="test_val",
            dtype=litertlm_builder.DType.STRING,
        )
    )

    with open(file_path, "wb") as f:
      builder.build(f)

    original_uuid = litertlm_test_util._get_uuid_from_file(file_path)
    self.assertIsNotNone(original_uuid)

    new_uuid = "00000000-0000-0000-0000-000000000000"
    litertlm_test_util.overwrite_uuid_in_file(file_path, new_uuid=new_uuid)

    updated_uuid = litertlm_test_util._get_uuid_from_file(file_path)
    self.assertEqual(updated_uuid, new_uuid)
    self.assertNotEqual(updated_uuid, original_uuid)


if __name__ == "__main__":
  absltest.main()
