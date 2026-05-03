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

"""Utility functions to help reading/writing LiteRT-LM files."""

from typing import BinaryIO

from litert_lm_builder import litertlm_core


def write_literlm_header_and_metadata(
    buff: BinaryIO, serialized_metadata: bytes | bytearray | memoryview
) -> int:
  """Writes a LiteRT-LM header to the given buffer.

  Args:
    buff: A file-like object to which to write the LiteRT-LM header.
    serialized_metadata: A serialized Flatbuffer containing the metadata for
      this ListeRT-LM file.

  Returns:
    The number of bytes written to `buff`.
  """
  # Write the magic bytes identifying this file as a LiteRT-LM file.
  offset = buff.write(litertlm_core.HEADER_MAGIC_BYTES)

  # Write the current LiteRT-LM version number.
  offset += buff.write(
      litertlm_core.LITERTLM_MAJOR_VERSION.to_bytes(4, 'little')
  )
  offset += buff.write(
      litertlm_core.LITERTLM_MINOR_VERSION.to_bytes(4, 'little')
  )
  offset += buff.write(
      litertlm_core.LITERTLM_PATCH_VERSION.to_bytes(4, 'little')
  )
  offset += buff.write(int(0).to_bytes(4, 'little'))  # Zero padding.

  # Write the offset of the end of the serialized metadata flatbuffer.
  assert offset == litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
  offset += buff.write(
      (
          litertlm_core.HEADER_BEGIN_BYTE_OFFSET + len(serialized_metadata)
      ).to_bytes(8, 'little')
  )

  # Append the serialized metadata.
  assert offset == litertlm_core.HEADER_BEGIN_BYTE_OFFSET
  offset += buff.write(serialized_metadata)

  # Return the number of bytes written.
  return offset
