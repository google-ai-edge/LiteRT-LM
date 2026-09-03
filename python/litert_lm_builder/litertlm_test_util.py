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

"""Test utility for LiteRT-LM files."""

import io
import pathlib
import struct
from litert_lm_builder import litertlm_peek
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema


def _get_uuid_from_file(file_path: pathlib.Path) -> str:
  """Extracts the UUID from a .litertlm file.

  Args:
    file_path: Path to the .litertlm file.

  Returns:
    The UUID string.

  Raises:
    ValueError: If no UUID is found or system metadata is missing.
  """
  metadata = litertlm_peek.read_litertlm_header(str(file_path), io.StringIO())

  system_metadata = metadata.SystemMetadata()
  if not system_metadata:
    raise ValueError(f"No system metadata found in file {file_path!r}.")

  for i in range(system_metadata.EntriesLength()):
    kvp = system_metadata.Entries(i)
    if not kvp:
      continue

    key_bytes = kvp.Key()
    if not key_bytes or key_bytes.decode("utf-8") != "uuid":
      continue

    if kvp.ValueType() != schema.VData.StringValue:
      continue

    union_table = kvp.Value()
    if not union_table:
      continue

    value_obj = schema.StringValue()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    value_bytes = value_obj.Value()
    if value_bytes is not None:
      return value_bytes.decode("utf-8")

  raise ValueError(f"No UUID found in file {file_path!r}.")


def overwrite_uuid_in_file(file_path: pathlib.Path, *, new_uuid: str) -> None:
  """Overwrites the UUID in a .litertlm file (Test-only.)

  Args:
    file_path: Path to the .litertlm file.
    new_uuid: The new UUID string (must be 36 characters).

  Raises:
    ValueError: If length mismatch, if no UUID is found in the file, or if the
      old UUID is not found in the header.
    OSError: If file operations fail.
  """
  old_uuid = _get_uuid_from_file(file_path)

  old_uuid_bytes = old_uuid.encode("utf-8")
  new_uuid_bytes = new_uuid.encode("utf-8")

  if len(old_uuid_bytes) != len(new_uuid_bytes):
    raise ValueError(
        f"New UUID length ({len(new_uuid_bytes)}) must match old UUID length"
        f" ({len(old_uuid_bytes)})."
    )

  try:
    with open(file_path, "r+b") as f:
      f.seek(litertlm_peek.litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
      header_end_offset, = struct.unpack("<Q", f.read(8))

      f.seek(litertlm_peek.litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
      header_data = f.read(
          header_end_offset
          - litertlm_peek.litertlm_core.HEADER_BEGIN_BYTE_OFFSET
      )

      if old_uuid_bytes not in header_data:
        raise ValueError(f"Old UUID not found in header of file {file_path!r}.")

      updated_header_data = header_data.replace(old_uuid_bytes, new_uuid_bytes)

      f.seek(litertlm_peek.litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
      f.write(updated_header_data)
  except OSError as e:
    raise OSError(f"Failed to open or modify file {file_path!r}") from e
