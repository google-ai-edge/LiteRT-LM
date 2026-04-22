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

"""Core library with shared constants and utilities for LiteRT-LM tools."""

import os
import sys

from absl import app

from litert_lm.schema.core import litertlm_header_schema_py_generated as schema

# --- File Format Constants ---
LITERTLM_MAJOR_VERSION = 1
LITERTLM_MINOR_VERSION = 5
LITERTLM_PATCH_VERSION = 0
BLOCK_SIZE = 16 * 1024
HEADER_BEGIN_BYTE_OFFSET = 32
HEADER_END_LOCATION_BYTE_OFFSET = 24

SECTION_DATA_TYPE_TO_STRING_MAP = {
    v: k for k, v in schema.AnySectionDataType.__dict__.items()
}


def any_section_data_type_to_string(data_type: int):
  """Converts AnySectionDataType enum to its string representation."""
  if data_type in SECTION_DATA_TYPE_TO_STRING_MAP:
    return SECTION_DATA_TYPE_TO_STRING_MAP[data_type]
  else:
    raise ValueError(f"Unknown AnySectionDataType value: {data_type}")


def path_exists(file_path: str) -> bool:
  """Checks if a file exists."""
  return os.path.exists(file_path)


def open_file(file_path: str, mode: str = "rb"):
  """Opens a file using the given mode."""
  return open(file_path, mode)


def run_app(main_func):
  """Shared run logic for LiteRT-LM tools."""
  app.run(main_func, sys.argv[:1])
