# Copyright 2026 Google LLC.
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

"""Utility functions for LiteRT LM tests."""

import json
import os
from typing import Any


def read_json_config(
    caller_file_path: str, test_function_name: str
) -> list[Any]:
  """Loads test data from a JSON file.

  Dynamically loads test data from a JSON file matching the caller's filename.

  Args:
      caller_file_path: The __file__ variable from the calling test script.
      test_function_name: The top-level JSON key to extract the array from.

  Returns:
      A list of test cases from the JSON file.

  Raises:
      FileNotFoundError: If the JSON file is not found.
      KeyError: If the test_function_name is not found in the JSON file.
  """

  base_dir = os.path.dirname(os.path.abspath(caller_file_path))
  file_name = os.path.basename(caller_file_path)
  json_name = file_name.replace(".py", ".json")

  json_path = os.path.join(base_dir, "test_data", json_name)

  try:
    with open(json_path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except OSError as e:
    raise OSError(
        f"CRITICAL: Failed to open test data file missing at {json_path}"
    ) from e

  if test_function_name not in data:
    raise KeyError(
        f"CRITICAL: Key '{test_function_name}' not found in {json_name}. "
        f"Available keys: {list(data.keys())}"
    )
  return data[test_function_name]
