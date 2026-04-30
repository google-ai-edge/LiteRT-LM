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

"""Utility functions for LiteRT LM evaluation."""

from typing import Any, Dict, List


def parse_unknown_args(unknown_args: List[str]) -> Dict[str, Any]:
  """Parses unknown command line arguments into a kwargs dictionary.

  This takes a list of command-line arguments (e.g., unparsed arguments from
  `argparse.ArgumentParser.parse_known_args`) and converts them into a
  dictionary.
  It handles boolean flags (e.g., `--flag` -> `{'flag': True}`) and key-value
  pairs (e.g., `--key value` -> `{'key': 'value'}`). Basic type casting is
  applied
  to string values to convert them to integers, floats, or booleans.

  Args:
    unknown_args: A list of string arguments from the command line.

  Returns:
    A dictionary of parsed arguments, where keys are the flag names (without
    leading dashes) and values are the appropriately typed arguments.

  Example:
    >>> parse_unknown_args(['--write_out', '--backend', 'CPU', '--limit',
    '10.5'])
    {'write_out': True, 'backend': 'CPU', 'limit': 10.5}
  """
  kwargs: Dict[str, Any] = {}
  i = 0
  while i < len(unknown_args):
    if unknown_args[i].startswith("--"):
      key = unknown_args[i].lstrip("-")
      # If the next item is a value (not another flag).
      if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
        val: Any = unknown_args[i + 1]
        # Basic type casting.
        if val.isdigit():
          val = int(val)
        elif val.replace(".", "", 1).isdigit():
          val = float(val)
        elif val.lower() == "true":
          val = True
        elif val.lower() == "false":
          val = False
        kwargs[key] = val
        i += 2
      else:
        kwargs[key] = True
        i += 1
    else:
      i += 1
  return kwargs
