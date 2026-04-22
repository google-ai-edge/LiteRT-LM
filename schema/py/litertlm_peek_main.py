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

r"""This tool is designed to inspect the contents of a .litertlm file.

It reads the file's header, system metadata, and section information, and prints
them to the console.

Example usage:
  bazel run  schema/py:litertlm_peek_main -- \
  --litertlm_file=<path/to/your/file.litertlm>
"""

import argparse
import struct
import sys

from litert_lm.schema.py import litertlm_core
from litert_lm.schema.py import litertlm_peek


def main(_):
  """Parses command-line arguments and runs the litertlm_peek tool."""
  parser = argparse.ArgumentParser(
      description="Inspect the contents of a LiteRT-LM file."
  )
  parser.add_argument(
      "--litertlm_file",
      type=str,
      required=True,
      help="The path to the LiteRT-LM file to inspect.",
  )
  parser.add_argument(
      "--dump_files_dir",
      type=str,
      default=None,
      help=(
          "The directory to dump the files in the LiteRT-LM file. If not"
          " provided, the files will not be dumped."
      ),
  )
  args = parser.parse_args()

  try:
    litertlm_peek.peek_litertlm_file(
        args.litertlm_file, args.dump_files_dir, sys.stdout
    )
  except (ValueError, FileNotFoundError, struct.error) as e:
    print(f"Error processing file: {e}", file=sys.stderr)
    sys.exit(1)


def run():
  """Entry point for console_scripts."""
  litertlm_core.run_app(main)


if __name__ == "__main__":
  run()
