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
r"""Pipeline to externalize weights of a .litertlm model.

This script:
1. Concatenates model parts if they are split.
2. Unpacks the .litertlm file using litertlm_peek.
3. Externalizes the TFLite model weights using externalize_tflite_flatbuffer.
4. Updates the generated model.toml.
5. Repacks the model using litertlm_builder.
"""

import argparse
import os
import pathlib
import shutil
import sys
from typing import Any

import tomli as tomllib

from litert.litert.tools import externalize_tflite_flatbuffer
from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_peek


def _format_toml_value(value: Any) -> str:
  if isinstance(value, str):
    return f'"{value}"'
  elif isinstance(value, bool):
    return "true" if value else "false"
  elif isinstance(value, int):
    return str(value)
  elif isinstance(value, float):
    return str(value)
  else:
    raise ValueError(f"Unsupported TOML value type: {type(value)}")


def _write_model_toml(
    dump_files_dir: str,
    system_metadata: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
  """Writes a model.toml file to the dump directory."""
  lines = []
  if system_metadata:
    lines.append("[system_metadata]")
    lines.append("entries = [")
    for entry in system_metadata:
      lines.append(
          f'  {{ key = "{entry["key"]}",'
          f' value_type = "{entry["value_type"]}",'
          f' value = {_format_toml_value(entry["value"])} }},'
      )
    lines.append("]")
    lines.append("")

  for section in sections:
    lines.append("[[section]]")
    for key, value in section.items():
      if key == "additional_metadata":
        if value:
          lines.append("additional_metadata = [")
          for m in value:
            lines.append(
                f'  {{ key = "{m["key"]}",'
                f' value_type = "{m["value_type"]}",'
                f' value = {_format_toml_value(m["value"])} }},'
            )
          lines.append("]")
      else:
        lines.append(f"{key} = {_format_toml_value(value)}")
    lines.append("")

  toml_path = os.path.join(dump_files_dir, "model.toml")
  with litertlm_core.open_file(toml_path, "w") as f:
    f.write("\n".join(lines))


def _sort_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Sorts sections so that TFLiteModels come before TFLiteWeights."""
  models = []
  weights = []
  others = []
  for section in sections:
    stype = section.get("section_type")
    if stype == "TFLiteModel":
      models.append(section)
    elif stype == "TFLiteWeights":
      weights.append(section)
    else:
      others.append(section)
  return others + models + weights


def main(_) -> None:
  parser = argparse.ArgumentParser(
      description="Externalize weights of a .litertlm model."
  )
  parser.add_argument(
      "--model_parts_dir",
      type=pathlib.Path,
      required=True,
      help=(
          "Directory containing the model parts (e.g."
          " gemma3-1b-hw.litertlm.part_*)"
      ),
  )
  parser.add_argument(
      "--model_prefix",
      type=str,
      default="gemma3-1b-hw.litertlm",
      help="Prefix of the model parts",
  )
  parser.add_argument(
      "--output_model",
      type=pathlib.Path,
      required=True,
      help="Path to write the final externalized .litertlm model",
  )
  parser.add_argument(
      "--work_dir",
      type=pathlib.Path,
      default=pathlib.Path(os.path.expanduser("~/tmp/litert_lm_externalize")),
      help="Temporary working directory",
  )
  parser.add_argument(
      "--repack_only",
      action="store_true",
      help=(
          "Only repack the model to apply layout sorting, without externalizing"
          " weights."
      ),
  )
  args = parser.parse_args()

  # 0. Setup work dir
  if args.work_dir.exists():
    shutil.rmtree(args.work_dir)
  args.work_dir.mkdir(parents=True, exist_ok=True)

  # 1. Concatenate model parts
  parts = sorted(list(args.model_parts_dir.glob(f"{args.model_prefix}.part*")))
  if not parts:
    # Try if the file is not split
    single_file = args.model_parts_dir / args.model_prefix
    if single_file.exists():
      print(f"Found single model file: {single_file}")
      concatenated_model = single_file
    else:
      raise FileNotFoundError(
          f"No model parts or single file found with prefix {args.model_prefix}"
          f" in {args.model_parts_dir}"
      )
  else:
    print(f"Concatenating {len(parts)} parts...")
    concatenated_model = args.work_dir / args.model_prefix
    with concatenated_model.open("wb") as outfile:
      for part in parts:
        print(f"  Appending {part}...")
        with part.open("rb") as infile:
          shutil.copyfileobj(infile, outfile)
    print(
        "Concatenated model size:"
        f" {concatenated_model.stat().st_size / 1024 / 1024:.2f}MB"
    )

  # 2. Unpack the .litertlm file
  print("Unpacking .litertlm file...")
  unpack_dir = args.work_dir / "unpacked"
  litertlm_peek.peek_litertlm_file(
      str(concatenated_model), str(unpack_dir), sys.stdout
  )

  # 3. Read the generated model.toml
  toml_path = unpack_dir / "model.toml"

  if not args.repack_only:
    with toml_path.open("rb") as f:
      toml_data = tomllib.load(f)

    system_metadata = toml_data.get("system_metadata", {}).get("entries", [])
    sections = toml_data.get("section", [])

    new_sections = []
    for section in sections:
      if section.get("section_type") == "TFLiteModel":
        model_type = section.get("model_type")
        original_data_path = section.get("data_path")
        print(
            f"Found TFLiteModel section of type {model_type}:"
            f" {original_data_path}"
        )

        # Externalize this model
        input_model_path = unpack_dir / original_data_path
        externalized_dir = unpack_dir / f"externalized_{model_type}"

        print(f"Externalizing {input_model_path} to {externalized_dir}...")
        group_name = "tflite_weights"
        externalize_tflite_flatbuffer.externalize(
            input_model=input_model_path,
            output_dir=externalized_dir,
            group_name=group_name,
            num_elements_threshold=256,
        )

        # Update the TFLiteModel section to point to the externalized model
        section["data_path"] = str(
            (externalized_dir / "model.tflite").relative_to(unpack_dir)
        )
        new_sections.append(section)

        # Create a new TFLiteWeights section
        weights_section = {
            "section_type": "TFLiteWeights",
            "model_type": model_type,
            "data_path": str(
                (externalized_dir / group_name).relative_to(unpack_dir)
            ),
        }
        if "additional_metadata" in section:
          weights_section["additional_metadata"] = [
              m
              for m in section["additional_metadata"]
              if m["key"] != "model_type"
          ]
        new_sections.append(weights_section)
      else:
        new_sections.append(section)

    # 4. Write back the updated TOML
    print("Writing updated model.toml...")
    sorted_sections = _sort_sections(new_sections)
    _write_model_toml(str(unpack_dir), system_metadata, sorted_sections)
  else:
    print("Repack only mode. Skipping externalization.")

  # 5. Repack the model
  print("Repacking model...")
  builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(str(toml_path))

  args.output_model.parent.mkdir(parents=True, exist_ok=True)
  with args.output_model.open("wb") as f:
    builder.build(f)

  print(f"Successfully built externalized model at: {args.output_model}")
  print(
      "Final model size:"
      f" {args.output_model.stat().st_size / 1024 / 1024:.2f}MB"
  )


if __name__ == "__main__":
  litertlm_core.run_app(main)
