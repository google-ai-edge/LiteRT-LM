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

r"""CLI tool for building LiteRT-LM files.

There are two ways to use this tool:

1. Building the file by specifying the components as CLI arguments:

```
bazel run //python/litert_lm_builder:litertlm_builder_cli -- \
  system_metadata --str Authors "ODML team" \
  llm_metadata --path llm.pb \
  tflite_model --path embedder.tflite --model_type embedder  --str_metadata model_version "1.0.1" \
  tflite_model --path model.tflite --model_type prefill_decode \
  sp_tokenizer --path sp.model \
  output --path output.litertlm
```
Notes:
- Constraints from litertlm_builder.py still apply.
- The order of the components in the CLI arguments determines the order of the
  sections in the output LiteRT-LM file.
- There can be multiple per section metadata.

2. Building the file by specifying the components as a TOML file:

TOML file example:
```
[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The ODML Authors" }
]

[[section]]
# Section 0: LlmMetadataProto Can be a text or binary proto file.
section_type = "LlmMetadata"
data_path = "PATH/TO/LLM_METADATA.pb"

[[section]]
# Section 1: SP_Tokenizer (you can also use HF_Tokenizer)
section_type = "SP_Tokenizer"
data_path = "PATH/TO/SP_TOKENIZER.model"

[[section]]
# Section 2: TFLiteModel (Embedder)
section_type = "TFLiteModel"
model_type = "EMBEDDER"
data_path = "PATH/TO/EMBEDDER.tflite"

[[section]]
# Section 3: TFLiteModel (Prefill/Decode)
section_type = "TFLiteModel"
model_type = "PREFILL_DECODE"
data_path = "PATH/TO/PREFILL_DECODE.tflite"
additional_metadata = [
  { key = "License", value_type = "String", value = "Example" }
  { key = "model_version", value_type = "String", value = "1.0.1" }
]
```

```
bazel run //python/litert_lm_builder:litertlm_builder_cli -- \
  toml --path example.toml output --path output.litertlm
```

"""

import argparse
import os
import sys
from typing import BinaryIO, cast

from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_core

_SUBCOMMANDS = (
    "toml",
    "system_metadata",
    "llm_metadata",
    "tflite_model",
    "tflite_weights",
    "sp_tokenizer",
    "hf_tokenizer",
    "output",
)


def _add_toml_parser(subparsers) -> None:
  """Adds a parser for TOML file to the subparsers."""
  toml_parser = subparsers.add_parser(
      "toml",
      description="Add a TOML file to the LiteRT-LM file.",
      help="Add a TOML file.",
  )
  toml_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the TOML file.",
  )


def _add_system_metadata_parser(subparsers) -> None:
  """Adds a parser for system metadata to the subparsers."""
  system_metadata_parser = subparsers.add_parser(
      "system_metadata",
      description=(
          "Add one or more system metadata key-value pairs to the LiteRT-LM"
          " file."
      ),
      help="Add system metadata.",
  )
  system_metadata_parser.add_argument(
      "--str",
      nargs=2,
      action="append",
      metavar=("KEY", "VALUE"),
      required=False,
      help=(
          "A string key-value pair for the system metadata. Can be specified"
          " multiple times."
      ),
  )
  system_metadata_parser.add_argument(
      "--int",
      nargs=2,
      action="append",
      metavar=("KEY", "VALUE"),
      required=False,
      help=(
          "An integer key-value pair for the system metadata. Can be specified"
          " multiple times."
      ),
  )


def _add_metadata_arguments(parser) -> None:
  """Adds arguments for metadata to the parser."""
  parser.add_argument(
      "--str_metadata",
      nargs=2,
      action="append",
      metavar=("KEY", "VALUE"),
      required=False,
      help=(
          "A string key-value pair for the metadata. Can be specified"
          " multiple times."
      ),
  )


def _add_llm_metadata_parser(subparsers) -> None:
  """Adds a parser for llm metadata to the subparsers."""
  llm_metadata_parser = subparsers.add_parser(
      "llm_metadata",
      description=(
          "Add llm metadata to the LiteRT-LM file. Can be a text or binary"
          " proto file."
      ),
      help="Add llm metadata.",
  )
  llm_metadata_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the llm metadata file.",
  )


def _add_tflite_model_parser(subparsers) -> None:
  """Adds a parser for tflite model to the subparsers."""
  tflite_model_parser = subparsers.add_parser(
      "tflite_model",
      description="Add a tflite model to the LiteRT-LM file.",
      help="Add a tflite model.",
  )
  tflite_model_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the tflite model file.",
  )
  tflite_model_parser.add_argument(
      "--model_type",
      type=str,
      required=True,
      choices=[
          str(model_type.value).lower().replace("tf_lite_", "")
          for model_type in litertlm_builder.TfLiteModelType
      ],
      help="The type of the tflite model.",
  )
  tflite_model_parser.add_argument(
      "--backend_constraint",
      type=str.lower,
      required=False,
      default=None,
      choices=list(litertlm_builder.Backend),
      help="A list of backend constraints for the tflite model.",
  )
  _add_metadata_arguments(tflite_model_parser)


def _add_tflite_weights_parser(subparsers) -> None:
  """Adds a parser for tflite weights to the subparsers."""
  tflite_weights_parser = subparsers.add_parser(
      "tflite_weights",
      description="Add tflite weights to the LiteRT-LM file.",
      help="Add tflite weights.",
  )
  tflite_weights_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the tflite weights file.",
  )
  tflite_weights_parser.add_argument(
      "--model_type",
      type=str,
      required=True,
      choices=[
          str(model_type.value).lower().replace("tf_lite_", "")
          for model_type in litertlm_builder.TfLiteModelType
      ],
      help="The type of the tflite model these weights correspond to.",
  )
  _add_metadata_arguments(tflite_weights_parser)


def _add_sentencepiece_tokenizer_parser(subparsers) -> None:
  """Adds a parser for sentencepiece tokenizer to the subparsers."""
  sp_tokenizer_parser = subparsers.add_parser(
      "sp_tokenizer",
      description="Add a sentencepiece tokenizer to the LiteRT-LM file.",
      help="Add a sentencepiece tokenizer.",
  )
  sp_tokenizer_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the sentencepiece tokenizer file.",
  )
  _add_metadata_arguments(sp_tokenizer_parser)


def _add_hf_tokenizer_parser(subparsers) -> None:
  """Adds a parser for huggingface tokenizer to the subparsers."""
  hf_tokenizer_parser = subparsers.add_parser(
      "hf_tokenizer",
      description="Add a huggingface tokenizer to the LiteRT-LM file.",
      help="Add a huggingface tokenizer.",
  )
  hf_tokenizer_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the huggingface tokenizer `tokenizer.json` file.",
  )
  _add_metadata_arguments(hf_tokenizer_parser)


def _add_output_path_parser(subparsers) -> None:
  """Adds an argument for the output path to the subparsers."""
  output_path_parser = subparsers.add_parser(
      "output",
      description="The path to the output LiteRT-LM file.",
      help="The path to the output LiteRT-LM file.",
  )
  output_path_parser.add_argument(
      "--path",
      type=str,
      required=True,
      help="The path to the output LiteRT-LM file.",
  )


def _build_parser() -> argparse.ArgumentParser:
  """Builds an argument parser for the litertlm_builder tool."""
  parser = argparse.ArgumentParser(
      description="Build a LiteRT-LM file from input files and metadata."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)
  _add_toml_parser(subparsers)
  _add_system_metadata_parser(subparsers)
  _add_llm_metadata_parser(subparsers)
  _add_tflite_model_parser(subparsers)
  _add_tflite_weights_parser(subparsers)
  _add_sentencepiece_tokenizer_parser(subparsers)
  _add_hf_tokenizer_parser(subparsers)
  _add_output_path_parser(subparsers)

  return parser


def _parse_args(parser: argparse.ArgumentParser) -> list[argparse.Namespace]:
  """Parses the command-line arguments.

  Args:
    parser: The argument parser to use.

  Returns:
    A list of parsed argument namespaces.

  Raises:
    ValueError: If there are unparsed arguments.
  """
  args = sys.argv[1:]
  if len(args) == 1 and args[0] in ["--help", "-h"]:
    print(parser.format_help())
    return []

  # We need to break the arguments into subcommands to ensure overlapping flags
  # are handled correctly. For example, "--path" is a flag for both
  # "llm_metadata" and "output".
  subcommands = []
  current_subcommand = []
  for arg in args:
    if arg in _SUBCOMMANDS:
      if current_subcommand:
        subcommands.append(current_subcommand)
      current_subcommand = [arg]
    else:
      assert current_subcommand, (
          f"No subcommand found for argument: {arg}. Use --help for a list of"
          " subcommands."
      )
      current_subcommand.append(arg)
  if current_subcommand:
    subcommands.append(current_subcommand)

  parsed_args = []
  for subcommand in subcommands:
    parsed, unparsed = parser.parse_known_args(args=subcommand)
    if unparsed:
      raise ValueError(
          f"Failed to parse all arguments. Unparsed args: {unparsed}"
      )
    parsed_args.append(parsed)
  return parsed_args


def _build_system_metadata(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds system metadata from the parsed arguments."""
  if args.str:
    for str_metadata in args.str:
      key, value = str_metadata
      builder.add_system_metadata(
          litertlm_builder.Metadata(
              key=key,
              value=value,
              dtype=litertlm_builder.DType.STRING,
          )
      )
  if args.int:
    for int_metadata in args.int:
      key, value = int_metadata
      builder.add_system_metadata(
          litertlm_builder.Metadata(
              key=key,
              value=int(value),
              dtype=litertlm_builder.DType.INT32,
          )
      )


def _get_metadata_from_args(
    args: argparse.Namespace,
) -> list[litertlm_builder.Metadata] | None:
  """Builds metadata from the parsed arguments."""
  metadata = []
  if hasattr(args, "str_metadata") and args.str_metadata:
    for str_metadata in args.str_metadata:
      key, value = str_metadata
      metadata.append(
          litertlm_builder.Metadata(
              key=key,
              value=value,
              dtype=litertlm_builder.DType.STRING,
          )
      )
  return metadata if metadata else None


def _build_llm_metadata(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds llm metadata from the parsed arguments."""
  metadata = _get_metadata_from_args(args)
  builder.add_llm_metadata(args.path, additional_metadata=metadata)


def _build_tflite_model(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds tflite model from the parsed arguments."""
  metadata = _get_metadata_from_args(args)
  builder.add_tflite_model(
      args.path,
      litertlm_builder.TfLiteModelType.get_enum_from_tf_free_value(
          args.model_type
      ),
      backend_constraint=args.backend_constraint,
      additional_metadata=metadata,
  )


def _build_tflite_weights(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds tflite weights from the parsed arguments."""
  metadata = _get_metadata_from_args(args)
  builder.add_tflite_weights(
      args.path,
      litertlm_builder.TfLiteModelType.get_enum_from_tf_free_value(
          args.model_type
      ),
      additional_metadata=metadata,
  )


def _build_sp_tokenizer(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds sentencepiece tokenizer from the parsed arguments."""
  metadata = _get_metadata_from_args(args)
  builder.add_sentencepiece_tokenizer(args.path, additional_metadata=metadata)


def _build_hf_tokenizer(
    args: argparse.Namespace,
    builder: litertlm_builder.LitertLmFileBuilder,
) -> None:
  """Builds huggingface tokenizer from the parsed arguments."""
  metadata = _get_metadata_from_args(args)
  builder.add_hf_tokenizer(args.path, additional_metadata=metadata)


def _build_litertlm_file(parsed_args: list[argparse.Namespace]) -> None:
  """Builds a LiteRT-LM file from the parsed arguments."""
  if "toml" in [pa.command for pa in parsed_args]:
    toml_path = None
    output_path = None
    for parsed_arg in parsed_args:
      match parsed_arg.command:
        case "output":
          output_path = parsed_arg.path
        case "toml":
          toml_path = parsed_arg.path
        case _:
          raise ValueError(
              "When using TOML, only output and toml are supported."
          )
    assert output_path, "Output path is required."
    assert toml_path, "TOML path is required."
    output_dir = os.path.dirname(output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with litertlm_core.open_file(output_path, "wb") as f:
      builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(toml_path)
      builder.build(f)
  else:
    builder = litertlm_builder.LitertLmFileBuilder()
    output_path = None
    for parsed_arg in parsed_args:
      match parsed_arg.command:
        case "system_metadata":
          _build_system_metadata(parsed_arg, builder)
        case "llm_metadata":
          _build_llm_metadata(parsed_arg, builder)
        case "tflite_model":
          _build_tflite_model(parsed_arg, builder)
        case "tflite_weights":
          _build_tflite_weights(parsed_arg, builder)
        case "sp_tokenizer":
          _build_sp_tokenizer(parsed_arg, builder)
        case "hf_tokenizer":
          _build_hf_tokenizer(parsed_arg, builder)
        case "output":
          output_path = parsed_arg.path
        case _:
          raise ValueError(f"Unknown subcommand: {parsed_arg.command}")

    assert output_path, "Output path is required."
    output_dir = os.path.dirname(output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with litertlm_core.open_file(output_path, "wb") as f:
      builder.build(cast(BinaryIO, f))

  print(f"LiteRT-LM file successfully created at {output_path}")


def main(_) -> None:
  parser = _build_parser()
  parsed_args = _parse_args(parser)
  if not parsed_args:
    return
  _build_litertlm_file(parsed_args)


def run():
  """Entry point for console_scripts."""
  litertlm_core.run_app(main)


if __name__ == "__main__":
  run()
