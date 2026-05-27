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

"""Command-line entry point for PEFT Audio LoRA conversion."""

from __future__ import annotations

import argparse
import json
import pathlib

from litert_lm_builder import audio_lora_converter


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Compile a PEFT audio LoRA directory into a LiteRT-LM Audio LoRA "
          "TFLite sidecar."
      )
  )
  parser.add_argument("--peft_lora_dir", required=True)
  parser.add_argument("--output_lora_tflite", required=True)
  parser.add_argument(
      "--target_tflite",
      help=(
          "Optional LoRA-ready audio encoder TFLite graph. When provided, the "
          "converter validates output tensor names, byte sizes, and orientation "
          "against the graph inputs."
      ),
  )
  parser.add_argument("--lora_rank", type=int)
  parser.add_argument("--lora_alpha", type=float)
  parser.add_argument(
      "--peft_key_regex",
      help=(
          "Optional regular expression used to select PEFT tensor keys before "
          "mapping. This is useful when one adapter contains both audio and "
          "language-model LoRA tensors."
      ),
  )
  parser.add_argument(
      "--payload_alignment",
      type=int,
      default=64 * 1024,
      help="Byte alignment for sidecar tensor payloads.",
  )
  parser.add_argument(
      "--allow_missing_target_inputs",
      action="store_true",
      help=(
          "Allow --target_tflite LoRA inputs that are absent from the PEFT "
          "adapter. Strict conversion fails on these by default."
      ),
  )
  parser.add_argument(
      "--allow_unmatched_peft_keys",
      action="store_true",
      help=(
          "Allow selected PEFT keys that do not map to the LiteRT-LM audio "
          "LoRA ABI. Strict conversion fails on these by default."
      ),
  )
  parser.add_argument(
      "--allow_unused_peft_keys",
      action="store_true",
      help=(
          "Allow selected PEFT keys that map to tensors absent from the target "
          "graph. Strict conversion fails on these by default."
      ),
  )
  parser.add_argument(
      "--allow_unconsumed_target_inputs",
      action="store_true",
      help=(
          "Allow --target_tflite LoRA inputs unused by any op. Strict "
          "conversion fails on these by default."
      ),
  )
  parser.add_argument("--report_json")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  report = audio_lora_converter.compile_peft_audio_lora(
      peft_lora_dir=args.peft_lora_dir,
      output_path=args.output_lora_tflite,
      target_tflite_path=args.target_tflite,
      lora_rank=args.lora_rank,
      lora_alpha=args.lora_alpha,
      payload_alignment=args.payload_alignment,
      peft_key_regex=args.peft_key_regex,
      fail_on_missing_target_inputs=not args.allow_missing_target_inputs,
      fail_on_unmatched_peft_keys=not args.allow_unmatched_peft_keys,
      fail_on_unused_peft_keys=not args.allow_unused_peft_keys,
      fail_on_unconsumed_target_inputs=not args.allow_unconsumed_target_inputs,
  )
  report_json = json.dumps(report.to_json_dict(), indent=2, sort_keys=True)
  if args.report_json:
    pathlib.Path(args.report_json).write_text(report_json + "\n", encoding="utf-8")
  print(report_json)


if __name__ == "__main__":
  main()
