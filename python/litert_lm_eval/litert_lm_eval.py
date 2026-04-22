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

r"""LiteRT LM Evaluation Runner.

This script acts as a unified entry point for evaluating LiteRT models using
various evaluation frameworks (like `lm_eval`). It provides a consistent set of
command-line flags across different underlying frameworks, simplifying the
pipeline setup.

It parses the unified flags, maps them to the specific arguments required by
the chosen framework, and then delegates execution to that framework's CLI.

Usage example:
  bazel run //python/litert_lm_eval:litert_lm_eval \
      --model_path /path/to/model.litertlm \
      --tasks mmlu,gsm8k \
      --backend CPU \
      --output_path /path/to/save/results.json

  # Using the escape hatch for framework-specific arguments
  bazel run //python/litert_lm_eval:litert_lm_eval \
      --model_path /path/to/model.litertlm \
      --tasks mmlu \
      --framework_args "limit=10"
"""

import argparse
import json
import sys

import lm_eval

from litert_lm_eval import utils
from litert_lm_eval.runners.lm_eval_runner import litert_lm_model  # pylint: disable=unused-import


def main():
  parser = argparse.ArgumentParser(description="LiteRT LM Eval Runner")

  parser.add_argument(
      "--model_path", type=str, required=True, help="Path to the model file."
  )
  parser.add_argument(
      "--tasks",
      type=str,
      required=True,
      help="Comma-separated list of tasks to run (e.g., 'mmlu,gsm8k').",
  )
  parser.add_argument(
      "--backend",
      type=str,
      default="CPU",
      choices=["CPU", "GPU"],
      help="Backend to use (e.g., 'CPU', 'GPU').",
  )

  parser.add_argument(
      "--num_fewshot",
      type=int,
      default=None,
      help="Number of examples in few-shot context.",
  )
  parser.add_argument(
      "--limit",
      type=float,
      default=None,
      help="Limit examples per task (integer count or fraction).",
  )

  # Escape hatch
  parser.add_argument(
      "--framework_args",
      type=str,
      default="",
      help=(
          "Additional arguments to pass strictly to the model constructor "
          "(comma-separated key=value pairs or flags)."
      ),
  )

  parser.add_argument(
      "--output_path",
      type=str,
      default=None,
      help="Path to save the evaluation results as a JSON file.",
  )

  args, unknown = parser.parse_known_args()

  # Construct the model_args string required by lm_eval.
  model_args_str = f"model_path={args.model_path},backend={args.backend}"

  if args.framework_args:
    model_args_str += f",{args.framework_args}"

  tasks = args.tasks.split(",") if args.tasks else []

  # Parse unknown args into kwargs for simple_evaluate.
  kwargs = utils.parse_unknown_args(unknown)

  print(f"Running evaluation with model 'litert_lm' on tasks: {tasks}")

  results = lm_eval.simple_evaluate(
      model="litert_lm",
      model_args=model_args_str,
      tasks=tasks,
      num_fewshot=args.num_fewshot,
      limit=args.limit,
      **kwargs,  # Pass any remaining flags.
  )

  if results is not None:
    print(json.dumps(results["results"], indent=2, default=str))
    if args.output_path:
      with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
      print(f"\nResults successfully saved to {args.output_path}")
    print("\nEvaluation successful.")


if __name__ == "__main__":
  main()
