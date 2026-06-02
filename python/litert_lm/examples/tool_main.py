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

"""Example of using tools with LiteRT-LM."""

from collections.abc import Sequence

from absl import app
from absl import flags

import litert_lm

_MODEL_PATH = flags.DEFINE_string(
    "model_path", None, "Path to the model file.", required=True
)


def product(numbers: Sequence[float]) -> float:
  """Get the product of a list of numbers.

  Args:
      numbers: The numbers, could be floating point.
  """
  print(f"Calling tool product with arg: {numbers}")
  res = 1.0
  for n in numbers:
    res *= n
  return res


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

  engine = litert_lm.Engine(
      _MODEL_PATH.value,
      litert_lm.Backend.CPU(),
  )

  tools = [product]

  with (
      engine as engine,
      engine.create_conversation(tools=tools) as conversation,
  ):
    print("LiteRT-LM Tool Example")
    user_input = "What is the product of 1.1, 2.2, 3.3 and 4.4?"

    # Send message (async streaming)
    # We use yellow for model output as in the Kotlin example
    for chunk in conversation.send_message_async(user_input):
      content_list = chunk.get("content", [])
      for item in content_list:
        if item.get("type") == "text":
          print("\033[33m", end="")
          print(item.get("text", ""), end="", flush=True)
          print("\033[0m", end="")
    print("")


if __name__ == "__main__":
  app.run(main)
