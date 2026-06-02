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

"""Example of using multimodal LiteRT-LM with audio."""

import os
from typing import Sequence

from absl import app
from absl import flags

import litert_lm

_MODEL_PATH = flags.DEFINE_string(
    "model_path", None, "Path to the model file.", required=True
)

_AUDIO_PATH = flags.DEFINE_string(
    "audio_path", None, "Path to the audio file.", required=True
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  audio_abs_path = os.path.abspath(_AUDIO_PATH.value)

  with (
      litert_lm.Engine(
          _MODEL_PATH.value, audio_backend=litert_lm.Backend.CPU()
      ) as engine,
      engine.create_conversation() as conversation,
  ):
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": audio_abs_path,
            },
            {"type": "text", "text": "Describe the audio."},
        ],
    }

    message = conversation.send_message(user_message)
    print(f"Response: {message['content'][0]['text']}")


if __name__ == "__main__":
  app.run(main)
