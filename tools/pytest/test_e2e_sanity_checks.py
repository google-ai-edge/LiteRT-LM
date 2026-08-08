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

"""End-to-end sanity checks for the LiteRT-LM engine.

This module contains tests to validate that the LiteRT-LM engine generates
expected text for a set of predefined prompts, loaded from a JSON configuration.
"""

import re
from typing import Callable

import pytest
from utils import read_json_config

TEST_CASES = read_json_config(__file__, "test_model_sanity")


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[case.get("id", f"case_{i}") for i, case in enumerate(TEST_CASES)]
)
def test_model_sanity(
    run_engine: Callable[..., str], test_case: dict[str, str]
) -> None:
  """Validates LiteRT-LM engine generates expected text.

  This test checks that the engine generates expected text for given prompts.
  Powered by test_data/test_e2e_sanity_checks.json

  Args:
    run_engine: A callable that takes a prompt string and returns the engine's
      output.
    test_case: A dictionary containing the test case, including "prompt" and
      "response".
  """

  prompt = test_case["prompt"]
  expected_response = test_case["response"]

  # Execute
  output = run_engine(prompt)
  clean_output = output.replace("\n", " ")

  # Validate
  assert re.search(expected_response, clean_output), (
      "FAILED: Model output did not match expected regex.\n"
      f"PROMPT: {prompt}\n"
      f"EXPECTED: {expected_response}\n"
      f"OUTPUT TAIL:\n...{clean_output[-500:]}"
  )
