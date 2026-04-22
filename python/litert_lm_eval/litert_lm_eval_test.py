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

"""Tests for litert_lm_eval system."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

# Mock out lm_eval and framework objects so imports succeed.
sys.modules[
    "litert_lm_eval.runners.lm_eval_runner.litert_lm_model"
] = mock.MagicMock()
mock_lm_eval = mock.MagicMock()
sys.modules["lm_eval"] = mock_lm_eval
sys.modules["lm_eval.api"] = mock_lm_eval.api
sys.modules["lm_eval.api.model"] = mock_lm_eval.api.model
sys.modules["lm_eval.api.model"].LM = object
sys.modules["lm_eval.api.registry"] = mock.MagicMock()
sys.modules["lm_eval.api.registry"].register_model = lambda x: lambda y: y

from litert_lm_eval import litert_lm_eval  # pylint: disable=g-import-not-at-top


class LitertLmEvalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_lm_eval.reset_mock()

  @mock.patch.object(
      sys,
      "argv",
      [
          "litert_lm_eval.py",
          "--model_path",
          "test_model.tflite",
          "--tasks",
          "mmlu",
          "--backend",
          "GPU",
      ],
  )
  def test_main_lm_eval_basic(self):
    mock_lm_eval.simple_evaluate.return_value = {"results": {"mmlu": 0.5}}

    litert_lm_eval.main()

    mock_lm_eval.simple_evaluate.assert_called_once_with(
        model="litert_lm",
        model_args="model_path=test_model.tflite,backend=GPU",
        tasks=["mmlu"],
        num_fewshot=None,
        limit=None,
    )

  @mock.patch.object(
      sys,
      "argv",
      [
          "litert_lm_eval.py",
          "--model_path",
          "test_model.tflite",
          "--tasks",
          "mmlu",
          "--framework_args",
          "max_length=1024",
      ],
  )
  def test_main_lm_eval_with_kwargs_and_framework_args(self):
    mock_lm_eval.simple_evaluate.return_value = {"results": {"mmlu": 0.5}}

    litert_lm_eval.main()

    mock_lm_eval.simple_evaluate.assert_called_once_with(
        model="litert_lm",
        model_args="model_path=test_model.tflite,backend=CPU,max_length=1024",
        tasks=["mmlu"],
        num_fewshot=None,
        limit=None,
    )


if __name__ == "__main__":
  absltest.main()
