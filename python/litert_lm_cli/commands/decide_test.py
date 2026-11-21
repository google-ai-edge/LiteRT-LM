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
"""Unit tests for the decide subcommand."""

import json
import os
from unittest import mock

from absl.testing import absltest
from click import testing

import litert_lm
from litert_lm import decision_engine
from litert_lm_cli import model
import litert_lm_cli.commands.decide as decide_cmd

_RESPONSE = {
    "answers": {
        "refund": {"type": "noul", "confidence": 0.75, "noul": 0.75},
    }
}


class DecideTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_model = mock.MagicMock(spec=model.Model)
    self.mock_model.exists.return_value = True
    self.mock_model.model_path = "/path/to/model.litertlm"
    self.enter_context(
        mock.patch.object(
            model.Model,
            "from_model_reference",
            return_value=self.mock_model,
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            model,
            "parse_backend",
            return_value=litert_lm.Backend.CPU(),
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            model,
            "resolve_config_option",
            side_effect=lambda value, *args, **kwargs: value,
            autospec=True,
        )
    )
    self.mock_engine = mock.MagicMock(spec=decision_engine.DecisionEngine)
    self.mock_engine_cls = self.enter_context(
        mock.patch.object(decision_engine, "DecisionEngine")
    )
    self.mock_engine_cls.return_value.__enter__.return_value = self.mock_engine
    self.mock_engine.decide_json.return_value = _RESPONSE
    self.runner = testing.CliRunner()

  def invoke(self, args, **kwargs):
    return self.runner.invoke(
        decide_cmd.decide_command, ["my-model"] + args, **kwargs
    )

  def decided_request(self):
    self.mock_engine.decide_json.assert_called_once()
    (request,), _ = self.mock_engine.decide_json.call_args
    return request

  def test_instructions_flags_build_noul_question(self):
    result = self.invoke([
        "-i",
        "I was charged twice.",
        "--id",
        "refund",
        "-p",
        "Refund requested?",
    ])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(json.loads(result.output), _RESPONSE)
    self.assertEqual(
        self.decided_request(),
        {
            "input": "I was charged twice.",
            "questions": {
                "refund": {"type": "noul", "instructions": "Refund requested?"}
            },
        },
    )
    _, kwargs = self.mock_engine_cls.call_args
    self.assertIsNone(kwargs["max_num_tokens"])
    self.assertIsNone(kwargs["cache_dir"])

  def test_empty_option_id_is_rejected_before_loading(self):
    result = self.invoke(["-i", "hi", "-p", "Topic?", "-o", ":no id"])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("option ID must not be empty", result.output)
    self.mock_engine_cls.assert_not_called()

  def test_options_default_to_choice_and_keep_order(self):
    result = self.invoke([
        "--input",
        "hi",
        "--instructions",
        "Topic?",
        "--option",
        "shipping",
        "-o",
        "billing:Invoices and charges",
        "--max-num-tokens",
        "512",
    ])

    self.assertEqual(result.exit_code, 0, result.output)
    question = self.decided_request()["questions"]["q0"]
    self.assertEqual(
        question,
        {
            "type": "choice",
            "instructions": "Topic?",
            "criteria": {"shipping": "", "billing": "Invoices and charges"},
        },
    )
    self.assertEqual(list(question["criteria"]), ["shipping", "billing"])
    _, kwargs = self.mock_engine_cls.call_args
    self.assertEqual(kwargs["max_num_tokens"], 512)

  def test_score_options_are_levels(self):
    result = self.invoke([
        "-i",
        "hi",
        "-t",
        "Score",
        "-p",
        "How urgent?",
        "-o",
        "Low: can wait",
        "-o",
        "High",
    ])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(
        self.decided_request()["questions"]["q0"],
        {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Low: can wait", "High"],
        },
    )

  def test_json_questions_and_flag_question(self):
    result = self.invoke([
        "-i",
        "hi",
        "-q",
        (
            '{"urgency": {"type": "score", "instructions": "How urgent?",'
            ' "criteria": ["low", "high"]}}'
        ),
        "-p",
        "Upset?",
    ])

    self.assertEqual(result.exit_code, 0, result.output)
    questions = self.decided_request()["questions"]
    self.assertEqual(list(questions), ["urgency", "q0"])
    self.assertEqual(
        questions["urgency"],
        {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "high"],
        },
    )

  def test_request_json_is_passed_through(self):
    request = {
        "model": "gemini",
        "input": ["a", {"type": "text", "text": "b"}],
        "questions": {"refund": {"type": "noul", "instructions": "p?"}},
        "temperature": 2.0,
    }
    result = self.invoke(["-r", json.dumps(request)])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self.decided_request(), request)
    self.assertEqual(json.loads(result.output), _RESPONSE)

  def test_request_file_is_passed_through(self):
    request = {
        "input": "charged twice",
        "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}},
    }
    path = os.path.join(self.create_tempdir().full_path, "request.json")
    with open(path, "w", encoding="utf-8") as f:
      json.dump(request, f)

    result = self.invoke(["-f", path])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self.decided_request(), request)

  def test_request_from_stdin(self):
    request = {
        "input": "charged twice",
        "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}},
    }

    result = self.invoke([], input=json.dumps(request))

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self.decided_request(), request)

  def test_input_from_stdin(self):
    result = self.invoke(["-p", "Refund?"], input="I was charged twice.\n")

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self.decided_request()["input"], "I was charged twice.")

  def test_non_ascii_output(self):
    self.mock_engine.decide_json.return_value = {
        "answers": {
            "q0": {
                "type": "choice",
                "confidence": 0.5,
                "choice": "账单",
                "probabilities": {"账单": 0.5, "物流": 0.5},
            }
        }
    }

    result = self.invoke(
        ["-i", "扣款", "-p", "部门？", "-o", "账单", "-o", "物流"]
    )

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('"choice": "账单"', result.output)

  def test_engine_error_is_reported(self):
    self.mock_engine.decide_json.side_effect = RuntimeError("bad request")

    result = self.invoke(["-i", "hi", "-p", "Refund?"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("bad request", result.output)

  def test_usage_errors(self):
    for args, message in (
        (["-i", "hi"], "Provide --instructions/-p"),
        (["-p", "Refund?"], "Provide --input/-i"),
        (["-i", "hi", "-o", "a"], "require --instructions/-p"),
        (["-i", "hi", "--id", "q"], "require --instructions/-p"),
        (["-i", "hi", "-p", "p?", "-t", "noul", "-o", "a"], "not used by noul"),
        (["-i", "hi", "-p", "p?", "-t", "boolean"], "Invalid value"),
        (["-i", "hi", "-q", "[]"], "must be a JSON object"),
        (
            ["-i", "hi", "-q", '{"q0": {}}', "-p", "p?"],
            "Duplicate question ID",
        ),
        (["-r", "{"], "Invalid JSON"),
        (["-r", "[]"], "must be a JSON object"),
        (["-r", "{}", "-i", "hi"], "cannot be combined"),
        (["-r", "{}", "--id", "q"], "cannot be combined"),
        (["-r", "{}", "-t", "choice"], "cannot be combined"),
        (["-r", "{}", "-o", "a"], "cannot be combined"),
        (["-r", "{}", "-p", "p?"], "cannot be combined"),
        (["-r", "{}", "-q", "{}"], "cannot be combined"),
    ):
      with self.subTest(args=args):
        result = self.invoke(args, input="")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(message, result.output)
    self.mock_engine_cls.assert_not_called()

  def test_missing_model(self):
    self.mock_model.exists.return_value = False

    result = self.invoke(["-i", "hi", "-p", "Refund?"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Failed to find model", result.output)


if __name__ == "__main__":
  absltest.main()
