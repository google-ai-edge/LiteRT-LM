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
"""Unit tests for LiteRT-LM Python DecisionEngine."""

import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest

import litert_lm
from litert_lm import _decision_json
from litert_lm import decision_engine


def _test_model_path() -> str:
  return os.path.join(
      absltest.get_default_test_srcdir(),
      "litert_lm/runtime/testdata/test_lm.litertlm",
  )


_INPUT = ["Customer support triage.", "I was charged twice for my order."]
_NOUL = {"type": "noul", "instructions": "p?"}


class DecisionEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = decision_engine.DecisionEngine(
        _test_model_path(),
        backend=litert_lm.Backend.CPU(),
        max_num_tokens=1024,
        cache_dir=":nocache",
    )

  def tearDown(self):
    self.engine.close()
    super().tearDown()

  def test_decide_noul_choice_and_score(self):
    questions = {
        "topic": decision_engine.ChoiceQuestion(
            "What is the message about?",
            criteria={
                "shipping": "Delivery and tracking",
                "billing": "Invoices and charges",
                "other": "",
            },
        ),
        "refund": decision_engine.NoulQuestion(
            "Does the customer ask for a refund?",
            true_description="A refund is requested",
        ),
        "urgency": decision_engine.ScoreQuestion(
            "How urgent is this message?",
            criteria=["Not urgent", "Somewhat urgent", "Very urgent"],
        ),
    }
    response = self.engine.decide(_INPUT, questions)
    # Answers follow the order of the questions.
    self.assertEqual(list(response.answers), ["topic", "refund", "urgency"])

    topic = response.answers["topic"]
    assert isinstance(topic, decision_engine.ChoiceAnswer)
    # Probabilities follow the order of the criteria.
    self.assertEqual(
        list(topic.probabilities), ["shipping", "billing", "other"]
    )
    self.assertAlmostEqual(sum(topic.probabilities.values()), 1.0, places=5)
    self.assertAlmostEqual(
        topic.confidence, topic.probabilities[topic.choice], places=6
    )

    refund = response.answers["refund"]
    assert isinstance(refund, decision_engine.NoulAnswer)
    self.assertBetween(refund.noul, 0.0, 1.0)
    self.assertEqual(refund.value, refund.noul >= 0.5)
    self.assertAlmostEqual(
        refund.confidence, max(refund.noul, 1 - refund.noul), places=6
    )

    urgency = response.answers["urgency"]
    assert isinstance(urgency, decision_engine.ScoreAnswer)
    self.assertLen(urgency.probabilities, 3)
    self.assertEqual(
        urgency.legend, ["Not urgent", "Somewhat urgent", "Very urgent"]
    )
    self.assertAlmostEqual(
        urgency.score,
        sum(k * p for k, p in enumerate(urgency.probabilities)),
        places=5,
    )
    self.assertAlmostEqual(
        urgency.confidence,
        urgency.probabilities[urgency.most_likely_level],
        places=6,
    )

    # No state is kept between calls, so repeating the call produces the same
    # answers.
    second = self.engine.decide(_INPUT, questions)
    self.assertEqual(second, response)

  def test_decide_json_matches_decide(self):
    question = decision_engine.NoulQuestion("Is a refund requested?")
    typed = self.engine.decide(" ".join(_INPUT), {"refund": question})
    raw = self.engine.decide_json({
        "model": "ignored",
        "input": {
            "type": "text",
            "text": " ".join(_INPUT),
            "annotations": [{"type": "url_citation"}],
        },
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": [
                    {"type": "text", "text": "Is a refund requested?"}
                ],
            },
        },
    })
    self.assertEqual(raw, typed.to_dict())
    self.assertEqual(
        list(raw["answers"]["refund"]), ["type", "confidence", "noul"]
    )

  def test_decide_json_response_format(self):
    request = {
        "input": _INPUT,
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": ["How urgent", "is this?"],
                "criteria": ["Low", "High"],
            },
            "topic": {
                "type": "choice",
                "instructions": "Topic?",
                "criteria": {"shipping": "", "billing": "Charges"},
            },
        },
        "temperature": 2.0,
    }
    response = self.engine.decide_json(json.dumps(request))
    self.assertEqual(list(response), ["answers"])
    self.assertEqual(list(response["answers"]), ["urgency", "topic"])
    urgency = response["answers"]["urgency"]
    self.assertEqual(
        list(urgency),
        ["type", "confidence", "score", "probabilities", "legend"],
    )
    self.assertEqual(list(urgency["probabilities"]), ["0", "1"])
    self.assertEqual(urgency["legend"], {"0": "Low", "1": "High"})
    topic = response["answers"]["topic"]
    self.assertEqual(
        list(topic), ["type", "confidence", "choice", "probabilities"]
    )
    self.assertEqual(list(topic["probabilities"]), ["shipping", "billing"])

  def test_temperature(self):
    response = self.engine.decide(
        _INPUT,
        {
            "topic": decision_engine.ChoiceQuestion(
                "Topic?", criteria={"billing": "", "shipping": "", "other": ""}
            )
        },
        temperature=1e6,
    )
    topic = response.answers["topic"]
    assert isinstance(topic, decision_engine.ChoiceAnswer)
    for p in topic.probabilities.values():
      self.assertAlmostEqual(p, 1 / 3, places=3)
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      self.engine.decide(
          _INPUT, {"q": decision_engine.NoulQuestion("Ok?")}, temperature=0.0
      )

  def test_non_ascii_text_and_option_ids_round_trip(self):
    response = self.engine.decide_json({
        "input": "我被重复扣款了，请退款。",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "这条消息属于哪个部门？",
                "criteria": {"物流": "配送", "账单": "扣款和发票"},
            },
            "level": {
                "type": "score",
                "instructions": "有多紧急？",
                "criteria": ["不紧急", "紧急"],
            },
        },
    })
    department = response["answers"]["department"]
    self.assertEqual(list(department["probabilities"]), ["物流", "账单"])
    self.assertIn(department["choice"], ("物流", "账单"))
    self.assertEqual(response["answers"]["level"]["legend"]["1"], "紧急")
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      self.engine.decide("hi", {"问题": decision_engine.NoulQuestion("Ok?")})

  def test_rejects_invalid_requests(self):
    question = decision_engine.NoulQuestion("Ok?")
    with self.assertRaisesRegex(ValueError, "questions"):
      self.engine.decide("hello", {})
    with self.assertRaisesRegex(ValueError, "input"):
      self.engine.decide([], {"q": question})
    bad_input: Any = [litert_lm.Content.Text("hi")]
    with self.assertRaises(TypeError):
      self.engine.decide(bad_input, {"q": question})
    bad_question: Any = {"type": "noul"}
    with self.assertRaises(TypeError):
      self.engine.decide("hello", {"q": bad_question})
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      self.engine.decide(
          "hello",
          {"q": decision_engine.ChoiceQuestion("Pick one", criteria={"a": ""})},
      )
    with self.assertRaisesRegex(RuntimeError, "UNIMPLEMENTED"):
      self.engine.decide(
          "hello",
          {
              "q": decision_engine.ChoiceQuestion(
                  "Pick one", criteria={f"o{i}": "" for i in range(27)}
              )
          },
      )

  def test_input_longer_than_context_is_invalid_argument_on_cpu(self):
    long_input = " ".join(f"word{i}" for i in range(200))
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      self.engine.decide(long_input, {"q": decision_engine.NoulQuestion("Ok?")})

  def test_decide_json_rejects_unsupported_content(self):
    with self.assertRaises(NotImplementedError):
      self.engine.decide_json({
          "input": ["Look:", {"type": "image", "uri": "file:///cat.png"}],
          "questions": {"q": _NOUL},
      })

  def test_closed_engine_raises(self):
    self.engine.close()
    self.engine.close()  # Closing twice is a no-op.
    with self.assertRaises(RuntimeError):
      self.engine.decide("hi", {"q": decision_engine.NoulQuestion("Ok?")})

  def test_close_after_shared_engine_closed_does_not_delete(self):
    base_engine = litert_lm.Engine(
        _test_model_path(),
        backend=litert_lm.Backend.CPU(),
        max_num_tokens=1024,
        cache_dir=":nocache",
    )
    shared_decision = decision_engine.DecisionEngine.from_engine(base_engine)
    lib = shared_decision._lib  # pylint: disable=protected-access
    base_engine.close()
    with mock.patch.object(
        lib, "litert_lm_experimental_decision_engine_delete"
    ) as mock_delete:
      shared_decision.close()
    # Deleting the native decision engine after its engine is not allowed.
    mock_delete.assert_not_called()

  def test_from_engine(self):
    base_engine = litert_lm.Engine(
        _test_model_path(),
        backend=litert_lm.Backend.CPU(),
        max_num_tokens=1024,
        cache_dir=":nocache",
    )
    shared_decision = decision_engine.DecisionEngine.from_engine(base_engine)
    self.addCleanup(shared_decision.close)
    questions = {"refund": decision_engine.NoulQuestion("Refund requested?")}
    first = shared_decision.decide(_INPUT, questions)
    with base_engine.create_session() as _:
      pass
    second = shared_decision.decide(_INPUT, questions)
    base_engine.close()
    with self.assertRaises(ValueError):
      decision_engine.DecisionEngine.from_engine(base_engine)
    # The shared engine must outlive the decision engine.
    with self.assertRaises(RuntimeError):
      shared_decision.decide(_INPUT, questions)
    first_refund = first.answers["refund"]
    second_refund = second.answers["refund"]
    assert isinstance(first_refund, decision_engine.NoulAnswer)
    assert isinstance(second_refund, decision_engine.NoulAnswer)
    self.assertAlmostEqual(first_refund.noul, second_refund.noul, places=5)

  def test_npu_backend_forwards_dispatch_lib_dir(self):
    fake_lib = mock.MagicMock()
    with (
        mock.patch.object(decision_engine, "_get_lib", return_value=fake_lib),
        mock.patch.object(
            decision_engine,
            "create_checked",
            side_effect=[101, 202],
        ),
        mock.patch.object(decision_engine, "call_checked") as mock_call,
    ):
      eng = decision_engine.DecisionEngine(
          "/tmp/fake.litertlm",
          backend=litert_lm.Backend.NPU(litert_dispatch_lib_dir="/opt/npu/lib"),
      )
      eng.close()
    mock_call.assert_any_call(
        fake_lib,
        "litert_lm_engine_settings_set_litert_dispatch_lib_dir",
        101,
        "/opt/npu/lib",
    )


class DecisionJsonTest(absltest.TestCase):

  def test_parses_all_forms(self):
    parsed = _decision_json.parse_request({
        "model": "ignored",
        "input": [
            "a",
            {"type": "text", "text": "b", "annotations": []},
        ],
        "questions": {
            "c": {
                "type": "choice",
                "instructions": {"type": "text", "text": "Pick."},
                "criteria": {"z": "", "a": "A"},
            },
            "n": {
                "type": "noul",
                "instructions": ["Is it", {"type": "text", "text": "true?"}],
                "criteria": {"false_description": "No"},
            },
            "s": {"type": "score", "instructions": "Rate.", "criteria": ["0"]},
            "e": {"type": "noul", "instructions": "Ok?"},
        },
        "temperature": 2,
    })
    self.assertEqual(parsed.input, ["a", "b"])
    self.assertEqual(list(parsed.questions), ["c", "n", "s", "e"])
    self.assertEqual(
        parsed.questions["c"],
        {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"z": "", "a": "A"},
        },
    )
    self.assertEqual(list(parsed.questions["c"]["criteria"]), ["z", "a"])
    self.assertEqual(parsed.questions["n"]["instructions"], "Is it\ntrue?")
    self.assertEqual(
        parsed.questions["n"]["criteria"], {"false_description": "No"}
    )
    self.assertNotIn("criteria", parsed.questions["e"])
    self.assertEqual(parsed.temperature, 2.0)
    self.assertIsNone(
        _decision_json.parse_request(
            {"input": "hi", "questions": {"q": _NOUL}}
        ).temperature
    )

  def test_json_string_keeps_key_order_and_rejects_duplicates(self):
    parsed = _decision_json.parse_request(
        '{"input": "hi", "questions": {"q": {"type": "choice",'
        ' "instructions": "p", "criteria": {"10": "", "2": "", "1": ""}}}}'
    )
    self.assertEqual(list(parsed.questions["q"]["criteria"]), ["10", "2", "1"])
    for request in (
        (
            '{"input": "hi", "questions": {"q": {"type": "noul",'
            ' "instructions": "p"}, "q": {"type": "noul", "instructions":'
            ' "r"}}}'
        ),
        (
            '{"input": "hi", "questions": {"q": {"type": "choice",'
            ' "instructions": "p", "criteria": {"a": "", "b": "", "a": "x"}}}}'
        ),
        '{"input": "hi", "input": "ho", "questions": {}}',
    ):
      with self.subTest(request=request):
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
          _decision_json.parse_request(request)
    with self.assertRaises(ValueError):
      _decision_json.parse_request("not json")

  def test_rejects_unknown_and_missing_fields(self):
    for request in (
        {"input": "hi", "questions": {"q": _NOUL}, "contents": "hi"},
        {"input": "hi", "questions": {"q": {**_NOUL, "prompt": "p?"}}},
        {
            "input": "hi",
            "questions": {"q": {**_NOUL, "criteria": {"true": "yes"}}},
        },
        {
            "input": {"type": "text", "text": "hi", "mime_type": "x"},
            "questions": {"q": _NOUL},
        },
        {"questions": {"q": _NOUL}},
        {"input": "hi"},
        {"input": [], "questions": {"q": _NOUL}},
        {"input": {"type": "text"}, "questions": {"q": _NOUL}},
        {"input": {"type": "html", "text": "hi"}, "questions": {"q": _NOUL}},
        {"input": "hi", "questions": {"q": {"type": "noul"}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "type": "boolean"}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "type": "Noul"}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "instructions": []}}},
        {
            "input": "hi",
            "questions": {"q": {**_NOUL, "instructions": {"type": "image"}}},
        },
        {"input": "hi", "questions": {"q": {**_NOUL, "type": "choice"}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "type": "score"}}},
    ):
      with self.subTest(request=request):
        with self.assertRaises(ValueError):
          _decision_json.parse_request(request)

  def test_rejects_wrong_types(self):
    for request in (
        [],
        {"model": 1, "input": "hi", "questions": {"q": _NOUL}},
        {"input": 1, "questions": {"q": _NOUL}},
        {"input": None, "questions": {"q": _NOUL}},
        {"input": ["hi", 2], "questions": {"q": _NOUL}},
        {"input": {"type": "text", "text": 1}, "questions": {"q": _NOUL}},
        {"input": {"type": 1, "text": "hi"}, "questions": {"q": _NOUL}},
        {"input": "hi", "questions": [_NOUL]},
        {"input": "hi", "questions": {"q": "p?"}},
        {"input": "hi", "questions": {"q": {**_NOUL, "type": True}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "instructions": 1}}},
        {"input": "hi", "questions": {"q": {**_NOUL, "criteria": None}}},
        {
            "input": "hi",
            "questions": {"q": {**_NOUL, "criteria": {"true_description": 1}}},
        },
        {
            "input": "hi",
            "questions": {
                "q": {"type": "choice", "instructions": "p", "criteria": ["a"]}
            },
        },
        {
            "input": "hi",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "p",
                    "criteria": {"a": 1, "b": ""},
                }
            },
        },
        {
            "input": "hi",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "p",
                    "criteria": {"0": "low"},
                }
            },
        },
        {
            "input": "hi",
            "questions": {
                "q": {"type": "score", "instructions": "p", "criteria": [1]}
            },
        },
        {"input": "hi", "questions": {"q": _NOUL}, "temperature": "1"},
        {"input": "hi", "questions": {"q": _NOUL}, "temperature": True},
        {"input": "hi", "questions": {"q": _NOUL}, "temperature": None},
    ):
      with self.subTest(request=request):
        with self.assertRaises(TypeError):
          bad_request: Any = request
          _decision_json.parse_request(bad_request)

  def test_invalid_temperature_is_value_error(self):
    for temperature in (0, -1.0, float("nan"), float("inf")):
      with self.subTest(temperature=temperature):
        with self.assertRaisesRegex(ValueError, "temperature"):
          _decision_json.parse_request({
              "input": "hi",
              "questions": {"q": _NOUL},
              "temperature": temperature,
          })
    with self.assertRaisesRegex(ValueError, "temperature"):
      _decision_json.parse_request(
          '{"input": "hi", "questions": {"q": {"type": "noul",'
          ' "instructions": "p"}}, "temperature": NaN}'
      )

  def test_media_input_is_not_implemented(self):
    for content_type in ("image", "audio", "video", "document"):
      with self.subTest(content_type=content_type):
        with self.assertRaisesRegex(NotImplementedError, content_type):
          _decision_json.parse_request({
              "input": [{"type": content_type, "data": "AAAA"}],
              "questions": {"q": _NOUL},
          })


class DecisionTypesTest(absltest.TestCase):

  def test_question_to_dict(self):
    self.assertEqual(
        decision_engine.NoulQuestion("p?").to_dict(),
        {"type": "noul", "instructions": "p?"},
    )
    self.assertEqual(
        decision_engine.NoulQuestion("p?", false_description="No").to_dict(),
        {
            "type": "noul",
            "instructions": "p?",
            "criteria": {"false_description": "No"},
        },
    )
    choice = decision_engine.ChoiceQuestion(
        "Topic?", criteria={"shipping": "", "billing": "Charges"}
    ).to_dict()
    self.assertEqual(
        choice,
        {
            "type": "choice",
            "instructions": "Topic?",
            "criteria": {"shipping": "", "billing": "Charges"},
        },
    )
    self.assertEqual(list(choice["criteria"]), ["shipping", "billing"])
    self.assertEqual(
        decision_engine.ScoreQuestion("Rate.", ("low", "high")).to_dict(),
        {"type": "score", "instructions": "Rate.", "criteria": ["low", "high"]},
    )

  def test_response_to_dict(self):
    response = decision_engine.DecisionResponse(
        answers={
            "refund": decision_engine.NoulAnswer(confidence=0.75, noul=0.25),
            "topic": decision_engine.ChoiceAnswer(
                confidence=0.5,
                choice="billing",
                probabilities={"billing": 0.5, "shipping": 0.5},
            ),
            "urgency": decision_engine.ScoreAnswer(
                confidence=0.75,
                score=0.25,
                probabilities=[0.75, 0.25],
                legend=["low", "high"],
            ),
        }
    )
    refund = response.answers["refund"]
    assert isinstance(refund, decision_engine.NoulAnswer)
    self.assertFalse(refund.value)
    urgency = response.answers["urgency"]
    assert isinstance(urgency, decision_engine.ScoreAnswer)
    self.assertEqual(urgency.most_likely_level, 0)
    self.assertEqual(
        response.to_dict(),
        {
            "answers": {
                "refund": {"type": "noul", "confidence": 0.75, "noul": 0.25},
                "topic": {
                    "type": "choice",
                    "confidence": 0.5,
                    "choice": "billing",
                    "probabilities": {"billing": 0.5, "shipping": 0.5},
                },
                "urgency": {
                    "type": "score",
                    "confidence": 0.75,
                    "score": 0.25,
                    "probabilities": {"0": 0.75, "1": 0.25},
                    "legend": {"0": "low", "1": "high"},
                },
            }
        },
    )


if __name__ == "__main__":
  absltest.main()
