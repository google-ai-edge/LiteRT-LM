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

"""Tests for litert_lm_model early stopping behavior."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

# Mock out lm_eval and framework objects so imports succeed.
mock_lm_eval = mock.MagicMock()
sys.modules["lm_eval"] = mock_lm_eval
sys.modules["lm_eval.api"] = mock.MagicMock()
sys.modules["lm_eval.api.model"] = mock.MagicMock()
sys.modules["lm_eval.api.model"].LM = object
sys.modules["lm_eval.api.registry"] = mock.MagicMock()
sys.modules["lm_eval.api.registry"].register_model = lambda x: lambda y: y

from litert_lm_eval.runners.lm_eval_runner import litert_lm_model  # pylint: disable=g-import-not-at-top


class LitertLmModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(litert_lm_model.litert_lm, "Engine", autospec=True)
    )

  def test_generate_until_stops_at_earliest_sequence(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # Mock the conversation to return a payload with multiple stop sequences.
    # Notice that '\n\n' appears before 'User:'.
    mock_conversation = mock.MagicMock()
    mock_conversation.send_message.return_value = {
        "content": [{
            "type": "text",
            "text": "The answer is 42.\n\nUser: What is next? Question:",
        }]
    }

    # Context manager setup.
    model.engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )
    model.engine.create_conversation.return_value.__exit__.return_value = None

    class MockRequest:

      def __init__(self):
        self.args = (
            "context prompt",
            {"until": ["User:", "\n\n", "Question:"]},
        )

    requests = [MockRequest()]
    res = model.generate_until(requests)

    # "\n\n" came first in the actual text response despite "User:" coming first
    # in the until array. So it should correctly split right at "\n\n".
    self.assertEqual(["The answer is 42."], res)

  def test_loglikelihood_is_greedy_true(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # Mock the session for scoring.
    mock_session = mock.MagicMock()
    mock_scoring_responses = mock.MagicMock()
    mock_scoring_responses.scores = [-1.5]
    mock_session.run_text_scoring.return_value = mock_scoring_responses
    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )
    model.engine.create_session.return_value.__exit__.return_value = None

    # Mock the conversation for greedy check.
    mock_conversation = mock.MagicMock()
    mock_conversation.send_message.return_value = {
        "content": [{
            "type": "text",
            "text": " world and some more text",
        }]
    }
    model.engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )
    model.engine.create_conversation.return_value.__exit__.return_value = None

    class MockRequest:

      def __init__(self):
        self.args = ("hello", " world")

    requests = [MockRequest()]
    res = model.loglikelihood(requests)

    # Generated text starts with " world", so is_greedy=True.
    self.assertEqual([(-1.5, True)], res)

  def test_loglikelihood_is_greedy_false(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # Mock the session for scoring.
    mock_session = mock.MagicMock()
    mock_scoring_responses = mock.MagicMock()
    mock_scoring_responses.scores = [-3.0]
    mock_session.run_text_scoring.return_value = mock_scoring_responses
    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )
    model.engine.create_session.return_value.__exit__.return_value = None

    # Mock the conversation for greedy check.
    mock_conversation = mock.MagicMock()
    mock_conversation.send_message.return_value = {
        "content": [{
            "type": "text",
            "text": " everyone",
        }]
    }
    model.engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )
    model.engine.create_conversation.return_value.__exit__.return_value = None

    class MockRequest:

      def __init__(self):
        self.args = ("hello", " world")

    requests = [MockRequest()]
    res = model.loglikelihood(requests)

    # Generated text does not start with " world", so is_greedy=False.
    self.assertEqual([(-3.0, False)], res)

  def test_loglikelihood_different_contexts_caches_greedy_check(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # Mock the session for scoring.
    mock_session = mock.MagicMock()
    mock_scoring_responses = mock.MagicMock()
    mock_scoring_responses.scores = [-1.5]
    mock_session.run_text_scoring.return_value = mock_scoring_responses
    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )

    # Mock the conversation for greedy check.
    mock_conversation = mock.MagicMock()
    mock_conversation.send_message.return_value = {
        "content": [{
            "type": "text",
            "text": " world",
        }]
    }
    model.engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )

    class MockRequest:

      def __init__(self, context, continuation):
        self.args = (context, continuation)

    requests = [
        MockRequest("hello", " world"),
        MockRequest("hello", " everyone"),
        MockRequest("hello different", " world"),
    ]

    res = model.loglikelihood(requests)

    # Only 2 unique contexts, so send_message should be called exactly twice.
    self.assertEqual(2, mock_conversation.send_message.call_count)
    self.assertLen(res, 3)
    # The returned greedy status should match the mocked returned " world"
    # string.
    self.assertEqual((-1.5, True), res[0])
    self.assertEqual((-1.5, False), res[1])
    self.assertEqual((-1.5, True), res[2])


if __name__ == "__main__":
  absltest.main()
