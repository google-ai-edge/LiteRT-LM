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

import base64
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
sys.modules["transformers"] = mock.MagicMock()

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

    # Mock the session for greedy check.
    mock_decode_responses = mock.MagicMock()
    mock_decode_responses.texts = [" world and some more text"]
    mock_session.run_decode.return_value = mock_decode_responses

    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )
    model.engine.create_session.return_value.__exit__.return_value = None

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

    # Mock the session for greedy check.
    mock_decode_responses = mock.MagicMock()
    mock_decode_responses.texts = [" everyone"]
    mock_session.run_decode.return_value = mock_decode_responses

    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
    )
    model.engine.create_session.return_value.__exit__.return_value = None

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

    # Mock the session for greedy check.
    mock_decode_responses = mock.MagicMock()
    mock_decode_responses.texts = [" world"]
    mock_session.run_decode.return_value = mock_decode_responses

    model.engine.create_session.return_value.__enter__.return_value = (
        mock_session
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

    # Only 2 unique contexts, so run_decode should be called exactly twice.
    self.assertEqual(2, mock_session.run_decode.call_count)
    self.assertLen(res, 3)
    # The returned greedy status should match the mocked returned " world"
    # string.
    self.assertEqual((-1.5, True), res[0])
    self.assertEqual((-1.5, False), res[1])
    self.assertEqual((-1.5, True), res[2])

  def test_tokenizer_name_default(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    self.assertEqual(model.tokenizer_name, "litert_lm")

  def test_apply_chat_template(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    chat_history = [
        {"role": "user", "content": "hello_world"},
        {"role": "user", "content": "foo"},
    ]
    res = model.apply_chat_template(chat_history)
    import json

    self.assertEqual(res, json.dumps(chat_history))

  def test_extract_text(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    response_content = [
        {"type": "text", "text": "Hello, "},
        {"type": "image", "image": b"123"},
        {"type": "text", "text": "world!"},
    ]
    self.assertEqual("Hello, world!", model._extract_text(response_content))

  def test_split_context(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # String context
    history, last_msg = model._split_context("hello")
    self.assertIsNone(history)
    self.assertEqual("hello", last_msg)

    # List context
    history, last_msg = model._split_context([
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello_back"},
    ])
    self.assertEqual([{"role": "user", "content": "hi"}], history)
    self.assertEqual({"role": "model", "content": "hello_back"}, last_msg)

    # Empty list context
    history, last_msg = model._split_context([])
    self.assertIsNone(history)
    self.assertEqual([], last_msg)

  def test_render_context(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    mock_conversation = mock.MagicMock()
    mock_conversation.render_message_to_string.side_effect = (
        lambda m: m if isinstance(m, str) else m["content"]
    )

    # String context
    self.assertEqual("hello", model._render_context(mock_conversation, "hello"))

    # List context
    self.assertEqual(
        "hihello",
        model._render_context(
            mock_conversation,
            [
                {"role": "user", "content": "hi"},
                {"role": "model", "content": "hello"},
            ],
        ),
    )

  def test_process_multimodal_message(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    # No mm args
    self.assertEqual("hello", model._process_multimodal_message("hello", {}))

    # String message with mm args
    mm_args = {"visual": [b"img1", b"img2"]}

    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "image",
                "blob": base64.b64encode(b"img1").decode("utf-8"),
            },
            {
                "type": "image",
                "blob": base64.b64encode(b"img2").decode("utf-8"),
            },
        ],
    }
    self.assertEqual(
        expected, model._process_multimodal_message("hello", mm_args)
    )

    # Dict message with mm args
    msg = {"role": "user", "content": "hello"}
    self.assertEqual(expected, model._process_multimodal_message(msg, mm_args))

    # Dict message with list content and mm args
    msg2 = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    self.assertEqual(expected, model._process_multimodal_message(msg2, mm_args))

    # PIL Image format support
    mock_pil_image = mock.MagicMock()
    # Mocking basic PIL Image attributes just in case
    mock_pil_image.mode = "RGB"
    mock_pil_image.size = (224, 224)

    def fake_save(buf, format):  # pylint: disable=redefined-builtin
      self.assertEqual(format, "PNG")
      buf.write(b"fake_png_bytes")

    mock_pil_image.save.side_effect = fake_save

    mm_args_pil = {"visual": [mock_pil_image]}

    expected_pil = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "image",
                "blob": base64.b64encode(b"fake_png_bytes").decode("utf-8"),
            },
        ],
    }
    self.assertEqual(
        expected_pil, model._process_multimodal_message("hello", mm_args_pil)
    )

  def test_process_multimodal_message_with_placeholders(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    mm_args = {"visual": [b"img1", b"img2"]}
    # Prompt containing explicitly interleaved image placeholders.
    prompt = "Look at <image> and also <image>."

    res = model._process_multimodal_message(prompt, mm_args)
    self.assertEqual(
        res["content"],
        [
            {"type": "text", "text": "Look at "},
            {
                "type": "image",
                "blob": base64.b64encode(b"img1").decode("utf-8"),
            },
            {"type": "text", "text": " and also "},
            {
                "type": "image",
                "blob": base64.b64encode(b"img2").decode("utf-8"),
            },
            {"type": "text", "text": "."},
        ],
    )

  def test_process_multimodal_message_too_many_placeholders_raises_error(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    mm_args = {"visual": [b"only_one_image"]}
    prompt = "Here are two placeholders: <image> and <image>."
    with self.assertRaises(ValueError):
      model._process_multimodal_message(prompt, mm_args)

  def test_is_json_list_validation(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")
    # Valid serialized list
    self.assertTrue(model._is_json_list('["item1", "item2"]'))
    # Regular string prompt
    self.assertFalse(model._is_json_list("Just a prompt."))
    # Serialized JSON object (not list)
    self.assertFalse(model._is_json_list('{"key": "value"}'))
    # Malformed JSON string
    self.assertFalse(model._is_json_list("[unterminated"))

  def test_generate_until_multimodal(self):
    model = litert_lm_model.LitertLmModelRunner(model_path="dummy_path")

    mock_conversation = mock.MagicMock()
    mock_conversation.send_message.return_value = {
        "content": [{"type": "text", "text": "response text"}]
    }

    model.engine.create_conversation.return_value.__enter__.return_value = (
        mock_conversation
    )

    class MockRequest:

      def __init__(self):
        # 3-element tuple with mm_args
        self.args = (
            "hello",
            {"until": ["\n"]},
            {"visual": [b"image_data"]},
        )

    requests = [MockRequest()]
    res = model.generate_until(requests)

    self.assertEqual(["response text"], res)
    mock_conversation.send_message.assert_called_once_with({
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "image",
                "blob": base64.b64encode(b"image_data").decode("utf-8"),
            },
        ],
    })

  def test_init_backends(self):
    # Specialized backend maps correctly to backend enum
    model = litert_lm_model.LitertLmModelRunner(
        model_path="dummy_path",
        backend="CPU",
        vision_backend="GPU",
    )
    self.assertEqual(
        model.vision_backend, litert_lm_model.litert_lm.Backend.GPU
    )


if __name__ == "__main__":
  absltest.main()
