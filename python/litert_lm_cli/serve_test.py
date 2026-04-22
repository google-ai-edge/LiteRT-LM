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

"""Unit tests for the LiteRT-LM serve command."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

# 1. Mock the C++ extension specifically to prevent loading it.
# This MUST happen before importing anything from litert_lm.
mock_litert_lm_ext = mock.MagicMock()

# These must be classes because they are passed to abc.ABCMeta.register()
# in litert_lm/__init__.py
mock_litert_lm_ext._Benchmark = type("_Benchmark", (), {})
mock_litert_lm_ext._Engine = type("_Engine", (), {})
mock_litert_lm_ext.Benchmark = type("Benchmark", (), {})
mock_litert_lm_ext.BenchmarkInfo = type("BenchmarkInfo", (), {})
mock_litert_lm_ext.Conversation = type("Conversation", (), {})
mock_litert_lm_ext.Engine = mock.Mock()
mock_litert_lm_ext.Session = type("Session", (), {})

sys.modules[
    "litert_lm.litert_lm_ext"
] = mock_litert_lm_ext

# 2. Now we can import the real litert_lm safely. It will use our mocked extension.
import litert_lm as mock_litert_lm
from litert_lm import interfaces

# 3. Explicitly override Engine and other classes with Mocks to ensure they don't
# point to the mocked extension's classes which might not behave like standard mocks.
mock_litert_lm.Engine = mock.Mock()
mock_litert_lm.set_min_log_severity = mock.Mock()

# 4. Also mock model as it imports litert_lm too.
mock_model_mod = mock.Mock(spec_set=["Model"])
mock_model_mod.Model = mock.Mock(spec_set=["from_model_id"])
mock_model_mod.Model.from_model_id = mock.Mock()
sys.modules["litert_lm_cli.model"] = (
    mock_model_mod
)

from litert_lm_cli import serve


class ServeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global state in serve.py
    serve._current_engine = None
    serve._current_model_id = None
    # Reset mocks
    mock_litert_lm.set_min_log_severity.reset_mock()
    mock_litert_lm.Engine.reset_mock()
    mock_model_mod.Model.from_model_id.reset_mock()
    mock_model_mod.Model.from_model_id.side_effect = None

  @parameterized.named_parameters(
      dict(
          testcase_name="user_text",
          gemini_content={"role": "user", "parts": [{"text": "Hello"}]},
          expected={
              "role": "user",
              "content": [{"type": "text", "text": "Hello"}],
          },
      ),
      dict(
          testcase_name="model_text",
          gemini_content={"role": "model", "parts": [{"text": "Hi"}]},
          expected={
              "role": "assistant",
              "content": [{"type": "text", "text": "Hi"}],
          },
      ),
      dict(
          testcase_name="default_role",
          gemini_content={"parts": [{"text": "No role"}]},
          expected={
              "role": "user",
              "content": [{"type": "text", "text": "No role"}],
          },
      ),
      dict(
          testcase_name="tool_call",
          gemini_content={
              "role": "model",
              "parts": [{
                  "functionCall": {
                      "name": "get_weather",
                      "args": {"location": "London"},
                  }
              }],
          },
          expected={
              "role": "assistant",
              "tool_calls": [{
                  "function": {
                      "name": "get_weather",
                      "arguments": {"location": "London"},
                  }
              }],
          },
      ),
      dict(
          testcase_name="tool_response",
          gemini_content={
              "role": "tool",
              "parts": [{
                  "functionResponse": {
                      "name": "get_weather",
                      "response": {"weather": "sunny"},
                  }
              }],
          },
          expected={
              "role": "tool",
              "content": [{
                  "type": "tool_response",
                  "name": "get_weather",
                  "response": {"weather": "sunny"},
              }],
          },
      ),
  )
  def test_gemini_to_litertlm_message(self, gemini_content, expected):
    self.assertEqual(serve.gemini_to_litertlm_message(gemini_content), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="assistant_text",
          litertlm_response={
              "role": "assistant",
              "content": [{"type": "text", "text": "Response text"}],
          },
          finish_reason="STOP",
          expected={
              "candidates": [{
                  "content": {
                      "role": "model",
                      "parts": [{"text": "Response text"}],
                  },
                  "finishReason": "STOP",
                  "index": 0,
              }]
          },
      ),
      dict(
          testcase_name="tool_calls",
          litertlm_response={
              "role": "assistant",
              "tool_calls": [{
                  "function": {
                      "name": "get_weather",
                      "arguments": {"location": "London"},
                  }
              }],
          },
          finish_reason="STOP",
          expected={
              "candidates": [{
                  "content": {
                      "role": "model",
                      "parts": [{
                          "functionCall": {
                              "name": "get_weather",
                              "args": {"location": "London"},
                          }
                      }],
                  },
                  "finishReason": "STOP",
                  "index": 0,
              }]
          },
      ),
      dict(
          testcase_name="streaming",
          litertlm_response={"content": [{"type": "text", "text": "Chunk"}]},
          finish_reason="",
          expected={
              "candidates": [{
                  "content": {
                      "role": "model",
                      "parts": [{"text": "Chunk"}],
                  },
                  "index": 0,
              }]
          },
      ),
      dict(
          testcase_name="custom_finish_reason",
          litertlm_response={"content": [{"type": "text", "text": "Text"}]},
          finish_reason="MAX_TOKENS",
          expected={
              "candidates": [{
                  "content": {
                      "role": "model",
                      "parts": [{"text": "Text"}],
                  },
                  "finishReason": "MAX_TOKENS",
                  "index": 0,
              }]
          },
      ),
  )
  def test_litertlm_to_gemini_response(
      self, litertlm_response, finish_reason, expected
  ):
    self.assertEqual(
        serve.litertlm_to_gemini_response(litertlm_response, finish_reason),
        expected,
    )

  def test_get_engine_caching(self):
    mock_model = mock.Mock(spec_set=["exists", "model_path"])
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/model"
    mock_model_mod.Model.from_model_id.return_value = mock_model

    mock_engine_instance = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_instance.__enter__.return_value = mock_engine_instance
    mock_engine_instance.__exit__.return_value = False
    mock_litert_lm.Engine.return_value = mock_engine_instance

    # First call - creates engine
    engine1 = serve.get_engine("test-model")
    self.assertEqual(engine1, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()

    # Second call with same ID - returns cached engine
    engine2 = serve.get_engine("test-model")
    self.assertEqual(engine2, mock_engine_instance)
    self.assertEqual(mock_litert_lm.Engine.call_count, 1)

  def test_get_engine_recovery_after_failure(self):
    def from_id_side_effect(model_id):
      m = mock.Mock(spec_set=["exists", "model_path"])
      m.exists.return_value = True
      m.model_path = f"/path/to/{model_id}"
      return m

    mock_model_mod.Model.from_model_id.side_effect = from_id_side_effect

    mock_engine_instance = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_instance.__enter__.return_value = mock_engine_instance
    mock_engine_instance.__exit__.return_value = False
    mock_litert_lm.Engine.return_value = mock_engine_instance

    # 1. Load model "A"
    serve.get_engine("A")
    self.assertEqual(serve._current_model_id, "A")

    # 2. Mock engine creation failure for model "B"
    mock_litert_lm.Engine.side_effect = RuntimeError("Failed to init")
    with self.assertRaises(RuntimeError):
      serve.get_engine("B")

    # Verify state was cleared
    self.assertIsNone(serve._current_engine)
    self.assertIsNone(serve._current_model_id)

    # 3. Load model "C" - should succeed
    mock_litert_lm.Engine.side_effect = None
    serve.get_engine("C")
    self.assertEqual(serve._current_model_id, "C")

  def test_model_id_regex_parsing(self):
    self.assertTrue(
        serve.GEN_CONTENT_RE.fullmatch(
            "/v1beta/models/gemma-2b:generateContent"
        )
    )
    self.assertTrue(
        serve.GEN_CONTENT_RE.fullmatch(
            "/v1beta/models/gemma-2b,cpu,1024:generateContent"
        )
    )
    self.assertTrue(
        serve.STREAM_GEN_CONTENT_RE.fullmatch(
            "/v1beta/models/gemma-2b:streamGenerateContent"
        )
    )
    self.assertFalse(
        serve.GEN_CONTENT_RE.fullmatch("/v1/models/gemma-2b:generateContent")
    )


if __name__ == "__main__":
  absltest.main()
