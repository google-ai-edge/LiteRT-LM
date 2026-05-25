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
mock_ffi = mock.MagicMock()
mock_ffi.LogSeverity = type("LogSeverity", (), {})
mock_ffi.set_min_log_severity = mock.Mock()

mock_benchmark = mock.MagicMock()
mock_benchmark.Benchmark = type("Benchmark", (), {})

mock_conversation = mock.MagicMock()
mock_conversation.Conversation = type("Conversation", (), {})

mock_engine = mock.MagicMock()
mock_engine.Engine = mock.Mock()

mock_session = mock.MagicMock()
mock_session.Session = type("Session", (), {})

sys.modules["litert_lm._ffi"] = (
    mock_ffi
)
sys.modules["litert_lm.benchmark"] = (
    mock_benchmark
)
sys.modules[
    "litert_lm.conversation"
] = mock_conversation
sys.modules["litert_lm.engine"] = (
    mock_engine
)
sys.modules["litert_lm.session"] = (
    mock_session
)

# 2. Now we can import the real litert_lm safely. It will use our mocked
# extension.
import litert_lm as mock_litert_lm
from litert_lm import interfaces

# 3. Explicitly override Engine and other classes with Mocks to ensure they don't
# point to the mocked extension's classes which might not behave like standard
# mocks.
mock_litert_lm.Engine = mock_engine.Engine
mock_litert_lm.set_min_log_severity = mock_ffi.set_min_log_severity

mock_model_mod = mock.Mock(spec_set=["Model", "parse_backend"])
mock_model_mod.Model = mock.Mock(spec_set=["from_model_id"])
mock_model_mod.Model.from_model_id = mock.Mock()
mock_model_mod.parse_backend = mock.Mock()
sys.modules["litert_lm_cli.model"] = (
    mock_model_mod
)
if "litert_lm_cli" in sys.modules:
  sys.modules[
      "litert_lm_cli"
  ].model = mock_model_mod

from litert_lm_cli.commands import serve
from litert_lm_cli.commands import serve_util


class ServeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Reset mocks.
    mock_litert_lm.set_min_log_severity.reset_mock()  # pytype: disable=attribute-error
    mock_litert_lm.Engine.reset_mock()  # pytype: disable=attribute-error
    mock_model_mod.Model.from_model_id.reset_mock()
    mock_model_mod.Model.from_model_id.side_effect = None
    mock_model_mod.parse_backend.reset_mock()
    mock_model_mod.parse_backend.return_value = interfaces.Backend.CPU()

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

    server = mock.MagicMock(spec=serve_util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None

    # First call creates the engine.
    engine1 = serve_util.get_or_initialize_server_engine(
        server, model_id="test-model"
    )
    self.assertEqual(engine1, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()  # pytype: disable=attribute-error
    self.assertEqual(server.litert_lm_engine, mock_engine_instance)
    self.assertEqual(server.model_id, "test-model")

    # Second call with same ID - returns cached engine.
    engine2 = serve_util.get_or_initialize_server_engine(
        server, model_id="test-model"
    )
    self.assertEqual(engine2, mock_engine_instance)
    self.assertEqual(mock_litert_lm.Engine.call_count, 1)  # pytype: disable=attribute-error

  def test_get_engine_switching_reinitializes(self):
    mock_model_a = mock.Mock(spec_set=["exists", "model_path"])
    mock_model_a.exists.return_value = True
    mock_model_a.model_path = "/path/to/model_a"

    mock_model_b = mock.Mock(spec_set=["exists", "model_path"])
    mock_model_b.exists.return_value = True
    mock_model_b.model_path = "/path/to/model_b"

    def from_model_id_side_effect(model_id):
      if model_id == "A":
        return mock_model_a
      if model_id == "B":
        return mock_model_b
      m = mock.Mock(spec_set=["exists"])
      m.exists.return_value = False
      return m

    mock_model_mod.Model.from_model_id.side_effect = from_model_id_side_effect

    mock_engine_a = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_a.__enter__.return_value = mock_engine_a

    mock_engine_b = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_b.__enter__.return_value = mock_engine_b

    def engine_side_effect(model_path, **unused_kwargs):
      if "model_a" in model_path:
        return mock_engine_a
      if "model_b" in model_path:
        return mock_engine_b
      return None

    mock_litert_lm.Engine.side_effect = engine_side_effect

    server = mock.MagicMock(spec=serve_util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None

    # Initialize with model A.
    engine1 = serve_util.get_or_initialize_server_engine(server, model_id="A")
    self.assertEqual(engine1, mock_engine_a)
    self.assertEqual(server.model_id, "A")
    mock_engine_a.__exit__.assert_not_called()

    # Switching to model B re-initializes (closes A, opens B).
    engine2 = serve_util.get_or_initialize_server_engine(server, model_id="B")
    self.assertEqual(engine2, mock_engine_b)
    self.assertEqual(server.model_id, "B")
    mock_engine_a.__exit__.assert_called_once_with(None, None, None)

  def test_get_engine_backend_switching_reinitializes(self):
    mock_model = mock.Mock(spec_set=["exists", "model_path"])
    mock_model.exists.return_value = True
    mock_model.model_path = "/path/to/model"
    mock_model_mod.Model.from_model_id.return_value = mock_model

    mock_engine_instance = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_instance.__enter__.return_value = mock_engine_instance
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = mock.MagicMock(spec=serve_util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None

    # Initialize with the CPU backend.
    serve_util.get_or_initialize_server_engine(
        server, model_id="model", backend=interfaces.Backend.CPU()
    )
    self.assertEqual(server.backend, interfaces.Backend.CPU())
    mock_engine_instance.__exit__.assert_not_called()

    # Switching to the GPU backend re-initializes.
    serve_util.get_or_initialize_server_engine(
        server, model_id="model", backend=interfaces.Backend.GPU()
    )
    self.assertEqual(server.backend, interfaces.Backend.GPU())
    mock_engine_instance.__exit__.assert_called_once_with(None, None, None)

  @parameterized.named_parameters(
      dict(
          testcase_name="model_only",
          model_spec="gemma",
          want_model="gemma",
          want_backend=None,
          want_max_tokens=None,
          want_error=None,
      ),
      dict(
          testcase_name="model_and_gpu",
          model_spec="gemma,gpu",
          want_model="gemma",
          want_backend=interfaces.Backend.GPU(),
          want_max_tokens=None,
          want_error=None,
      ),
      dict(
          testcase_name="model_cpu_and_tokens",
          model_spec="gemma,cpu,1024",
          want_model="gemma",
          want_backend=interfaces.Backend.CPU(),
          want_max_tokens=1024,
          want_error=None,
      ),
      dict(
          testcase_name="model_and_tokens_without_backend",
          model_spec="gemma,,1024",
          want_model="gemma",
          want_backend=None,
          want_max_tokens=1024,
          want_error=None,
      ),
      dict(
          testcase_name="invalid_trailing_comma",
          model_spec="gemma,",
          want_model=None,
          want_backend=None,
          want_max_tokens=None,
          want_error="Trailing comma in model spec: gemma,",
      ),
      dict(
          testcase_name="invalid_backend",
          model_spec="gemma,invalid_backend",
          want_model=None,
          want_backend=None,
          want_max_tokens=None,
          want_error="Unavailable backend 'invalid_backend'",
      ),
      dict(
          testcase_name="invalid_tokens",
          model_spec="gemma,gpu,invalid_tokens",
          want_model=None,
          want_backend=None,
          want_max_tokens=None,
          want_error="Invalid max_tokens: invalid_tokens",
      ),
      dict(
          testcase_name="invalid_extra_parameter",
          model_spec="gemma,gpu,1024,extra",
          want_model=None,
          want_backend=None,
          want_max_tokens=None,
          want_error="Too many parameters in model spec: gemma,gpu,1024,extra",
      ),
  )
  def test_parse_model_spec(
      self, model_spec, want_model, want_backend, want_max_tokens, want_error
  ):
    if want_error is not None:
      with self.assertRaisesRegex(ValueError, want_error):
        serve_util.parse_model_spec(model_spec)
    else:
      spec = serve_util.parse_model_spec(model_spec)
      self.assertEqual(spec.model_id, want_model)
      self.assertEqual(spec.backend, want_backend)
      self.assertEqual(spec.max_num_tokens, want_max_tokens)

  @parameterized.named_parameters(
      dict(
          testcase_name="model_only",
          path="/v1beta/models/gemma3-1b:generateContent",
          want_model="gemma3-1b",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error=None,
      ),
      dict(
          testcase_name="model_and_backend_cpu",
          path="/v1beta/models/gemma3-1b,cpu:generateContent",
          want_model="gemma3-1b",
          want_backend=interfaces.Backend.CPU(),
          want_max_tokens=None,
          want_stream=False,
          want_error=None,
      ),
      dict(
          testcase_name="model_and_backend_gpu",
          path="/v1beta/models/gemma3-1b,gpu:generateContent",
          want_model="gemma3-1b",
          want_backend=interfaces.Backend.GPU(),
          want_max_tokens=None,
          want_stream=False,
          want_error=None,
      ),
      dict(
          testcase_name="model_backend_and_max_tokens",
          path="/v1beta/models/gemma3-1b,cpu,8192:generateContent",
          want_model="gemma3-1b",
          want_backend=interfaces.Backend.CPU(),
          want_max_tokens=8192,
          want_stream=False,
          want_error=None,
      ),
      dict(
          testcase_name="model_max_tokens_without_backend",
          path="/v1beta/models/gemma3-1b,,8192:generateContent",
          want_model="gemma3-1b",
          want_backend=None,
          want_max_tokens=8192,
          want_stream=False,
          want_error=None,
      ),
      dict(
          testcase_name="invalid_max_tokens",
          path="/v1beta/models/gemma3-1b,cpu,abc:generateContent",
          want_model="",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error="Invalid max_tokens: abc",
      ),
      dict(
          testcase_name="invalid_format_trailing_comma",
          path="/v1beta/models/gemma3-1b,:generateContent",
          want_model="",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error="Trailing comma in model spec: gemma3-1b,",
      ),
      dict(
          testcase_name="invalid_format_extra_slashes",
          path="/v1beta/models/gemma3-1b/gpu:generateContent",
          want_model="",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error="Not Found",
      ),
      dict(
          testcase_name="unsupported_backend_rejects_gracefully",
          path="/v1beta/models/gemma3-1b,tpu:generateContent",
          want_model="",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error=(
              "Unavailable backend 'tpu', available backends are 'cpu' and"
              " 'gpu'"
          ),
      ),
      dict(
          testcase_name="valid_stream_endpoint",
          path="/v1beta/models/gemma3-1b:streamGenerateContent",
          want_model="gemma3-1b",
          want_backend=None,
          want_max_tokens=None,
          want_stream=True,
          want_error=None,
      ),
      dict(
          testcase_name="invalid_endpoint",
          path="/v1beta/models/gemma3-1b:unknownEndpoint",
          want_model="",
          want_backend=None,
          want_max_tokens=None,
          want_stream=False,
          want_error="Not Found",
      ),
  )
  def test_parse_model_and_backend_from_path(
      self,
      path,
      want_model,
      want_backend,
      want_max_tokens,
      want_stream,
      want_error,
  ):
    req = serve.parse_model_and_backend(path)
    if want_error is not None:
      self.assertEqual(req.error_msg, want_error)
    else:
      self.assertIsNone(req.error_msg)
      self.assertEqual(req.model_id, want_model)
      self.assertEqual(req.backend, want_backend)
      self.assertEqual(req.max_num_tokens, want_max_tokens)
      self.assertEqual(req.is_stream, want_stream)

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
