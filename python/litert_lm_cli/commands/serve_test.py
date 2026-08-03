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

import http.server
import socket
import sys
import threading
from unittest import mock
import urllib.request

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
mock_model_mod.Model = mock.Mock(spec_set=["from_model_id", "get_all_models"])
mock_model_mod.Model.from_model_id = mock.Mock()
mock_model_mod.Model.get_all_models = mock.Mock()
mock_model_mod.parse_backend = mock.Mock()
sys.modules["litert_lm_cli.model"] = (
    mock_model_mod
)
if "litert_lm_cli" in sys.modules:
  sys.modules[
      "litert_lm_cli"
  ].model = mock_model_mod

from litert_lm_cli.commands import gemini_handler
from litert_lm_cli.commands import openai_handler
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
    mock_model_mod.Model.get_all_models.reset_mock()
    mock_model_mod.Model.get_all_models.side_effect = None
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
  def test_litertlm_message_from_gemini(self, gemini_content, expected):
    self.assertEqual(
        gemini_handler.litertlm_message_from_gemini(gemini_content), expected
    )

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
  def test_gemini_response_from_litertlm(
      self, litertlm_response, finish_reason, expected
  ):
    self.assertEqual(
        gemini_handler.gemini_response_from_litertlm(
            litertlm_response, finish_reason
        ),
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



  @parameterized.named_parameters(
      dict(
          testcase_name="gen_content_standard",
          regex_type="gen",
          path="/v1beta/models/gemma-2b:generateContent",
          expected=True,
      ),
      dict(
          testcase_name="gen_content_with_params",
          regex_type="gen",
          path="/v1beta/models/gemma-2b,cpu,1024:generateContent",
          expected=True,
      ),
      dict(
          testcase_name="stream_gen_content",
          regex_type="stream",
          path="/v1beta/models/gemma-2b:streamGenerateContent",
          expected=True,
      ),
      dict(
          testcase_name="invalid_version",
          regex_type="gen",
          path="/v1/models/gemma-2b:generateContent",
          expected=False,
      ),
  )
  def test_model_id_regex_parsing(self, regex_type, path, expected):
    regex = (
        gemini_handler.GEN_CONTENT_RE
        if regex_type == "gen"
        else gemini_handler.STREAM_GEN_CONTENT_RE
    )
    match = regex.fullmatch(path)
    if expected:
      self.assertIsNotNone(match)
    else:
      self.assertIsNone(match)

  @mock.patch.object(http.server.HTTPServer, "__init__", autospec=True)
  def test_litert_lm_server_ipv6(self, mock_super_init):
    serve_util.LiteRTLMServer(("::1", 8000), mock.MagicMock())
    mock_super_init.assert_called_once()
    args, _ = mock_super_init.call_args
    self_arg, _, _ = args
    self.assertEqual(self_arg.address_family, socket.AF_INET6)

  @mock.patch.object(http.server.HTTPServer, "__init__", autospec=True)
  def test_litert_lm_server_ipv4(self, mock_super_init):
    serve_util.LiteRTLMServer(("127.0.0.1", 8000), mock.MagicMock())
    mock_super_init.assert_called_once()
    args, _ = mock_super_init.call_args
    self_arg, _, _ = args
    self.assertEqual(
        getattr(self_arg, "address_family", socket.AF_INET), socket.AF_INET
    )

  def test_build_name_by_tool_call_id_map(self):
    messages = [
        {"role": "user", "content": "What is the weather in London?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "London"}',
                },
            }],
        },
    ]

    # Build mapping.
    name_by_tool_call_id = openai_handler._build_name_by_tool_call_id_map(
        messages
    )
    self.assertEqual(name_by_tool_call_id, {"call_123": "get_weather"})

  def test_translate_openai_message_tool_resolution(self):
    message = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Weather in London is sunny.",
    }
    name_by_tool_call_id = {"call_123": "get_weather"}

    # Translate the tool message.
    translated = openai_handler._translate_openai_message(
        message, name_by_tool_call_id
    )

    expected = {
        "role": "tool",
        "content": [{
            "type": "tool_response",
            "name": "get_weather",
            "response": "Weather in London is sunny.",
        }],
    }
    self.assertEqual(translated, expected)

  def test_translate_openai_message_tool_resolution_unknown_name(self):
    message = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Weather in London is sunny.",
    }
    # Empty mapping to trigger failure.
    name_by_tool_call_id = {}

    # Translate the tool message should raise ValueError.
    with self.assertRaisesRegex(
        ValueError, "No matching tool call found for tool_call_id"
    ):
      openai_handler._translate_openai_message(message, name_by_tool_call_id)

  def test_translate_openai_message_tool_resolution_missing_tool_call_id(self):
    message = {
        "role": "tool",
        "content": "Weather in London is sunny.",
    }
    name_by_tool_call_id = {"call_123": "get_weather"}

    with self.assertRaisesRegex(
        ValueError, "Tool message must have a tool_call_id"
    ):
      openai_handler._translate_openai_message(message, name_by_tool_call_id)

  def test_translate_openai_message_tool_resolution_none_mapping(self):
    message = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Weather in London is sunny.",
    }

    with self.assertRaisesRegex(
        ValueError, "No matching tool call found for tool_call_id"
    ):
      openai_handler._translate_openai_message(message, None)

  def test_cors_headers_disabled_by_default(self):
    server = serve_util.LiteRTLMServer(
        ("127.0.0.1", 0), openai_handler.OpenAIHandler
    )
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # Test OPTIONS
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          method="OPTIONS",
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Methods"))

      # Test GET
      mock_model_mod.Model.get_all_models.return_value = []
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/models", method="GET"
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    finally:
      server.shutdown()
      thread.join()

  def test_cors_headers_wildcard(self):
    server = serve_util.LiteRTLMServer(
        ("127.0.0.1", 0), openai_handler.OpenAIHandler, allowed_origins=("*",)
    )
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # Test OPTIONS
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          method="OPTIONS",
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Methods"),
            "GET, POST, OPTIONS",
        )

      # Test GET
      mock_model_mod.Model.get_all_models.return_value = []
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/models", method="GET"
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")

    finally:
      server.shutdown()
      thread.join()

  def test_cors_headers_restricted(self):
    allowed = ("http://localhost:3000", "http://example.com")
    server = serve_util.LiteRTLMServer(
        ("127.0.0.1", 0), openai_handler.OpenAIHandler, allowed_origins=allowed
    )
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # Test matched origin
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/models",
          method="GET",
          headers={"Origin": "http://localhost:3000"},
      )
      mock_model_mod.Model.get_all_models.return_value = []
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:3000",
        )
        self.assertEqual(resp.headers.get("Vary"), "Origin")
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Methods"),
            "GET, POST, OPTIONS",
        )
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Headers"),
            "Content-Type, Authorization, X-Requested-With",
        )

      # Test unmatched origin
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/models",
          method="GET",
          headers={"Origin": "http://evil.com"},
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    finally:
      server.shutdown()
      thread.join()


if __name__ == "__main__":
  absltest.main()
