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
import json
import socket
import sys
import threading
from typing import Any
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
# pylint: disable=g-import-not-at-top
import litert_lm as mock_litert_lm
from litert_lm import interfaces

# 3. Explicitly override Engine and other classes with Mocks to ensure they
# don't point to the mocked extension's classes which might not behave like
# standard mocks.
mock_litert_lm.Engine = mock_engine.Engine
mock_litert_lm.set_min_log_severity = mock_ffi.set_min_log_severity

mock_model_mod = mock.Mock(
    spec_set=["Model", "parse_backend", "resolve_config_option"]
)
mock_model_mod.Model = mock.Mock(
    spec_set=[
        "from_model_id",
        "from_model_path",
        "from_model_reference",
        "get_all_models",
    ]
)


def _from_model_reference(ref: Any) -> Any:
  return mock_model_mod.Model.from_model_id(ref)


mock_model_mod.Model.from_model_id = mock.Mock()
mock_model_mod.Model.from_model_path = mock.Mock()
mock_model_mod.Model.from_model_reference = mock.Mock(
    side_effect=_from_model_reference
)
mock_model_mod.Model.get_all_models = mock.Mock()
mock_model_mod.parse_backend = mock.Mock()
mock_model_mod.resolve_config_option = mock.Mock(
    side_effect=lambda value, model_obj, config_key, label=None: value
)
sys.modules["litert_lm_cli.model"] = (
    mock_model_mod
)
if "litert_lm_cli" in sys.modules:
  sys.modules[  # pyrefly: ignore[missing-attribute]
      "litert_lm_cli"
  ].model = mock_model_mod

from litert_lm_cli.commands.serve import gemini_handler
from litert_lm_cli.commands.serve import openai_chat_completions
from litert_lm_cli.commands.serve import openai_common
from litert_lm_cli.commands.serve import openai_embeddings
from litert_lm_cli.commands.serve import openai_handler
from litert_lm_cli.commands.serve import openai_models
from litert_lm_cli.commands.serve import openai_responses
from litert_lm_cli.commands.serve import util

# pylint: enable=g-import-not-at-top


class ServeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Reset mocks.
    mock_litert_lm.set_min_log_severity.reset_mock()  # pyrefly: ignore[missing-attribute]
    mock_litert_lm.Engine.reset_mock()  # pyrefly: ignore[missing-attribute]
    mock_litert_lm.Engine.side_effect = None  # pyrefly: ignore[missing-attribute]
    mock_model_mod.Model.from_model_id.reset_mock()
    mock_model_mod.Model.from_model_id.side_effect = None
    mock_model_mod.Model.from_model_reference.reset_mock()
    mock_model_mod.Model.from_model_reference.side_effect = (
        _from_model_reference
    )
    mock_model_mod.Model.get_all_models.reset_mock()
    mock_model_mod.Model.get_all_models.side_effect = None
    mock_model_mod.parse_backend.reset_mock()
    mock_model_mod.parse_backend.return_value = interfaces.Backend.CPU()
    mock_model_mod.resolve_config_option.reset_mock()
    mock_model_mod.resolve_config_option.side_effect = (
        lambda value, model_obj, config_key, label=None: value
    )

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
          litertlm_response=mock_litert_lm.Message.model(
              mock_litert_lm.Contents.of("Response text")
          ),
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
          litertlm_response=mock_litert_lm.Message.model(
              tool_calls=[
                  mock_litert_lm.ToolCall(
                      name="get_weather",
                      arguments={"location": "London"},
                  )
              ]
          ),
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
          litertlm_response=mock_litert_lm.Message.model(
              mock_litert_lm.Contents.of("Chunk")
          ),
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
          litertlm_response=mock_litert_lm.Message.model(
              mock_litert_lm.Contents.of("Text")
          ),
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

    server = mock.MagicMock(spec=util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None

    # First call creates the engine.
    engine1 = util.get_or_initialize_server_engine(
        server, model_id="test-model"
    )
    self.assertEqual(engine1, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()  # pyrefly: ignore[missing-attribute]
    self.assertEqual(server.litert_lm_engine, mock_engine_instance)
    self.assertEqual(server.model_id, "test-model")

    # Second call with same ID - returns cached engine.
    engine2 = util.get_or_initialize_server_engine(
        server, model_id="test-model"
    )
    self.assertEqual(engine2, mock_engine_instance)
    self.assertEqual(mock_litert_lm.Engine.call_count, 1)  # pyrefly: ignore[missing-attribute]

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

    server = mock.MagicMock(spec=util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None
    server.activation_data_type = None

    # Initialize with model A.
    engine1 = util.get_or_initialize_server_engine(server, model_id="A")
    self.assertEqual(engine1, mock_engine_a)
    self.assertEqual(server.model_id, "A")
    mock_engine_a.__exit__.assert_not_called()

    # Switching to model B re-initializes (closes A, opens B).
    engine2 = util.get_or_initialize_server_engine(server, model_id="B")
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
    util.LiteRTLMServer(("::1", 8000), mock.MagicMock())
    mock_super_init.assert_called_once()
    args, _ = mock_super_init.call_args
    self_arg, _, _ = args
    self.assertEqual(self_arg.address_family, socket.AF_INET6)

  @mock.patch.object(http.server.HTTPServer, "__init__", autospec=True)
  def test_litert_lm_server_ipv4(self, mock_super_init):
    util.LiteRTLMServer(("127.0.0.1", 8000), mock.MagicMock())
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
    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
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
    server = util.LiteRTLMServer(
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
    server = util.LiteRTLMServer(
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

  def test_parse_response_format_valid(self):
    self.assertIsNone(openai_handler._parse_response_format({}))
    self.assertIsNone(
        openai_handler._parse_response_format({"response_format": None})
    )
    self.assertIsNone(
        openai_handler._parse_response_format(
            {"response_format": {"type": "text"}}
        )
    )

    rf_json = openai_handler._parse_response_format(
        {"response_format": {"type": "json_object"}}
    )
    self.assertIsNotNone(rf_json)
    self.assertEqual(rf_json.type, interfaces.ResponseFormat.Type.JSON_OBJECT)
    self.assertEqual(rf_json.schema_or_pattern, "{}")

    schema_dict = {"type": "object", "properties": {"a": {"type": "string"}}}
    rf_json_schema = openai_handler._parse_response_format({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": schema_dict},
        }
    })
    self.assertIsNotNone(rf_json_schema)
    self.assertEqual(
        rf_json_schema.type, interfaces.ResponseFormat.Type.JSON_OBJECT
    )

    rf_regex = openai_handler._parse_response_format(
        {"response_format": {"type": "regex", "pattern": "[0-9]{3}"}}
    )
    self.assertIsNotNone(rf_regex)
    self.assertEqual(rf_regex.type, interfaces.ResponseFormat.Type.REGEX)
    self.assertEqual(rf_regex.schema_or_pattern, "[0-9]{3}")

  @parameterized.named_parameters(
      dict(
          testcase_name="not_a_dict",
          body={"response_format": "invalid"},
          err_msg="response_format must be a dict",
      ),
      dict(
          testcase_name="unsupported_type",
          body={"response_format": {"type": "unsupported"}},
          err_msg="Unsupported response_format type",
      ),
      dict(
          testcase_name="missing_json_schema",
          body={"response_format": {"type": "json_schema"}},
          err_msg="json_schema response_format requires a dict or str schema",
      ),
      dict(
          testcase_name="missing_regex_pattern",
          body={"response_format": {"type": "regex"}},
          err_msg="regex response_format requires a string pattern/regex",
      ),
  )
  def test_parse_response_format_invalid(self, body, err_msg):
    with self.assertRaisesRegex(ValueError, err_msg):
      openai_handler._parse_response_format(body)

  @parameterized.named_parameters(
      dict(
          testcase_name="model_only",
          input_str="gemma3-1b",
          expected=("gemma3-1b", None, None),
      ),
      dict(
          testcase_name="model_and_backend",
          input_str="gemma3-1b,gpu",
          expected=("gemma3-1b", "gpu", None),
      ),
      dict(
          testcase_name="model_backend_and_tokens",
          input_str="gemma3-1b,gpu,32768",
          expected=("gemma3-1b", "gpu", 32768),
      ),
      dict(
          testcase_name="model_and_tokens_default_backend",
          input_str="gemma3-1b,,32768",
          expected=("gemma3-1b", None, 32768),
      ),
      dict(
          testcase_name="trailing_comma",
          input_str="gemma3-1b,gpu,",
          expected=("gemma3-1b", "gpu", None),
      ),
      dict(
          testcase_name="whitespace_stripping",
          input_str="gemma-2-2b-it, CPU , 2048 ",
          expected=("gemma-2-2b-it", "CPU", 2048),
      ),
  )
  def test_parse_model_parameter_valid(self, input_str, expected):
    self.assertEqual(openai_handler._parse_model_parameter(input_str), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_model_id",
          input_str="",
          err_msg="model_id cannot be empty",
      ),
      dict(
          testcase_name="not_a_string",
          input_str=123,
          err_msg="model parameter must be a string",
      ),
      dict(
          testcase_name="invalid_max_tokens",
          input_str="gemma3-1b,gpu,notanint",
          err_msg="Invalid max_num_tokens",
      ),
      dict(
          testcase_name="non_positive_max_tokens",
          input_str="gemma3-1b,gpu,0",
          err_msg="max_num_tokens must be a positive integer",
      ),
      dict(
          testcase_name="negative_max_tokens",
          input_str="gemma3-1b,gpu,-100",
          err_msg="max_num_tokens must be a positive integer",
      ),
      dict(
          testcase_name="too_many_parts",
          input_str="gemma3-1b,gpu,32768,extra",
          err_msg="Too many comma-separated components",
      ),
  )
  def test_parse_model_parameter_invalid(self, input_str, err_msg):
    with self.assertRaisesRegex(ValueError, err_msg):
      openai_handler._parse_model_parameter(input_str)

  def test_get_engine_backend_and_max_tokens_override(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma3-1b"
    mock_m.model_id = "gemma3-1b"

    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_engine_instance = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_instance.__enter__.return_value = mock_engine_instance
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = mock.MagicMock(spec=util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None
    server.vision_backend = None
    server.audio_backend = None
    server.activation_data_type = None

    engine = util.get_or_initialize_server_engine(
        server, model_id="gemma3-1b", backend="gpu", max_num_tokens=32768
    )
    self.assertEqual(engine, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()  # pyrefly: ignore[missing-attribute]
    _, kwargs = mock_litert_lm.Engine.call_args  # pyrefly: ignore[missing-attribute]
    self.assertEqual(kwargs.get("max_num_tokens"), 32768)
    self.assertTrue(kwargs.get("use_ringbuffers_local_attention"))
    self.assertEqual(server.max_num_tokens, 32768)

  def test_get_engine_activation_data_type_from_config(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma3-1b"
    mock_m.model_id = "gemma3-1b"

    mock_model_mod.Model.from_model_id.return_value = mock_m
    mock_model_mod.resolve_config_option.side_effect = (
        lambda value, model_obj, config_key, label=None: (
            "fp16" if config_key == "activation_data_type" else value
        )
    )
    mock_litert_lm.ActivationDataType.from_str.side_effect = (  # pyrefly: ignore[missing-attribute]
        lambda s: mock_litert_lm.ActivationDataType.FLOAT16
        if s == "fp16"
        else None
    )

    mock_engine_instance = mock.MagicMock(spec=interfaces.AbstractEngine)
    mock_engine_instance.__enter__.return_value = mock_engine_instance
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = mock.MagicMock(spec=util.LiteRTLMServer)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None
    server.vision_backend = None
    server.audio_backend = None
    server.activation_data_type = None

    engine = util.get_or_initialize_server_engine(server, model_id="gemma3-1b")
    self.assertEqual(engine, mock_engine_instance)
    mock_litert_lm.Engine.assert_called_once()  # pyrefly: ignore[missing-attribute]
    _, kwargs = mock_litert_lm.Engine.call_args  # pyrefly: ignore[missing-attribute]
    self.assertEqual(
        kwargs.get("activation_data_type"),
        mock_litert_lm.ActivationDataType.FLOAT16,
    )

  def test_normalize_embedding_input(self):
    self.assertEqual(
        openai_handler._normalize_embedding_input("hello"), ["hello"]
    )
    self.assertEqual(
        openai_handler._normalize_embedding_input(["hello", "world"]),
        ["hello", "world"],
    )
    self.assertEqual(
        openai_handler._normalize_embedding_input([101, 2045]),
        [["101", "2045"]],
    )
    self.assertEqual(
        openai_handler._normalize_embedding_input(
            {"type": "text", "text": "hello"}
        ),
        ["hello"],
    )
    self.assertEqual(
        openai_handler._normalize_embedding_input(
            [{"type": "text", "text": "hello"}]
        ),
        [["hello"]],
    )

    with self.assertRaisesRegex(ValueError, "input cannot be empty"):
      openai_handler._normalize_embedding_input("")

    with self.assertRaisesRegex(ValueError, "input cannot be empty"):
      openai_handler._normalize_embedding_input([])

  def test_embedding_base64_and_l2_normalize(self):
    vec = [3.0, 4.0]
    normalized = openai_handler._l2_normalize(vec)
    self.assertAlmostEqual(normalized[0], 0.6)
    self.assertAlmostEqual(normalized[1], 0.8)

    b64_str = openai_handler._embedding_to_base64([1.0, 2.0])
    self.assertIsInstance(b64_str, str)
    self.assertNotEmpty(b64_str)

  def test_openai_embeddings_success(self):
    endpoint = "/v1/embeddings"
    mock_emb_engine = mock.MagicMock()
    mock_emb_engine.compute_embedding_batch.return_value = [
        mock.MagicMock(embedding=[0.1, 0.2, 0.3]),
        mock.MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_get_engine = self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler, "_get_embedding_engine", autospec=True
        )
    )
    mock_get_engine.return_value = mock_emb_engine

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps({
          "model": "embedding-gemma",
          "input": ["hello", "world"],
      }).encode("utf-8")

      req = urllib.request.Request(
          f"http://127.0.0.1:{port}{endpoint}",
          data=data,
          headers={"Content-Type": "application/json"},
      )

      with urllib.request.urlopen(req) as response:
        self.assertEqual(response.getcode(), 200)
        res_body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(res_body["object"], "list")
        self.assertEqual(res_body["model"], "embedding-gemma")
        self.assertLen(res_body["data"], 2)
        self.assertEqual(res_body["data"][0]["object"], "embedding")
        self.assertEqual(res_body["data"][0]["index"], 0)
        self.assertEqual(res_body["data"][0]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(res_body["data"][1]["index"], 1)
        self.assertEqual(res_body["data"][1]["embedding"], [0.4, 0.5, 0.6])
        self.assertIn("usage", res_body)
    finally:
      server.shutdown()
      thread.join()

  def test_openai_embeddings_base64_and_dimensions(self):
    mock_emb_engine = mock.MagicMock()
    mock_emb_engine.compute_embedding_batch.return_value = [
        mock.MagicMock(embedding=[3.0, 4.0, 0.0]),
    ]
    mock_get_engine = self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler, "_get_embedding_engine", autospec=True
        )
    )
    mock_get_engine.return_value = mock_emb_engine

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps({
          "model": "embedding-gemma",
          "input": "hello",
          "encoding_format": "base64",
          "dimensions": 2,
      }).encode("utf-8")

      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/embeddings",
          data=data,
          headers={"Content-Type": "application/json"},
      )

      with urllib.request.urlopen(req) as response:
        self.assertEqual(response.getcode(), 200)
        res_body = json.loads(response.read().decode("utf-8"))
        emb = res_body["data"][0]["embedding"]
        self.assertIsInstance(emb, str)
        # Expected vector after slicing to 2 and L2-normalizing [3.0, 4.0] is
        # [0.6, 0.8].
        expected_b64 = openai_handler._embedding_to_base64([0.6, 0.8])
        self.assertEqual(emb, expected_b64)
    finally:
      server.shutdown()
      thread.join()

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_model",
          body={"input": "hello"},
          err_code=400,
      ),
      dict(
          testcase_name="missing_input",
          body={"model": "gemma"},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_encoding_format",
          body={"model": "gemma", "input": "hello", "encoding_format": "xml"},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_dimensions",
          body={"model": "gemma", "input": "hello", "dimensions": -5},
          err_code=400,
      ),
      dict(
          testcase_name="non_int_dimensions",
          body={"model": "gemma", "input": "hello", "dimensions": "invalid"},
          err_code=400,
      ),
      dict(
          testcase_name="bool_dimensions",
          body={"model": "gemma", "input": "hello", "dimensions": True},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_model_parameter",
          body={"model": "gemma,gpu,-1", "input": "hello"},
          err_code=400,
      ),
  )
  def test_openai_embeddings_errors(self, body, err_code):
    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps(body).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/embeddings",
          data=data,
          headers={"Content-Type": "application/json"},
      )

      with self.assertRaises(urllib.error.HTTPError) as cm:
        urllib.request.urlopen(req)
      self.assertEqual(cm.exception.code, err_code)
    finally:
      server.shutdown()
      thread.join()

  def test_openai_embeddings_dimensions_exceed_vector_and_inference_error(self):
    mock_emb_engine = mock.MagicMock()
    mock_emb_engine.compute_embedding_batch.return_value = [
        mock.MagicMock(embedding=[0.1, 0.2, 0.3]),
    ]
    mock_get_engine = self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler, "_get_embedding_engine", autospec=True
        )
    )
    mock_get_engine.return_value = mock_emb_engine

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps({
          "model": "embedding-gemma",
          "input": "hello",
          "dimensions": 10,
      }).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/embeddings",
          data=data,
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as cm:
        urllib.request.urlopen(req)
      self.assertEqual(cm.exception.code, 400)

      mock_emb_engine.compute_embedding_batch.side_effect = RuntimeError("boom")
      data_ok = json.dumps({
          "model": "embedding-gemma",
          "input": "hello",
      }).encode("utf-8")
      req_err = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/embeddings",
          data=data_ok,
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as cm_err:
        urllib.request.urlopen(req_err)
      self.assertEqual(cm_err.exception.code, 500)
    finally:
      server.shutdown()
      thread.join()

  def test_handle_responses_non_streaming_headers_and_body(self):
    handler = mock.MagicMock()
    handler.headers_sent = False
    conv = mock.MagicMock()
    conv.send_message.return_value = mock_litert_lm.Message.from_json(
        {"content": [{"type": "text", "text": "Hello from responses"}]}
    )

    openai_responses._handle_responses(
        handler,
        conv,
        "Say hello",
        False,
        now_str="20260918",
        created_ts=1234567890,
        model_id="gemma3",
    )

    self.assertTrue(handler.headers_sent)
    handler.send_response.assert_called_once_with(200)
    handler.send_header.assert_called_once_with(
        "Content-Type", "application/json"
    )
    handler.end_headers.assert_called_once_with()
    written_bytes = handler.wfile.write.call_args[0][0]
    parsed = json.loads(written_bytes.decode("utf-8"))
    self.assertEqual(parsed["id"], "resp_20260918")
    self.assertEqual(
        parsed["output"][0]["content"][0]["text"], "Hello from responses"
    )

  def test_openai_responses_non_streaming_and_streaming(self):
    mock_conv = mock.MagicMock()
    mock_conv.__enter__.return_value = mock_conv
    mock_conv.__exit__.return_value = False
    mock_conv.send_message.return_value = mock_litert_lm.Message.from_json(
        {"content": [{"type": "text", "text": "Non-streaming answer"}]}
    )
    mock_conv.send_message_async.return_value = [
        mock_litert_lm.Message.from_json(
            {"content": [{"type": "text", "text": "Streamed "}]}
        ),
        mock_litert_lm.Message.from_json(
            {"content": [{"type": "text", "text": "answer"}]}
        ),
    ]
    mock_engine_instance = mock.MagicMock()
    mock_engine_instance.create_conversation.return_value = mock_conv
    self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler,
            "_get_engine",
            return_value=mock_engine_instance,
        )
    )

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # 1. Non-streaming with dict input
      data = json.dumps({
          "model": "gemma3",
          "input": {"role": "user", "content": "Hello"},
          "stream": False,
      }).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/responses",
          data=data,
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(
            body["output"][0]["content"][0]["text"], "Non-streaming answer"
        )

      # 2. Streaming
      data_stream = json.dumps({
          "model": "gemma3",
          "input": "Hello",
          "stream": True,
      }).encode("utf-8")
      req_stream = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/responses",
          data=data_stream,
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req_stream) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        raw = resp.read().decode("utf-8")
        self.assertIn("event: response.created", raw)
        self.assertIn("event: response.output_text.delta", raw)
        self.assertIn("event: response.completed", raw)
        self.assertIn("data: [DONE]", raw)
    finally:
      server.shutdown()
      thread.join()

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_model",
          body={"input": "hello"},
          err_code=400,
      ),
      dict(
          testcase_name="missing_input",
          body={"model": "gemma3"},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_model_parameter",
          body={"model": "gemma3,gpu,-1", "input": "hello"},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_dict_input",
          body={
              "model": "gemma3",
              "input": {"role": "tool", "tool_call_id": "missing"},
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_thinking_config",
          body={
              "model": "gemma3",
              "input": "hello",
              "reasoning_effort": "bogus",
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_response_format",
          body={
              "model": "gemma3",
              "input": "hello",
              "response_format": "bogus",
          },
          err_code=400,
      ),
  )
  def test_openai_responses_errors(self, body, err_code):
    mock_engine_instance = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler,
            "_get_engine",
            return_value=mock_engine_instance,
        )
    )
    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps(body).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/responses",
          data=data,
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as cm:
        urllib.request.urlopen(req)
      self.assertEqual(cm.exception.code, err_code)
    finally:
      server.shutdown()
      thread.join()

  def test_openai_chat_completions_formatter_and_proxy_tool(self):
    formatter = openai_chat_completions._OpenAIChatCompletionsFormatter(
        now_str="20260918",
        created_ts=1234567890,
        model_id="gemma3",
        include_usage=True,
    )
    tool_delta_bytes = formatter.format_tool_call_delta([{
        "function": {
            "name": "get_weather",
            "arguments": {"location": "NYC"},
        }
    }])
    tool_delta_str = tool_delta_bytes.decode("utf-8")
    self.assertIn("data: ", tool_delta_str)
    payload = json.loads(tool_delta_str[len("data: ") :])
    self.assertEqual(
        payload["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather",
    )

    usage_bytes = formatter.format_usage(
        {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    )
    self.assertIn('"total_tokens": 8', usage_bytes.decode("utf-8"))

    tool_def = {"type": "function", "function": {"name": "fn"}}
    proxy_tool = openai_chat_completions._ProxyTool(tool_def)
    self.assertEqual(proxy_tool.get_tool_description(), tool_def)
    with self.assertRaises(NotImplementedError):
      proxy_tool.execute({})

  def test_openai_chat_completions_non_streaming_and_streaming(self):
    mock_conv = mock.MagicMock()
    mock_conv.__enter__.return_value = mock_conv
    mock_conv.__exit__.return_value = False
    mock_conv.get_benchmark_info.return_value = mock.MagicMock(
        last_prefill_token_count=10, last_decode_token_count=5
    )
    mock_conv.send_message_async.return_value = [
        mock_litert_lm.Message.from_json(
            {"channels": {"thought": "thinking..."}}
        ),
        mock_litert_lm.Message.from_json(
            {"content": [{"type": "text", "text": "Hello world"}]}
        ),
        mock_litert_lm.Message.from_json({
            "tool_calls": [{
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "NYC"},
                }
            }]
        }),
    ]
    mock_engine_instance = mock.MagicMock()
    mock_engine_instance.create_conversation.return_value = mock_conv
    self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler,
            "_get_engine",
            return_value=mock_engine_instance,
        )
    )

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # 1. Non-streaming with tools and input fallback (no messages array)
      data = json.dumps({
          "model": "gemma3",
          "input": {"role": "user", "content": "Weather in NYC?"},
          "tools": [{"type": "function", "function": {"name": "get_weather"}}],
          "stream": False,
      }).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=data,
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )

      # 2. Streaming with messages array and include_usage
      data_stream = json.dumps({
          "model": "gemma3",
          "messages": [{"role": "user", "content": "Hi"}],
          "stream": True,
          "stream_options": {"include_usage": True},
      }).encode("utf-8")
      req_stream = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=data_stream,
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req_stream) as resp:
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        raw = resp.read().decode("utf-8")
        self.assertIn("Hello world", raw)
        self.assertIn("get_weather", raw)
        self.assertIn("data: [DONE]", raw)
    finally:
      server.shutdown()
      thread.join()

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_model",
          body={"messages": [{"role": "user", "content": "hi"}]},
          err_code=400,
      ),
      dict(
          testcase_name="invalid_model_parameter",
          body={
              "model": "gemma3,gpu,-1",
              "messages": [{"role": "user", "content": "hi"}],
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_messages",
          body={
              "model": "gemma3",
              "messages": [{"role": "tool", "tool_call_id": "missing"}],
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_input_dict",
          body={
              "model": "gemma3",
              "input": {"role": "tool", "tool_call_id": "missing"},
          },
          err_code=400,
      ),
      dict(
          testcase_name="missing_messages_and_input",
          body={"model": "gemma3"},
          err_code=400,
      ),
      dict(
          testcase_name="bool_max_completion_tokens",
          body={
              "model": "gemma3",
              "messages": [{"role": "user", "content": "hi"}],
              "max_completion_tokens": True,
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_sampler_config",
          body={
              "model": "gemma3",
              "messages": [{"role": "user", "content": "hi"}],
              "temperature": -1.0,
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_thinking_config",
          body={
              "model": "gemma3",
              "messages": [{"role": "user", "content": "hi"}],
              "reasoning_effort": "bogus",
          },
          err_code=400,
      ),
      dict(
          testcase_name="invalid_response_format",
          body={
              "model": "gemma3",
              "messages": [{"role": "user", "content": "hi"}],
              "response_format": "bogus",
          },
          err_code=400,
      ),
  )
  def test_openai_chat_completions_errors(self, body, err_code):
    mock_engine_instance = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler,
            "_get_engine",
            return_value=mock_engine_instance,
        )
    )
    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      data = json.dumps(body).encode("utf-8")
      req = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=data,
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as cm:
        urllib.request.urlopen(req)
      self.assertEqual(cm.exception.code, err_code)
    finally:
      server.shutdown()
      thread.join()

  def test_openai_handler_headers_sent_and_inference_error(self):
    handler = object.__new__(openai_handler.OpenAIHandler)
    handler._headers_sent = False
    handler.wfile = mock.MagicMock()
    handler.wfile.closed = False
    handler.send_error = mock.MagicMock()

    self.assertFalse(handler.headers_sent)
    handler.headers_sent = True
    self.assertTrue(handler.headers_sent)

    # When headers_sent is False, handle_inference_error sends 500
    handler.headers_sent = False
    handler.handle_inference_error(RuntimeError("failure"), "gemma3", "hi")
    handler.send_error.assert_called_once()
    self.assertEqual(handler.send_error.call_args[0][0], 500)

    # When headers_sent is True, handle_inference_error does not call send_error
    handler.send_error.reset_mock()
    handler.headers_sent = True
    handler.handle_inference_error(RuntimeError("failure"), "gemma3", "hi")
    handler.send_error.assert_not_called()

  def test_get_or_initialize_server_embedding_engine(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/embedding-model"
    mock_m.model_id = "embedding-model"

    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_emb_engine_instance = mock.MagicMock()
    with mock.patch.object(
        mock_litert_lm, "EmbeddingEngine", return_value=mock_emb_engine_instance
    ) as mock_emb_engine_cls:
      server = mock.MagicMock(spec=util.LiteRTLMServer)
      server.litert_lm_embedding_engine = None
      server.embedding_model_id = None
      server.embedding_backend = None
      server.embedding_vision_backend = None
      server.embedding_audio_backend = None

      # First call initializes engine
      engine1 = util.get_or_initialize_server_embedding_engine(
          server, model_id="embedding-model"
      )
      self.assertEqual(engine1, mock_emb_engine_instance)
      mock_emb_engine_cls.assert_called_once()

      # Second call returns cached engine
      engine2 = util.get_or_initialize_server_embedding_engine(
          server, model_id="embedding-model"
      )
      self.assertEqual(engine2, mock_emb_engine_instance)
      self.assertEqual(mock_emb_engine_cls.call_count, 1)

  def test_openai_models_listing_and_errors(self):
    m1 = mock.Mock(spec_set=["model_id", "model_path"])
    m1.model_id = "m1"
    m1.model_path = "/path/to/m1"
    m2 = mock.Mock(spec_set=["model_id", "model_path"])
    m2.model_id = "m2"
    m2.model_path = "/path/to/m2"
    mock_model_mod.Model.get_all_models.return_value = [m1, m2]

    def _getmtime_side_effect(p):
      if p == "/path/to/m1":
        return 1000
      raise OSError("missing")

    self.enter_context(
        mock.patch.object(
            openai_models.os.path, "getmtime", side_effect=_getmtime_side_effect
        )
    )

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
      with urllib.request.urlopen(req) as resp:
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(body["data"][0]["created"], 1000)
        self.assertEqual(body["data"][1]["created"], 0)

      # GET 404
      with self.assertRaises(urllib.error.HTTPError) as cm_get:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/unknown")
      self.assertEqual(cm_get.exception.code, 404)

      # POST 404
      req_post_404 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/unknown", data=b"{}"
      )
      with self.assertRaises(urllib.error.HTTPError) as cm_post:
        urllib.request.urlopen(req_post_404)
      self.assertEqual(cm_post.exception.code, 404)

      # POST invalid JSON -> 400
      req_bad_json = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/responses",
          data=b"not-json",
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as cm_json:
        urllib.request.urlopen(req_bad_json)
      self.assertEqual(cm_json.exception.code, 400)

      # GET /v1/models exception -> 500
      mock_model_mod.Model.get_all_models.side_effect = RuntimeError("fail")
      with self.assertRaises(urllib.error.HTTPError) as cm_500:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models")
      self.assertEqual(cm_500.exception.code, 500)
    finally:
      mock_model_mod.Model.get_all_models.side_effect = None
      server.shutdown()
      server.server_close()
      thread.join()

  def test_openai_embeddings_multimodal_content_parts(self):
    self.assertEqual(openai_embeddings.l2_normalize([0.0, 0.0]), [0.0, 0.0])
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        })
    )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "image_url", "image_url": {"url": "file:///tmp/test.png"}}
        )
    )
    with self.assertRaises(ValueError):
      openai_embeddings._parse_embedding_content_part(
          {"type": "image_url", "image_url": {"url": "data:image/png,raw"}}
      )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "input_audio", "input_audio": {"data": "aGVsbG8="}}
        )
    )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "image", "blob": "aGVsbG8="}
        )
    )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "image", "path": "/tmp/a.png"}
        )
    )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "audio", "blob": "aGVsbG8="}
        )
    )
    self.assertIsNotNone(
        openai_embeddings._parse_embedding_content_part(
            {"type": "audio", "path": "/tmp/a.wav"}
        )
    )
    with self.assertRaises(ValueError):
      openai_embeddings._parse_embedding_content_part({"type": "unknown"})
    with self.assertRaises(ValueError):
      openai_embeddings._parse_embedding_content_part(12.34)
    self.assertEqual(
        openai_embeddings.normalize_embedding_input([[1, 2], [3, 4]]),
        [["1", "2"], ["3", "4"]],
    )
    with self.assertRaises(ValueError):
      openai_embeddings.normalize_embedding_input(12345)

  def test_openai_common_multimodal_and_thinking(self):
    self.assertIsNotNone(
        openai_common.parse_thinking_config({"reasoning_effort": "none"})
    )
    for effort in ("minimal", "low", "medium", "high", "xhigh"):
      self.assertIsNotNone(
          openai_common.parse_thinking_config({"reasoning_effort": effort})
      )
    with self.assertRaises(ValueError):
      openai_common.parse_thinking_config({"reasoning_effort": 123})

    assistant_msg = {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }],
    }
    translated_assistant = openai_common.translate_openai_message(assistant_msg)
    self.assertEqual(
        translated_assistant["tool_calls"][0]["function"]["arguments"],
        {"q": "test"},
    )

    multimodal_msg = {
        "role": "user",
        "content": [
            "raw_string_part",
            {"type": "text", "text": "Look at this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
            {
                "type": "image_url",
                "image_url": {"url": "file:///tmp/test.png"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "aGVsbG8="},
            },
            {"type": "custom", "value": 42},
        ],
    }
    translated_user = openai_common.translate_openai_message(multimodal_msg)
    self.assertLen(translated_user["content"], 6)
    self.assertEqual(
        translated_user["content"][2], {"type": "image", "blob": "aGVsbG8="}
    )
    self.assertEqual(
        translated_user["content"][3],
        {"type": "image", "path": "/tmp/test.png"},
    )
    self.assertEqual(
        translated_user["content"][4], {"type": "audio", "blob": "aGVsbG8="}
    )

    with self.assertRaises(ValueError):
      openai_common.translate_openai_message({
          "role": "user",
          "content": [{
              "type": "image_url",
              "image_url": {"url": "data:image/png,notbase64"},
          }],
      })

  def test_openai_chat_completions_conversation_cache_hit_non_streaming(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma"
    mock_m.model_id = "gemma"
    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_engine_instance = mock.MagicMock()
    mock_conv = mock.MagicMock()
    mock_conv.get_benchmark_info.return_value.last_prefill_token_count = 10
    mock_conv.get_benchmark_info.return_value.last_decode_token_count = 5
    type(mock_conv).token_count = mock.PropertyMock(side_effect=[15, 30])
    mock_engine_instance.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_conv.send_message_async.side_effect = [
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            })
        ]),
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "I am doing great!"}],
            })
        ]),
    ]
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      # Turn 1: initial message
      req1 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "messages": [{"role": "user", "content": "Hello"}],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req1) as response:
        self.assertEqual(response.getcode(), 200)
        res1 = json.loads(response.read().decode("utf-8"))
        self.assertEqual(res1["choices"][0]["message"]["content"], "Hi there!")
        self.assertEqual(res1["usage"]["prompt_tokens"], 10)
        self.assertEqual(
            res1["usage"]["prompt_tokens_details"], {"cached_tokens": 0}
        )

      self.assertEqual(mock_engine_instance.create_conversation.call_count, 1)
      self.assertIs(server.litert_lm_conversation, mock_conv)
      mock_conv.__exit__.assert_not_called()

      # Turn 2: continuing the same conversation
      req2 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "messages": [
                  {"role": "user", "content": "Hello"},
                  {"role": "assistant", "content": "Hi there!"},
                  {"role": "user", "content": "How are you?"},
              ],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req2) as response:
        self.assertEqual(response.getcode(), 200)
        res2 = json.loads(response.read().decode("utf-8"))
        self.assertEqual(
            res2["choices"][0]["message"]["content"], "I am doing great!"
        )
        self.assertEqual(res2["usage"]["prompt_tokens"], 25)
        self.assertEqual(
            res2["usage"]["prompt_tokens_details"], {"cached_tokens": 15}
        )

      # Reused cached conversation without creating a second one.
      self.assertEqual(mock_engine_instance.create_conversation.call_count, 1)
      self.assertEqual(mock_conv.send_message_async.call_count, 2)
      self.assertEqual(
          mock_conv.send_message_async.call_args_list[1].args[0],
          {"role": "user", "content": "How are you?"},
      )
    finally:
      server.close_conversation()
      server.shutdown()
      thread.join()

  def test_openai_chat_completions_conversation_cache_hit_streaming(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma"
    mock_m.model_id = "gemma"
    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_engine_instance = mock.MagicMock()
    mock_conv = mock.MagicMock()
    mock_conv.get_benchmark_info.return_value.last_prefill_token_count = 10
    mock_conv.get_benchmark_info.return_value.last_decode_token_count = 5
    mock_engine_instance.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_conv.send_message_async.side_effect = [
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "Streamed "}],
            }),
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "reply"}],
            }),
        ]),
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "Second streamed reply"}],
            })
        ]),
    ]
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      req1 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "stream": True,
              "messages": [{"role": "user", "content": "Hi"}],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req1) as response:
        self.assertEqual(response.getcode(), 200)
        _ = response.read().decode("utf-8")

      req2 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "stream": True,
              "messages": [
                  {"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Streamed reply"},
                  {"role": "user", "content": "Follow up"},
              ],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req2) as response:
        self.assertEqual(response.getcode(), 200)
        _ = response.read().decode("utf-8")

      self.assertEqual(mock_engine_instance.create_conversation.call_count, 1)
      self.assertEqual(mock_conv.send_message_async.call_count, 2)
    finally:
      server.close_conversation()
      server.shutdown()
      thread.join()

  def test_openai_chat_completions_conversation_cache_hit_tool_calling(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma"
    mock_m.model_id = "gemma"
    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_engine_instance = mock.MagicMock()
    mock_conv = mock.MagicMock()
    mock_conv.get_benchmark_info.return_value.last_prefill_token_count = 10
    mock_conv.get_benchmark_info.return_value.last_decode_token_count = 5
    mock_engine_instance.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_conv.send_message_async.side_effect = [
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Taipei"},
                    }
                }],
            })
        ]),
        iter([
            mock_litert_lm.Message.from_json({
                "role": "assistant",
                "content": [{"type": "text", "text": "It is sunny in Taipei."}],
            })
        ]),
    ]
    mock_litert_lm.Engine.return_value = mock_engine_instance

    tools_spec = [{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}},
    }]

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      req1 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "tools": tools_spec,
              "messages": [{"role": "user", "content": "Weather in Taipei?"}],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req1) as response:
        res1 = json.loads(response.read().decode("utf-8"))
        assistant_msg = res1["choices"][0]["message"]
        tc_id = assistant_msg["tool_calls"][0]["id"]

      req2 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "tools": tools_spec,
              "messages": [
                  {"role": "user", "content": "Weather in Taipei?"},
                  assistant_msg,
                  {
                      "role": "tool",
                      "tool_call_id": tc_id,
                      "content": "Sunny, 28C",
                  },
              ],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req2) as response:
        res2 = json.loads(response.read().decode("utf-8"))
        self.assertEqual(
            res2["choices"][0]["message"]["content"], "It is sunny in Taipei."
        )

      self.assertEqual(mock_engine_instance.create_conversation.call_count, 1)
      self.assertEqual(mock_conv.send_message_async.call_count, 2)
    finally:
      server.close_conversation()
      server.shutdown()
      thread.join()

  def test_openai_chat_completions_conversation_cache_miss_closes_old(self):
    mock_m = mock.Mock(spec_set=["exists", "model_path", "model_id"])
    mock_m.exists.return_value = True
    mock_m.model_path = "/path/to/gemma"
    mock_m.model_id = "gemma"
    mock_model_mod.Model.from_model_id.return_value = mock_m

    mock_engine_instance = mock.MagicMock()
    mock_conv1 = mock.MagicMock()
    mock_conv1.get_benchmark_info.return_value.last_prefill_token_count = 10
    mock_conv1.get_benchmark_info.return_value.last_decode_token_count = 5
    mock_conv2 = mock.MagicMock()
    mock_conv2.get_benchmark_info.return_value.last_prefill_token_count = 10
    mock_conv2.get_benchmark_info.return_value.last_decode_token_count = 5
    cm1 = mock.MagicMock()
    cm1.__enter__.return_value = mock_conv1
    cm2 = mock.MagicMock()
    cm2.__enter__.return_value = mock_conv2
    mock_engine_instance.create_conversation.side_effect = [cm1, cm2]

    mock_conv1.send_message_async.return_value = iter([
        mock_litert_lm.Message.from_json({
            "role": "assistant",
            "content": [{"type": "text", "text": "First"}],
        })
    ])
    mock_conv2.send_message_async.return_value = iter([
        mock_litert_lm.Message.from_json({
            "role": "assistant",
            "content": [{"type": "text", "text": "Second"}],
        })
    ])
    mock_litert_lm.Engine.return_value = mock_engine_instance

    server = util.LiteRTLMServer(("127.0.0.1", 0), openai_handler.OpenAIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
      req1 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "messages": [{"role": "user", "content": "Topic A"}],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req1) as response:
        self.assertEqual(response.getcode(), 200)

      req2 = urllib.request.Request(
          f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps({
              "model": "gemma",
              "messages": [{"role": "user", "content": "Unrelated Topic B"}],
          }).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req2) as response:
        self.assertEqual(response.getcode(), 200)

      self.assertEqual(mock_engine_instance.create_conversation.call_count, 2)
      mock_conv1.__exit__.assert_called_once_with(None, None, None)
      self.assertIs(server.litert_lm_conversation, mock_conv2)
    finally:
      server.close_conversation()
      server.shutdown()
      thread.join()


if __name__ == "__main__":
  absltest.main()
