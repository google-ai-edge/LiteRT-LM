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

import collections.abc
import http.client
import json
import pathlib
import threading
from unittest import mock
import urllib.error
import urllib.request

from absl.testing import absltest

import litert_lm
from litert_lm_cli import config as cli_config
from litert_lm_cli import model
from litert_lm_cli.commands import openai_handler
from litert_lm_cli.commands import serve_util


def _parse_sse_events(
    lines: collections.abc.Iterable[str],
) -> list[dict[str, str]]:
  events = []
  current_event = {}
  for line in lines:
    if line.startswith("event: "):
      current_event["event"] = line[len("event: ") :]
    elif line.startswith("data: "):
      current_event["data"] = line[len("data: ") :]
    elif not line and current_event:
      events.append(current_event)
      current_event = {}
  return events


class ServeOpenAIStreamingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.server = serve_util.LiteRTLMServer(
        ("localhost", 0), openai_handler.OpenAIHandler
    )
    self.port = self.server.server_port

    self.server_thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self.server_thread.start()

    self.model_path = (
        pathlib.Path(absltest.get_default_test_srcdir())
        / "google3/runtime/e2e_tests/data/gemma3-1b-it-int4.litertlm"
    )

  def tearDown(self):
    if (
        hasattr(self.server, "litert_lm_engine")
        and self.server.litert_lm_engine is not None
    ):
      try:
        self.server.litert_lm_engine.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self.server.litert_lm_engine = None
    self.server.shutdown()
    self.server.server_close()
    self.server_thread.join()
    super().tearDown()

  def test_openai_responses_streaming(self):

    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps(
        {"model": "gemma3", "input": "Say hi", "stream": True}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      self.assertEqual(response.getheader("Content-Type"), "text/event-stream")

      lines = response.read().decode("utf-8").split("\n")

      events = _parse_sse_events(lines)

      self.assertNotEmpty(events)

      with self.subTest(name="Verify created event"):
        created_event = events[0]
        self.assertEqual(created_event["event"], "response.created")
        created_data = json.loads(created_event["data"])
        self.assertIn("id", created_data)
        self.assertEqual(created_data["status"], "in_progress")

      with self.subTest(name="Verify delta events"):
        delta_events = [
            e for e in events if e.get("event") == "response.output_text.delta"
        ]
        self.assertNotEmpty(delta_events)
        for de in delta_events:
          delta_data = json.loads(de["data"])
          self.assertIn("delta", delta_data)
          self.assertIn("text", delta_data["delta"])

      with self.subTest(name="Verify completed event"):
        completed_event = next(
            e for e in events if e.get("event") == "response.completed"
        )
        completed_data = json.loads(completed_event["data"])
        self.assertIn("id", completed_data)
        self.assertEqual(completed_data["status"], "completed")

      with self.subTest(name="Verify DONE message"):
        self.assertIn("data: [DONE]", lines)

  def test_openai_responses_streaming_client_disconnect(self):

    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps(
        {"model": "gemma3", "input": "Count to 50", "stream": True}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    response = urllib.request.urlopen(req, timeout=60)
    self.assertEqual(response.getcode(), 200)

    for line in response:
      line_str = line.decode("utf-8")
      if line_str.startswith("event: response.output_text.delta"):
        data_line = next(response).decode("utf-8")
        self.assertStartsWith(data_line, "data: ")
        break
    else:
      self.fail("Stream ended early without delta event")

    # This tests a scenario where a client makes a request and exits before the
    # response is completed. Note: this assumes prefill is already complete.
    # TODO: b/508348544 - There are other scenarios where a client can cause the
    # server to hang.
    response.close()

    conn = http.client.HTTPConnection("localhost", self.port, timeout=15)
    try:
      conn.request(
          "POST",
          "/v1/responses",
          body=json.dumps({"model": "gemma3", "input": "Hi"}).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      try:
        response2 = conn.getresponse()
      except Exception as e:  # pylint: disable=broad-exception-caught
        self.fail(f"Second request failed (timed out as expected?): {e!r}")

      self.assertEqual(response2.status, 200)
      res_body2 = json.loads(response2.read().decode("utf-8"))
      self.assertIn("id", res_body2)
    finally:
      conn.close()

  def test_get_models(self):
    mock_get_all = self.enter_context(
        mock.patch.object(model.Model, "get_all_models", autospec=True)
    )
    model1 = model.Model(model_id="cpu_model", model_path="/path/to/cpu_model")
    model2 = model.Model(model_id="gpu_model", model_path="/path/to/gpu_model")
    mock_get_all.return_value = [model1, model2]

    self.enter_context(
        mock.patch.object(
            openai_handler.os.path, "getmtime", return_value=123456789
        )
    )

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/models",
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      self.assertEqual(response.getheader("Content-Type"), "application/json")

      res_body = json.loads(response.read().decode("utf-8"))
      self.assertEqual(res_body["object"], "list")

      # We expect cpu_model and gpu_model.
      model_ids = [m["id"] for m in res_body["data"]]
      self.assertEqual(model_ids, ["cpu_model", "gpu_model"])
      for m in res_body["data"]:
        self.assertEqual(m["object"], "model")
        self.assertEqual(m["created"], 123456789)
        self.assertEqual(m["owned_by"], "litert-lm")

  def test_parse_thinking_config(self):
    config_none = openai_handler._parse_thinking_config(
        {"reasoning_effort": "none"}
    )
    self.assertIsNotNone(config_none)
    self.assertFalse(config_none.enable_thinking)
    self.assertEqual(config_none.thinking_token_budget, 0)

    for effort in ("minimal", "low", "medium", "high", "xhigh"):
      config = openai_handler._parse_thinking_config(
          {"reasoning_effort": effort}
      )
      self.assertIsNotNone(config)
      self.assertTrue(config.enable_thinking)
      self.assertEqual(config.thinking_token_budget, -1)

    with self.assertRaises(ValueError):
      openai_handler._parse_thinking_config(
          {"reasoning_effort": "invalid_effort"}
      )

    with self.assertRaises(ValueError):
      openai_handler._parse_thinking_config({"reasoning_effort": 10})

    with self.assertRaises(ValueError):
      openai_handler._parse_thinking_config({"reasoning_effort": True})

    # Test fallback to config.json when reasoning_effort is not provided
    with mock.patch.object(cli_config, "get_model_config") as mock_get_cfg:
      mock_get_cfg.return_value = cli_config.ModelConfig(thinking=True)
      cfg = openai_handler._parse_thinking_config({}, model_id="gemma3")
      self.assertIsNotNone(cfg)
      self.assertTrue(cfg.enable_thinking)
      self.assertEqual(cfg.thinking_token_budget, -1)

      mock_get_cfg.return_value = cli_config.ModelConfig(thinking_budget=10)
      cfg = openai_handler._parse_thinking_config({}, model_id="gemma3")
      self.assertIsNotNone(cfg)
      self.assertTrue(cfg.enable_thinking)
      self.assertEqual(cfg.thinking_token_budget, 10)

      mock_get_cfg.return_value = cli_config.ModelConfig(thinking_budget=0)
      cfg = openai_handler._parse_thinking_config({}, model_id="gemma3")
      self.assertIsNotNone(cfg)
      self.assertFalse(cfg.enable_thinking)
      self.assertEqual(cfg.thinking_token_budget, 0)

      mock_get_cfg.return_value = cli_config.ModelConfig(thinking=False)
      cfg = openai_handler._parse_thinking_config({"model": "gemma3"})
      self.assertIsNotNone(cfg)
      self.assertFalse(cfg.enable_thinking)
      self.assertEqual(cfg.thinking_token_budget, 0)

      # When reasoning_effort is explicitly given, it overrides config.json
      mock_get_cfg.return_value = cli_config.ModelConfig(thinking=False)
      cfg = openai_handler._parse_thinking_config(
          {"reasoning_effort": "low", "model": "gemma3"}
      )
      self.assertIsNotNone(cfg)
      self.assertTrue(cfg.enable_thinking)
      self.assertEqual(cfg.thinking_token_budget, -1)

  def test_openai_responses_invalid_reasoning_effort(self):
    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps(
        {"model": "gemma3", "input": "Say hi", "reasoning_effort": "invalid"}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with self.assertRaises(urllib.error.HTTPError) as cm:
      urllib.request.urlopen(req)
    self.assertEqual(cm.exception.code, 400)

  def test_openai_responses_with_reasoning_effort(self):
    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps(
        {"model": "gemma3", "input": "Say hi", "reasoning_effort": "low"}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)

  def test_openai_chat_completions_usage(self):
    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps({
        "model": "gemma3",
        "messages": [{"role": "user", "content": "Say hi"}],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      res_body = json.loads(response.read().decode("utf-8"))
      self.assertIn("usage", res_body)
      usage = res_body["usage"]
      self.assertIn("prompt_tokens", usage)
      self.assertIn("completion_tokens", usage)
      self.assertIn("total_tokens", usage)
      self.assertEqual(
          usage["total_tokens"],
          usage["prompt_tokens"] + usage["completion_tokens"],
      )
      self.assertIn("completion_tokens_details", usage)
      self.assertEqual(
          usage["completion_tokens_details"], {"reasoning_tokens": 0}
      )
      self.assertNotIn("prompt_tokens_details", usage)

  def test_openai_chat_completions_streaming_usage(self):
    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps({
        "model": "gemma3",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      lines = response.read().decode("utf-8").split("\n")
      chunks = []
      for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
          chunks.append(json.loads(line[len("data: ") :]))

      self.assertNotEmpty(chunks)
      usage_chunk = chunks[-1]
      self.assertEqual(usage_chunk["choices"], [])
      self.assertIn("usage", usage_chunk)
      self.assertIsNotNone(usage_chunk["usage"])
      usage = usage_chunk["usage"]
      self.assertIn("prompt_tokens", usage)
      self.assertIn("completion_tokens", usage)
      self.assertIn("total_tokens", usage)
      self.assertIn("completion_tokens_details", usage)
      self.assertEqual(
          usage["completion_tokens_details"], {"reasoning_tokens": 0}
      )

  def test_compute_token_usage_benchmark_info(self):
    mock_conv = mock.MagicMock()
    mock_conv.get_benchmark_info.return_value = litert_lm.BenchmarkInfo(
        init_time_in_second=0.1,
        time_to_first_token_in_second=0.05,
        last_prefill_token_count=15,
        last_prefill_tokens_per_second=300.0,
        last_decode_token_count=10,
        last_decode_tokens_per_second=200.0,
    )

    usage = openai_handler._compute_token_usage(mock_conv)
    self.assertEqual(usage["prompt_tokens"], 15)
    self.assertEqual(usage["completion_tokens"], 10)
    self.assertEqual(usage["total_tokens"], 25)
    self.assertEqual(
        usage["completion_tokens_details"], {"reasoning_tokens": 0}
    )

    usage_with_reasoning = openai_handler._compute_token_usage(
        mock_conv, reasoning_tokens=4
    )
    self.assertEqual(
        usage_with_reasoning["completion_tokens_details"],
        {"reasoning_tokens": 4},
    )

  def test_openai_chat_completions_thinking_tokens_usage(self):
    mock_engine = mock.MagicMock()
    mock_conv = mock.MagicMock()
    mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )
    mock_get_engine = self.enter_context(
        mock.patch.object(
            openai_handler.OpenAIHandler, "_get_engine", autospec=True
        )
    )
    mock_get_engine.return_value = mock_engine

    mock_conv.send_message_async.return_value = [
        {"role": "assistant", "channels": {"thought": "Thinking..."}},
        {"role": "assistant", "channels": {"thought": "more"}},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    mock_conv.get_benchmark_info.return_value = litert_lm.BenchmarkInfo(
        init_time_in_second=0.1,
        time_to_first_token_in_second=0.05,
        last_prefill_token_count=5,
        last_prefill_tokens_per_second=100.0,
        last_decode_token_count=3,
        last_decode_tokens_per_second=100.0,
    )

    data = json.dumps({
        "model": "gemma3",
        "messages": [{"role": "user", "content": "Say hi"}],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      res_body = json.loads(response.read().decode("utf-8"))
      self.assertIn("usage", res_body)
      usage = res_body["usage"]
      self.assertEqual(
          usage["completion_tokens_details"], {"reasoning_tokens": 2}
      )
      self.assertEqual(res_body["choices"][0]["message"]["content"], "Hi")


if __name__ == "__main__":
  absltest.main()
