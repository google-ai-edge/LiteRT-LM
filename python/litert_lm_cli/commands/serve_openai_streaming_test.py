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
import os
import pathlib
import threading
from unittest import mock
import urllib.request

from absl.testing import absltest

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
      except Exception as e:
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

    mock_is_gpu_only = self.enter_context(
        mock.patch.object(openai_handler, "_is_gpu_only_model", autospec=True)
    )

    def is_gpu_only_side_effect(model_path: str) -> bool:
      if "cpu_model" in model_path:
        return False
      elif "gpu_model" in model_path:
        return True
      return False

    mock_is_gpu_only.side_effect = is_gpu_only_side_effect

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/models",
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      self.assertEqual(response.getheader("Content-Type"), "application/json")

      res_body = json.loads(response.read().decode("utf-8"))
      self.assertEqual(res_body["object"], "list")

      # We expect cpu_model, cpu_model,gpu and gpu_model,gpu
      # gpu_model (CPU version) should NOT be present.
      model_ids = [m["id"] for m in res_body["data"]]
      self.assertEqual(
          model_ids, ["cpu_model", "cpu_model,gpu", "gpu_model,gpu"]
      )
      for m in res_body["data"]:
        self.assertEqual(m["object"], "model")
        self.assertEqual(m["created"], 123456789)
        self.assertEqual(m["owned_by"], "litert-lm")


if __name__ == "__main__":
  absltest.main()
