import http.server
import json
import pathlib
import threading
from unittest import mock
import urllib.request

from absl.testing import absltest

from litert_lm_cli import model
from litert_lm_cli import serve


class ServeOpenAIIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global state in serve.py to ensure each test starts fresh
    serve._current_engine = None
    serve._current_model_id = None

    # Start the server on a free ephemeral port
    self.server = http.server.HTTPServer(("localhost", 0), serve.OpenAIHandler)
    self.port = self.server.server_port

    self.server_thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self.server_thread.start()

    # The real model path provided via 'data' in BUILD
    self.model_path = (
        pathlib.Path(absltest.get_default_test_srcdir())
        / "google3/runtime/e2e_tests/data/gemma3-1b-it-int4.litertlm"
    )

  def tearDown(self):
    self.server.shutdown()
    self.server.server_close()
    self.server_thread.join()
    super().tearDown()

  def test_openai_responses(self):
    self.assertTrue(
        self.model_path.exists(), f"Model not found at {self.model_path}"
    )

    # Use self.enter_context instead of with statement for patching
    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps({"model": "gemma3", "input": "Say hi"}).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      res_body = json.loads(response.read().decode("utf-8"))

      # Verify top-level fields
      self.assertIn("id", res_body)
      self.assertIn("output", res_body)

      # Unpack list and verify structure
      [output_item] = res_body["output"]
      self.assertEqual(output_item.get("role"), "assistant")
      self.assertEqual(output_item.get("status"), "completed")

      [content_item] = output_item["content"]
      self.assertEqual(content_item.get("type"), "output_text")
      self.assertNotEmpty(content_item.get("text"))


if __name__ == "__main__":
  absltest.main()
