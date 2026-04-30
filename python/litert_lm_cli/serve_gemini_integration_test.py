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

"""Integration tests for the LiteRT-LM serve command.

These tests verify the Gemini-compatible API server using the official
google.genai Python SDK and a real gemma3-1b-it-int4-litertlm model.
"""

import http.server
import os
import threading
from unittest import mock

from absl.testing import absltest
from google import genai

from litert_lm_cli import model
from litert_lm_cli import serve


class ServeIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global state in serve.py to ensure each test starts fresh
    serve._current_engine = None
    serve._current_model_id = None

    # Start the server on a free ephemeral port
    self.server = http.server.HTTPServer(("localhost", 0), serve.GeminiHandler)
    self.port = self.server.server_port

    self.server_thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self.server_thread.start()

    # The real model path provided via 'data' in BUILD
    self.model_path = os.path.join(
        absltest.get_default_test_srcdir(),
        "google3/runtime/e2e_tests/data/gemma3-1b-it-int4.litertlm",
    )

  def tearDown(self):
    # Shutdown the local server cleanly
    self.server.shutdown()
    self.server.server_close()
    self.server_thread.join()
    super().tearDown()

  def test_genai_generate_content(self):
    # Ensure the model exists at the expected path
    self.assertTrue(
        os.path.exists(self.model_path), f"Model not found at {self.model_path}"
    )

    client = genai.Client(
        api_key="NOT_REQUIRED",
        http_options={"base_url": f"http://localhost:{self.port}"},
    )

    # Mock model.Model.from_model_id to point to our 'data' model path
    with mock.patch.object(
        model.Model, "from_model_id", autospec=True
    ) as mock_from_id:
      mock_from_id.return_value = model.Model(
          model_id="gemma3", model_path=self.model_path
      )

      response = client.models.generate_content(
          model="gemma3", contents="Say hi"
      )

      self.assertNotEmpty(response.text)
      print(f"\n[Generate Content Response]: {response.text}")

  def test_genai_stream_generate_content(self):
    self.assertTrue(
        os.path.exists(self.model_path), f"Model not found at {self.model_path}"
    )

    client = genai.Client(
        api_key="NOT_REQUIRED",
        http_options={"base_url": f"http://localhost:{self.port}"},
    )

    with mock.patch.object(
        model.Model, "from_model_id", autospec=True
    ) as mock_from_id:
      mock_from_id.return_value = model.Model(
          model_id="gemma3", model_path=self.model_path
      )

      response_stream = client.models.generate_content_stream(
          model="gemma3", contents="Hello!"
      )

      full_text = ""
      chunks_received = 0
      print("\n[Stream Generate Content Response]: ", end="")
      for chunk in response_stream:
        if chunk.text:
          full_text += chunk.text
          print(chunk.text, end="", flush=True)
        chunks_received += 1
      print()

      self.assertNotEmpty(full_text)
      self.assertGreater(chunks_received, 0)


if __name__ == "__main__":
  absltest.main()
