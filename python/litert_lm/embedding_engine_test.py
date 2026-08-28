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
"""Tests for LiteRT-LM EmbeddingEngine."""

import math
import pathlib
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import litert_lm

FLAGS = flags.FLAGS


class EmbeddingEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = str(
        pathlib.Path(FLAGS.test_srcdir)
        / "google3/runtime/testdata/test_embedding.litertlm"
    )

  def test_embedding_response_dataclass(self):
    resp1 = litert_lm.EmbeddingResponse(embedding=[1.0, 2.0, 3.0])
    resp2 = litert_lm.EmbeddingResponse(embedding=[1.0, 2.0, 3.0])
    resp3 = litert_lm.EmbeddingResponse(embedding=[1.0, 2.0, 4.0])

    self.assertEqual(resp1, resp2)
    self.assertNotEqual(resp1, resp3)

  def test_embedding_options_dataclass(self):
    opts_default = litert_lm.EmbeddingOptions()
    self.assertIsNone(opts_default.normalize)
    self.assertIsNone(opts_default.insert_special_tokens)
    self.assertIsNone(opts_default.input_overflow_strategy)
    self.assertIsNone(opts_default.output_size)
    self.assertIsNone(opts_default.vision_tokens_per_image)

    opts_custom = litert_lm.EmbeddingOptions(
        normalize=False,
        insert_special_tokens=True,
        input_overflow_strategy=litert_lm.InputOverflowStrategy.TRUNCATE,
        output_size=128,
        vision_tokens_per_image=70,
    )
    self.assertFalse(opts_custom.normalize)
    self.assertTrue(opts_custom.insert_special_tokens)
    self.assertEqual(
        opts_custom.input_overflow_strategy,
        litert_lm.InputOverflowStrategy.TRUNCATE,
    )
    self.assertEqual(opts_custom.output_size, 128)
    self.assertEqual(opts_custom.vision_tokens_per_image, 70)

  def test_compute_embedding_with_output_size(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      response = engine.compute_embedding(
          contents="'s",
          options=litert_lm.EmbeddingOptions(normalize=True, output_size=64),
      )
      self.assertIsInstance(response, litert_lm.EmbeddingResponse)
      self.assertLen(response.embedding, 64)
      norm = math.sqrt(sum(x * x for x in response.embedding))
      self.assertAlmostEqual(norm, 1.0, places=4)
    finally:
      engine.close()

  def test_compute_embedding_batch_with_output_size(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      responses = engine.compute_embedding_batch(
          contents_batch=["'s", "'s"],
          options=litert_lm.EmbeddingOptions(normalize=True, output_size=64),
      )
      self.assertLen(responses, 2)
      self.assertLen(responses[0].embedding, 64)
      self.assertLen(responses[1].embedding, 64)
    finally:
      engine.close()

  def test_compute_embedding_single(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      response = engine.compute_embedding(
          contents="'s",
          options=litert_lm.EmbeddingOptions(normalize=True),
      )
      self.assertIsInstance(response, litert_lm.EmbeddingResponse)
      self.assertNotEmpty(response.embedding)

      # Verify L2 normalization
      norm = math.sqrt(sum(x * x for x in response.embedding))
      self.assertAlmostEqual(norm, 1.0, places=4)
    finally:
      engine.close()

  def test_compute_embedding_with_overflow_strategy(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      response = engine.compute_embedding(
          contents="'s",
          options=litert_lm.EmbeddingOptions(
              input_overflow_strategy=litert_lm.InputOverflowStrategy.TRUNCATE
          ),
      )
      self.assertIsInstance(response, litert_lm.EmbeddingResponse)
      self.assertNotEmpty(response.embedding)
    finally:
      engine.close()

  def test_compute_embedding_content_object(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      content = litert_lm.Content.Text("'s")
      response = engine.compute_embedding(
          contents=content,
          options=litert_lm.EmbeddingOptions(normalize=False),
      )
      self.assertIsInstance(response, litert_lm.EmbeddingResponse)
      self.assertNotEmpty(response.embedding)
    finally:
      engine.close()

  def test_compute_embedding_batch(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      batch = ["'s", "'s"]
      responses = engine.compute_embedding_batch(
          contents_batch=batch,
          options=litert_lm.EmbeddingOptions(normalize=True),
      )
      self.assertLen(responses, 2)
      self.assertNotEmpty(responses[0].embedding)
      self.assertNotEmpty(responses[1].embedding)
      self.assertEqual(len(responses[0].embedding), len(responses[1].embedding))
    finally:
      engine.close()

  def test_context_manager(self):
    with litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    ) as engine:
      response = engine.compute_embedding("'s")
      self.assertNotEmpty(response.embedding)

  def test_max_input_length_and_vision_tokens_properties(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path,
        backend=litert_lm.Backend.CPU(),
        max_input_length=512,
    )
    try:
      self.assertEqual(engine.max_input_length, 512)
      self.assertIsNone(engine.vision_tokens_per_image)
      response = engine.compute_embedding("'s")
      self.assertNotEmpty(response.embedding)
    finally:
      engine.close()

    with self.assertRaises(RuntimeError):
      litert_lm.EmbeddingEngine(
          model_path=self.model_path,
          backend=litert_lm.Backend.CPU(),
          vision_tokens_per_image=280,
      )

  def test_closed_raises_runtime_error(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    engine.close()
    with self.assertRaises(RuntimeError):
      engine.compute_embedding("'s")
    with self.assertRaises(RuntimeError):
      engine.compute_embedding_batch(["'s"])

  def test_invalid_model_path_raises_runtime_error(self):
    with self.assertRaises(RuntimeError):
      litert_lm.EmbeddingEngine(
          model_path="/invalid/path/to/nonexistent/model.litertlm"
      )

  def test_create_c_input_data_multimodal_image_and_audio(self):
    mock_lib = mock.MagicMock()
    mock_lib.litert_lm_input_data_create.side_effect = [101, 201]

    image_content = litert_lm.Content.ImageBytes(bytes=b"fake_image_bytes")
    ptr = litert_lm.embedding_engine._create_c_input_data(
        mock_lib, image_content
    )
    self.assertEqual(ptr, 101)
    self.assertEqual(mock_lib.litert_lm_input_data_create.call_count, 1)
    self.assertEqual(
        mock_lib.litert_lm_input_data_create.call_args_list[0][0][0],
        litert_lm._ffi.InputDataType.IMAGE,
    )

    audio_content = litert_lm.Content.AudioBytes(bytes=b"fake_audio_bytes")
    ptr_audio = litert_lm.embedding_engine._create_c_input_data(
        mock_lib, audio_content
    )
    self.assertEqual(ptr_audio, 201)
    self.assertEqual(mock_lib.litert_lm_input_data_create.call_count, 2)
    self.assertEqual(
        mock_lib.litert_lm_input_data_create.call_args_list[1][0][0],
        litert_lm._ffi.InputDataType.AUDIO,
    )


if __name__ == "__main__":
  absltest.main()
