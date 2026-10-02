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

import ctypes
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

  def test_compute_embedding_with_cpu_thread_count(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path,
        backend=litert_lm.Backend.CPU(thread_count=4),
    )
    try:
      response = engine.compute_embedding(contents="'s")
      self.assertIsInstance(response, litert_lm.EmbeddingResponse)
      self.assertNotEmpty(response.embedding)
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

  def test_min_max_input_length_and_vision_tokens_properties(self):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path,
        backend=litert_lm.Backend.CPU(),
        min_input_length=64,
        max_input_length=128,
    )
    try:
      self.assertEqual(engine.min_input_length, 64)
      self.assertEqual(engine.max_input_length, 128)
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

  @parameterized.named_parameters(
      ("single", False),
      ("batch", True),
  )
  def test_options_setter_failure_propagates_runtime_error(self, batch):
    engine = litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    )
    try:
      with mock.patch.object(
          litert_lm.embedding_engine,
          "call_checked",
          autospec=True,
          side_effect=RuntimeError("setter failed"),
      ):
        with self.assertRaisesRegex(RuntimeError, "setter failed"):
          if batch:
            engine.compute_embedding_batch(
                ["'s"], options=litert_lm.EmbeddingOptions(normalize=True)
            )
          else:
            engine.compute_embedding(
                "'s", options=litert_lm.EmbeddingOptions(normalize=True)
            )
    finally:
      engine.close()

  def test_invalid_model_path_raises_runtime_error(self):
    with self.assertRaises(RuntimeError):
      litert_lm.EmbeddingEngine(
          model_path="/invalid/path/to/nonexistent/model.litertlm"
      )

  def test_invalid_model_path_error_includes_c_status(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "litert_lm_embedding_engine_create failed with status NOT_FOUND",
    ):
      litert_lm.EmbeddingEngine(
          model_path="/invalid/path/to/nonexistent/model.litertlm"
      )

  def test_engine_create_failure_includes_c_status(self):
    with self.assertRaisesRegex(
        RuntimeError, "litert_lm_embedding_engine_create failed with status"
    ):
      litert_lm.EmbeddingEngine(
          model_path=self.model_path,
          backend=litert_lm.Backend.CPU(),
          max_input_length=512,
      )

  def test_c_options_getters_round_trip(self):
    lib = litert_lm._ffi._get_lib()
    options_ptr = litert_lm.embedding_engine._create_c_options(
        lib,
        litert_lm.EmbeddingOptions(
            normalize=False,
            insert_special_tokens=False,
            input_overflow_strategy=litert_lm.InputOverflowStrategy.TRUNCATE,
            output_size=64,
            vision_tokens_per_image=70,
        ),
    )
    try:

      def get(name, out_type):
        return litert_lm._ffi.get_checked(lib, name, out_type, options_ptr)

      self.assertFalse(
          get("litert_lm_embedding_options_get_normalize", ctypes.c_bool)
      )
      self.assertFalse(
          get(
              "litert_lm_embedding_options_get_insert_special_tokens",
              ctypes.c_bool,
          )
      )
      self.assertEqual(
          get(
              "litert_lm_embedding_options_get_input_overflow_strategy",
              ctypes.c_int,
          ),
          litert_lm.InputOverflowStrategy.TRUNCATE,
      )
      self.assertEqual(
          get("litert_lm_embedding_options_get_output_size", ctypes.c_int), 64
      )
      self.assertEqual(
          get(
              "litert_lm_embedding_options_get_vision_tokens_per_image",
              ctypes.c_int,
          ),
          70,
      )
    finally:
      lib.litert_lm_embedding_options_delete(options_ptr)

  def test_c_options_unset_optional_getters_return_not_found(self):
    lib = litert_lm._ffi._get_lib()
    options_ptr = litert_lm._ffi.create_checked(
        lib, "litert_lm_embedding_options_create"
    )
    try:
      for name in (
          "litert_lm_embedding_options_get_output_size",
          "litert_lm_embedding_options_get_vision_tokens_per_image",
      ):
        with self.subTest(name=name):
          out = ctypes.c_int(42)
          status = getattr(lib, name)(options_ptr, ctypes.byref(out))
          self.assertEqual(status, litert_lm._ffi.StatusCode.NOT_FOUND)
          self.assertEqual(out.value, 42)
    finally:
      lib.litert_lm_embedding_options_delete(options_ptr)

  def test_c_null_handle_returns_invalid_argument(self):
    lib = litert_lm._ffi._get_lib()
    out = ctypes.c_size_t(42)
    status = lib.litert_lm_embedding_response_get_size(None, ctypes.byref(out))
    self.assertEqual(status, litert_lm._ffi.StatusCode.INVALID_ARGUMENT)
    self.assertEqual(out.value, 42)
    with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
      litert_lm._ffi.get_checked(
          lib,
          "litert_lm_embedding_options_get_output_size",
          ctypes.c_int,
          None,
      )

  def test_c_responses_get_at_out_of_range(self):
    lib = litert_lm._ffi._get_lib()
    with litert_lm.EmbeddingEngine(
        model_path=self.model_path, backend=litert_lm.Backend.CPU()
    ) as engine:
      text = b"'s"
      input_ptr = litert_lm._ffi.create_checked(
          lib,
          "litert_lm_input_data_create",
          litert_lm._ffi.InputDataType.TEXT,
          text,
          len(text),
      )
      try:
        request = (ctypes.c_void_p * 1)(input_ptr)
        batch = (ctypes.POINTER(ctypes.c_void_p) * 1)(
            ctypes.cast(request, ctypes.POINTER(ctypes.c_void_p))
        )
        num_inputs = (ctypes.c_size_t * 1)(1)
        responses_ptr = litert_lm._ffi.create_checked(
            lib,
            "litert_lm_embedding_engine_compute_embedding_batch",
            engine._engine_ptr,  # pylint: disable=protected-access
            batch,
            num_inputs,
            1,
            None,
        )
      finally:
        lib.litert_lm_input_data_delete(input_ptr)
      try:
        out = ctypes.c_void_p()
        status = lib.litert_lm_embedding_responses_get_at(
            responses_ptr, 1, ctypes.byref(out)
        )
        self.assertEqual(status, litert_lm._ffi.StatusCode.OUT_OF_RANGE)
        self.assertIsNone(out.value)
      finally:
        lib.litert_lm_embedding_responses_delete(responses_ptr)

  def test_create_c_input_data_multimodal_image_and_audio(self):
    mock_lib = mock.MagicMock()
    handles = iter([101, 201])

    def fake_input_data_create(unused_type, unused_data, unused_size, out):
      # `out` is a `ctypes.byref` to the handle to fill in.
      out._obj.value = next(handles)  # pylint: disable=protected-access
      return litert_lm._ffi.StatusCode.OK

    mock_lib.litert_lm_input_data_create.side_effect = fake_input_data_create

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
