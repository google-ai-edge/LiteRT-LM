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
"""Tests for model info extraction API."""

import pathlib
from unittest import mock

from absl import flags
from absl.testing import absltest

import litert_lm

FLAGS = flags.FLAGS


_OK = litert_lm._ffi.StatusCode.OK  # pylint: disable=protected-access
_NOT_FOUND = litert_lm._ffi.StatusCode.NOT_FOUND  # pylint: disable=protected-access
_INVALID_ARGUMENT = litert_lm._ffi.StatusCode.INVALID_ARGUMENT  # pylint: disable=protected-access


def _writes_out(value, status=_OK):
  """Returns a fake C API function that writes `value` to its out-parameter.

  The fake mirrors the Option A C API contract: it returns `status`, and writes
  `value` to the trailing out-parameter (passed via `ctypes.byref`) only when
  `status` is OK.

  Args:
    value: The value to write to the out-parameter on success.
    status: The status code to return.
  """

  def fake(*args):
    if status == _OK:
      args[-1]._obj.value = value  # pylint: disable=protected-access
    return int(status)

  return fake


def _fake_lengths(values):
  """Returns a fake two-pass `*_selection` C API function reporting `values`."""

  def fake(unused_handle, lengths, max_size, out_count):
    if lengths is not None:
      for i in range(min(max_size, len(values))):
        lengths[i] = values[i]
    out_count._obj.value = len(values)  # pylint: disable=protected-access
    return int(_OK)

  return fake


def _make_mock_lib(model_type=litert_lm.LiteRtLmModelType.LLM):
  """Returns a mock C library that loads a handle of 12345 of `model_type`."""
  mock_lib = mock.MagicMock()
  mock_lib.litert_lm_loaded_file_create.side_effect = _writes_out(12345)
  mock_lib.litert_lm_loaded_file_model_type.side_effect = _writes_out(
      int(model_type)
  )
  mock_lib.litert_lm_get_last_error_message.return_value = b"fake error"
  return mock_lib


class ModelInfoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = (
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

  def test_model_info_load(self):
    model_info = litert_lm.ModelInfo(self.model_path)

    # Check simple capability flags (expect False for the legacy test model)
    self.assertIsNotNone(model_info.llm)
    assert model_info.llm is not None
    self.assertIsNone(model_info.embedding)
    self.assertFalse(model_info.llm.supports_thinking())
    self.assertFalse(model_info.llm.supports_function_calling())
    self.assertFalse(model_info.llm.has_speculative_decoding_support())
    self.assertEqual(model_info.model_type, litert_lm.ModelType.LLM)
    self.assertFalse(model_info.is_embedding_model)
    self.assertTrue(model_info.is_llm_model)
    self.assertEqual(model_info.max_vision_token_budget, -1)
    self.assertIsNone(model_info.vision_signature_selection)
    self.assertIsNone(model_info.min_runtime_version)
    self.assertEqual(model_info.max_context_tokens, 128)
    self.assertFalse(model_info.is_dynamic_context)
    self.assertFalse(model_info.llm.is_dynamic_context)

    # Verify modality-specific backends for text (defaults to CPU and GPU)
    self.assertEqual(
        model_info.supported_backends_for_modality(
            litert_lm.LiteRtLmModality.TEXT
        ),
        ["cpu", "gpu"],
    )
    # Verify modality-specific backends for vision (not present -> empty)
    self.assertEqual(
        model_info.supported_backends_for_modality(
            litert_lm.LiteRtLmModality.VISION
        ),
        [],
    )
    self.assertEqual(
        model_info.npu_brand_for_modality(litert_lm.LiteRtLmModality.TEXT),
        litert_lm.LiteRtLmNpuBrand.UNKNOWN,
    )
    self.assertIsNone(
        model_info.soc_name_for_modality(litert_lm.LiteRtLmModality.TEXT)
    )

    # Modalities
    self.assertTrue(model_info.input_modalities.text)
    self.assertFalse(model_info.input_modalities.vision)
    self.assertFalse(model_info.input_modalities.audio)
    self.assertFalse(model_info.input_modalities.video)

    # Sampler default parameters (from test model config)
    sampler_config = model_info.llm.default_sampler_params
    self.assertIsInstance(sampler_config, litert_lm.SamplerConfig)
    self.assertEqual(sampler_config.temperature, 0.0)
    self.assertEqual(sampler_config.top_k, 1)
    top_p = sampler_config.top_p
    self.assertIsNotNone(top_p)
    self.assertAlmostEqual(top_p, 0.7)

  def test_model_info_non_existent_file(self):
    with self.assertRaises(FileNotFoundError):
      litert_lm.ModelInfo("/non/existent/path")

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_model_info_destructor_deletes_handle(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists  # Unused.
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info._handle, 12345)

    model_info.__del__()
    mock_lib.litert_lm_loaded_file_delete.assert_called_once_with(12345)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_model_info_creation_failure_raises_runtime_error(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_create.side_effect = _writes_out(
        None, status=_NOT_FOUND
    )

    with self.assertRaisesRegex(
        RuntimeError,
        "Failed to load model info for model: /invalid/model.litertlm.*"
        "NOT_FOUND.*fake error",
    ):
      litert_lm.ModelInfo("/invalid/model.litertlm")

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_model_info_destructor_noop_when_handle_none(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib

    model_info = litert_lm.ModelInfo("/fake/path")
    model_info._handle = None  # Clear handle manually
    model_info.__del__()
    mock_lib.litert_lm_loaded_file_delete.assert_not_called()

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_model_info_destructor_noop_when_handle_not_set(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib

    # Create uninitialized model_info object
    model_info = object.__new__(litert_lm.ModelInfo)
    try:
      model_info.__del__()
    except AttributeError as e:
      self.fail(f"__del__ raised AttributeError on uninitialized object: {e}")

  def test_model_info_context_manager(self):
    with litert_lm.ModelInfo(self.model_path) as model_info:
      self.assertIsNotNone(model_info.llm)
      assert model_info.llm is not None
      self.assertFalse(model_info.llm.supports_thinking())
      llm = model_info.llm
    # Outside context block, model_info should be closed
    with self.assertRaises(RuntimeError):
      _ = llm.supports_thinking()

  def test_model_info_close_explicit(self):
    model_info = litert_lm.ModelInfo(self.model_path)
    llm = model_info.llm
    self.assertIsNotNone(llm)
    assert llm is not None
    model_info.close()
    with self.assertRaises(RuntimeError):
      _ = llm.supports_thinking()
    with self.assertRaises(RuntimeError):
      _ = llm.default_sampler_params
    with self.assertRaises(RuntimeError):
      _ = model_info.input_modalities

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_max_vision_token_budget(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_max_vision_token_budget.side_effect = (
        _writes_out(280)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.max_vision_token_budget, 280)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_max_vision_token_budget_not_found_returns_minus_one(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_max_vision_token_budget.side_effect = (
        _writes_out(280, status=_NOT_FOUND)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.max_vision_token_budget, -1)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_getter_error_raises_runtime_error(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_max_vision_token_budget.side_effect = (
        _writes_out(280, status=_INVALID_ARGUMENT)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    with self.assertRaisesRegex(
        RuntimeError,
        "litert_lm_loaded_file_max_vision_token_budget failed with status"
        " INVALID_ARGUMENT.*fake error",
    ):
      _ = model_info.max_vision_token_budget

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_vision_signature_selection(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib

    mock_lib.litert_lm_loaded_file_vision_signature_selection.side_effect = (
        _fake_lengths([64, 256])
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.vision_signature_selection, [64, 256])

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_vision_signature_selection_not_found(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_vision_signature_selection.side_effect = (
        _writes_out(0, status=_NOT_FOUND)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNone(model_info.vision_signature_selection)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_min_runtime_version(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_min_runtime_version.side_effect = (
        _writes_out(b"0.12.3")
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.min_runtime_version, "0.12.3")

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_min_runtime_version_not_set(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_min_runtime_version.side_effect = (
        _writes_out(None)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNone(model_info.min_runtime_version)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_supported_backends_for_modality(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib

    def fake_supported_backends(unused_handle, unused_modality, *args):
      fake_lengths = _fake_lengths([
          int(litert_lm.LiteRtLmBackendType.GPU),
          int(litert_lm.LiteRtLmBackendType.CPU),
      ])
      return fake_lengths(unused_handle, *args)

    mock_lib.litert_lm_loaded_file_modality_supported_backends.side_effect = (
        fake_supported_backends
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(
        model_info.supported_backends_for_modality(
            litert_lm.LiteRtLmModality.VISION
        ),
        ["gpu", "cpu"],
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_npu_brand_for_modality(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_modality_npu_brand.side_effect = _writes_out(
        int(litert_lm.LiteRtLmNpuBrand.MEDIATEK)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(
        model_info.npu_brand_for_modality(litert_lm.LiteRtLmModality.AUDIO),
        litert_lm.LiteRtLmNpuBrand.MEDIATEK,
    )
    assert_brand = mock_lib.litert_lm_loaded_file_modality_npu_brand
    assert_brand.assert_called_once()
    self.assertEqual(
        assert_brand.call_args.args[:2],
        (12345, int(litert_lm.LiteRtLmModality.AUDIO)),
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_soc_name_for_modality(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_modality_soc_name.side_effect = _writes_out(
        b"SM8750"
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(
        model_info.soc_name_for_modality(litert_lm.LiteRtLmModality.TEXT),
        "SM8750",
    )
    assert_fn = mock_lib.litert_lm_loaded_file_modality_soc_name
    assert_fn.assert_called_once()
    self.assertEqual(
        assert_fn.call_args.args[:2],
        (12345, int(litert_lm.LiteRtLmModality.TEXT)),
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_soc_name_for_modality_none(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_modality_soc_name.side_effect = _writes_out(
        None
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNone(
        model_info.soc_name_for_modality(litert_lm.LiteRtLmModality.TEXT)
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_max_context_tokens(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_max_context_tokens.side_effect = _writes_out(
        4096
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.max_context_tokens, 4096)
    mock_lib.litert_lm_loaded_file_max_context_tokens.assert_called_once()
    self.assertEqual(
        mock_lib.litert_lm_loaded_file_max_context_tokens.call_args.args[0],
        12345,
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_max_context_tokens_not_found_returns_zero(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_max_context_tokens.side_effect = _writes_out(
        4096, status=_NOT_FOUND
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(model_info.max_context_tokens, 0)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_is_dynamic_context(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_is_dynamic_context.side_effect = _writes_out(
        True
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertTrue(model_info.is_dynamic_context)
    mock_lib.litert_lm_loaded_file_is_dynamic_context.assert_called_once()
    self.assertEqual(
        mock_lib.litert_lm_loaded_file_is_dynamic_context.call_args.args[0],
        12345,
    )

  def test_model_info_load_embedding_model(self):
    embedding_model_path = (
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_embedding.litertlm"
    )
    model_info = litert_lm.ModelInfo(embedding_model_path)

    self.assertEqual(model_info.model_type, litert_lm.ModelType.EMBEDDING)
    self.assertTrue(model_info.is_embedding_model)
    self.assertFalse(model_info.is_llm_model)
    self.assertIsNone(model_info.llm)
    self.assertIsNotNone(model_info.embedding)
    assert model_info.embedding is not None
    self.assertEqual(model_info.embedding.dimension, 768)
    self.assertEqual(model_info.embedding.signature_selection, [128])
    self.assertEqual(model_info.max_context_tokens, 128)
    self.assertFalse(model_info.is_dynamic_context)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_is_embedding_and_llm_model(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib(litert_lm.LiteRtLmModelType.EMBEDDING)
    mock_get_lib.return_value = mock_lib

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertEqual(
        model_info.model_type, litert_lm.LiteRtLmModelType.EMBEDDING
    )
    self.assertTrue(model_info.is_embedding_model)
    self.assertFalse(model_info.is_llm_model)
    self.assertIsNotNone(model_info.embedding)
    self.assertIsNone(model_info.llm)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_embedding_dimension(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib(litert_lm.LiteRtLmModelType.EMBEDDING)
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_embedding_dimension.side_effect = (
        _writes_out(768)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNotNone(model_info.embedding)
    assert model_info.embedding is not None
    self.assertEqual(model_info.embedding.dimension, 768)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_embedding_dimension_unset(self, unused_mock_exists, mock_get_lib):
    mock_lib = _make_mock_lib(litert_lm.LiteRtLmModelType.EMBEDDING)
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_embedding_dimension.side_effect = (
        _writes_out(768, status=_NOT_FOUND)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNotNone(model_info.embedding)
    assert model_info.embedding is not None
    self.assertIsNone(model_info.embedding.dimension)

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_embedding_signature_selection(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib(litert_lm.LiteRtLmModelType.EMBEDDING)
    mock_get_lib.return_value = mock_lib

    mock_lib.litert_lm_loaded_file_embedding_signature_selection.side_effect = (
        _fake_lengths([128, 256, 512])
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNotNone(model_info.embedding)
    assert model_info.embedding is not None
    self.assertEqual(
        model_info.embedding.signature_selection, [128, 256, 512]
    )

  @mock.patch(
      "litert_lm.model_info._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_embedding_signature_selection_unset(
      self, unused_mock_exists, mock_get_lib
  ):
    mock_lib = _make_mock_lib(litert_lm.LiteRtLmModelType.EMBEDDING)
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_embedding_signature_selection.side_effect = (
        _writes_out(0, status=_NOT_FOUND)
    )

    model_info = litert_lm.ModelInfo("/fake/path")
    self.assertIsNotNone(model_info.embedding)
    assert model_info.embedding is not None
    self.assertIsNone(model_info.embedding.signature_selection)


if __name__ == "__main__":
  absltest.main()
