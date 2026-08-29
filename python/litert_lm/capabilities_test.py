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
"""Tests for model capabilities extraction API."""

import pathlib
from unittest import mock

from absl import flags
from absl.testing import absltest

import litert_lm

FLAGS = flags.FLAGS


class CapabilitiesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = (
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

  def test_capabilities_load(self):
    capabilities = litert_lm.Capabilities(self.model_path)

    # Check simple capability flags (expect False for the legacy test model)
    self.assertFalse(capabilities.supports_thinking())
    self.assertFalse(capabilities.supports_function_calling())
    self.assertFalse(capabilities.has_speculative_decoding_support())
    self.assertEqual(capabilities.max_vision_token_budget, -1)

    # Modalities
    self.assertTrue(capabilities.input_modalities.text)
    self.assertFalse(capabilities.input_modalities.vision)
    self.assertFalse(capabilities.input_modalities.audio)
    self.assertFalse(capabilities.input_modalities.video)

    # Sampler default parameters (from test model config)
    sampler_config = capabilities.default_sampler_params
    self.assertIsInstance(sampler_config, litert_lm.SamplerConfig)
    self.assertEqual(sampler_config.temperature, 0.0)
    self.assertEqual(sampler_config.top_k, 1)
    top_p = sampler_config.top_p
    self.assertIsNotNone(top_p)
    self.assertAlmostEqual(top_p, 0.7)

  def test_capabilities_non_existent_file(self):
    with self.assertRaises(FileNotFoundError):
      litert_lm.Capabilities("/non/existent/path")

  @mock.patch(
      "litert_lm.capabilities._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_capabilities_destructor_deletes_handle(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists  # Unused.
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_create.return_value = 12345

    capabilities = litert_lm.Capabilities("/fake/path")
    self.assertEqual(capabilities._handle, 12345)

    capabilities.__del__()
    mock_lib.litert_lm_loaded_file_delete.assert_called_once_with(12345)

  @mock.patch(
      "litert_lm.capabilities._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_capabilities_creation_failure_raises_runtime_error(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_create.return_value = 0

    with self.assertRaises(RuntimeError):
      litert_lm.Capabilities("/invalid/model.litertlm")

  @mock.patch(
      "litert_lm.capabilities._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_capabilities_destructor_noop_when_handle_none(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_create.return_value = 12345

    capabilities = litert_lm.Capabilities("/fake/path")
    capabilities._handle = None  # Clear handle manually
    capabilities.__del__()
    mock_lib.litert_lm_loaded_file_delete.assert_not_called()

  @mock.patch(
      "litert_lm.capabilities._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_capabilities_destructor_noop_when_handle_not_set(
      self, mock_exists, mock_get_lib
  ):
    del mock_exists
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib

    # Create uninitialized capabilities object
    capabilities = object.__new__(litert_lm.Capabilities)
    try:
      capabilities.__del__()
    except AttributeError as e:
      self.fail(f"__del__ raised AttributeError on uninitialized object: {e}")

  def test_capabilities_context_manager(self):
    with litert_lm.Capabilities(self.model_path) as capabilities:
      self.assertFalse(capabilities.supports_thinking())
    # Outside context block, capabilities should be closed
    with self.assertRaises(RuntimeError):
      _ = capabilities.supports_thinking()

  def test_capabilities_close_explicit(self):
    capabilities = litert_lm.Capabilities(self.model_path)
    capabilities.close()
    with self.assertRaises(RuntimeError):
      _ = capabilities.supports_thinking()
    with self.assertRaises(RuntimeError):
      _ = capabilities.default_sampler_params
    with self.assertRaises(RuntimeError):
      _ = capabilities.input_modalities

  @mock.patch(
      "litert_lm.capabilities._ffi._get_lib"
  )
  @mock.patch("os.path.exists", return_value=True)
  def test_max_vision_token_budget(self, unused_mock_exists, mock_get_lib):
    mock_lib = mock.MagicMock()
    mock_get_lib.return_value = mock_lib
    mock_lib.litert_lm_loaded_file_create.return_value = 12345
    mock_lib.litert_lm_loaded_file_max_vision_token_budget.return_value = 280

    capabilities = litert_lm.Capabilities("/fake/path")
    self.assertEqual(capabilities.max_vision_token_budget, 280)


if __name__ == "__main__":
  absltest.main()
