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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm
from litert_lm_builder import litertlm_builder
from litert_lm_cli import config
from litert_lm_cli import model


class ModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Ensure local config doesn't interfere with standard tests
    patcher_model = mock.patch.object(
        config, "get_model_config", return_value=config.ModelConfig()
    )
    patcher_model.start()
    self.addCleanup(patcher_model.stop)

  @parameterized.named_parameters(
      ("default_backend", None, None, "cpu", None),
      ("cpu_backend", "cpu", None, "cpu", None),
      ("gpu_backend", "gpu", None, "gpu", None),
      ("npu_backend", "npu", None, "npu", None),
      ("cpu_with_threads", "cpu", 4, "cpu", 4),
      ("default_with_threads", None, 4, "cpu", 4),
  )
  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend(
      self,
      backend,
      cpu_thread_count,
      expected_type_str,
      expected_thread_count,
      mock_default_backend,
  ):
    mock_default_backend.return_value = "cpu"
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    # Mock NPU to avoid RuntimeError on Linux
    with mock.patch.object(
        litert_lm.Backend, "NPU", autospec=True
    ) as mock_npu_class:
      result = model.parse_backend(
          backend=backend,
          model_obj=mock_model,
          cpu_thread_count=cpu_thread_count,
      )

      if expected_type_str == "cpu":
        self.assertIsInstance(result, litert_lm.Backend.CPU)
        self.assertEqual(result.thread_count, expected_thread_count)
      elif expected_type_str == "gpu":
        self.assertIsInstance(result, litert_lm.Backend.GPU)
      elif expected_type_str == "npu":
        mock_npu_class.assert_called_once()
        self.assertEqual(result, mock_npu_class.return_value)

  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_gpu_constraint_explicit_cpu(
      self, mock_default_backend
  ):
    mock_default_backend.return_value = "gpu"
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    # Explicit 'cpu' should be used, even if model defaults to 'gpu'
    result = model.parse_backend(backend="cpu", model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.CPU)

  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_gpu_constraint_default(self, mock_default_backend):
    mock_default_backend.return_value = "gpu"
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    # Default (None) should use model default backend (gpu)
    result = model.parse_backend(backend=None, model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.GPU)

  @parameterized.named_parameters(
      (
          "artisan_always_gpu",
          litertlm_builder.TfLiteModelType.ARTISAN_TEXT_DECODER.value,
          None,
          "gpu",
          None,
      ),
      (
          "artisan_always_gpu_uppercase",
          litertlm_builder.TfLiteModelType.ARTISAN_TEXT_DECODER.value.upper(),
          None,
          "gpu",
          None,
      ),
      (
          "prefill_decode_gpu",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE.value,
          "gpu",
          "gpu",
          None,
      ),
      (
          "prefill_decode_gpu_uppercase",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE.value.upper(),
          "gpu",
          "gpu",
          None,
      ),
      (
          "prefill_decode_cpu",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE.value,
          "cpu",
          "cpu",
          None,
      ),
      (
          "prefill_decode_none",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE.value,
          None,
          "cpu",
          None,
      ),
      ("other_model_gpu_ignored", "other_model", "gpu", "cpu", None),
      (
          "prefill_decode_multi",
          litertlm_builder.TfLiteModelType.PREFILL_DECODE.value,
          "gpu,npu",
          "gpu",
          None,
      ),
      (
          "audio_encoder_hw_cpu",
          litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
          "cpu",
          "cpu",
          {litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value},
      ),
      (
          "audio_encoder_hw_gpu",
          litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
          "gpu",
          "gpu",
          {litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value},
      ),
      (
          "vision_encoder_cpu",
          litertlm_builder.TfLiteModelType.VISION_ENCODER.value,
          "cpu",
          "cpu",
          {litertlm_builder.TfLiteModelType.VISION_ENCODER.value},
      ),
      (
          "vision_encoder_gpu",
          litertlm_builder.TfLiteModelType.VISION_ENCODER.value,
          "gpu",
          "gpu",
          {litertlm_builder.TfLiteModelType.VISION_ENCODER.value},
      ),
      (
          "audio_encoder_hw_ignored_for_main",
          litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW.value,
          "gpu",
          "cpu",
          None,
      ),
  )
  @mock.patch.object(model, "litertlm_peek")
  def test_model_default_backend(
      self,
      model_type,
      backend_constraint,
      expected_backend,
      target_model_types,
      mock_peek,
  ):
    mock_metadata = mock.Mock()
    mock_peek.read_litertlm_header.return_value = mock_metadata

    mock_section_metadata = mock.Mock()
    mock_metadata.SectionMetadata.return_value = mock_section_metadata

    mock_section = mock.Mock()
    mock_section_metadata.ObjectsLength.return_value = 1
    mock_section_metadata.Objects.return_value = mock_section

    mock_peek.get_model_type.return_value = model_type

    if backend_constraint is not None:
      mock_item = mock.Mock()
      mock_section.ItemsLength.return_value = 1
      mock_section.Items.return_value = mock_item
      mock_peek.kvp_to_dict.return_value = {
          "key": "backend_constraint",
          "value": backend_constraint,
          "value_type": "String",
      }
    else:
      mock_section.ItemsLength.return_value = 0

    if target_model_types is None:
      result = model.model_default_backend("dummy_path")
    else:
      result = model.model_default_backend(
          "dummy_path", target_model_types=target_model_types
      )
    self.assertEqual(result, expected_backend)


class ParseBackendWithConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config._clear_cache()

  @mock.patch.object(config, "get_model_config")
  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_with_config(
      self, mock_model_default, mock_get_model_config
  ):
    mock_model_default.return_value = "cpu"
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    # Default mocks
    mock_get_model_config.return_value = config.ModelConfig()

    # 1. Should use backend from get_model_config
    mock_get_model_config.return_value = config.ModelConfig(backend="gpu")
    result = model.parse_backend(backend=None, model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.GPU)
    mock_get_model_config.assert_called_once_with("dummy_model_id")

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 2. Explicit backend overrides config
    mock_get_model_config.return_value = config.ModelConfig(backend="cpu")
    result = model.parse_backend(backend="gpu", model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.GPU)

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 3. Config-defined threads should be used if CLI threads is None
    mock_get_model_config.return_value = config.ModelConfig(
        backend="cpu", cpu_thread_count=4
    )
    result = model.parse_backend(backend=None, model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.CPU)
    self.assertEqual(result.thread_count, 4)

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 4. Explicit CLI threads overrides config threads
    mock_get_model_config.return_value = config.ModelConfig(
        backend="cpu", cpu_thread_count=4
    )
    result = model.parse_backend(
        backend=None, cpu_thread_count=8, model_obj=mock_model
    )
    self.assertIsInstance(result, litert_lm.Backend.CPU)
    self.assertEqual(result.thread_count, 8)

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 5. Config-defined gpu_decode_steps_per_sync should be used if CLI is None
    mock_get_model_config.return_value = config.ModelConfig(
        backend="gpu", gpu_decode_steps_per_sync=4
    )
    result = model.parse_backend(backend=None, model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.GPU)
    self.assertEqual(result.gpu_decode_steps_per_sync, 4)

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 6. Explicit CLI gpu_decode_steps_per_sync overrides config
    mock_get_model_config.return_value = config.ModelConfig(
        backend="gpu", gpu_decode_steps_per_sync=4
    )
    result = model.parse_backend(
        backend=None, gpu_decode_steps_per_sync=8, model_obj=mock_model
    )
    self.assertIsInstance(result, litert_lm.Backend.GPU)
    self.assertEqual(result.gpu_decode_steps_per_sync, 8)

    # Reset mocks
    mock_get_model_config.reset_mock()

    # 5. Config threads should be used with default backend
    mock_model_default.return_value = "cpu"
    mock_get_model_config.return_value = config.ModelConfig(cpu_thread_count=4)
    result = model.parse_backend(backend=None, model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.CPU)
    self.assertEqual(result.thread_count, 4)

  @mock.patch.object(config, "get_model_config")
  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_auxiliary_ignores_config(
      self, mock_model_default, mock_get_model_config
  ):
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    # Config has backend="gpu" and cpu_thread_count=4
    mock_get_model_config.return_value = config.ModelConfig(
        backend="gpu", cpu_thread_count=4
    )

    # When resolving vision backend (auxiliary):
    # It should IGNORE the config backend ("gpu") and threads (4),
    # and fall back to the model's default backend (which we mock as "cpu").
    mock_model_default.return_value = "cpu"
    result = model.parse_backend(
        backend=None,
        model_obj=mock_model,
        target_model_types={"vision_encoder"},  # Auxiliary type
        label="vision",
    )
    self.assertIsInstance(result, litert_lm.Backend.CPU)
    # Thread count should be None (default), NOT 4 from config!
    self.assertIsNone(result.thread_count)

    # When resolving audio backend (auxiliary):
    # It should also IGNORE the config.
    mock_model_default.return_value = "cpu"
    result = model.parse_backend(
        backend=None,
        model_obj=mock_model,
        target_model_types={"audio_encoder_hw"},  # Auxiliary type
        label="audio",
    )
    self.assertIsInstance(result, litert_lm.Backend.CPU)
    self.assertIsNone(result.thread_count)

  @mock.patch.object(config, "get_model_config")
  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_auxiliary_uses_specific_config(
      self, mock_model_default, mock_get_model_config
  ):
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"
    mock_model.model_id = "dummy_model_id"

    mock_get_model_config.return_value = config.ModelConfig(
        vision_backend="gpu", audio_backend="cpu"
    )

    mock_model_default.return_value = "cpu"
    vision_result = model.parse_backend(
        backend=None,
        model_obj=mock_model,
        target_model_types={"vision_encoder"},
        label="vision",
    )
    self.assertIsInstance(vision_result, litert_lm.Backend.GPU)

    audio_result = model.parse_backend(
        backend=None,
        model_obj=mock_model,
        target_model_types={"audio_encoder_hw"},
        label="audio",
    )
    self.assertIsInstance(audio_result, litert_lm.Backend.CPU)

  def test_from_model_path(self):
    path = "/home/user/.litert-lm/models/gemma4--2b/model.litertlm"
    m1 = model.Model.from_model_path(path)
    self.assertEqual(m1.model_id, "gemma4/2b")
    self.assertEqual(m1.model_path, path)

  @mock.patch.object(config, "get_model_config")
  def test_resolve_config_option(self, mock_get_model_config):
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_id = "dummy_model_id"

    mock_get_model_config.return_value = config.ModelConfig(
        cache="memory",
        max_num_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        seed=42,
        gpu_decode_steps_per_sync=4,
        speculative_decoding=True,
        thinking=True,
        thinking_budget=50,
    )

    # Explicit value provided by user
    self.assertEqual(
        model.resolve_config_option("disk", mock_model, "cache"), "disk"
    )

    # Fallback to config
    self.assertEqual(
        model.resolve_config_option(None, mock_model, "cache"), "memory"
    )
    self.assertEqual(
        model.resolve_config_option(None, mock_model, "max_num_tokens"), 1024
    )
    self.assertEqual(
        model.resolve_config_option(None, mock_model, "temperature"), 0.7
    )
    self.assertEqual(
        model.resolve_config_option(None, mock_model, "top_p"), 0.9
    )
    self.assertEqual(model.resolve_config_option(None, mock_model, "top_k"), 40)
    self.assertEqual(model.resolve_config_option(None, mock_model, "seed"), 42)
    self.assertEqual(
        model.resolve_config_option(
            None, mock_model, "gpu_decode_steps_per_sync"
        ),
        4,
    )
    self.assertTrue(
        model.resolve_config_option(None, mock_model, "speculative_decoding")
    )
    self.assertTrue(model.resolve_config_option(None, mock_model, "thinking"))
    self.assertEqual(
        model.resolve_config_option(None, mock_model, "thinking_budget"), 50
    )

    # Unset field returns None
    mock_get_model_config.return_value = config.ModelConfig()
    self.assertIsNone(model.resolve_config_option(None, mock_model, "cache"))

    path3 = "/home/user/.litert-lm/models/google--gemma3-1b-it/model.litertlm"
    m3 = model.Model.from_model_path(path3)
    self.assertEqual(m3.model_id, "google/gemma3-1b-it")
    self.assertEqual(m3.model_path, path3)

    m2 = model.Model.from_model_path("/custom/path/custom_model.litertlm")
    self.assertEqual(m2.model_id, "custom_model.litertlm")
    self.assertEqual(m2.model_path, "/custom/path/custom_model.litertlm")


if __name__ == "__main__":
  absltest.main()
