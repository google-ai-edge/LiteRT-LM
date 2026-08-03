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
from litert_lm_cli import model


class ModelTest(parameterized.TestCase):

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

    # Explicit 'cpu' should be used, even if model defaults to 'gpu'
    result = model.parse_backend(backend="cpu", model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.CPU)

  @mock.patch.object(model, "model_default_backend")
  def test_parse_backend_gpu_constraint_default(self, mock_default_backend):
    mock_default_backend.return_value = "gpu"
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = "dummy_path"

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


if __name__ == "__main__":
  absltest.main()
