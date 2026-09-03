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
"""Unit tests for the LiteRT-LM describe command."""

from unittest import mock

from absl.testing import absltest
from click import testing

import litert_lm
from litert_lm_cli import model
import litert_lm_cli.commands.describe as describe_cmd


class DescribeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_model = mock.MagicMock(spec=model.Model)
    self.mock_model.exists.return_value = True
    self.mock_model.model_path = "/path/to/model.litertlm"

    self.mock_from_reference = self.enter_context(
        mock.patch.object(
            model.Model,
            "from_model_reference",
            return_value=self.mock_model,
            autospec=True,
        )
    )

    self.mock_capabilities = mock.MagicMock(spec=litert_lm.Capabilities)
    self.mock_capabilities.max_vision_token_budget = -1
    self.mock_capabilities.vision_signature_selection = None
    self.mock_capabilities.max_context_tokens = 0
    self.mock_capabilities.is_dynamic_context = False
    self.mock_capabilities.supported_backends_for_modality.return_value = [
        "cpu",
        "gpu",
    ]
    self.mock_capabilities.soc_name_for_modality.return_value = None
    self.mock_capabilities_cls = self.enter_context(
        mock.patch.object(
            litert_lm,
            "Capabilities",
            return_value=self.mock_capabilities,
            autospec=True,
        )
    )

  def test_describe_success(self):
    self.mock_capabilities.supports_thinking.return_value = True
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = True

    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=True, audio=False, video=False
    )
    self.mock_capabilities.max_vision_token_budget = 280
    self.mock_capabilities.vision_signature_selection = [280]
    self.mock_capabilities.min_runtime_version = "0.12.3"
    self.mock_capabilities.max_context_tokens = 2048
    self.mock_capabilities.is_dynamic_context = True
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["my-model-id"],
    )

    self.assertEqual(result.exit_code, 0)
    self.mock_from_reference.assert_called_once_with("my-model-id")
    self.mock_capabilities_cls.assert_called_once_with(
        "/path/to/model.litertlm"
    )

    # Verify stdout report formatting
    self.assertEqual(
        result.output.count("========================================"), 3
    )
    self.assertIn("LiteRT-LM Model Capabilities Report", result.output)
    self.assertIn("[LLM Capabilities]", result.output)
    self.assertIn("File: /path/to/model.litertlm", result.output)
    self.assertIn("Supports Function Call: NO", result.output)
    self.assertIn("Supports Thinking:      YES", result.output)
    self.assertIn("Speculative Decoding:   YES", result.output)
    self.assertIn("Max Vision Token Budget: 280", result.output)
    self.assertIn("Sampler Temp:           0.70", result.output)
    self.assertIn("Sampler Top K:          40", result.output)
    self.assertIn("Sampler Top P:          0.90", result.output)
    self.assertIn("Max Context Tokens:     2048", result.output)
    self.assertIn("Is Dynamic Context:     YES", result.output)
    self.assertIn("Input Modalities:       Text Vision", result.output)
    self.assertIn("Text Backends:          CPU GPU", result.output)
    self.assertIn("Vision Backends:        CPU GPU", result.output)
    self.assertIn("Min Runtime Version:    0.12.3", result.output)
    self.assertIn("Vision Signature Selection: [280]", result.output)

  def test_describe_model_multimodal_audio_video(self):
    self.mock_capabilities.supports_thinking.return_value = False
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = False

    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=True, audio=True, video=True
    )
    self.mock_capabilities.min_runtime_version = None
    self.mock_capabilities.vision_signature_selection = None
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["my-model-id"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn(
        "Input Modalities:       Text Vision Audio Video", result.output
    )
    self.assertIn("Text Backends:          CPU GPU", result.output)
    self.assertIn("Vision Backends:        CPU GPU", result.output)
    self.assertIn("Audio Backends:         CPU GPU", result.output)
    self.assertIn("Video Backends:         CPU GPU", result.output)
    self.assertIn("Min Runtime Version:    -1", result.output)
    self.assertIn("Vision Signature Selection: -1", result.output)

  def test_describe_model_not_found(self):
    self.mock_model.exists.return_value = False

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["missing-model-id"],
    )

    self.assertEqual(result.exit_code, 1)
    self.assertIn(
        "Error: Failed to find model 'missing-model-id'", result.output
    )
    self.mock_capabilities_cls.assert_not_called()

  def test_describe_load_error(self):
    self.mock_capabilities_cls.side_effect = RuntimeError("Failed to open file")

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["bad-model-id"],
    )

    self.assertEqual(result.exit_code, 1)
    self.assertIn(
        "Error: Failed to load capabilities for model 'bad-model-id': Failed to"
        " open file",
        result.output,
    )

  def test_describe_from_huggingface(self):
    mock_resolve = self.enter_context(
        mock.patch.object(
            describe_cmd.cli_helpers,
            "resolve_model_file",
            return_value="model.litertlm",
        )
    )
    mock_download = self.enter_context(
        mock.patch.object(
            describe_cmd.huggingface_download,
            "download_from_huggingface",
            return_value="/path/to/downloaded/model.litertlm",
        )
    )
    mock_hf_model = mock.MagicMock(spec=model.Model)
    mock_hf_model.exists.return_value = True
    mock_hf_model.model_path = "/path/to/downloaded/model.litertlm"
    mock_from_path = self.enter_context(
        mock.patch.object(
            model.Model,
            "from_model_path",
            return_value=mock_hf_model,
        )
    )

    self.mock_capabilities.supports_thinking.return_value = True
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = True
    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=False, audio=False, video=False
    )
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.0,
        top_k=None,
        top_p=0.0,
    )

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["--from-huggingface-repo", "org/repo"],
    )

    self.assertEqual(result.exit_code, 0)
    mock_resolve.assert_called_once_with("org/repo", None)
    mock_download.assert_called_once_with(
        repo_id="org/repo",
        filename="model.litertlm",
        token=None,
    )
    mock_from_path.assert_called_once_with("/path/to/downloaded/model.litertlm")
    self.mock_capabilities_cls.assert_called_once_with(
        "/path/to/downloaded/model.litertlm"
    )

    self.assertIn("File: /path/to/downloaded/model.litertlm", result.output)
    self.assertIn("Supports Thinking:      YES", result.output)

  def test_describe_no_args_prompts_imported(self):
    mock_resolve = self.enter_context(
        mock.patch.object(
            describe_cmd.cli_helpers,
            "resolve_model_file",
            return_value="gemma3-1b",
        )
    )
    self.mock_capabilities.supports_thinking.return_value = True
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = True
    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=False, audio=False, video=False
    )
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.0,
        top_k=None,
        top_p=0.0,
    )

    runner = testing.CliRunner()
    result = runner.invoke(describe_cmd.describe_model)

    self.assertEqual(result.exit_code, 0)
    mock_resolve.assert_called_once_with(None, None)
    self.mock_from_reference.assert_called_once_with("gemma3-1b")
    self.mock_capabilities_cls.assert_called_once_with(
        "/path/to/model.litertlm"
    )
    self.assertIn("File: /path/to/model.litertlm", result.output)

  def test_describe_with_default_backend_and_soc_name(self):
    self.mock_capabilities.supports_thinking.return_value = True
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = False
    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=False, audio=False, video=False
    )
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.0,
        top_k=None,
        top_p=0.0,
    )
    self.mock_capabilities.supported_backends_for_modality.return_value = [
        "npu",
    ]
    self.mock_capabilities.npu_brand_for_modality.return_value = (
        litert_lm.LiteRtLmNpuBrand.QUALCOMM
    )
    self.mock_capabilities.soc_name_for_modality.return_value = "SM8750"

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["my-model-id"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn("Text Backends:          NPU", result.output)
    self.assertIn("Text Default Backend:   NPU", result.output)
    self.assertIn("Text SoC Name:          Qualcomm QNN SM8750", result.output)

  def test_describe_with_intel_npu_brand(self):
    self.mock_capabilities.supports_thinking.return_value = False
    self.mock_capabilities.supports_function_calling.return_value = False
    self.mock_capabilities.has_speculative_decoding_support.return_value = False
    self.mock_capabilities.input_modalities = litert_lm.SupportedModalities(
        text=True, vision=False, audio=False, video=False
    )
    self.mock_capabilities.default_sampler_params = litert_lm.SamplerConfig(
        temperature=0.0,
        top_k=None,
        top_p=0.0,
    )
    self.mock_capabilities.supported_backends_for_modality.return_value = [
        "npu",
    ]
    self.mock_capabilities.npu_brand_for_modality.return_value = (
        litert_lm.LiteRtLmNpuBrand.INTEL
    )
    self.mock_capabilities.soc_name_for_modality.return_value = "LunarLake"

    runner = testing.CliRunner()
    result = runner.invoke(
        describe_cmd.describe_model,
        ["my-model-id"],
    )

    self.assertEqual(result.exit_code, 0)
    self.assertIn("Text Backends:          NPU", result.output)
    self.assertIn("Text Default Backend:   NPU", result.output)
    self.assertIn("Text SoC Name:          Intel NPU LunarLake", result.output)


if __name__ == "__main__":
  absltest.main()
