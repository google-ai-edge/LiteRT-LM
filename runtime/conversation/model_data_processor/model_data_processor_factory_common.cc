// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/model_data_processor/model_data_processor_factory_common.h"

#include <optional>
#include <string>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

absl::StatusOr<std::string> GetTokenString(
    const proto::TokenUnion& token_union) {
  if (token_union.has_token_str()) {
    return token_union.token_str();
  }
  return absl::InvalidArgumentError(
      "token_str field is not set in TokenUnion.");
}

absl::StatusOr<DataProcessorConfig> CreateGemma3DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  Gemma3DataProcessorConfig config;
  if (model_type.has_gemma3n()) {
    proto::Gemma3N gemma3n = model_type.gemma3n();
    if (gemma3n.has_start_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(config.boi_token,
                            GetTokenString(gemma3n.start_of_image_token()));
    }
    if (gemma3n.has_end_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(config.eoi_token,
                            GetTokenString(gemma3n.end_of_image_token()));
    }
    if (gemma3n.has_start_of_audio_token()) {
      ABSL_ASSIGN_OR_RETURN(config.boa_token,
                            GetTokenString(gemma3n.start_of_audio_token()));
    }
    if (gemma3n.has_end_of_audio_token()) {
      ABSL_ASSIGN_OR_RETURN(config.eoa_token,
                            GetTokenString(gemma3n.end_of_audio_token()));
    }
    const auto& default_gemma3n = proto::Gemma3N::default_instance();
    if (gemma3n.image_tensor_height() !=
        default_gemma3n.image_tensor_height()) {
      config.image_tensor_height = gemma3n.image_tensor_height();
    }
    if (gemma3n.image_tensor_width() != default_gemma3n.image_tensor_width()) {
      config.image_tensor_width = gemma3n.image_tensor_width();
    }
  } else if (model_type.has_gemma3()) {
    proto::Gemma3 gemma3 = model_type.gemma3();
    if (gemma3.has_start_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(config.boi_token,
                            GetTokenString(gemma3.start_of_image_token()));
    }
    if (gemma3.has_end_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(config.eoi_token,
                            GetTokenString(gemma3.end_of_image_token()));
    }
    const auto& default_gemma3 = proto::Gemma3::default_instance();
    if (gemma3.image_tensor_height() != default_gemma3.image_tensor_height()) {
      config.image_tensor_height = gemma3.image_tensor_height();
    }
    if (gemma3.image_tensor_width() != default_gemma3.image_tensor_width()) {
      config.image_tensor_width = gemma3.image_tensor_width();
    }
  } else {
    return absl::InvalidArgumentError(
        "Gemma3N or Gemma3 LlmModelType is required to create "
        "Gemma3DataProcessorConfig.");
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateGemma4DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_gemma4()) {
    return absl::InvalidArgumentError(
        "Gemma4 LlmModelType is required to create "
        "Gemma4DataProcessorConfig.");
  }
  Gemma4DataProcessorConfig config;
  proto::Gemma4 gemma4 = model_type.gemma4();
  if (gemma4.has_start_of_image_token()) {
    ABSL_ASSIGN_OR_RETURN(config.boi_token,
                          GetTokenString(gemma4.start_of_image_token()));
  }
  if (gemma4.has_end_of_image_token()) {
    ABSL_ASSIGN_OR_RETURN(config.eoi_token,
                          GetTokenString(gemma4.end_of_image_token()));
  }
  if (gemma4.has_start_of_audio_token()) {
    ABSL_ASSIGN_OR_RETURN(config.boa_token,
                          GetTokenString(gemma4.start_of_audio_token()));
  }
  if (gemma4.has_end_of_audio_token()) {
    ABSL_ASSIGN_OR_RETURN(config.eoa_token,
                          GetTokenString(gemma4.end_of_audio_token()));
  }
  const auto& default_gemma4 = proto::Gemma4::default_instance();
  if (gemma4.code_fence_start() != default_gemma4.code_fence_start()) {
    config.code_fence_start = gemma4.code_fence_start();
  }
  if (gemma4.code_fence_end() != default_gemma4.code_fence_end()) {
    config.code_fence_end = gemma4.code_fence_end();
  }
  if (gemma4.syntax_type() != default_gemma4.syntax_type()) {
    config.syntax_type = gemma4.syntax_type();
  }
  if (gemma4.escape_fence_strings() != default_gemma4.escape_fence_strings()) {
    config.escape_fence_strings = gemma4.escape_fence_strings();
  }
  if (gemma4.tool_code_regex() != default_gemma4.tool_code_regex()) {
    config.tool_code_regex = gemma4.tool_code_regex();
  }
  if (gemma4.open_quote() != default_gemma4.open_quote()) {
    config.open_quote = gemma4.open_quote();
  }
  if (gemma4.close_quote() != default_gemma4.close_quote()) {
    config.close_quote = gemma4.close_quote();
  }
  if (gemma4.function_response_start() !=
      default_gemma4.function_response_start()) {
    config.function_response_start = gemma4.function_response_start();
  }
  if (gemma4.constraint_mode() != default_gemma4.constraint_mode()) {
    switch (gemma4.constraint_mode()) {
      case proto::CONSTRAINT_MODE_FUNCTION_CALL_ONLY:
        config.constraint_mode =
            Gemma4DataProcessorConfig::ConstraintMode::kFunctionCallOnly;
        break;
      case proto::CONSTRAINT_MODE_TEXT_AND_OR:
      default:
        config.constraint_mode =
            Gemma4DataProcessorConfig::ConstraintMode::kTextAndOr;
        break;
    }
  }
  if (gemma4.patch_width() != default_gemma4.patch_width()) {
    config.patch_width = gemma4.patch_width();
  }
  if (gemma4.patch_height() != default_gemma4.patch_height()) {
    config.patch_height = gemma4.patch_height();
  }
  if (gemma4.max_num_patches() != default_gemma4.max_num_patches()) {
    config.max_num_patches = gemma4.max_num_patches();
  }
  if (gemma4.pooling_kernel_size() != default_gemma4.pooling_kernel_size()) {
    config.pooling_kernel_size = gemma4.pooling_kernel_size();
  }
  if (gemma4.merge_patches() != default_gemma4.merge_patches()) {
    config.merge_patches = gemma4.merge_patches();
  }
  if (gemma4.skip_mel_spectrogram_extraction() !=
      default_gemma4.skip_mel_spectrogram_extraction()) {
    config.skip_mel_spectrogram_extraction =
        gemma4.skip_mel_spectrogram_extraction();
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateGenericDataProcessorConfig(
    const proto::LlmModelType& model_type,
    const std::optional<proto::ToolCallConfig>& tool_call_config) {
  if (!model_type.has_generic_model()) {
    return absl::InvalidArgumentError(
        "GenericModel LlmModelType is required to create "
        "GenericDataProcessorConfig.");
  }
  GenericDataProcessorConfig config;
  if (tool_call_config.has_value()) {
    ABSL_RETURN_IF_ERROR(
        GenericDataProcessor::ValidateToolCallConfig(*tool_call_config));
    config.tool_call_config = tool_call_config;
  }
  proto::GenericModel generic_model = model_type.generic_model();

  if (generic_model.image_enabled() || generic_model.audio_enabled()) {
    MultimodalConfig multi_config;
    multi_config.image_enabled = generic_model.image_enabled();
    multi_config.audio_enabled = generic_model.audio_enabled();

    if (generic_model.has_delimiter_regex()) {
      multi_config.processing_config.delimiter_regex =
          generic_model.delimiter_regex();
    }
    if (generic_model.has_image_token_regex()) {
      multi_config.processing_config.image_token_regex =
          generic_model.image_token_regex();
    }
    if (generic_model.has_audio_token_regex()) {
      multi_config.processing_config.audio_token_regex =
          generic_model.audio_token_regex();
    }

    int height = 0;
    int width = 0;
    if (generic_model.has_image_tensor_height()) {
      height = generic_model.image_tensor_height();
    }
    if (generic_model.has_image_tensor_width()) {
      width = generic_model.image_tensor_width();
    }
    multi_config.image_preprocess_parameter.SetTargetDimensions(
        Dimensions({1, height, width, 3}));

    if (generic_model.has_patch_width() || generic_model.has_patch_height() ||
        generic_model.has_max_num_patches() ||
        generic_model.has_pooling_kernel_size()) {
      multi_config.image_preprocess_parameter.SetPatchifyConfig(
          ImagePreprocessParameter::PatchifyConfig{
              .patch_width = generic_model.patch_width(),
              .patch_height = generic_model.patch_height(),
              .max_num_patches = generic_model.max_num_patches(),
              .pooling_kernel_size = generic_model.pooling_kernel_size(),
          });
    }

    if (generic_model.has_start_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(
          multi_config.processing_config.boi_token,
          GetTokenString(generic_model.start_of_image_token()));
    }
    if (generic_model.has_end_of_image_token()) {
      ABSL_ASSIGN_OR_RETURN(multi_config.processing_config.eoi_token,
                            GetTokenString(generic_model.end_of_image_token()));
    }
    if (generic_model.has_image_prefix()) {
      multi_config.processing_config.image_prefix =
          generic_model.image_prefix();
    }
    if (generic_model.has_image_suffix()) {
      multi_config.processing_config.image_suffix =
          generic_model.image_suffix();
    }
    multi_config.processing_config.add_image_end =
        generic_model.add_image_end();

    int sample_rate_hz = 16000;
    int num_channels = 1;
    int frame_length = 512;
    int hop_length = 160;
    int fft_length = 1024;
    float input_scale = 32768.0;
    float pre_emphasis_factor = 0.97;
    int num_mel_bins = 128;
    float mel_low_hz = 125.0;
    float mel_high_hz = 7500.0;
    float mel_floor = 1e-6;
    bool normalize_mel = true;
    bool add_floor_to_mel_before_log = false;
    bool semicausal_padding = false;
    bool non_zero_hanning = true;
    bool periodic_hanning = true;
    AudioPreprocessorConfig::FftPaddingType fft_padding_type =
        AudioPreprocessorConfig::FftPaddingType::kRight;
    bool skip_mel_spectrogram_extraction = false;
    bool buffer_last_frame = false;

    if (generic_model.has_audio_sample_rate_hz()) {
      sample_rate_hz = generic_model.audio_sample_rate_hz();
    }
    if (generic_model.has_audio_num_channels()) {
      num_channels = generic_model.audio_num_channels();
    }
    if (generic_model.has_audio_frame_length()) {
      frame_length = generic_model.audio_frame_length();
    }
    if (generic_model.has_audio_hop_length()) {
      hop_length = generic_model.audio_hop_length();
    }
    if (generic_model.has_audio_fft_length()) {
      fft_length = generic_model.audio_fft_length();
    }
    if (generic_model.has_audio_input_scale()) {
      input_scale = generic_model.audio_input_scale();
    }
    if (generic_model.has_audio_pre_emphasis_factor()) {
      pre_emphasis_factor = generic_model.audio_pre_emphasis_factor();
    }
    if (generic_model.has_audio_num_mel_bins()) {
      num_mel_bins = generic_model.audio_num_mel_bins();
    }
    if (generic_model.has_audio_mel_low_hz()) {
      mel_low_hz = generic_model.audio_mel_low_hz();
    }
    if (generic_model.has_audio_mel_high_hz()) {
      mel_high_hz = generic_model.audio_mel_high_hz();
    }
    if (generic_model.has_audio_mel_floor()) {
      mel_floor = generic_model.audio_mel_floor();
    }
    normalize_mel = generic_model.audio_normalize_mel();
    add_floor_to_mel_before_log =
        generic_model.audio_add_floor_to_mel_before_log();
    semicausal_padding = generic_model.audio_semicausal_padding();
    non_zero_hanning = generic_model.audio_non_zero_hanning();
    periodic_hanning = generic_model.audio_periodic_hanning();
    skip_mel_spectrogram_extraction =
        generic_model.skip_mel_spectrogram_extraction();
    if (generic_model.has_audio_fft_padding_type()) {
      switch (generic_model.audio_fft_padding_type()) {
        case litert::lm::proto::FFT_PADDING_TYPE_RIGHT:
          fft_padding_type = AudioPreprocessorConfig::FftPaddingType::kRight;
          break;
        case litert::lm::proto::FFT_PADDING_TYPE_CENTER:
          fft_padding_type = AudioPreprocessorConfig::FftPaddingType::kCenter;
          break;
        default:
          break;
      }
    }

    multi_config.audio_preprocessor_config = AudioPreprocessorConfig::Create(
        sample_rate_hz, num_channels, frame_length, hop_length, fft_length,
        input_scale, pre_emphasis_factor, num_mel_bins, mel_low_hz, mel_high_hz,
        mel_floor, normalize_mel, add_floor_to_mel_before_log,
        semicausal_padding, non_zero_hanning, periodic_hanning,
        fft_padding_type, skip_mel_spectrogram_extraction, buffer_last_frame);

    if (generic_model.has_start_of_audio_token()) {
      ABSL_ASSIGN_OR_RETURN(
          multi_config.processing_config.boa_token,
          GetTokenString(generic_model.start_of_audio_token()));
    }
    if (generic_model.has_end_of_audio_token()) {
      ABSL_ASSIGN_OR_RETURN(multi_config.processing_config.eoa_token,
                            GetTokenString(generic_model.end_of_audio_token()));
    }
    if (generic_model.has_audio_prefix()) {
      multi_config.processing_config.audio_prefix =
          generic_model.audio_prefix();
    }
    if (generic_model.has_audio_suffix()) {
      multi_config.processing_config.audio_suffix =
          generic_model.audio_suffix();
    }
    multi_config.processing_config.add_audio_end =
        generic_model.add_audio_end();

    config.multimodal = multi_config;
  }

  return config;
}

absl::Status ValidateVisualTokenBudget(const DataProcessorArguments& args,
                                       int max_vision_tokens_per_image) {
  std::optional<int> visual_token_budget;
  if (const auto* gemma4_args =
          std::get_if<Gemma4DataProcessorArguments>(&args)) {
    visual_token_budget = gemma4_args->visual_token_budget;
  } else if (const auto* generic_args =
                 std::get_if<GenericDataProcessorArguments>(&args)) {
    visual_token_budget = generic_args->visual_token_budget;
  }
  if (visual_token_budget.has_value()) {
    if (*visual_token_budget <= 0) {
      return absl::InvalidArgumentError(
          "Visual token budget must be positive.");
    }
    if (*visual_token_budget > max_vision_tokens_per_image) {
      return absl::InvalidArgumentError(
          absl::StrCat("Visual token budget (", *visual_token_budget,
                       ") cannot be larger than the engine's max vision "
                       "tokens per image (",
                       max_vision_tokens_per_image, ")."));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
