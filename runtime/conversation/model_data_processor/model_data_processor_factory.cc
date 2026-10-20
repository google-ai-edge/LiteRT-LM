// Copyright 2025 The ODML Authors.
// Copyright (C) 2026 Samsung Electronics Co. LTD.
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

#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#ifdef ENABLE_FASTVLM_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/fastvlm_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#ifdef ENABLE_FUNCTION_GEMMA_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/function_gemma_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#ifdef ENABLE_GEMMA3_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#ifdef ENABLE_GEMMA4_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#ifdef ENABLE_LFM2_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/lfm2_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/lfm2_data_processor_config.h"
#ifdef ENABLE_MINICPM5_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/minicpm5_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/minicpm5_data_processor_config.h"
#ifdef ENABLE_MINICPMV_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/minicpmv_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory_common.h"
#ifdef ENABLE_QWEN3_DATA_PROCESSOR
#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"
#endif
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {
namespace {

void EnsureBuiltinProcessorsRegistered() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    auto& reg = ModelDataProcessorRegistry::Instance();
    reg.Register<GenericDataProcessor, GenericDataProcessorConfig>();
#ifdef ENABLE_GEMMA3_DATA_PROCESSOR
    reg.Register<Gemma3DataProcessor, Gemma3DataProcessorConfig>();
#endif
#ifdef ENABLE_GEMMA4_DATA_PROCESSOR
    reg.Register<Gemma4DataProcessor, Gemma4DataProcessorConfig>();
#endif
#ifdef ENABLE_FUNCTION_GEMMA_DATA_PROCESSOR
    reg.Register<FunctionGemmaDataProcessor,
                 FunctionGemmaDataProcessorConfig>();
#endif
#ifdef ENABLE_QWEN3_DATA_PROCESSOR
    reg.Register<Qwen3DataProcessor, Qwen3DataProcessorConfig>();
#endif
#ifdef ENABLE_FASTVLM_DATA_PROCESSOR
    reg.Register<FastVlmDataProcessor, FastVlmDataProcessorConfig>();
#endif
#ifdef ENABLE_MINICPM5_DATA_PROCESSOR
    reg.Register<MiniCpm5DataProcessor, MiniCpm5DataProcessorConfig>();
#endif
#ifdef ENABLE_MINICPMV_DATA_PROCESSOR
    reg.Register<MiniCpmVDataProcessor, MiniCpmVDataProcessorConfig>();
#endif
#ifdef ENABLE_LFM2_DATA_PROCESSOR
    reg.Register<Lfm2DataProcessor, Lfm2DataProcessorConfig>();
#endif
  });
}

}  // namespace

absl::StatusOr<DataProcessorConfig> CreateFunctionGemmaDataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_function_gemma()) {
    return absl::InvalidArgumentError(
        "FunctionGemma LlmModelType is required to create "
        "FunctionGemmaDataProcessorConfig.");
  }
  FunctionGemmaDataProcessorConfig config;
  proto::FunctionGemma function_gemma = model_type.function_gemma();
  const auto& default_function_gemma = proto::FunctionGemma::default_instance();
  if (function_gemma.code_fence_start() !=
      default_function_gemma.code_fence_start()) {
    config.code_fence_start = function_gemma.code_fence_start();
  }
  if (function_gemma.code_fence_end() !=
      default_function_gemma.code_fence_end()) {
    config.code_fence_end = function_gemma.code_fence_end();
  }
  if (function_gemma.syntax_type() != default_function_gemma.syntax_type()) {
    config.syntax_type = function_gemma.syntax_type();
  }
  if (function_gemma.escape_fence_strings() !=
      default_function_gemma.escape_fence_strings()) {
    config.escape_fence_strings = function_gemma.escape_fence_strings();
  }
  if (function_gemma.tool_code_regex() !=
      default_function_gemma.tool_code_regex()) {
    config.tool_code_regex = function_gemma.tool_code_regex();
  }
  if (function_gemma.use_template_for_fc_format() !=
      default_function_gemma.use_template_for_fc_format()) {
    config.use_template_for_fc_format =
        function_gemma.use_template_for_fc_format();
  }
  if (function_gemma.open_quote() != default_function_gemma.open_quote()) {
    config.open_quote = function_gemma.open_quote();
  }
  if (function_gemma.close_quote() != default_function_gemma.close_quote()) {
    config.close_quote = function_gemma.close_quote();
  }
  if (function_gemma.function_response_start() !=
      default_function_gemma.function_response_start()) {
    config.function_response_start = function_gemma.function_response_start();
  }
  if (function_gemma.constraint_mode() !=
      default_function_gemma.constraint_mode()) {
    switch (function_gemma.constraint_mode()) {
      case proto::CONSTRAINT_MODE_FUNCTION_CALL_ONLY:
        config.constraint_mode =
            FunctionGemmaDataProcessorConfig::ConstraintMode::kFunctionCallOnly;
        break;
      case proto::CONSTRAINT_MODE_TEXT_AND_OR:
      default:
        config.constraint_mode =
            FunctionGemmaDataProcessorConfig::ConstraintMode::kTextAndOr;
        break;
    }
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateFastVlmDataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_fast_vlm()) {
    return absl::InvalidArgumentError(
        "FastVlm LlmModelType is required to create "
        "FastVlmDataProcessorConfig.");
  }
  FastVlmDataProcessorConfig config;
  proto::FastVlm fast_vlm = model_type.fast_vlm();
  const auto& default_fast_vlm = proto::FastVlm::default_instance();
  if (fast_vlm.image_tensor_height() !=
      default_fast_vlm.image_tensor_height()) {
    config.image_tensor_height = fast_vlm.image_tensor_height();
  }
  if (fast_vlm.image_tensor_width() != default_fast_vlm.image_tensor_width()) {
    config.image_tensor_width = fast_vlm.image_tensor_width();
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateQwen3DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_qwen3() && !model_type.has_qwen2p5()) {
    return absl::InvalidArgumentError(
        "Qwen3 or Qwen2.5 LlmModelType is required to create "
        "Qwen3DataProcessorConfig.");
  }
  Qwen3DataProcessorConfig config;
  if (model_type.has_qwen3()) {
    if (model_type.qwen3().has_code_fence_start()) {
      config.code_fence_start = model_type.qwen3().code_fence_start();
    }
    if (model_type.qwen3().has_code_fence_end()) {
      config.code_fence_end = model_type.qwen3().code_fence_end();
    }
    if (model_type.qwen3().has_escape_fence_strings()) {
      config.escape_fence_strings = model_type.qwen3().escape_fence_strings();
    }
    if (model_type.qwen3().has_tool_code_regex()) {
      config.tool_code_regex = model_type.qwen3().tool_code_regex();
    }
  }
  if (model_type.has_qwen2p5()) {
    if (model_type.qwen2p5().has_code_fence_start()) {
      config.code_fence_start = model_type.qwen2p5().code_fence_start();
    }
    if (model_type.qwen2p5().has_code_fence_end()) {
      config.code_fence_end = model_type.qwen2p5().code_fence_end();
    }
    if (model_type.qwen2p5().has_escape_fence_strings()) {
      config.escape_fence_strings = model_type.qwen2p5().escape_fence_strings();
    }
    if (model_type.qwen2p5().has_tool_code_regex()) {
      config.tool_code_regex = model_type.qwen2p5().tool_code_regex();
    }
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateLfm2DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_lfm2()) {
    return absl::InvalidArgumentError(
        "Lfm2 LlmModelType is required to create Lfm2DataProcessorConfig.");
  }
  Lfm2DataProcessorConfig config;
  const proto::Lfm2& lfm2 = model_type.lfm2();
  const auto& default_lfm2 = proto::Lfm2::default_instance();
  if (lfm2.has_start_of_image_token()) {
    ABSL_ASSIGN_OR_RETURN(config.boi_token,
                          GetTokenString(lfm2.start_of_image_token()));
  }
  if (lfm2.has_end_of_image_token()) {
    ABSL_ASSIGN_OR_RETURN(config.eoi_token,
                          GetTokenString(lfm2.end_of_image_token()));
  }
  if (lfm2.patch_width() != default_lfm2.patch_width()) {
    config.patch_width = lfm2.patch_width();
  }
  if (lfm2.patch_height() != default_lfm2.patch_height()) {
    config.patch_height = lfm2.patch_height();
  }
  if (lfm2.max_num_patches() != default_lfm2.max_num_patches()) {
    config.max_num_patches = lfm2.max_num_patches();
  }
  if (lfm2.pooling_kernel_size() != default_lfm2.pooling_kernel_size()) {
    config.pooling_kernel_size = lfm2.pooling_kernel_size();
  }
  if (lfm2.image_tensor_height() != default_lfm2.image_tensor_height()) {
    config.image_height = lfm2.image_tensor_height();
  }
  if (lfm2.image_tensor_width() != default_lfm2.image_tensor_width()) {
    config.image_width = lfm2.image_tensor_width();
  }
  if (!lfm2.normalization_mean().empty()) {
    config.normalization_mean.assign(lfm2.normalization_mean().begin(),
                                     lfm2.normalization_mean().end());
  }
  if (!lfm2.normalization_std().empty()) {
    config.normalization_std.assign(lfm2.normalization_std().begin(),
                                    lfm2.normalization_std().end());
  }
  if (lfm2.normalization_rescale_factor() !=
      default_lfm2.normalization_rescale_factor()) {
    config.normalization_rescale_factor = lfm2.normalization_rescale_factor();
  }
  if (lfm2.has_code_fence_start()) {
    config.code_fence_start = lfm2.code_fence_start();
  }
  if (lfm2.has_code_fence_end()) {
    config.code_fence_end = lfm2.code_fence_end();
  }
  if (lfm2.has_escape_fence_strings()) {
    config.escape_fence_strings = lfm2.escape_fence_strings();
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateMiniCpm5DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_minicpm5()) {
    return absl::InvalidArgumentError(
        "MiniCPM5 LlmModelType is required to create "
        "MiniCpm5DataProcessorConfig.");
  }
  MiniCpm5DataProcessorConfig config;
  const proto::MiniCPM5& minicpm5 = model_type.minicpm5();
  if (minicpm5.has_code_fence_start()) {
    config.code_fence_start = minicpm5.code_fence_start();
  }
  if (minicpm5.has_code_fence_end()) {
    config.code_fence_end = minicpm5.code_fence_end();
  }
  if (minicpm5.has_escape_fence_strings()) {
    config.escape_fence_strings = minicpm5.escape_fence_strings();
  }
  if (minicpm5.has_tool_code_regex()) {
    config.tool_code_regex = minicpm5.tool_code_regex();
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateMiniCpmVDataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_minicpmv4()) {
    return absl::InvalidArgumentError(
        "MiniCpmV4 LlmModelType is required to create "
        "MiniCpmVDataProcessorConfig.");
  }
  return MiniCpmVDataProcessorConfig();
}

absl::StatusOr<DataProcessorConfig> CreateDataProcessorConfigFromLlmModelType(
    const proto::LlmModelType& model_type,
    const std::optional<proto::ToolCallConfig>& tool_call_config) {
  if (tool_call_config.has_value() && !model_type.has_generic_model()) {
    return absl::InvalidArgumentError(
        "tool_call_config is only supported for generic_model.");
  }
  switch (model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3:
    case proto::LlmModelType::kGemma3N:
      return CreateGemma3DataProcessorConfig(model_type);
    case proto::LlmModelType::kLfm2:
      return CreateLfm2DataProcessorConfig(model_type);
    case proto::LlmModelType::kMinicpm5:
      return CreateMiniCpm5DataProcessorConfig(model_type);
    case proto::LlmModelType::kMinicpmv4:
      return CreateMiniCpmVDataProcessorConfig(model_type);
    case proto::LlmModelType::kGemma4:
      return CreateGemma4DataProcessorConfig(model_type);
    case proto::LlmModelType::kQwen3:
    case proto::LlmModelType::kQwen2P5:
      return CreateQwen3DataProcessorConfig(model_type);
    case proto::LlmModelType::kGenericModel:
      return CreateGenericDataProcessorConfig(model_type, tool_call_config);
    case proto::LlmModelType::kFastVlm:
      return CreateFastVlmDataProcessorConfig(model_type);
    case proto::LlmModelType::kFunctionGemma:
      return CreateFunctionGemmaDataProcessorConfig(model_type);
    default:
      return absl::InvalidArgumentError("Unsupported model type");
  }
}

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateModelDataProcessor(
    const DataProcessorConfig& config, std::optional<Preface> preface,
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    bool enable_constrained_decoding) {
  EnsureBuiltinProcessorsRegistered();
  return ModelDataProcessorRegistry::Instance().Create(
      config, std::move(preface), tokenizer, stop_token_ids,
      enable_constrained_decoding);
}

}  // namespace litert::lm
