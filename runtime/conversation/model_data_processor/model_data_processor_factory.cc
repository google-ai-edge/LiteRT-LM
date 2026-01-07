// Copyright 2025 The ODML Authors.
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
#include <string>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::string> GetTokenString(
    const proto::TokenUnion& token_union) {
  if (token_union.has_token_str()) {
    return token_union.token_str();
  } else {
    return absl::InvalidArgumentError("TokenUnion does not have a token_str.");
  }
}

absl::StatusOr<DataProcessorConfig> CreateGemma3DataProcessorConfig(
    const proto::LlmModelType& model_type) {
  Gemma3DataProcessorConfig config;
  if (model_type.has_gemma3n()) {
    proto::Gemma3N gemma3n = model_type.gemma3n();
    if (gemma3n.has_start_of_image_token()) {
      ASSIGN_OR_RETURN(config.boi_token,
                       GetTokenString(gemma3n.start_of_image_token()));
    }
    if (gemma3n.has_end_of_image_token()) {
      ASSIGN_OR_RETURN(config.eoi_token,
                       GetTokenString(gemma3n.end_of_image_token()));
    }
    if (gemma3n.has_start_of_audio_token()) {
      ASSIGN_OR_RETURN(config.boa_token,
                       GetTokenString(gemma3n.start_of_audio_token()));
    }
    if (gemma3n.has_end_of_audio_token()) {
      ASSIGN_OR_RETURN(config.eoa_token,
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
      ASSIGN_OR_RETURN(config.boi_token,
                       GetTokenString(gemma3.start_of_image_token()));
    }
    if (gemma3.has_end_of_image_token()) {
      ASSIGN_OR_RETURN(config.eoi_token,
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
  if (function_gemma.constraint_mode() !=
      default_function_gemma.constraint_mode()) {
    switch (function_gemma.constraint_mode()) {
      case proto::CONSTRAINT_MODE_FUNCTION_CALL_ONLY:
        config.constraint_mode = ConstraintMode::kFunctionCallOnly;
        break;
      case proto::CONSTRAINT_MODE_TEXT_AND_OR:
      default:
        config.constraint_mode = ConstraintMode::kTextAndOr;
        break;
    }
  }
  return config;
}

absl::StatusOr<DataProcessorConfig> CreateGenericDataProcessorConfig(
    const proto::LlmModelType& model_type) {
  if (!model_type.has_generic_model()) {
    return absl::InvalidArgumentError(
        "GenericModel LlmModelType is required to create "
        "GenericDataProcessorConfig.");
  }
  GenericDataProcessorConfig config;
  if (model_type.generic_model().has_model_role()) {
    config.model_role = model_type.generic_model().model_role();
  }
  if (model_type.generic_model().has_force_string_content()) {
    config.force_string_content =
        model_type.generic_model().force_string_content();
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

absl::StatusOr<DataProcessorConfig> CreateDataProcessorConfigFromLlmModelType(
    const proto::LlmModelType& model_type) {
  switch (model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3:
    case proto::LlmModelType::kGemma3N:
      return CreateGemma3DataProcessorConfig(model_type);
    case proto::LlmModelType::kQwen3:
    case proto::LlmModelType::kQwen2P5:
      return CreateQwen3DataProcessorConfig(model_type);
    case proto::LlmModelType::kGenericModel:
      return CreateGenericDataProcessorConfig(model_type);
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
    bool enable_constrained_decoding, PromptTemplateCapabilities capabilities) {
  if (std::holds_alternative<Gemma3DataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating Gemma3DataProcessor";
    return Gemma3DataProcessor::Create(
        std::get<Gemma3DataProcessorConfig>(config), preface, tokenizer,
        stop_token_ids, enable_constrained_decoding);
  } else if (std::holds_alternative<Qwen3DataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating Qwen3DataProcessor";
    return Qwen3DataProcessor::Create(
        std::get<Qwen3DataProcessorConfig>(config), preface);
  } else if (std::holds_alternative<GenericDataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating GenericDataProcessor";
    return GenericDataProcessor::Create(
        std::get<GenericDataProcessorConfig>(config), capabilities);
  } else if (std::holds_alternative<FunctionGemmaDataProcessorConfig>(config)) {
    ABSL_LOG(INFO) << "Creating FunctionGemmaDataProcessor";
    return FunctionGemmaDataProcessor::Create(
        std::get<FunctionGemmaDataProcessorConfig>(config), preface, tokenizer,
        stop_token_ids, enable_constrained_decoding);
  } else {
    return absl::InvalidArgumentError("Unsupported data processor config type");
  }
}

}  // namespace litert::lm
