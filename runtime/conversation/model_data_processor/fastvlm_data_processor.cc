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

#include "runtime/conversation/model_data_processor/fastvlm_data_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<FastVlmDataProcessor>>
FastVlmDataProcessor::Create(
    FastVlmDataProcessorConfig config, std::optional<Preface> preface,
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    bool enable_constrained_decoding) {
  Gemma3DataProcessorConfig gemma3_config;
  gemma3_config.boi_token = config.boi_token;
  gemma3_config.eoi_token = config.eoi_token;
  gemma3_config.image_tensor_height = config.image_tensor_height;
  gemma3_config.image_tensor_width = config.image_tensor_width;

  ASSIGN_OR_RETURN(auto impl, Gemma3DataProcessor::Create(
                                  gemma3_config, preface, tokenizer,
                                  stop_token_ids, enable_constrained_decoding));
  return absl::WrapUnique(new FastVlmDataProcessor(config, std::move(impl)));
}

absl::StatusOr<std::vector<InputData>>
FastVlmDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const FastVlmDataProcessorArguments& args) const {
  return impl_->ToInputDataVector(rendered_template_prompt, messages,
                                  Gemma3DataProcessorArguments{});
}

absl::StatusOr<Message> FastVlmDataProcessor::ToMessageImpl(
    const Responses& responses,
    const FastVlmDataProcessorArguments& args) const {
  return impl_->ToMessage(responses, Gemma3DataProcessorArguments{});
}

absl::Status FastVlmDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                     FastVlmDataProcessorArguments>& other) {
  const FastVlmDataProcessor& other_fastvlm =
      static_cast<const FastVlmDataProcessor&>(other);
  return impl_->CloneState(*other_fastvlm.impl_);
}

}  // namespace litert::lm
