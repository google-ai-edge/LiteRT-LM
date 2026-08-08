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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPM5_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPM5_DATA_PROCESSOR_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpm5_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// MiniCpm5DataProcessor is a ModelDataProcessor for MiniCPM5 models.
class MiniCpm5DataProcessor
    : public TypeSafeModelDataProcessor<MiniCpm5DataProcessorConfig,
                                        MiniCpm5DataProcessorArguments> {
 public:
  static absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Create(
      MiniCpm5DataProcessorConfig config,
      std::optional<Preface> preface = std::nullopt);

  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override {
    return tools;
  }

  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  absl::string_view CodeFenceStart() const override;

  absl::string_view CodeFenceEnd() const override;

  const MiniCpm5DataProcessorConfig& GetConfig() const override {
    return config_;
  }

 private:
  explicit MiniCpm5DataProcessor(MiniCpm5DataProcessorConfig config,
                                 std::optional<Preface> preface)
      : config_(std::move(config)), preface_(std::move(preface)) {}

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const MiniCpm5DataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const MiniCpm5DataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<MiniCpm5DataProcessorConfig,
                                       MiniCpm5DataProcessorArguments>& other)
      override {
    ABSL_VLOG(1) << "MiniCpm5DataProcessor::CloneStateImpl is a no-op.";
    return absl::OkStatus();
  }

  MiniCpm5DataProcessorConfig config_;
  std::optional<Preface> preface_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPM5_DATA_PROCESSOR_H_
