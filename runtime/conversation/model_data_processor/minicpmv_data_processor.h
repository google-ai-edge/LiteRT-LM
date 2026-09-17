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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Data processor for MiniCPM-V-4 multi-slice vision (fused navit + resampler).
//
// Replaces the chat-template marker "<image_soft_token>" with the official
// layout (a thumbnail plus optional <slice> cells) and one InputImage per
// slice for the stock map-based Encode (images / positions_xy / vit_positions
// -> 64 soft tokens). The LLM placeholders are filled via
// EmbeddingLookupMultiModal.
class MiniCpmVDataProcessor
    : public TypeSafeModelDataProcessor<MiniCpmVDataProcessorConfig,
                                        MiniCpmVDataProcessorArguments> {
 public:
  // Creates a MiniCpmVDataProcessor instance. If `image_preprocessor` is null,
  // the platform default ImagePreprocessor::Create() is used.
  static absl::StatusOr<std::unique_ptr<MiniCpmVDataProcessor>> Create(
      MiniCpmVDataProcessorConfig config,
      std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr);

  // Returns the config of the MiniCpmVDataProcessor.
  const MiniCpmVDataProcessorConfig& GetConfig() const override {
    return config_;
  }

  // Formats tool declarations.
  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override;

  // Returns the start of tool call blocks.
  absl::string_view CodeFenceStart() const override { return ""; }

  // Returns the end of tool call blocks.
  absl::string_view CodeFenceEnd() const override { return ""; }

 private:
  MiniCpmVDataProcessor(MiniCpmVDataProcessorConfig config,
                        std::unique_ptr<ImagePreprocessor> image_preprocessor)
      : config_(config),
        image_preprocessor_(std::move(image_preprocessor)) {}

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const MiniCpmVDataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const MiniCpmVDataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<MiniCpmVDataProcessorConfig,
                                       MiniCpmVDataProcessorArguments>& other)
      override;

  MiniCpmVDataProcessorConfig config_;
  // Decodes image bytes for PreprocessImageSliced. Held here so that the
  // backing image codec is set up once per model, not once per image.
  std::unique_ptr<ImagePreprocessor> image_preprocessor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_
