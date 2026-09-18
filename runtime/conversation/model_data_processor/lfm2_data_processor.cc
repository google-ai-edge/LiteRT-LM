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
#include "runtime/conversation/model_data_processor/lfm2_data_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl        // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl        // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl      // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json           // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/lfm2_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/engine/io_types.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/preprocessor/stb_image_preprocessor.h"  // IWYU pragma: keep

namespace litert::lm {

namespace {
using ::nlohmann::ordered_json;

// The number of channels (RGB) the image encoder expects.
constexpr int kImageChannels = 3;
}  // namespace

absl::StatusOr<std::unique_ptr<Lfm2DataProcessor>> Lfm2DataProcessor::Create(
    Lfm2DataProcessorConfig config, std::optional<Preface> preface,
    const ::litert::support::Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    bool enable_constrained_decoding) {
  if (enable_constrained_decoding) {
    return absl::FailedPreconditionError(
        "Constrained decoding is not supported for Lfm2DataProcessor.");
  }
  return absl::WrapUnique(new Lfm2DataProcessor(
      std::move(config), preface, std::make_unique<StbImagePreprocessor>()));
}

absl::StatusOr<std::unique_ptr<Constraint>> Lfm2DataProcessor::CreateConstraint(
    const ordered_json& tools) const {
  return absl::FailedPreconditionError(
      "Constrained decoding is not supported for Lfm2DataProcessor.");
}

absl::string_view Lfm2DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Lfm2DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

absl::StatusOr<std::vector<InputData>> Lfm2DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const Lfm2DataProcessorArguments& args) const {
  MultimodalPromptProcessingConfig multi_config{
      .delimiter_regex = R"regex((<\|image_start\|>|<image>))regex",
      .image_token_regex = R"regex((<\|image_start\|>|<image>))regex",
      .boi_token = config_.boi_token,
      .image_suffix = config_.eoi_token,
      .add_image_end = false,
  };
  using ImageParam = ::litert::support::ImagePreprocessParameter;
  ImageParam image_params;
  image_params.SetPatchifyConfig(ImageParam::PatchifyConfig{
      .patch_width = config_.patch_width,
      .patch_height = config_.patch_height,
      .max_num_patches = config_.max_num_patches,
      .pooling_kernel_size = config_.pooling_kernel_size,
      // LFM2 VL has a single image encoder input and does not consume
      // the per-patch positions tensor.
      .emit_positions = false});
  image_params.SetTargetDimensions(
      Dimensions{1, config_.image_height, config_.image_width, kImageChannels});
  image_params.SetNormalizationConfig(ImageParam::NormalizationConfig{
      .mean = config_.normalization_mean,
      .std = config_.normalization_std,
      .rescale_factor = config_.normalization_rescale_factor});
  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      /*audio_preprocessor=*/nullptr, multi_config, image_params);
}

absl::StatusOr<Message> Lfm2DataProcessor::ToMessageImpl(
    const Responses& responses, const Lfm2DataProcessorArguments& args) const {
  return ResponseTextToMessage(
      responses.GetTexts()[0], preface_, config_.code_fence_start,
      config_.code_fence_end, SyntaxType::kPython,
      {.escape_fence_strings = config_.escape_fence_strings,
       .return_error_on_parse_failure = ReturnErrorOnParseFailure()});
}

}  // namespace litert::lm
