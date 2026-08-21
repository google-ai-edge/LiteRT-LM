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

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl         // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl       // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl   // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json            // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/lfm2_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/preprocessor/stb_image_preprocessor.h"  // IWYU pragma: keep
#include "re2/re2.h"  // from @com_googlesource_code_re2  // from @com_googlesource_code_re2
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

absl::StatusOr<ordered_json> Lfm2DataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  // For LFM2, the template input is the same as the message.
  return message;
}

absl::StatusOr<ordered_json> Lfm2DataProcessor::FormatTools(
    const ordered_json& tools) const {
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  // For LFM2, return tools as-is without formatting.
  return tools;
}

absl::StatusOr<std::unique_ptr<Constraint>> Lfm2DataProcessor::CreateConstraint(
    const ordered_json& tools) const {
  return absl::FailedPreconditionError(
      "Constrained decoding is not supported for Lfm2DataProcessor.");
}

absl::string_view Lfm2DataProcessor::CodeFenceStart() const { return ""; }

absl::string_view Lfm2DataProcessor::CodeFenceEnd() const { return ""; }

absl::StatusOr<std::vector<InputData>> Lfm2DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const Lfm2DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  // Find all images contained in the messages.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        if (item["type"] == "image") {
          ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                                LoadItemData(item));
          image_files.push_back(std::move(mmap_file));
        }
      }
    }
  }
  // Use the boi_token as the delimiter to find image placeholders in the
  // prompt. The placeholder in the prompt is "<image>".
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
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

  RE2 re_delimiter(R"regex((<\|image_start\|>|<image>))regex");
  // Replace the "<image>" placeholder with the actual image data.
  // Note: We need to find "<image>" but not "<image|>" (eoi_token).
  // The placeholder is specifically "<image>" which is the boi_token.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    input_data.emplace_back(
        InputText(std::string(text_part) + config_.boi_token));
    if (image_files.empty()) {
      return absl::InvalidArgumentError(
          "Provided less images than expected in the prompt.");
    }
    auto image_file = std::move(image_files.front());
    image_files.pop_front();

    auto process_status =
        image_preprocessor_
            ? image_preprocessor_->Preprocess(
                  InputImage(
                      std::string(static_cast<const char*>(image_file->data()),
                                  image_file->length())),
                  image_params)
            : InputImage(
                  std::string(static_cast<const char*>(image_file->data()),
                              image_file->length()));
    if (!process_status.ok()) {
      return process_status.status();
    }
    auto preprocessed_image = std::move(*process_status);

    input_data.emplace_back(InputImage(std::move(preprocessed_image)));
    input_data.emplace_back(InputText(config_.eoi_token));
  }
  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }
  // Add the remaining text in the prompt.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }

  return input_data;
}

absl::StatusOr<Message> Lfm2DataProcessor::ToMessageImpl(
    const Responses& responses, const Lfm2DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json message = {{"role", "assistant"}};
  message["content"] = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return message;
}

absl::Status Lfm2DataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<Lfm2DataProcessorConfig,
                                     Lfm2DataProcessorArguments>& other) {
  // LFM2 has no state to clone (image_preprocessor_ is ignored as requested).
  return absl::OkStatus();
}

absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
Lfm2DataProcessor::RenderSingleTurnTemplate(
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context) const {
  const auto& json_preface = std::get<JsonPreface>(preface);
  std::string prefill_text = "";
  bool is_first_part = false;
  bool is_last_part = false;
  if (!current_is_appending_message) {
    is_first_part = true;
  }
  if (!append_message) {
    is_last_part = true;
  }
  bool new_is_appending_message = current_is_appending_message;
  if (is_first_part) {
    new_is_appending_message = true;
  }
  if (is_last_part) {
    new_is_appending_message = false;
  }
  bool is_role_changed = false;
  if (!history.empty()) {
    const auto& last_message = history.back();
    // If the last message is in appending state and the current message is
    // different role, then we need to add a closing message to the prefill.
    if (current_is_appending_message &&
        (last_message["role"] != message["role"] &&
         last_message["role"] != "system")) {
      is_role_changed = true;
      PromptTemplateInput closing_tmpl_input;
      nlohmann::ordered_json closing_message = {
          {"role", last_message["role"]},
          {"content", ""},
      };
      ABSL_ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                            MessageToTemplateInput(closing_message));
      closing_tmpl_input.extra_context["message"] = message_tmpl_input;
      closing_tmpl_input.extra_context["is_appending_to_prefill"] = true;
      closing_tmpl_input.extra_context["is_first_part"] = false;
      closing_tmpl_input.extra_context["is_last_part"] = true;
      closing_tmpl_input.add_generation_prompt = false;
      ABSL_ASSIGN_OR_RETURN(std::string closing_text,
                            prompt_template.Apply(closing_tmpl_input));
      prefill_text += closing_text;
    }
  } else {
    PromptTemplateInput preface_tmpl_input;
    if (!json_preface.messages.empty() || !json_preface.tools.empty() ||
        !json_preface.extra_context.is_null()) {
      preface_tmpl_input.messages.push_back(
          Message{{"role", "user"}, {"content", ""}});
      preface_tmpl_input.add_generation_prompt = false;
      if (extra_context.has_value()) {
        for (const auto& [key, value] : extra_context.value().items()) {
          preface_tmpl_input.extra_context[key] = value;
        }
      }
      ABSL_ASSIGN_OR_RETURN(std::string preface_text,
                            prompt_template.Apply(preface_tmpl_input));
      prefill_text += preface_text;
    }
  }
  if (message.is_object()) {
    PromptTemplateInput tmpl_input;
    ABSL_ASSIGN_OR_RETURN(tmpl_input.extra_context["message"],
                          MessageToTemplateInput(message));
    tmpl_input.extra_context["is_appending_to_prefill"] = true;
    tmpl_input.extra_context["is_first_part"] =
        is_first_part || is_role_changed;
    tmpl_input.extra_context["is_last_part"] = is_last_part;
    tmpl_input.add_generation_prompt = !new_is_appending_message;
    if (extra_context.has_value()) {
      for (const auto& [key, value] : extra_context.value().items()) {
        tmpl_input.extra_context[key] = value;
      }
    }
    ABSL_ASSIGN_OR_RETURN(std::string new_text,
                          prompt_template.Apply(tmpl_input));
    prefill_text += new_text;
  }
  return ModelDataProcessor::SingleTurnTemplateRenderResult{
      prefill_text, new_is_appending_message};
}
}  // namespace litert::lm
