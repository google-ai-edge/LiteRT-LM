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

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {

using ::nlohmann::ordered_json;

bool IsImage(absl::string_view part) { return part == "<image_soft_token>"; }

}  // namespace

absl::StatusOr<std::unique_ptr<FastVlmDataProcessor>>
FastVlmDataProcessor::Create(FastVlmDataProcessorConfig config,
                             const PromptTemplateCapabilities& capabilities) {
  return absl::WrapUnique(new FastVlmDataProcessor(
      config, capabilities, std::make_unique<StbImagePreprocessor>()));
}

absl::StatusOr<ordered_json> FastVlmDataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  if (message["content"].is_string() && capabilities_.requires_typed_content) {
    return ordered_json::object(
        {{"role", message["role"]},
         {"content", ordered_json::array(
                         {{{"type", "text"}, {"text", message["content"]}}})}});
  } else if (message["content"].is_array() && message["content"].size() == 1 &&
             message["content"][0]["type"] == "text" &&
             !capabilities_.requires_typed_content) {
    return ordered_json::object({{"role", message["role"]},
                                 {"content", message["content"][0]["text"]}});
  } else {
    return message;
  }
}

absl::StatusOr<ordered_json> FastVlmDataProcessor::FormatTools(
    const ordered_json& tools) const {
  return tools;
}

absl::StatusOr<std::vector<InputData>>
FastVlmDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const FastVlmDataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;

  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                         LoadItemData(item));
        if (item["type"] == "image") {
          image_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  RE2 re_delimiter("(<image_soft_token>)");
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));

  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    if (IsImage(part)) {
      input_data.emplace_back(InputText(std::string(text_part)));

      if (image_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less images than expected in the prompt.");
      }
      auto image_file = std::move(image_files.front());
      image_files.pop_front();
      ASSIGN_OR_RETURN(auto preprocessed_image,
                       image_preprocessor_->Preprocess(
                           InputImage(std::string(
                               static_cast<const char*>(image_file->data()),
                               image_file->length())),
                           image_params));
      input_data.emplace_back(InputImage(std::move(preprocessed_image)));
    }
  }

  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }

  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }

  return input_data;
}

absl::StatusOr<Message> FastVlmDataProcessor::ToMessageImpl(
    const Responses& responses,
    const FastVlmDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json content = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return ordered_json::object({{"role", "assistant"}, {"content", content}});
}

absl::Status FastVlmDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<FastVlmDataProcessorConfig,
                                     FastVlmDataProcessorArguments>& other) {
  return absl::OkStatus();
}

}  // namespace litert::lm
