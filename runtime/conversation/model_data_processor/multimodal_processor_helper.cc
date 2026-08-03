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

#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "support/preprocessor/image_preprocessor.h"  // from @litert
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    std::optional<ImagePreprocessParameter> image_params,
    std::optional<int> visual_token_budget) {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  std::deque<std::unique_ptr<MemoryMappedFile>> audio_files;

  // 1. Find all images and audio contained in the messages.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        if (!item.contains("type")) {
          continue;
        }
        ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                         LoadItemData(item));
        if (item["type"] == "image") {
          image_files.push_back(std::move(mmap_file));
        } else if (item["type"] == "audio") {
          audio_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  // 2. Setup regex delimiters and matchers.
  RE2 re_delimiter(config.delimiter_regex);
  RE2 re_image(config.image_token_regex);
  RE2 re_audio(config.audio_token_regex);

  absl::string_view prompt_view = rendered_template_prompt;
  const char* start = prompt_view.data();
  std::string part;

  // 3. Setup image preprocess parameters if preprocessor is present.
  ImagePreprocessParameter img_params;
  if (image_preprocessor != nullptr) {
    if (!image_params.has_value()) {
      return absl::FailedPreconditionError(
          "Image preprocessor is configured but no image parameters are "
          "provided.");
    }
    img_params = *image_params;
    if (image_params->GetPatchifyConfig().has_value()) {
      int max_num_patches = image_params->GetPatchifyConfig()->max_num_patches;
      if (visual_token_budget.has_value()) {
        int budget = visual_token_budget.value();
        if (budget <= 0) {
          return absl::InvalidArgumentError(
              "Visual token budget must be positive.");
        }
        max_num_patches = std::min(
            max_num_patches,
            budget * image_params->GetPatchifyConfig()->pooling_kernel_size *
                image_params->GetPatchifyConfig()->pooling_kernel_size);
      }
      img_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
          .patch_width = image_params->GetPatchifyConfig()->patch_width,
          .patch_height = image_params->GetPatchifyConfig()->patch_height,
          .max_num_patches = max_num_patches,
          .pooling_kernel_size =
              image_params->GetPatchifyConfig()->pooling_kernel_size});
    }
  }

  // 4. Parse the rendered prompt and replace placeholders.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();

    if (RE2::FullMatch(part, re_image)) {
      if (image_preprocessor == nullptr) {
        return absl::FailedPreconditionError(
            "Image placeholder found but no image preprocessor is configured.");
      }

      std::string text =
          absl::StrCat(text_part, config.image_prefix, config.boi_token);
      if (!text.empty()) {
        input_data.emplace_back(InputText(std::move(text)));
      }

      if (image_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less images than expected in the prompt.");
      }
      auto image_file = std::move(image_files.front());
      image_files.pop_front();

      ASSIGN_OR_RETURN(auto preprocessed_image,
                       image_preprocessor->Preprocess(
                           InputImage(std::string(
                               static_cast<const char*>(image_file->data()),
                               image_file->length())),
                           img_params));
      input_data.emplace_back(InputImage(std::move(preprocessed_image)));

      if (config.add_image_end) {
        input_data.emplace_back(InputImageEnd());
      }
      if (!config.image_suffix.empty()) {
        input_data.emplace_back(InputText(config.image_suffix));
      }
    } else if (RE2::FullMatch(part, re_audio)) {
      if (audio_preprocessor == nullptr) {
        return absl::FailedPreconditionError(
            "Audio placeholder found but no audio preprocessor is configured.");
      }

      std::string text =
          absl::StrCat(text_part, config.audio_prefix, config.boa_token);
      if (!text.empty()) {
        input_data.emplace_back(InputText(std::move(text)));
      }

      if (audio_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less audio than expected in the prompt.");
      }
      auto audio_file = std::move(audio_files.front());
      audio_files.pop_front();

      ASSIGN_OR_RETURN(auto preprocessed_audio,
                       audio_preprocessor->Preprocess(InputAudio(std::string(
                           static_cast<const char*>(audio_file->data()),
                           audio_file->length()))));
      audio_preprocessor->Reset();

      input_data.emplace_back(InputAudio(std::move(preprocessed_audio)));

      if (config.add_audio_end) {
        input_data.emplace_back(InputAudioEnd());
      }
      if (!config.audio_suffix.empty()) {
        input_data.emplace_back(InputText(config.audio_suffix));
      }
    }
  }

  // 5. Verification.
  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }
  if (!audio_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more audio than expected in the prompt.");
  }

  // 6. Append remaining prompt text.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }

  return input_data;
}

}  // namespace litert::lm
