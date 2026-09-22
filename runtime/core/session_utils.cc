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

#include "runtime/core/session_utils.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

absl::StatusOr<std::string> MaybeGetBosString(
    const SessionConfig& session_config, support::Tokenizer& tokenizer) {
  auto bos_token_id = session_config.GetStartTokenId();
  std::string bos_string = "";
  if (bos_token_id >= 0) {
    ABSL_ASSIGN_OR_RETURN(bos_string, tokenizer.TokenIdsToText({bos_token_id}));
  }
  return bos_string;
}

absl::StatusOr<InputText> StringToProcessedInputText(
    absl::string_view text, const SessionConfig& session_config,
    support::Tokenizer& tokenizer,
    const std::optional<BenchmarkInfo>& benchmark_info) {
  auto bos_token_id = session_config.GetStartTokenId();
  std::string bos_string = "";
  if (bos_token_id >= 0) {
    ABSL_ASSIGN_OR_RETURN(bos_string, tokenizer.TokenIdsToText({bos_token_id}));
  }
  bool bos_token_found = false;
  if (!bos_string.empty() && absl::StartsWith(text, bos_string)) {
    text = text.substr(bos_string.size());
    bos_token_found = true;
  }

  int benchmark_prefill_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_prefill_token_count =
        benchmark_info->GetBenchmarkParams().num_prefill_tokens();
    ABSL_RETURN_IF_ERROR(
        const_cast<BenchmarkInfo&>(*benchmark_info).TimeTextToTokenIdsStart());
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer.TextToTokenIds(text));
  if (benchmark_prefill_token_count > 0) {
    // If benchmark is enabled, we will use the benchmark prefill token
    // count to set the prefill token count.
    ids.resize(benchmark_prefill_token_count);
  } else if (bos_token_found) {
    ids.insert(ids.begin(), session_config.GetStartTokenId());
  }
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(const_cast<BenchmarkInfo&>(*benchmark_info)
                             .TimeTextToTokenIdsEnd(ids.size()));
  }
  ABSL_ASSIGN_OR_RETURN(auto ids_buffer, tokenizer.TokenIdsToTensorBuffer(ids));
  return InputText(std::move(ids_buffer));
}

absl::StatusOr<std::vector<InputData>> ApplyPromptTemplates(
    const std::vector<InputData>& contents, ContentType content_type,
    const SessionConfig& session_config, support::Tokenizer& tokenizer,
    bool is_first_turn) {
  ABSL_ASSIGN_OR_RETURN(std::string bos_string,
                        MaybeGetBosString(session_config, tokenizer));

  std::vector<InputData> templated_contents;
  if (!session_config.GetApplyPromptTemplateInSession()) {
    RET_CHECK(content_type == ContentType::kNA);
    if (is_first_turn && !bos_string.empty()) {
      templated_contents.push_back(InputText(bos_string));
    }
    for (int i = 0; i < contents.size(); ++i) {
      const auto& content = contents[i];
      ABSL_ASSIGN_OR_RETURN(auto content_copy, CreateInputDataCopy(content));
      templated_contents.emplace_back(std::move(content_copy));
    }
    return templated_contents;
  }

  RET_CHECK(content_type != ContentType::kNA);

  if (is_first_turn && !bos_string.empty()) {
    templated_contents.push_back(InputText(bos_string));
  }

  if (is_first_turn) {
    RET_CHECK(content_type == ContentType::kFirst);
  };

  std::string turn_prefix = session_config.GetPromptTemplates().user().prefix();
  std::string turn_suffix =
      absl::StrCat(session_config.GetPromptTemplates().user().suffix(),
                   session_config.GetPromptTemplates().model().prefix());
  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    const bool is_first_chunk = i == 0;
    const bool is_text_chunk = std::holds_alternative<InputText>(content);

    if (is_text_chunk) {
      ABSL_ASSIGN_OR_RETURN(absl::string_view raw_text,
                            std::get<InputText>(content).GetRawTextString());

      // Check if the input starts with the BOS string. If it does, return an
      // error. This is to prevent the user from including the BOS string in the
      // input. This is also needed for the current implementation as tokenizer
      // will treat the BOS string differently from other strings. If the BOS
      // string is empty, it means the BOS token id is not valid. In this case,
      // we will not check for the BOS string in the input.
      if (!bos_string.empty() && absl::StartsWith(raw_text, bos_string)) {
        return absl::InvalidArgumentError(
            "Input contains bos control token. Control token should not be "
            "included in the input.");
      }

      std::string templated_text;
      if (is_first_chunk && (content_type == ContentType::kFirst)) {
        templated_text = absl::StrCat(turn_prefix, raw_text);
      } else if (content_type == ContentType::kLast) {
        templated_text = absl::StrCat(raw_text, turn_suffix);
      } else {
        templated_text = raw_text;
      }

      if (!templated_text.empty()) {
        templated_contents.push_back(InputText(std::move(templated_text)));
      }
    } else {
      if (is_first_chunk && (content_type == ContentType::kFirst) &&
          !turn_prefix.empty()) {
        templated_contents.push_back(InputText(turn_prefix));
      }
      ABSL_ASSIGN_OR_RETURN(auto content_copy, CreateInputDataCopy(content));
      templated_contents.emplace_back(std::move(content_copy));
      if ((content_type == ContentType::kLast) && !turn_suffix.empty()) {
        templated_contents.push_back(InputText(turn_suffix));
      }
    }
  }

  return templated_contents;
}

absl::StatusOr<std::vector<InputData>> PreprocessContents(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    support::Tokenizer& tokenizer,
    const std::optional<BenchmarkInfo>& benchmark_info) {
  std::vector<InputData> preprocessed_contents;
  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    if (const auto* input_text = std::get_if<InputText>(&content)) {
      if (input_text->IsTensorBuffer()) {
        ABSL_ASSIGN_OR_RETURN(auto input_text_copy, input_text->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_text_copy));
      } else {
        ABSL_ASSIGN_OR_RETURN(auto templated_text,
                              input_text->GetRawTextString());
        if (templated_text.empty()) {
          // We skip empty input text contents in the final preprocessed
          // version.
          continue;
        }
        ABSL_ASSIGN_OR_RETURN(
            auto processed_input_text,
            StringToProcessedInputText(templated_text, session_config,
                                       tokenizer, benchmark_info));
        preprocessed_contents.emplace_back(std::move(processed_input_text));
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&content)) {
      if (input_image->IsTensorBuffer() || input_image->IsTensorBufferMap()) {
        ABSL_ASSIGN_OR_RETURN(auto input_image_copy, input_image->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_image_copy));
      } else {
        return absl::InternalError(
            "Image must be preprocessed before being used in SessionAdvanced.");
      }
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&content)) {
      preprocessed_contents.emplace_back(InputImageEnd());
    } else if (const auto* input_audio = std::get_if<InputAudio>(&content)) {
      if (input_audio->IsTensorBuffer() || input_audio->IsAudioEmbeddings()) {
        ABSL_ASSIGN_OR_RETURN(auto input_audio_copy, input_audio->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_audio_copy));
      } else {
        return absl::InternalError(
            "Audio must be preprocessed before being used in SessionAdvanced.");
      }
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&content)) {
      preprocessed_contents.emplace_back(InputAudioEnd());
    } else {
      return absl::InternalError(
          "Unsupported input type in preprocessed_contents.");
    }
  }
  return preprocessed_contents;
}

absl::StatusOr<int> CalculateInputDataTokens(
    const std::vector<InputData>& contents, const Engine& engine) {
  int total_tokens = 0;
  std::optional<VisionExecutorProperties> vision_props;
  std::optional<AudioExecutorProperties> audio_props;

  for (size_t i = 0; i < contents.size(); ++i) {
    const auto& input = contents[i];
    if (const auto* text = std::get_if<InputText>(&input)) {
      if (auto raw_str = text->GetRawTextString(); raw_str.ok()) {
        ABSL_ASSIGN_OR_RETURN(auto token_ids, const_cast<support::Tokenizer&>(
                                                  engine.GetTokenizer())
                                                  .TextToTokenIds(*raw_str));
        total_tokens += token_ids.size();
      } else if (auto tensor = text->GetPreprocessedTextTensor(); tensor.ok()) {
        ABSL_ASSIGN_OR_RETURN(
            auto ids_vec, support::Tokenizer::TensorBufferToTokenIds(**tensor));
        for (const auto& ids : ids_vec) {
          total_tokens += ids.size();
        }
      }
    } else if (const auto* image = std::get_if<InputImage>(&input)) {
      if (!vision_props.has_value()) {
        ABSL_ASSIGN_OR_RETURN(vision_props,
                              engine.GetVisionExecutorProperties());
      }
      int token_length = vision_props->num_tokens_per_image;
      if (vision_props->patch_num_shrink_factor.has_value() &&
          image->IsTensorBufferMap()) {
        ABSL_ASSIGN_OR_RETURN(const auto* map,
                              image->GetPreprocessedImageTensorMap());
        auto it = map->find("positions_xy");
        if (it != map->end()) {
          LITERT_ASSIGN_OR_RETURN(auto type, it->second.TensorType());
          auto dims = type.Layout().Dimensions();
          if (dims.size() >= 2) {
            int num_patches = dims[1];
            int shrink = vision_props->patch_num_shrink_factor.value();
            if (shrink <= 0) {
              return absl::InvalidArgumentError(
                  "vision_properties.patch_num_shrink_factor must be strictly "
                  "positive.");
            }
            token_length = (num_patches + shrink - 1) / shrink;
          }
        }
      }
      total_tokens += token_length;
    } else if (const auto* audio = std::get_if<InputAudio>(&input)) {
      if (!audio_props.has_value()) {
        ABSL_ASSIGN_OR_RETURN(audio_props, engine.GetAudioExecutorProperties());
      }
      ABSL_ASSIGN_OR_RETURN(const auto* buffer,
                            audio->GetPreprocessedAudioTensor());
      LITERT_ASSIGN_OR_RETURN(auto type, buffer->TensorType());
      auto dims = type.Layout().Dimensions();
      int input_sequence_length = 0;
      if (dims.size() >= 2) {
        input_sequence_length = dims[dims.size() - 2];
      }
      if (input_sequence_length <= 0) {
        return absl::InvalidArgumentError(
            "Invalid or empty audio sequence length in TensorBuffer.");
      }
      int shrink = audio_props->audio_shrink_factor;
      if (shrink <= 0) {
        return absl::InvalidArgumentError(
            "audio_properties.audio_shrink_factor must be strictly positive.");
      }
      if (audio_props->is_streaming_model) {
        int window_size = audio_props->streaming_chunk_size;
        int overlap_size = audio_props->streaming_chunk_overlap_size;
        int stride = window_size - overlap_size;
        if (stride <= 0 || window_size <= overlap_size) {
          return absl::InvalidArgumentError(
              "Invalid audio streaming chunk/overlap size.");
        }
        int chunk_output_tokens = stride / shrink;
        bool is_flush = i + 1 < contents.size() &&
                        std::holds_alternative<InputAudioEnd>(contents[i + 1]);
        if (!is_flush) {
          if (input_sequence_length >= window_size) {
            int num_full_chunks =
                (input_sequence_length - overlap_size) / stride;
            total_tokens += num_full_chunks * chunk_output_tokens;
          }
        } else if (input_sequence_length > overlap_size) {
          int num_chunks =
              (input_sequence_length - overlap_size + stride - 1) / stride;
          int last_chunk_len = std::min(
              window_size, input_sequence_length - (num_chunks - 1) * stride);
          int last_chunk_tokens = std::min(
              chunk_output_tokens, (last_chunk_len + shrink - 1) / shrink);
          total_tokens +=
              (num_chunks - 1) * chunk_output_tokens + last_chunk_tokens;
        }
      } else {
        total_tokens += (input_sequence_length + shrink - 1) / shrink;
      }
    } else if (std::holds_alternative<InputImageEnd>(input) ||
               std::holds_alternative<InputAudioEnd>(input)) {
      total_tokens += 1;
    }
  }

  return total_tokens;
}

}  // namespace litert::lm
