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

#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_end_of_multi_modal.h"
#include "runtime/components/embedding_lookup/embedding_lookup_multi_modal.h"
#include "runtime/components/embedding_lookup/embedding_lookup_text.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::StatusOr<std::unique_ptr<EmbeddingLookupManager>>
EmbeddingLookupManager::Create(
    litert::Environment& env,
    const litert::Model* absl_nonnull text_embedding_model,
    absl::flat_hash_map<int, const litert::Model*>&
        end_of_multi_modal_embedding_models,
    bool fully_supports_multi_modal, std::optional<std::string> signature_key) {
  auto embedding_lookup_manager = std::make_unique<EmbeddingLookupManager>();
  RETURN_IF_ERROR(embedding_lookup_manager->Initialize(
      env, text_embedding_model, end_of_multi_modal_embedding_models,
      fully_supports_multi_modal, signature_key));
  return std::move(embedding_lookup_manager);
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupManager>>
EmbeddingLookupManager::Create(
    litert::Environment& env,
    const litert::Model* absl_nonnull text_embedding_model,
    bool fully_supports_multi_modal, std::optional<std::string> signature_key) {
  absl::flat_hash_map<int, const litert::Model*>
      end_of_multi_modal_embedding_models;
  return Create(env, text_embedding_model, end_of_multi_modal_embedding_models,
                fully_supports_multi_modal, signature_key);
}

absl::Status EmbeddingLookupManager::UpdateMultiModalEmbeddings(
    const ::litert::lm::ExecutorInputs& inputs) {
  auto vision_embeddings = inputs.GetVisionEmbeddingsPtr();
  if (vision_embeddings.ok() && *vision_embeddings != nullptr) {
    if (!fully_supports_multi_modal_) {
      return absl::InvalidArgumentError(
          "When fully_supports_multi_modal_ is false, multimodal embeddings "
          "must not be provided. Their entries will be default to the 0th "
          "embedding value of the text embedding table.");
    }
    auto vision_embedding_lookup = EmbeddingLookupMultiModal::Create(
        *vision_embeddings, ::litert::lm::ExecutorVisionData::kSpecialToken);
    if (!vision_embedding_lookup.ok()) {
      return vision_embedding_lookup.status();
    }
    multi_modal_embedding_lookups_.push_back(
        std::move(*vision_embedding_lookup));
  }

  auto audio_embeddings = inputs.GetAudioEmbeddingsPtr();
  if (audio_embeddings.ok() && *audio_embeddings != nullptr) {
    if (!fully_supports_multi_modal_) {
      return absl::InvalidArgumentError(
          "When fully_supports_multi_modal_ is false, multimodal embeddings "
          "must not be provided. Their entries will be default to the 0th "
          "embedding value of the text embedding table.");
    }
    auto audio_embedding_lookup = EmbeddingLookupMultiModal::Create(
        *audio_embeddings, ::litert::lm::ExecutorAudioData::kSpecialToken);
    if (!audio_embedding_lookup.ok()) {
      return audio_embedding_lookup.status();
    }
    multi_modal_embedding_lookups_.push_back(
        std::move(*audio_embedding_lookup));
  }

  return absl::OkStatus();
}

absl::Status EmbeddingLookupManager::CleanupMultiModalEmbeddings() {
  multi_modal_embedding_lookups_.clear();
  return absl::OkStatus();
}

absl::Status EmbeddingLookupManager::LookupDecode(
    int token, std::vector<float>& output_vector) {
  if (text_embedding_lookup_ == nullptr) {
    return absl::InternalError(
        "Text embedding lookup is null. Please ensure that the "
        "EmbeddingLookupManager is initialized properly.");
  }
  const size_t floats_per_token = text_embedding_lookup_->GetFloatsPerToken();
  output_vector.resize(floats_per_token);

  if (token < 0) {
    return absl::InvalidArgumentError(
        "Multimodal embeddings are not supported during decode.");
  }

  return text_embedding_lookup_->LookupDecode(token, output_vector);
}

absl::Status EmbeddingLookupManager::LookupDecode(
    int token, litert::TensorBuffer* output_tensor) {
  if (text_embedding_lookup_ == nullptr) {
    return absl::InternalError(
        "Text embedding lookup is null. Please ensure that the "
        "EmbeddingLookupManager is initialized properly.");
  }

  if (token < 0) {
    return absl::InvalidArgumentError(
        "Multimodal embeddings are not supported during decode.");
  }

  return text_embedding_lookup_->LookupDecode(token, output_tensor);
}

absl::Status EmbeddingLookupManager::LookupPrefill(
    int token, std::vector<float>& output_vector) {
  if (text_embedding_lookup_ == nullptr) {
    return absl::InternalError(
        "Text embedding lookup is null. Please ensure that the "
        "EmbeddingLookupManager is initialized properly.");
  }
  const size_t floats_per_token = text_embedding_lookup_->GetFloatsPerToken();
  output_vector.resize(floats_per_token);

  if (token >= 0) {
    return text_embedding_lookup_->LookupPrefill(token, output_vector);
  } else if (fully_supports_multi_modal_) {
    for (const auto& embedding_lookup : multi_modal_embedding_lookups_) {
      RETURN_IF_ERROR(embedding_lookup->LookupPrefill(token, output_vector));
    }
    for (const auto& embedding_lookup : end_of_multi_modal_embedding_lookups_) {
      RETURN_IF_ERROR(embedding_lookup->LookupPrefill(token, output_vector));
    }
  } else {
    // If fully_supports_multi_modal_ is false, then we need to fill in the
    // missing embeddings with the default embedding vector.
    memcpy(output_vector.data(),
           text_embedding_lookup_->GetDefaultEmbeddingVector().data(),
           floats_per_token * sizeof(float));
  }
  return absl::OkStatus();
}

absl::Status EmbeddingLookupManager::LookupPrefill(
    absl::Span<const int> tokens, litert::TensorBuffer* output_tensor,
    size_t token_offset) {
  if (text_embedding_lookup_ == nullptr) {
    return absl::InternalError(
        "Text embedding lookup is null. Please ensure that the "
        "EmbeddingLookupManager is initialized properly.");
  }
  const size_t floats_per_token = text_embedding_lookup_->GetFloatsPerToken();
  const size_t byte_offset = token_offset * sizeof(float) * floats_per_token;

  RETURN_IF_ERROR(text_embedding_lookup_->LookupPrefill(tokens, output_tensor,
                                                        byte_offset));

  if (fully_supports_multi_modal_) {
    for (const auto& embedding_lookup : multi_modal_embedding_lookups_) {
      RETURN_IF_ERROR(
          embedding_lookup->LookupPrefill(tokens, output_tensor, byte_offset));
    }
    for (const auto& embedding_lookup : end_of_multi_modal_embedding_lookups_) {
      RETURN_IF_ERROR(
          embedding_lookup->LookupPrefill(tokens, output_tensor, byte_offset));
    }
  } else {
    // If fully_supports_multi_modal_ is false, then we need to fill in the
    // missing embeddings with the default embedding vector.
    const size_t bytes_per_token = floats_per_token * sizeof(float);
    LITERT_ASSIGN_OR_RETURN(auto output_tensor_size,
                            output_tensor->PackedSize());
    uint8_t* output_tensor_ptr = nullptr;
    uint8_t* output_tensor_last_ptr = nullptr;
    const auto* default_embedding =
        text_embedding_lookup_->GetDefaultEmbeddingVector().data();
    for (int i = 0; i < tokens.size(); ++i) {
      if (tokens[i] < 0) {
        if (output_tensor_ptr == nullptr) {
          LITERT_ASSIGN_OR_RETURN(
              auto output_tensor_ptr_void,
              output_tensor->Lock(::litert::TensorBuffer::LockMode::kWrite));
          output_tensor_ptr = static_cast<uint8_t*>(output_tensor_ptr_void);
          output_tensor_last_ptr =
              output_tensor_ptr + output_tensor_size - bytes_per_token;
          output_tensor_ptr += byte_offset + i * bytes_per_token;
        }
        memcpy(output_tensor_ptr, default_embedding, bytes_per_token);
      }
      if (output_tensor_ptr != nullptr) {
        output_tensor_ptr += bytes_per_token;
        if (output_tensor_ptr > output_tensor_last_ptr) {
          break;
        }
      }
    }
    if (output_tensor_ptr != nullptr) {
      LITERT_RETURN_IF_ERROR(output_tensor->Unlock());
    }
  }
  // Remove fully used multi modal embedding lookups.
  std::erase_if(multi_modal_embedding_lookups_,
                [](const auto& embedding_lookup) {
                  return !embedding_lookup->HasRemainingEmbeddings();
                });
  return absl::OkStatus();
}

absl::Status EmbeddingLookupManager::Initialize(
    litert::Environment& env,
    const litert::Model* absl_nonnull text_embedding_model,
    absl::flat_hash_map<int, const litert::Model*>&
        end_of_multi_modal_embedding_models,
    bool fully_supports_multi_modal, std::optional<std::string> signature_key) {
  if (!fully_supports_multi_modal &&
      !end_of_multi_modal_embedding_models.empty()) {
    return absl::InvalidArgumentError(
        "When fully_supports_multi_modal is false, "
        "end_of_multi_modal_embedding_models must be empty.");
  }
  fully_supports_multi_modal_ = fully_supports_multi_modal;
  ASSIGN_OR_RETURN(text_embedding_lookup_,
                   EmbeddingLookupText::Create(
                       env, std::move(text_embedding_model), signature_key));
  for (const auto& [special_token, embedding_model] :
       end_of_multi_modal_embedding_models) {
    ASSIGN_OR_RETURN(auto end_of_multi_modal_embedding_lookup,
                     EndOfMultiModalEmbedding::Create(
                         env, std::move(embedding_model), special_token));
    end_of_multi_modal_embedding_lookups_.push_back(
        std::move(end_of_multi_modal_embedding_lookup));
  }
  return absl::OkStatus();
}

litert::Expected<bool> EmbeddingLookupManager::IsFullyAccelerated() const {
  if (text_embedding_lookup_ == nullptr) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Text embedding lookup has not been created.");
  }
  if (auto res = text_embedding_lookup_->IsFullyAccelerated();
      !res.HasValue() || !*res) {
    return res;
  }
  for (const auto& lookup : end_of_multi_modal_embedding_lookups_) {
    if (auto res = lookup->IsFullyAccelerated(); !res.HasValue() || !*res) {
      return res;
    }
  }
  return true;
}

}  // namespace litert::lm
