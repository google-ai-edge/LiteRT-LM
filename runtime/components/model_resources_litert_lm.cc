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

#include "runtime/components/model_resources_litert_lm.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"

#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
#include "runtime/components/sentencepiece_tokenizer.h"
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
#include "runtime/components/huggingface_tokenizer.h"
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesLitertLm::Create(
    std::unique_ptr<LitertLmLoader> litert_lm_loader) {
  return absl::WrapUnique(
      new ModelResourcesLitertLm(std::move(litert_lm_loader)));
};

absl::StatusOr<const litert::Model*> ModelResourcesLitertLm::GetTFLiteModel(
    ModelType model_type) {
  auto it = model_map_.find(model_type);
  if (it != model_map_.end()) {
    return it->second.get();
  }

  litert::BufferRef<uint8_t> buffer_ref =
      litert_lm_loader_->GetTFLiteModel(model_type);
  ABSL_LOG(INFO) << "model_type: " << ModelTypeToString(model_type);
  ABSL_LOG(INFO) << "litert model size: " << buffer_ref.Size();
  if (buffer_ref.Size() == 0) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
  model_map_[model_type] = std::make_unique<litert::Model>(std::move(model));
  return model_map_[model_type].get();
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelBackendConstraint(ModelType model_type) {
  return litert_lm_loader_->GetTFLiteModelBackendConstraint(model_type);
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelPreferActivationType(
    ModelType model_type) {
  return litert_lm_loader_->GetTFLiteModelPreferActivationType(model_type);
}

absl::StatusOr<absl::string_view> ModelResourcesLitertLm::GetTFLiteModelBuffer(
    ModelType model_type) {
  litert::BufferRef<uint8_t> buffer_ref =
      litert_lm_loader_->GetTFLiteModel(model_type);

  ABSL_LOG(INFO) << "model_type: " << ModelTypeToString(model_type);
  ABSL_LOG(INFO) << "litert model size: " << buffer_ref.Size();
  if (buffer_ref.Size() == 0) {
    return absl::NotFoundError(absl::StrCat(ModelTypeToString(model_type),
                                            " not found in the model."));
  }
  return buffer_ref.StrView();
};

absl::StatusOr<std::unique_ptr<Tokenizer>>
ModelResourcesLitertLm::GetTokenizer() {
#if !defined(ENABLE_SENTENCEPIECE_TOKENIZER) && \
    !defined(ENABLE_HUGGINGFACE_TOKENIZER)
  return absl::UnimplementedError(
      "Tokenizers cannot be used. Neither ENABLE_SENTENCEPIECE_TOKENIZER nor "
      "ENABLE_HUGGINGFACE_TOKENIZER are defined during build.");
#endif  // !ENABLE_SENTENCEPIECE_TOKENIZER && !ENABLE_HUGGINGFACE_TOKENIZER

  auto sp_tokenizer = litert_lm_loader_->GetSentencePieceTokenizer();
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
  if (sp_tokenizer) {
    return SentencePieceTokenizer::CreateFromBuffer(sp_tokenizer->StrView());
  }
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

  auto hf_tokenizer = litert_lm_loader_->GetHuggingFaceTokenizer();
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
  if (hf_tokenizer) {
    std::string json_data(hf_tokenizer->StrData(), hf_tokenizer->Size());
    return HuggingFaceTokenizer::CreateFromJson(json_data);
  }
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

  if (sp_tokenizer) {
    return absl::UnimplementedError(
        "SentencePiece tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_SENTENCEPIECE_TOKENIZER=1.");
  } else if (hf_tokenizer) {
    return absl::UnimplementedError(
        "HuggingFace tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_HUGGINGFACE_TOKENIZER=1.");
  } else {
    return absl::NotFoundError("No tokenizer found in the model.");
  }
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesLitertLm::GetLlmMetadata() {
  if (llm_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetLlmMetadata();
    auto llm_metadata = std::make_unique<proto::LlmMetadata>();
    if (!llm_metadata->ParseFromString(
            std::string(buffer_ref.StrView()))) {  // NOLINT
      return absl::InternalError("Failed to parse LlmMetadata");
    }
    llm_metadata_ = std::move(llm_metadata);
  }
  return llm_metadata_.get();
};

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesLitertLm::GetScopedFile() {
  return litert_lm_loader_->GetScopedFile();
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesLitertLm::GetWeightsSectionOffset(ModelType model_type) {
  return litert_lm_loader_->GetSectionLocation(
      BufferKey(schema::AnySectionDataType_TFLiteWeights, model_type));
}

}  // namespace litert::lm
