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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/status_macros.h"  //NOLINT

#if !DISABLE_SENTENCEPIECE_TOKENIZER
#include "runtime/components/sentencepiece_tokenizer.h"
#endif  // !DISABLE_SENTENCEPIECE_TOKENIZER

#if !DISABLE_HUGGINGFACE_TOKENIZER
#include "runtime/components/huggingface_tokenizer.h"
#endif  // !DISABLE_HUGGINGFACE_TOKENIZER

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesLitertLm::Create(
    std::unique_ptr<LitertLmLoader> litert_lm_loader) {
  return absl::WrapUnique(
      new ModelResourcesLitertLm(std::move(litert_lm_loader)));
};

absl::StatusOr<std::shared_ptr<litert::Model>>
ModelResourcesLitertLm::GetTFLiteModel(ModelType model_type) {
  if (model_map_.find(model_type) != model_map_.end()) {
    return model_map_[model_type];
  }
  litert::BufferRef<uint8_t> buffer_ref =
      litert_lm_loader_->GetTFLiteModel(model_type);
  ABSL_LOG(INFO) << "model_type: " << ModelTypeToString(model_type);
  ABSL_LOG(INFO) << "litert model size: " << buffer_ref.Size();
  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
  model_map_[model_type] = std::make_shared<litert::Model>(std::move(model));
  return model_map_[model_type];
}

absl::StatusOr<std::shared_ptr<Tokenizer>>
ModelResourcesLitertLm::GetTokenizer() {
  if (tokenizer_ != nullptr) {
    return tokenizer_;
  }

  // Get both tokenizers. The loader will only return the first available
  // tokenizer.
  auto sp_tokenizer = litert_lm_loader_->GetSentencePieceTokenizer();
  auto hf_tokenizer = litert_lm_loader_->GetHuggingFaceTokenizer();

#if !DISABLE_SENTENCEPIECE_TOKENIZER
  if (sp_tokenizer) {
    ASSIGN_OR_RETURN(  // NOLINT
        auto tokenizer,
        SentencePieceTokenizer::CreateFromBuffer(sp_tokenizer->StrView()));
    tokenizer_ = std::move(tokenizer);
    return tokenizer_;
  }
#endif  // !DISABLE_SENTENCEPIECE_TOKENIZER

#if !DISABLE_HUGGINGFACE_TOKENIZER
  if (hf_tokenizer) {
    std::string json_data(hf_tokenizer->StrData(),
                          hf_tokenizer->StrData() + hf_tokenizer->Size());
    ASSIGN_OR_RETURN(  // NOLINT
        auto tokenizer, HuggingFaceTokenizer::CreateFromJson(json_data));
    litert_lm_loader_->ClearHuggingFaceTokenizerJson();
    tokenizer_ = std::move(tokenizer);
    return tokenizer_;
  }
#endif  // !DISABLE_HUGGINGFACE_TOKENIZER

#if DISABLE_SENTENCEPIECE_TOKENIZER
  if (sp_tokenizer) {
    return absl::UnimplementedError(
        "SentencePiece tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_SENTENCEPIECE_TOKENIZER.");
  }
#endif  // !DISABLE_SENTENCEPIECE_TOKENIZER

#if DISABLE_HUGGINGFACE_TOKENIZER
  if (hf_tokenizer) {
    return absl::UnimplementedError(
        "HuggingFace tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_HUGGINGFACE_TOKENIZER.");
  }
#endif  // !DISABLE_HUGGINGFACE_TOKENIZER

  return absl::NotFoundError("No tokenizer found in the model.");
}

absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
ModelResourcesLitertLm::GetLlmMetadata() {
  if (llm_metadata_ != nullptr) {
    return llm_metadata_;
  }
  auto buffer_ref = litert_lm_loader_->GetLlmMetadata();
  llm_metadata_ = std::make_shared<proto::LlmMetadata>();
  llm_metadata_->ParseFromString(std::string(buffer_ref.StrView()));  // NOLINT
  return llm_metadata_;
};

}  // namespace litert::lm
