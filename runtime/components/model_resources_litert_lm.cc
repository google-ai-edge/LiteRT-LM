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

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

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
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"

#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

namespace litert::lm {

namespace {

absl::StatusOr<litert::Model> CreateModelFromFileSection(ScopedFile& model_file,
                                                         uint64_t begin_offset,
                                                         uint64_t end_offset) {
  if (end_offset <= begin_offset) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid LiteRT-LM section range: [", begin_offset, ", ",
                     end_offset, ")"));
  }

  LITERT_ASSIGN_OR_RETURN(auto dup_file, model_file.Duplicate());
  LITERT_ASSIGN_OR_RETURN(int fd, dup_file.Release());
  auto model_or =
      litert::Model::CreateFromFd(fd, begin_offset, end_offset - begin_offset);
#if defined(_WIN32)
  _close(fd);
#else
  close(fd);
#endif
  LITERT_ASSIGN_OR_RETURN(auto model, model_or);
  return model;
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesLitertLm::Create(
    std::unique_ptr<LitertLmLoader> litert_lm_loader,
    bool enable_file_backed_model_loading) {
  return absl::WrapUnique(new ModelResourcesLitertLm(
      std::move(litert_lm_loader), enable_file_backed_model_loading));
};

absl::StatusOr<const litert::Model*> ModelResourcesLitertLm::GetTFLiteModel(
    ModelType model_type) {
  auto it = model_map_.find(model_type);
  if (it != model_map_.end()) {
    return it->second.get();
  }

  if (enable_file_backed_model_loading_) {
    auto scoped_file = litert_lm_loader_->GetScopedFile();
    auto section_location = litert_lm_loader_->GetSectionLocation(
        BufferKey(schema::AnySectionDataType_TFLiteModel, model_type));
    if (scoped_file.ok() && section_location.ok()) {
      auto model_from_section = CreateModelFromFileSection(
          scoped_file->get(), section_location->first,
          section_location->second);
      if (!model_from_section.ok()) {
        if (model_from_section.status().code() !=
            absl::StatusCode::kUnimplemented) {
          return model_from_section.status();
        }
        ABSL_VLOG(1) << "File-backed LiteRT model loading is unsupported; "
                        "falling back to buffer-backed loading.";
      } else {
        auto& model = model_map_[model_type];
        model = std::make_unique<litert::Model>(
            std::move(model_from_section).value());
        return model.get();
      }
    }
  }

  litert::BufferRef<uint8_t> buffer_ref =
      litert_lm_loader_->GetTFLiteModel(model_type);
  ABSL_VLOG(1) << "model_type: " << ModelTypeToString(model_type);
  ABSL_VLOG(1) << "litert model size: " << buffer_ref.Size();
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

  ABSL_VLOG(1) << "model_type: " << ModelTypeToString(model_type);
  ABSL_VLOG(1) << "litert model size: " << buffer_ref.Size();
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
}

absl::StatusOr<const proto::ExecutorMetadata*>
ModelResourcesLitertLm::GetExecutorMetadata() {
  if (executor_metadata_ == nullptr) {
    auto buffer_ref_or = litert_lm_loader_->GetExecutorMetadata();
    if (!buffer_ref_or.has_value()) {
      return absl::NotFoundError(
          "ExecutorMetadata not found in the model file.");
    }
    auto executor_metadata = std::make_unique<proto::ExecutorMetadata>();
    if (!executor_metadata->ParseFromString(
            std::string(buffer_ref_or->StrView()))) {  // NOLINT
      return absl::InternalError("Failed to parse ExecutorMetadata");
    }
    executor_metadata_ = std::move(executor_metadata);
  }
  return executor_metadata_.get();
}

absl::StatusOr<const proto::EmbeddingMetadata*>
ModelResourcesLitertLm::GetEmbeddingMetadata() {
  if (embedding_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetEmbeddingMetadata();
    if (!buffer_ref.has_value()) {
      return absl::NotFoundError("No EmbeddingMetadata found in the model.");
    }
    auto embedding_metadata = std::make_unique<proto::EmbeddingMetadata>();
    if (!embedding_metadata->ParseFromString(
            std::string(buffer_ref->StrView()))) {  // NOLINT
      return absl::InternalError("Failed to parse EmbeddingMetadata");
    }
    embedding_metadata_ = std::move(embedding_metadata);
  }
  return embedding_metadata_.get();
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

absl::StatusOr<FileRegion>
ModelResourcesLitertLm::GetTFLiteModelSectionFileRegion(
    ModelType model_type) {
  LITERT_ASSIGN_OR_RETURN(
      auto location,
      litert_lm_loader_->GetSectionLocation(
          BufferKey(schema::AnySectionDataType_TFLiteModel, model_type)));
  return FileRegion{
      .offset = location.first,
      .size = location.second - location.first,
  };
}

}  // namespace litert::lm
