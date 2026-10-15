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
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::lm {

using ::litert::support::HuggingFaceTokenizer;

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
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<const litert::Model*> ModelResourcesLitertLm::GetTFLiteModel(
    absl::string_view model_type_str) {
  std::string key = absl::AsciiStrToLower(model_type_str);
  auto it = model_map_.find(key);
  if (it != model_map_.end()) {
    return it->second.get();
  }

  if (enable_file_backed_model_loading_) {
    auto scoped_file = litert_lm_loader_->GetScopedFile();
    auto section_location = litert_lm_loader_->GetSectionLocation(
        BufferKey(schema::AnySectionDataType_TFLiteModel, key));
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
        auto& model = model_map_[key];
        model = std::make_unique<litert::Model>(
            std::move(model_from_section).value());
        return model.get();
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(litert::BufferRef<uint8_t> buffer_ref,
                        GetTFLiteModelSectionBuffer(key));
  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
  model_map_[key] = std::make_unique<litert::Model>(std::move(model));
  return model_map_[key].get();
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelBackendConstraint(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBackendConstraint(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelBackendConstraint(
    absl::string_view model_type_str) {
  return litert_lm_loader_->GetTFLiteModelBackendConstraint(model_type_str);
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelPreferActivationType(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelPreferActivationType(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
ModelResourcesLitertLm::GetTFLiteModelPreferActivationType(
    absl::string_view model_type_str) {
  return litert_lm_loader_->GetTFLiteModelPreferActivationType(model_type_str);
}

absl::StatusOr<absl::string_view> ModelResourcesLitertLm::GetTFLiteModelBuffer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBuffer(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<absl::string_view> ModelResourcesLitertLm::GetTFLiteModelBuffer(
    absl::string_view model_type_str) {
  std::string key = absl::AsciiStrToLower(model_type_str);
  ABSL_ASSIGN_OR_RETURN(litert::BufferRef<uint8_t> buffer_ref,
                        GetTFLiteModelSectionBuffer(key));
  return buffer_ref.StrView();
}

absl::StatusOr<litert::BufferRef<uint8_t>>
ModelResourcesLitertLm::GetTFLiteModelSectionBuffer(absl::string_view key) {
  auto section_buffer = litert_lm_loader_->GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_TFLiteModel, key));
  if (absl::IsNotFound(section_buffer.status())) {
    return absl::NotFoundError(absl::StrCat(key, " not found in the model."));
  }
  // A section that exists but fails to map is a broken model file, not a
  // missing section; propagate it so engine creation fails instead of
  // producing an executor that silently runs without this model.
  ABSL_ASSIGN_OR_RETURN(litert::BufferRef<uint8_t> buffer_ref,
                        std::move(section_buffer));
  ABSL_VLOG(1) << "model_type: " << key;
  ABSL_VLOG(1) << "litert model size: " << buffer_ref.Size();
  if (buffer_ref.Size() == 0) {
    return absl::NotFoundError(absl::StrCat(key, " not found in the model."));
  }
  return buffer_ref;
}

absl::StatusOr<std::unique_ptr<Tokenizer>>
ModelResourcesLitertLm::GetTokenizer() {
  return GetTokenizer(proto::LlmMetadata::TF_LITE_PREFILL_DECODE);
}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResourcesLitertLm::GetTokenizer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTokenizer(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResourcesLitertLm::GetTokenizer(
    absl::string_view model_type_str) {
  auto sp_tokenizer =
      litert_lm_loader_->GetSentencePieceTokenizer(model_type_str);
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
  if (sp_tokenizer) {
    return SentencePieceTokenizer::CreateFromBuffer(sp_tokenizer->StrView());
  }
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

  auto hf_tokenizer =
      litert_lm_loader_->GetHuggingFaceTokenizer(model_type_str);
  if (hf_tokenizer) {
    std::string json_data(hf_tokenizer->StrData(), hf_tokenizer->Size());
    return HuggingFaceTokenizer::CreateFromJson(json_data);
  }

  if (sp_tokenizer) {
    return absl::UnimplementedError(
        "SentencePiece tokenizer found, but LiteRT LM was built with "
        "--define=DISABLE_SENTENCEPIECE_TOKENIZER=1.");
  } else {
    return absl::NotFoundError(absl::StrCat(
        "No tokenizer found in the model for model type: ", model_type_str));
  }
}

absl::StatusOr<const Tokenizer*> ModelResourcesLitertLm::GetOrCreateTokenizer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetOrCreateTokenizer(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<const Tokenizer*> ModelResourcesLitertLm::GetOrCreateTokenizer(
    absl::string_view model_type_str) {
  std::string key = absl::AsciiStrToLower(model_type_str);
  if (auto it = tokenizer_map_.find(key); it != tokenizer_map_.end()) {
    return it->second.get();
  }
  ABSL_ASSIGN_OR_RETURN(auto tokenizer, GetTokenizer(key));
  return tokenizer_map_.emplace(std::move(key), std::move(tokenizer))
      .first->second.get();
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesLitertLm::GetLlmMetadata() {
  if (llm_metadata_ == nullptr) {
    // Non-LLM containers (e.g. TTS, ASR) carry no LlmMetadata section; report
    // NotFound instead of dereferencing an empty optional.
    if (!litert_lm_loader_->HasLlmMetadata()) {
      return absl::NotFoundError("LlmMetadata not found in the model file.");
    }
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

absl::StatusOr<const proto::Text2SpeechMetadata*>
ModelResourcesLitertLm::GetText2SpeechMetadata() {
  if (text2speech_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetText2SpeechMetadata();
    if (!buffer_ref.has_value()) {
      return absl::NotFoundError("No Text2SpeechMetadata found in the model.");
    }
    auto text2speech_metadata = std::make_unique<proto::Text2SpeechMetadata>();
    if (!text2speech_metadata->ParseFromArray(buffer_ref->Data(),
                                              buffer_ref->Size())) {
      return absl::InternalError("Failed to parse Text2SpeechMetadata");
    }
    text2speech_metadata_ = std::move(text2speech_metadata);
  }
  return text2speech_metadata_.get();
}

absl::StatusOr<const proto::Speech2TextMetadata*>
ModelResourcesLitertLm::GetSpeech2TextMetadata() {
  if (speech2text_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetSpeech2TextMetadata();
    if (!buffer_ref.has_value()) {
      return absl::NotFoundError("No Speech2TextMetadata found in the model.");
    }
    auto speech2text_metadata = std::make_unique<proto::Speech2TextMetadata>();
    if (!speech2text_metadata->ParseFromArray(buffer_ref->Data(),
                                              buffer_ref->Size())) {
      return absl::InternalError("Failed to parse Speech2TextMetadata");
    }
    speech2text_metadata_ = std::move(speech2text_metadata);
  }
  return speech2text_metadata_.get();
}

absl::StatusOr<const proto::Text2ImageMetadata*>
ModelResourcesLitertLm::GetText2ImageMetadata() {
  if (text2image_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetText2ImageMetadata();
    if (!buffer_ref.has_value()) {
      return absl::NotFoundError("No Text2ImageMetadata found in the model.");
    }
    auto text2image_metadata = std::make_unique<proto::Text2ImageMetadata>();
    if (!text2image_metadata->ParseFromArray(buffer_ref->Data(),
                                             buffer_ref->Size())) {
      return absl::InternalError("Failed to parse Text2ImageMetadata");
    }
    text2image_metadata_ = std::move(text2image_metadata);
  }
  return text2image_metadata_.get();
}

absl::StatusOr<const proto::Any2AnyMetadata*>
ModelResourcesLitertLm::GetAny2AnyMetadata() {
  if (any2any_metadata_ == nullptr) {
    auto buffer_ref = litert_lm_loader_->GetAny2AnyMetadata();
    if (!buffer_ref.has_value()) {
      return absl::NotFoundError("No Any2AnyMetadata found in the model.");
    }
    auto any2any_metadata = std::make_unique<proto::Any2AnyMetadata>();
    if (!any2any_metadata->ParseFromArray(buffer_ref->Data(),
                                          buffer_ref->Size())) {
      return absl::InternalError("Failed to parse Any2AnyMetadata");
    }
    any2any_metadata_ = std::move(any2any_metadata);
  }
  return any2any_metadata_.get();
}

absl::StatusOr<absl::string_view>
ModelResourcesLitertLm::GetGenericBinaryDataBuffer(absl::string_view name) {
  auto buffer_ref = litert_lm_loader_->GetGenericBinaryData(name);
  if (!buffer_ref.has_value() || buffer_ref->Size() == 0) {
    return absl::NotFoundError(
        "GenericBinaryData section not found in the model.");
  }
  return buffer_ref->StrView();
}

std::vector<std::string> ModelResourcesLitertLm::GetGenericBinaryDataNames()
    const {
  return litert_lm_loader_->GetGenericBinaryDataNames();
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesLitertLm::GetScopedFile() {
  return litert_lm_loader_->GetScopedFile();
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesLitertLm::GetWeightsSectionOffset(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetWeightsSectionOffset(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesLitertLm::GetWeightsSectionOffset(
    absl::string_view model_type_str) {
  return litert_lm_loader_->GetSectionLocation(
      BufferKey(schema::AnySectionDataType_TFLiteWeights, model_type_str));
}

absl::StatusOr<FileRegion>
ModelResourcesLitertLm::GetTFLiteModelSectionFileRegion(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelSectionFileRegion(
      TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<FileRegion>
ModelResourcesLitertLm::GetTFLiteModelSectionFileRegion(
    absl::string_view model_type_str) {
  ABSL_ASSIGN_OR_RETURN(
      auto location,
      litert_lm_loader_->GetSectionLocation(
          BufferKey(schema::AnySectionDataType_TFLiteModel, model_type_str)));
  return FileRegion{
      .offset = location.first,
      .size = location.second - location.first,
  };
}

}  // namespace litert::lm
