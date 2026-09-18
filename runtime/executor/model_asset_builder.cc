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

#include "runtime/executor/model_asset_builder.h"

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_sections.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

namespace litert::lm {

namespace {

template <typename ProtoT>
absl::StatusOr<ProtoT> ParseProtoFromFile(absl::string_view path,
                                          absl::string_view type_name) {
  ABSL_ASSIGN_OR_RETURN(ScopedFile scoped_file, ScopedFile::Open(path));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mapped_file,
                        MemoryMappedFile::Create(scoped_file.file()));
  absl::string_view content(reinterpret_cast<const char*>(mapped_file->data()),
                            mapped_file->length());
  ProtoT proto;
  const bool is_textproto = absl::EndsWith(path, ".pbtxt") ||
                            absl::EndsWith(path, ".pbtext") ||
                            absl::EndsWith(path, ".textproto");
  if (is_textproto) {
    if constexpr (std::is_base_of_v<proto2::Message, ProtoT>) {
      if (!google::protobuf::TextFormat::ParseFromString(content, &proto)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to parse textproto ", type_name, " from file: ", path));
      }
      return proto;
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Textproto parsing is not supported in protobuf lite "
                       "builds for ",
                       type_name, ": ", path));
    }
  }
  if (!proto.ParseFromString(content)) {
    // Fallback to textproto in case a text file was provided without standard
    // extension.
    if constexpr (std::is_base_of_v<proto2::Message, ProtoT>) {
      if (google::protobuf::TextFormat::ParseFromString(content, &proto)) {
        return proto;
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse ", type_name, " from file: ", path));
  }
  return proto;
}

}  // namespace

ModelAssetBuilder& ModelAssetBuilder::SetLlmMetadata(
    proto::LlmMetadata llm_metadata) {
  config_.llm_metadata = std::move(llm_metadata);
  llm_metadata_path_ = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetLlmMetadataFromFile(
    absl::string_view path) {
  llm_metadata_path_ = std::string(path);
  config_.llm_metadata = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetExecutorMetadata(
    proto::ExecutorMetadata executor_metadata) {
  config_.executor_metadata = std::move(executor_metadata);
  executor_metadata_path_ = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetExecutorMetadataFromFile(
    absl::string_view path) {
  executor_metadata_path_ = std::string(path);
  config_.executor_metadata = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetEmbeddingMetadata(
    proto::EmbeddingMetadata embedding_metadata) {
  config_.embedding_metadata = std::move(embedding_metadata);
  embedding_metadata_path_ = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetEmbeddingMetadataFromFile(
    absl::string_view path) {
  embedding_metadata_path_ = std::string(path);
  config_.embedding_metadata = std::nullopt;
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::AddTFLiteModel(
    absl::string_view path, ModelType model_type,
    std::optional<std::string> backend_constraint,
    std::optional<std::string> prefer_activation_type) {
  config_.tflite_models[model_type] = TFLiteModelSection{
      .path = std::string(path),
      .backend_constraint = std::move(backend_constraint),
      .prefer_activation_type = std::move(prefer_activation_type),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::AddTFLiteModel(
    std::shared_ptr<ScopedFile> scoped_file, ModelType model_type,
    std::optional<std::string> backend_constraint,
    std::optional<std::string> prefer_activation_type) {
  config_.tflite_models[model_type] = TFLiteModelSection{
      .scoped_file = std::move(scoped_file),
      .backend_constraint = std::move(backend_constraint),
      .prefer_activation_type = std::move(prefer_activation_type),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::AddTFLiteModel(
    std::shared_ptr<MemoryMappedFile> mapped_file, ModelType model_type,
    std::optional<std::string> backend_constraint,
    std::optional<std::string> prefer_activation_type) {
  config_.tflite_models[model_type] = TFLiteModelSection{
      .mapped_file = std::move(mapped_file),
      .backend_constraint = std::move(backend_constraint),
      .prefer_activation_type = std::move(prefer_activation_type),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::AddTFLiteWeights(absl::string_view path,
                                                       ModelType model_type) {
  config_.tflite_weights[model_type] = TFLiteWeightsSection{
      .path = std::string(path),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::AddTFLiteWeights(
    std::shared_ptr<ScopedFile> scoped_file, ModelType model_type) {
  config_.tflite_weights[model_type] = TFLiteWeightsSection{
      .scoped_file = std::move(scoped_file),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetSentencePieceTokenizer(
    absl::string_view path, ModelType model_type) {
  config_.tokenizers[model_type] = TokenizerSection{
      .format = TokenizerFormat::kSentencePiece,
      .path = std::string(path),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetSentencePieceTokenizerFromBuffer(
    absl::string_view buffer, ModelType model_type) {
  config_.tokenizers[model_type] = TokenizerSection{
      .format = TokenizerFormat::kSentencePiece,
      .buffer = std::string(buffer),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetHuggingFaceTokenizer(
    absl::string_view path, ModelType model_type) {
  config_.tokenizers[model_type] = TokenizerSection{
      .format = TokenizerFormat::kHuggingFaceJson,
      .path = std::string(path),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetHuggingFaceTokenizerFromBuffer(
    absl::string_view buffer, ModelType model_type) {
  config_.tokenizers[model_type] = TokenizerSection{
      .format = TokenizerFormat::kHuggingFaceJson,
      .buffer = std::string(buffer),
  };
  return *this;
}

ModelAssetBuilder& ModelAssetBuilder::SetCacheKeyPath(
    absl::string_view cache_key_path) {
  cache_key_path_ = std::string(cache_key_path);
  return *this;
}

absl::StatusOr<ModelAssets> ModelAssetBuilder::Build() const {
  ModelResourcesSections::Config built_config = config_;
  if (llm_metadata_path_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(built_config.llm_metadata,
                          ParseProtoFromFile<proto::LlmMetadata>(
                              *llm_metadata_path_, "LlmMetadata"));
  }
  if (executor_metadata_path_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(built_config.executor_metadata,
                          ParseProtoFromFile<proto::ExecutorMetadata>(
                              *executor_metadata_path_, "ExecutorMetadata"));
  }
  if (embedding_metadata_path_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(built_config.embedding_metadata,
                          ParseProtoFromFile<proto::EmbeddingMetadata>(
                              *embedding_metadata_path_, "EmbeddingMetadata"));
  }

  auto shared_config =
      std::make_shared<ModelResourcesSections::Config>(std::move(built_config));
  ModelAssets::ModelResourcesFactory factory =
      [shared_config](bool enable_file_backed_model_loading)
      -> absl::StatusOr<std::unique_ptr<ModelResources>> {
    ModelResourcesSections::Config config_copy = *shared_config;
    config_copy.enable_file_backed_model_loading =
        enable_file_backed_model_loading;
    return ModelResourcesSections::Create(std::move(config_copy));
  };
  return ModelAssets::Create(std::move(factory), cache_key_path_);
}

}  // namespace litert::lm
