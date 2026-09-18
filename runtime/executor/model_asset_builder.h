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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_ASSET_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_ASSET_BUILDER_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_sections.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Builder for constructing a ModelAssets instance at runtime from individual
// section files or in-memory protos/buffers without requiring a monolithic
// .litertlm bundle.
class ModelAssetBuilder {
 public:
  ModelAssetBuilder() = default;

  // Sets LlmMetadata from a proto object or a file path (.pb binary or .pbtxt
  // textproto).
  ModelAssetBuilder& SetLlmMetadata(proto::LlmMetadata llm_metadata);
  ModelAssetBuilder& SetLlmMetadataFromFile(absl::string_view path);

  // Sets ExecutorMetadata from a proto object or a file path (.pb or .pbtxt).
  ModelAssetBuilder& SetExecutorMetadata(
      proto::ExecutorMetadata executor_metadata);
  ModelAssetBuilder& SetExecutorMetadataFromFile(absl::string_view path);

  // Sets EmbeddingMetadata from a proto object or a file path (.pb or .pbtxt).
  ModelAssetBuilder& SetEmbeddingMetadata(
      proto::EmbeddingMetadata embedding_metadata);
  ModelAssetBuilder& SetEmbeddingMetadataFromFile(absl::string_view path);

  // Adds a TFLite model section for the given ModelType (defaults to
  // ModelType::kTfLitePrefillDecode).
  ModelAssetBuilder& AddTFLiteModel(
      absl::string_view path,
      ModelType model_type = ModelType::kTfLitePrefillDecode,
      std::optional<std::string> backend_constraint = std::nullopt,
      std::optional<std::string> prefer_activation_type = std::nullopt);
  ModelAssetBuilder& AddTFLiteModel(
      std::shared_ptr<ScopedFile> scoped_file,
      ModelType model_type = ModelType::kTfLitePrefillDecode,
      std::optional<std::string> backend_constraint = std::nullopt,
      std::optional<std::string> prefer_activation_type = std::nullopt);
  ModelAssetBuilder& AddTFLiteModel(
      std::shared_ptr<MemoryMappedFile> mapped_file,
      ModelType model_type = ModelType::kTfLitePrefillDecode,
      std::optional<std::string> backend_constraint = std::nullopt,
      std::optional<std::string> prefer_activation_type = std::nullopt);

  // Adds an external TFLite weights file for the given ModelType.
  ModelAssetBuilder& AddTFLiteWeights(
      absl::string_view path,
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  ModelAssetBuilder& AddTFLiteWeights(
      std::shared_ptr<ScopedFile> scoped_file,
      ModelType model_type = ModelType::kTfLitePrefillDecode);

  // Sets SentencePiece or HuggingFace tokenizer for the given ModelType.
  ModelAssetBuilder& SetSentencePieceTokenizer(
      absl::string_view path,
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  ModelAssetBuilder& SetSentencePieceTokenizerFromBuffer(
      absl::string_view buffer,
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  ModelAssetBuilder& SetHuggingFaceTokenizer(
      absl::string_view path,
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  ModelAssetBuilder& SetHuggingFaceTokenizerFromBuffer(
      absl::string_view buffer,
      ModelType model_type = ModelType::kTfLitePrefillDecode);

  // Optional identifier/path used for cache key derivation when caching is
  // enabled.
  ModelAssetBuilder& SetCacheKeyPath(absl::string_view cache_key_path);

  // Validates sections (parsing metadata files if paths were supplied) and
  // builds a ModelAssets instance.
  absl::StatusOr<ModelAssets> Build() const;

 private:
  ModelResourcesSections::Config config_;
  std::optional<std::string> llm_metadata_path_;
  std::optional<std::string> executor_metadata_path_;
  std::optional<std::string> embedding_metadata_path_;
  std::string cache_key_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_ASSET_BUILDER_H_
