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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_SECTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_SECTIONS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

struct TFLiteModelSection {
  std::string path;
  std::shared_ptr<ScopedFile> scoped_file;
  std::shared_ptr<MemoryMappedFile> mapped_file;
  std::optional<std::string> backend_constraint;
  std::optional<std::string> prefer_activation_type;
};

struct TFLiteWeightsSection {
  std::string path;
  std::shared_ptr<ScopedFile> scoped_file;
};

enum class TokenizerFormat {
  kSentencePiece,
  kHuggingFaceJson,
};

struct TokenizerSection {
  TokenizerFormat format;
  std::string path;
  std::string buffer;
};

// ModelResources implementation backed by individual section files or buffers
// constructed at runtime, without requiring a monolithic .litertlm bundle.
class ModelResourcesSections : public ModelResources {
 public:
  struct Config {
    std::optional<proto::LlmMetadata> llm_metadata;
    std::optional<proto::ExecutorMetadata> executor_metadata;
    std::optional<proto::EmbeddingMetadata> embedding_metadata;
    absl::flat_hash_map<ModelType, TFLiteModelSection> tflite_models;
    absl::flat_hash_map<ModelType, TFLiteWeightsSection> tflite_weights;
    absl::flat_hash_map<ModelType, TokenizerSection> tokenizers;
    bool enable_file_backed_model_loading = false;
  };

  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(Config config);

  ~ModelResourcesSections() override = default;

  // ModelResources overrides:
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override;
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override;
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override;
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile(
      ModelType model_type) override;
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override;
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override;
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override;
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override;
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      ModelType model_type) override;
  absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      ModelType model_type) override;
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata() override;
  absl::StatusOr<const proto::EmbeddingMetadata*> GetEmbeddingMetadata()
      override;

 private:
  explicit ModelResourcesSections(Config config) : config_(std::move(config)) {}

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetOrOpenModelScopedFile(
      ModelType model_type);
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetOrOpenWeightsScopedFile(
      ModelType model_type);
  absl::StatusOr<std::shared_ptr<MemoryMappedFile>> GetOrMapModelFile(
      ModelType model_type);

  Config config_;
  absl::flat_hash_map<ModelType, std::shared_ptr<ScopedFile>>
      model_scoped_files_;
  absl::flat_hash_map<ModelType, std::shared_ptr<ScopedFile>>
      weights_scoped_files_;
  absl::flat_hash_map<ModelType, std::shared_ptr<MemoryMappedFile>>
      mapped_model_files_;
  absl::flat_hash_map<ModelType, std::unique_ptr<litert::Model>> model_map_;
  absl::flat_hash_map<ModelType, std::unique_ptr<Tokenizer>> tokenizer_map_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_SECTIONS_H_
