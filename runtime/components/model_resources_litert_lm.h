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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Model resources for the litert lm model.
class ModelResourcesLitertLm : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<LitertLmLoader> litert_lm_loader,
      bool enable_file_backed_model_loading = false);

  using ModelResources::GetOrCreateTokenizer;
  using ModelResources::GetTFLiteModel;
  using ModelResources::GetTFLiteModelBackendConstraint;
  using ModelResources::GetTFLiteModelBuffer;
  using ModelResources::GetTFLiteModelPreferActivationType;
  using ModelResources::GetTFLiteModelSectionFileRegion;
  using ModelResources::GetTokenizer;
  using ModelResources::GetWeightsSectionOffset;

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      absl::string_view model_type_str) override;

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override;

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str) override;

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str) override;

  // Returns the tokenizer from the *.litertlm file. If both SentencePiece and
  // HuggingFace tokenizer are present and supported by the current build
  // configuration, the SentencePiece tokenizer will be used.
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;

  // Returns the tokenizer for the specified model type from the *.litertlm
  // file.
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer(
      absl::string_view model_type_str) override;

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata() override;

  absl::StatusOr<const proto::EmbeddingMetadata*> GetEmbeddingMetadata()
      override;

  absl::StatusOr<const proto::Text2SpeechMetadata*> GetText2SpeechMetadata()
      override;

  absl::StatusOr<const proto::Speech2TextMetadata*> GetSpeech2TextMetadata()
      override;

  absl::StatusOr<const proto::Text2ImageMetadata*> GetText2ImageMetadata()
      override;

  absl::StatusOr<const proto::Any2AnyMetadata*> GetAny2AnyMetadata() override;

  absl::StatusOr<absl::string_view> GetGenericBinaryDataBuffer(
      absl::string_view name) override;

  std::vector<std::string> GetGenericBinaryDataNames() const override;

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override;

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str) override;

  // Returns the TFLite model section files region.
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str) override;

 protected:
  explicit ModelResourcesLitertLm(
      std::unique_ptr<LitertLmLoader> litert_lm_loader,
      bool enable_file_backed_model_loading)
      : litert_lm_loader_(std::move(litert_lm_loader)),
        enable_file_backed_model_loading_(enable_file_backed_model_loading) {}

  // The litert lm loader, used to mmap the tokenizer and tflite model etc from
  // the .litertlm model file.
  std::unique_ptr<LitertLmLoader> litert_lm_loader_;
  bool enable_file_backed_model_loading_;

  absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<const Tokenizer*> GetOrCreateTokenizer(
      absl::string_view model_type_str) override;

 private:
  // Returns the TFLite model section for the lower-cased model type `key`.
  // Returns kNotFound when the section is absent (or empty) and propagates the
  // loader's error when the section exists but could not be mapped.
  absl::StatusOr<litert::BufferRef<uint8_t>> GetTFLiteModelSectionBuffer(
      absl::string_view key);

  absl::flat_hash_map<std::string, std::unique_ptr<litert::Model>> model_map_;
  absl::flat_hash_map<std::string, std::unique_ptr<Tokenizer>> tokenizer_map_;
  std::unique_ptr<proto::LlmMetadata> llm_metadata_;
  std::unique_ptr<proto::EmbeddingMetadata> embedding_metadata_;
  std::unique_ptr<proto::Text2SpeechMetadata> text2speech_metadata_;
  std::unique_ptr<proto::Speech2TextMetadata> speech2text_metadata_;
  std::unique_ptr<proto::Text2ImageMetadata> text2image_metadata_;
  std::unique_ptr<proto::Any2AnyMetadata> any2any_metadata_;
  std::unique_ptr<proto::ExecutorMetadata> executor_metadata_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_H_
