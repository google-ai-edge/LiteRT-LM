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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EMBEDDING_ENGINE_IMPL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EMBEDDING_ENGINE_IMPL_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/litert_util.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

struct SpecialTokens {
  std::vector<int> bos_token_ids;
  std::vector<int> eos_token_ids;
  std::vector<int> start_of_image_token_ids;
  std::vector<int> end_of_image_token_ids;
  std::vector<int> start_of_audio_token_ids;
  std::vector<int> end_of_audio_token_ids;
  bool has_end_of_vision_model = false;
  bool has_end_of_audio_model = false;
};

class EmbeddingEngineImpl : public EmbeddingEngine {
 public:
  // Creates an EmbeddingEngineImpl instance from ModelResources,
  // OwnedEnvironment, Tokenizer, and EmbeddingEngineSettings.
  static absl::StatusOr<std::unique_ptr<EmbeddingEngine>> Create(
      std::unique_ptr<ModelResources> resources,
      std::unique_ptr<OwnedEnvironment> env,
      std::unique_ptr<::litert::support::Tokenizer> tokenizer,
      EmbeddingEngineSettings settings,
      std::optional<BenchmarkInfo> benchmark_info = std::nullopt);

  // Creates an EmbeddingEngineImpl instance from EmbeddingEngineSettings.
  static absl::StatusOr<std::unique_ptr<EmbeddingEngine>> Create(
      EmbeddingEngineSettings settings);

  // Constructs an `EmbeddingEngineImpl` with a LiteRT environment, a tokenizer
  // and executors.
  EmbeddingEngineImpl(
      std::unique_ptr<OwnedEnvironment> env,
      std::unique_ptr<::litert::support::Tokenizer> tokenizer,
      std::unique_ptr<EmbeddingExecutorBase> embedding_executor,
      std::unique_ptr<VisionExecutor> vision_executor = nullptr,
      std::unique_ptr<AudioExecutor> audio_executor = nullptr,
      std::optional<BenchmarkInfo> benchmark_info = std::nullopt,
      SpecialTokens special_tokens = {},
      std::unique_ptr<::litert::support::ImagePreprocessor> image_preprocessor =
          nullptr,
      std::optional<::litert::support::ImagePreprocessParameter>
          image_preprocess_parameter = std::nullopt,
      std::unique_ptr<::litert::support::AudioPreprocessor> audio_preprocessor =
          nullptr,
      std::optional<proto::EmbeddingMetadata> metadata = std::nullopt,
      std::optional<SelectedTextSignaturesInfo> selected_text_signatures_info =
          std::nullopt,
      std::optional<SelectedVisionSignatureInfo>
          selected_vision_signature_info = std::nullopt);

  ~EmbeddingEngineImpl() override = default;

  // Computes the embedding response for the given single request.
  absl::StatusOr<EmbeddingResponse> ComputeEmbedding(
      const std::vector<InputData>& contents,
      const EmbeddingOptions& options) override;

  // Computes a batch of embedding responses for the given batch of requests.
  absl::StatusOr<std::vector<EmbeddingResponse>> ComputeEmbeddingBatch(
      const std::vector<std::vector<InputData>>& contents,
      const EmbeddingOptions& options) override;

  // Returns the benchmark info of the engine.
  absl::optional<BenchmarkInfo> GetBenchmarkInfo() override;

  // Returns the mutable benchmark info of the engine.
  BenchmarkInfo* GetMutableBenchmarkInfo() override;

  // Returns the special tokens configured for the engine.
  const SpecialTokens& GetSpecialTokens() const { return special_tokens_; }

  // Returns the image preprocess parameter configured for the engine.
  const std::optional<::litert::support::ImagePreprocessParameter>&
  GetImagePreprocessParameter() const {
    return image_preprocess_parameter_;
  }

  // Returns the image preprocessor configured for the engine.
  const ::litert::support::ImagePreprocessor* GetImagePreprocessor() const {
    return image_preprocessor_.get();
  }

  // Returns the audio preprocessor configured for the engine.
  const ::litert::support::AudioPreprocessor* GetAudioPreprocessor() const {
    return audio_preprocessor_.get();
  }

  // Returns the embedding metadata of the engine.
  const std::optional<proto::EmbeddingMetadata>& GetEmbeddingMetadata()
      const override;

  // Returns the selected text encoder signatures info if auto-selection was
  // performed during engine creation, or nullopt otherwise.
  const std::optional<SelectedTextSignaturesInfo>&
  GetSelectedTextSignaturesInfo() const override {
    return selected_text_signatures_info_;
  }

  // Returns the selected vision signature info if auto-selection was performed
  // during engine creation, or nullopt otherwise.
  const std::optional<SelectedVisionSignatureInfo>&
  GetSelectedVisionSignatureInfo() const override {
    return selected_vision_signature_info_;
  }

 private:
  absl::StatusOr<std::vector<InputData>> InsertSpecialTokens(
      const std::vector<InputData>& contents) const;

  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& contents, const EmbeddingOptions& options);

  absl::StatusOr<EmbeddingResponse> ComputeEmbeddingInternal(
      const ExecutorInputs& inputs, const EmbeddingOptions& options);

  std::unique_ptr<OwnedEnvironment> env_;
  std::unique_ptr<::litert::support::Tokenizer> tokenizer_;
  std::unique_ptr<EmbeddingExecutorBase> embedding_executor_;
  std::unique_ptr<VisionExecutor> vision_executor_;
  std::unique_ptr<AudioExecutor> audio_executor_;
  std::optional<BenchmarkInfo> benchmark_info_;
  SpecialTokens special_tokens_;
  std::unique_ptr<::litert::support::ImagePreprocessor> image_preprocessor_;
  std::optional<::litert::support::ImagePreprocessParameter>
      image_preprocess_parameter_;
  std::unique_ptr<::litert::support::AudioPreprocessor> audio_preprocessor_;
  std::optional<proto::EmbeddingMetadata> metadata_;
  std::optional<SelectedTextSignaturesInfo> selected_text_signatures_info_;
  std::optional<SelectedVisionSignatureInfo> selected_vision_signature_info_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EMBEDDING_ENGINE_IMPL_H_
