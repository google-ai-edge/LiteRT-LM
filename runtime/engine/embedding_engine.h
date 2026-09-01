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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/model_signature_utils.h"
#include "runtime/proto/embedding_metadata.pb.h"

namespace litert::lm {

// Options for configuring the embedding computation.
struct EmbeddingOptions {
  // If true, the output embedding vector will be L2-normalized.
  bool normalize = true;

  // If true, automatically inserts the BOS, EOS, start of image, end of image,
  // start of audio, and end of audio tokens.
  bool insert_special_tokens = true;

  // Strategy for handling inputs longer than the maximum supported signature
  // length.
  InputOverflowStrategy input_overflow_strategy = InputOverflowStrategy::kError;
  // The number of vision soft tokens to generate per image. If set, overrides
  // the default max_num_patches parameter for image preprocessing during this
  // computation.
  std::optional<int> vision_tokens_per_image = std::nullopt;

  // The output embedding size to truncate the embedding to. If not set, uses
  // the default output embedding size. Must be non-negative and not exceed the
  // default output embedding size.
  std::optional<int> output_size = std::nullopt;
};

// Represents the result of an embedding computation.
struct EmbeddingResponse {
  // The computed embedding vector.
  std::vector<float> embedding;
  // The length of the input.
  int input_length = 0;
  // If the input was truncated, contains the length it was truncated to.
  std::optional<int> truncated_length = std::nullopt;
  // The number of chunks the input was split into.
  int num_chunks = 1;
};

// Interface for embedding models.
class EmbeddingEngine {
 public:
  virtual ~EmbeddingEngine() = default;

  // Computes embedding response for the given single request.
  //
  // - contents: The input data (text, audio, image) to embed.
  // - options: Configuration options for embedding computation.
  // - returns: The embedding response, or an error status on failure.
  virtual absl::StatusOr<EmbeddingResponse> ComputeEmbedding(
      const std::vector<InputData>& contents,
      const EmbeddingOptions& options) = 0;

  // Computes a batch of embedding responses for the given batch of requests.
  //
  // - contents: A batch of requests, where each request is a vector of
  //   InputData.
  // - options: Configuration options applied to all requests in the batch.
  // - returns: A vector of embedding responses corresponding to each input
  //   request, or an error status on failure.
  virtual absl::StatusOr<std::vector<EmbeddingResponse>> ComputeEmbeddingBatch(
      const std::vector<std::vector<InputData>>& contents,
      const EmbeddingOptions& options) = 0;

  // Returns the benchmark info of the engine.
  virtual absl::optional<BenchmarkInfo> GetBenchmarkInfo() = 0;

  // Returns the mutable benchmark info of the engine.
  virtual BenchmarkInfo* GetMutableBenchmarkInfo() = 0;

  // Returns the embedding metadata of the engine.
  virtual const std::optional<proto::EmbeddingMetadata>& GetEmbeddingMetadata()
      const = 0;

  // Returns the selected text encoder signatures info if auto-selection was
  // performed during engine creation, or nullopt otherwise.
  virtual const std::optional<SelectedTextSignaturesInfo>&
  GetSelectedTextSignaturesInfo() const = 0;

  // Returns the selected vision signature info if auto-selection was performed
  // during engine creation, or nullopt otherwise.
  virtual const std::optional<SelectedVisionSignatureInfo>&
  GetSelectedVisionSignatureInfo() const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_H_
