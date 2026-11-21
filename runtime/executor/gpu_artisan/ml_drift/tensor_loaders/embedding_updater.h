// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_EMBEDDING_UPDATER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_EMBEDDING_UPDATER_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/utils/model_data.h"

namespace odml::infra::gpu {

// The primes used for ngrammer ID calculation.
constexpr std::array<int, 8> kNgrammerPrimes = {645641, 645647, 645649, 645661,
                                                645683, 645691, 645703, 645713};

using TokensTransformFn = std::function<int(int)>;

struct EmbeddingRaw {
  ml_drift::DataType data_type;
  std::unique_ptr<odml::infra::llm_utils::DataHolder<uint8_t>> weights;
  std::unique_ptr<odml::infra::llm_utils::DataHolder<uint8_t>> scales;
};

struct EmbeddingWeights {
  ml_drift::GpuModelBuilder::TensorHandle weights;
  ml_drift::GpuModelBuilder::TensorHandle scales;
  std::vector<uint8_t> weights_data;
  std::vector<float> scales_data;
};

// Updates the embedding tensors based on the passed in tokens.
// If tokens_transform_fn is provided, it will be applied to the tokens before
// they are passed to the embedding updater, for multimodality special logic.
void UpdateEmbeddings(
    const std::vector<EmbeddingRaw>& embeddings,
    std::vector<EmbeddingWeights>& weights, absl::Span<const int> tokens,
    int max_emb_size,
    TokensTransformFn tokens_transform_fn = [](int token) { return token; });

// Updates the ngrammer embedding which includes logic to convert `tokens` and
// `previous_token` to the ngrammer token IDs.
void UpdateNgrammerEmbeddings(const std::vector<EmbeddingRaw>& embeddings,
                              std::vector<EmbeddingWeights>& weights,
                              int previous_token, absl::Span<const int> tokens,
                              int vocabulary_size, int ngram_vocabulary_size);

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_EMBEDDING_UPDATER_H_
