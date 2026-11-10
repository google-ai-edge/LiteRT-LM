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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/utils/model_data.h"

namespace odml::infra::gpu {
namespace {

void UpdateEmbeddings(const std::vector<EmbeddingRaw>& embeddings,
                      std::vector<EmbeddingWeights>& weights,
                      const std::vector<absl::Span<const int>>& tokens,
                      int max_emb_size) {
  for (int i = 0; i < weights.size(); ++i) {
    int seq_size = weights[i].scales.tensor_desc.GetBHWCShape().c;
    size_t row_size =
        weights[i].weights.tensor_desc.GetBHWCShape().c / seq_size;

    for (int j = 0; j < tokens[i].size(); ++j) {
      int token = tokens[i][j];
      // vision or audio tokens are negative, and the embedding of the two
      // modality are inputs from modality adapter.
      // We also might be passed audio hard tokens which are larger than
      // the vocab/ngram size.
      if (token < 0 || token >= max_emb_size) {
        continue;
      }
      weights[i].scales_data[j] =
          *(reinterpret_cast<float*>(embeddings[i].scales->GetData().data()) +
            token);

      absl::Span<uint8_t> quantized_weights =
          embeddings[i].weights->GetData().subspan(token * row_size, row_size);
      std::memcpy(weights[i].weights_data.data() + (j * row_size),
                  quantized_weights.data(), row_size);
    }
  }
}

}  // namespace

void UpdateEmbeddings(const std::vector<EmbeddingRaw>& embeddings,
                      std::vector<EmbeddingWeights>& weights,
                      absl::Span<const int> tokens, int max_emb_size,
                      TokensTransformFn tokens_transform_fn) {
  std::vector<int> tokens_copy(tokens.size());
  absl::c_transform(
      tokens, tokens_copy.begin(),
      [&tokens_transform_fn](int token) { return tokens_transform_fn(token); });
  std::vector<absl::Span<const int>> all_tokens(weights.size(), tokens_copy);
  UpdateEmbeddings(embeddings, weights, all_tokens, max_emb_size);
}

void UpdateNgrammerEmbeddings(const std::vector<EmbeddingRaw>& embeddings,
                              std::vector<EmbeddingWeights>& weights,
                              int previous_token, absl::Span<const int> tokens,
                              int vocabulary_size, int ngram_vocabulary_size) {
  // First collect bigram IDs.
  std::vector<int32_t> bigram_ids(tokens.size());
  for (int i = 0; i < tokens.size(); ++i) {
    int32_t last_token = i == 0 ? previous_token : tokens[i - 1];
    // Vision or audio tokens are negative or >= vocabulary_size, and should not
    // use Ngrammer embedding.
    if (tokens[i] < 0 || tokens[i] >= vocabulary_size) {
      continue;
    }
    last_token =
        last_token < 0 || last_token >= vocabulary_size ? 0 : last_token;
    bigram_ids[i] = last_token * vocabulary_size + tokens[i];
  }

  // Then generate per-dimension ngram IDs.
  std::vector<std::vector<int>> per_dim_ids(weights.size());
  std::vector<absl::Span<const int>> all_tokens(per_dim_ids.size());
  for (int i = 0; i < per_dim_ids.size(); ++i) {
    int prime = kNgrammerPrimes[i];
    for (int32_t bigram_id : bigram_ids) {
      int32_t ngram_id = (bigram_id * (i + 1) + i + 1) % prime;
      if (ngram_id < 0) {
        ngram_id += prime;
      }
      per_dim_ids[i].push_back(ngram_id % ngram_vocabulary_size);
    }
    all_tokens[i] = absl::MakeSpan(per_dim_ids[i]);
  }
  UpdateEmbeddings(embeddings, weights, all_tokens,
                   std::max(ngram_vocabulary_size, vocabulary_size));
}

}  // namespace odml::infra::gpu
