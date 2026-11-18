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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_

#include <vector>

namespace litert::omni::any2any {

// Continuous audio embeddings produced by a single encoder chunk invocation.
struct EncodedAudioChunk {
  // Flattened `[valid_tokens, llm_hidden_size]` float32 embeddings.
  std::vector<float> embeddings;
  // Number of valid embedding tokens in this chunk.
  int valid_tokens = 0;
};

// Continuous vision embeddings produced by `Any2AnyImageStage` for a single
// `ImageInput`. Empty (`embeddings.empty()` and `num_tokens == 0`) for empty
// `ImageInput` passthrough items.
struct Any2AnyImageEmbeddings {
  // Flattened `[num_tokens, llm_hidden_size]` float32 vision embeddings.
  // For multi-slice vision encoders (e.g., MiniCPM-o), contains the overview
  // slice followed by `grid_x * grid_y` sub-slices in row-major order, each
  // contributing `tokens_per_slice` tokens.
  std::vector<float> embeddings;
  // Total number of vision embedding tokens in `embeddings`.
  int num_tokens = 0;
  // Number of vision embedding tokens per encoder slice (equals `num_tokens`
  // for single-pass vision encoders).
  int tokens_per_slice = 0;
  // Optional 2D high-resolution sub-slice grid dimensions (`0` when unsliced).
  int grid_x = 0;
  int grid_y = 0;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_
