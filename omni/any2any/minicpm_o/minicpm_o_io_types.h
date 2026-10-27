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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IO_TYPES_H_

#include <vector>

namespace litert::omni::any2any {

// Intermediate modality embedding representation produced by modality stages
// (TextStage, AudioStage, ImageStage) for a single sequenced input item.
struct MiniCpmOEmbeddingSegment {
  // Flattened token embeddings in LLM hidden dimension space with shape
  // `[num_tokens, llm_hidden_size]`.
  std::vector<float> embeddings;
  // Corresponding token IDs for text/control tokens, or empty for non-token
  // continuous modality embeddings.
  std::vector<int> token_ids;
  // Number of embedding tokens represented in `embeddings`.
  int num_tokens = 0;
  // Whether this segment marks the end of the current user input turn
  // (`EndOfInputTurn`).
  bool is_turn_end = false;
};

// Output of Stage 1 (MiniCpmOFrontendStage): interleaved multimodal prompt
// embeddings in LLM hidden dimension space ([prompt_len, llm_hidden_size]).
struct MiniCpmOFrontendOutput {
  // Flattened interleaved multimodal prompt embeddings in LLM hidden dimension
  // space with shape `[prompt_len, llm_hidden_size]`.
  std::vector<float> prompt_embeddings;
  // Total number of tokens across all interleaved modality segments in the
  // prompt.
  int prompt_len = 0;
  // Concatenated token IDs from text and control token segments in the prompt.
  std::vector<int> token_ids;
  // Whether downstream LLM and TTS stages should synthesize speech output for
  // this turn.
  bool generate_audio = true;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_IO_TYPES_H_
