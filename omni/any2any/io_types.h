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

#include <string>
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

// Interleaved multimodal prompt embeddings and token IDs produced by an
// Any2Any frontend stage for consumption by `Any2AnyLlmStage`.
struct Any2AnyFrontendOutput {
  // Flattened interleaved multimodal prompt embeddings in LLM hidden dimension
  // space with shape `[prompt_len, llm_hidden_size]`.
  std::vector<float> prompt_embeddings;
  // Total number of tokens across all interleaved modality segments in the
  // prompt.
  int prompt_len = 0;
  // Concatenated token IDs across the prompt (e.g., text, control, and
  // modality framing/placeholder tokens).
  std::vector<int> token_ids;
  // Whether downstream LLM and TTS stages should synthesize speech output for
  // this turn.
  bool generate_audio = true;
};

// Output of `Any2AnyLlmStage`: generated assistant text, token IDs, and
// optional LLM last-layer hidden states for downstream TTS conditioning.
struct Any2AnyLlmOutput {
  // Detokenized assistant text response generated for the turn (empty when
  // `detokenize_output` is false or no non-stop tokens were generated).
  std::string text;
  // Prompt token IDs (`[prompt_len]`) forwarded for downstream TTS prompt
  // prefill when `HiddenStateMode::kPostTokenWithPrompt` and audio conditioning
  // are enabled.
  std::vector<int> prompt_token_ids;
  // Flattened LLM last-layer hidden states over the prompt
  // (`[prompt_len, llm_hidden_size]`) when
  // `HiddenStateMode::kPostTokenWithPrompt` and audio conditioning are enabled.
  std::vector<float> prompt_hidden_states;
  // Sequence of generated LLM token IDs (excluding any stop token).
  std::vector<int> generated_token_ids;
  // Flattened LLM last-layer hidden states for the generated tokens with shape
  // `[num_tokens, llm_hidden_size]`.
  std::vector<float> llm_hidden_states;
  // Total number of generated tokens represented in `generated_token_ids` and
  // `llm_hidden_states`.
  int num_tokens = 0;
  // Whether downstream TTS and vocoder stages should synthesize speech audio
  // for this output.
  bool generate_audio = true;
  // Whether this output marks the final LLM chunk for the current turn.
  bool is_final = true;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_
