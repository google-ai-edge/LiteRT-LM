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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_IO_TYPES_H_

#include <vector>

namespace litert::omni::tts {

// Output payload for the Qwen3-TTS frontend prompt framing stage.
struct Qwen3TtsFrontendOutput {
  // Input prompt text token IDs framed with prefix and suffix control tokens.
  std::vector<int> token_ids;
  // Precomputed or projected prefix prompt text embeddings.
  std::vector<float> prompt_embeddings;
  // Number of active token embeddings in prompt_embeddings.
  int prompt_len = 0;
  // Trailing suffix prompt text embeddings.
  std::vector<float> trailing_embeddings;
  // Number of active token embeddings in trailing_embeddings.
  int trailing_len = 0;
  // Padding embedding vector used for TTS generation token slots.
  std::vector<float> tts_pad_embedding;
};

// Output payload for the Qwen3-TTS acoustic predictor (Talker + MTP) stage.
struct Qwen3TtsAcousticOutput {
  // RVQ audio codebook token frames matrix of shape [num_frames,
  // num_codebooks].
  std::vector<std::vector<int>> rvq_frames;
  // Codec hidden state feature representations of shape [num_frames,
  // hidden_dim].
  std::vector<float> codec_features;
};

// Output payload for the Qwen3-TTS latent decoder stage.
struct Qwen3TtsLatentOutput {
  // Decoded codec latent feature representations passed to the vocoder stage.
  std::vector<float> codec_features;
  // Predicted RVQ audio codebook token frames matrix of shape [num_frames,
  // num_codebooks].
  std::vector<std::vector<int>> rvq_frames;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_IO_TYPES_H_
