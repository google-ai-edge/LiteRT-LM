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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_ENGINE_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_ENGINE_DECODER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_lm_engine_runner.h"
#include "runtime/engine/io_types.h"

namespace litert::omni::asr {

// Autoregressive language model decoder powered by LiteRtLmEngineRunner.
// Takes audio feature embeddings, constructs multimodal prefill inputs
// (optional prompt + audio + audio_end), and decodes output tokens in one
// pass using the LiteRT-LM Engine.
class LmEngineDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  // Creates an LmEngineDecoder instance.
  //
  // Args:
  //   engine_runner: Non-null pointer to LiteRtLmEngineRunner.
  //   prompt: Optional text prompt prefix (e.g. transcription instruction).
  //   max_output_tokens: Maximum number of decoded tokens, or <= 0 for default.
  //   decode_start_token_id: Optional start token ID.
  //   decode_stop_token_id: Optional explicit stop token ID to terminate
  //     decoding, in addition to any stop tokens in the model metadata.
  //   decode_skip_until_token_id: If >= 0, tokens up to and including this
  //     ID are skipped from the decoded output.
  static absl::StatusOr<std::unique_ptr<LmEngineDecoder>> Create(
      LiteRtLmEngineRunner* absl_nonnull engine_runner, std::string prompt = "",
      int max_output_tokens = -1, int decode_start_token_id = -1,
      int decode_stop_token_id = -1, int decode_skip_until_token_id = -1);

  ~LmEngineDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs) override;

 private:
  LmEngineDecoder(LiteRtLmEngineRunner* absl_nonnull engine_runner,
                  std::string prompt, lm::DecodeConfig decode_config,
                  int decode_start_token_id, int decode_stop_token_id,
                  int decode_skip_until_token_id,
                  absl::flat_hash_set<int> stop_tokens);

  LiteRtLmEngineRunner* const absl_nonnull engine_runner_;
  const std::string prompt_;
  const lm::DecodeConfig decode_config_;
  const int decode_start_token_id_;
  const int decode_stop_token_id_;
  const int decode_skip_until_token_id_;
  const absl::flat_hash_set<int> stop_tokens_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_ENGINE_DECODER_H_
