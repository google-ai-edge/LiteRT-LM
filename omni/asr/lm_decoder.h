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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_DECODER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_lm_runner.h"

namespace litert::omni::asr {

// Autoregressive language model decoder using LiteRtLmRunner.
// Takes audio prompt embeddings, executes prefill via
// LiteRtLmRunner::Prefill(), and generates tokens in an autoregressive
// loop via LiteRtLmRunner::Decode().
class LmDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  // Creates an LmDecoder instance.
  //
  // Args:
  //   lm_runner: Non-null pointer to LiteRtLmRunner managing model execution.
  //   decode_start_token_id: Optional initial prompt token ID fed into the
  //     decoder at step 0 (e.g. BOS or language token). Defaults to 0 if < 0.
  //   decode_stop_token_id: Optional explicit stop token ID to terminate
  //     decoding (e.g. EOS), in addition to any stop tokens configured in the
  //     model's LLM metadata.
  //   decode_skip_until_token_id: If >= 0, all decoded tokens prior to and
  //     including this token ID are skipped and omitted from the output. This
  //     is used for models (such as Qwen3-ASR) that output a task prefix or
  //     special tag (e.g. `<asr_text>`) before the actual transcript text.
  //   max_decode_steps: Maximum number of autoregressive decoding steps to
  //     execute before terminating.
  static absl::StatusOr<std::unique_ptr<LmDecoder>> Create(
      LiteRtLmRunner* absl_nonnull lm_runner, int decode_start_token_id = -1,
      int decode_stop_token_id = -1, int decode_skip_until_token_id = -1,
      int max_decode_steps = 128);

  ~LmDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs) override;

 private:
  LmDecoder(LiteRtLmRunner* absl_nonnull lm_runner, int decode_start_token_id,
            int decode_stop_token_id, int decode_skip_until_token_id,
            int max_decode_steps, absl::flat_hash_set<int> stop_tokens);

  LiteRtLmRunner* const absl_nonnull lm_runner_;
  const int decode_start_token_id_;
  const int decode_stop_token_id_;
  const int decode_skip_until_token_id_;
  const int max_decode_steps_;
  const absl::flat_hash_set<int> stop_tokens_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LM_DECODER_H_
