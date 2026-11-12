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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_STATEFUL_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_STATEFUL_DECODER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/speech2text/litert_speech_recognizer.h"
#include "omni/speech2text/speech_recognizer.h"

namespace litert::omni::asr {

// Stateful autoregressive speech recognizer decoder using StatefulLiteRtRunner.
// Manages recurrent KV cache state tensors across decoding steps for an
// autoregressive decode signature, updating KV cache in place via ping-pong
// double buffering.
//
// Difference from LmDecoder and why StatefulDecoder is needed:
// - LmDecoder is designed specifically for decoder-only LLM-based ASR models
//   (such as Qwen3-ASR) driven by LiteRtLmRunner, where audio
//   features are projected and interleaved into the text token embedding
//   sequence, and generation is driven by LLM Prefill() and Decode() APIs.
// - In contrast, encoder-decoder ASR models (such as Whisper and Moonshine)
//   have distinct acoustic encoders producing cross-attention hidden states
//   [1, time, d_model] and decoders requiring separate self-attention KV cache
//   and cross-attention key/value states. They cannot be executed through
//   LiteRtLmRunner, which does not support cross-attention inputs or
//   multi-input encoder-decoder signatures.
// - StatefulDecoder fills this gap by driving stateful encoder-decoder models
//   using StatefulLiteRtRunner, managing arbitrary recurrent state tensors with
//   zero-copy ping-pong double-buffering.
class StatefulDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  // Creates a StatefulDecoder instance.
  //
  // Args:
  //   runner: Non-null pointer to LiteRtRunner that implements the decode
  //     signature.
  //   signature_name: Name of the decode signature (defaults to "decode").
  //   num_non_state_inputs: Number of non-state inputs (e.g. token_id [1, 1],
  //     encoder_hidden_states [1, time, d_model]). Default is 2.
  //   num_non_state_outputs: Number of non-state outputs (e.g. logits [1, 1,
  //     vocab]). Default is 1.
  //   decode_start_token_ids: Optional prefix prompt tokens to seed decoding.
  //   decode_stop_token_id: Optional token ID that terminates decoding.
  //   decode_skip_until_token_id: If >= 0, tokens up to and including this ID
  //     are omitted from output.
  //   max_decode_steps: Maximum number of autoregressive decoding steps.
  static absl::StatusOr<std::unique_ptr<StatefulDecoder>> Create(
      LiteRtRunner* absl_nonnull runner,
      absl::string_view signature_name = "decode",
      size_t num_non_state_inputs = 2, size_t num_non_state_outputs = 1,
      std::vector<int> decode_start_token_ids = {},
      int decode_stop_token_id = -1, int decode_skip_until_token_id = -1,
      int max_decode_steps = 256);

  ~StatefulDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs,
      std::optional<DecodeRange> decode_range) override;

  using Decoder::Decode;

 protected:
  StatefulDecoder(std::unique_ptr<StatefulLiteRtRunner> stateful_runner,
                  size_t token_input_index, size_t num_logits_per_token,
                  std::vector<int> decode_start_token_ids,
                  int decode_stop_token_id, int decode_skip_until_token_id,
                  int max_decode_steps);

  // Resets stateful runner and initializes non-state step inputs with encoder
  // outputs.
  absl::Status ResetAndInitStepInputs(
      absl::Span<const ::litert::TensorBuffer> encoder_outputs);

  // Feeds a single token to the stateful model, updates recurrent state via
  // ping-pong double buffering, and returns logits [num_logits_per_token_].
  // The returned span is valid until the next call to Step().
  absl::StatusOr<absl::Span<const float>> Step(int32_t token_id);

  // Feeds prompt prefix tokens and returns the argmax token ID after the last
  // prompt token, or -1 if prompt tokens is empty.
  absl::StatusOr<int32_t> FeedPromptTokens(
      absl::Span<const int> start_token_ids);

  StatefulLiteRtRunner& stateful_runner() { return *stateful_runner_; }
  size_t token_input_index() const { return token_input_index_; }
  size_t num_logits_per_token() const { return num_logits_per_token_; }
  const std::vector<int>& decode_start_token_ids() const {
    return decode_start_token_ids_;
  }
  int decode_stop_token_id() const { return decode_stop_token_id_; }
  int decode_skip_until_token_id() const { return decode_skip_until_token_id_; }
  int max_decode_steps() const { return max_decode_steps_; }

 protected:
  struct RunnerContext {
    std::unique_ptr<StatefulLiteRtRunner> stateful_runner;
    size_t token_input_index = 0;
    size_t num_logits_per_token = 0;
  };

  static absl::StatusOr<RunnerContext> InitRunnerContext(
      LiteRtRunner* absl_nonnull runner, absl::string_view signature_name,
      size_t num_non_state_inputs, size_t num_non_state_outputs);

 private:
  std::unique_ptr<StatefulLiteRtRunner> stateful_runner_;
  const size_t token_input_index_;
  const size_t num_logits_per_token_;
  const std::vector<int> decode_start_token_ids_;
  const int decode_stop_token_id_;
  const int decode_skip_until_token_id_;
  const int max_decode_steps_;
  std::vector<::litert::TensorBuffer> step_inputs_;
  std::optional<::litert::TensorBufferScopedLock> cached_output_lock_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_STATEFUL_DECODER_H_
