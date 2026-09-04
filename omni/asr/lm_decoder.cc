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

#include "omni/asr/lm_decoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_utils.h"
#include "runtime/proto/token.pb.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<LmDecoder>> LmDecoder::Create(
    LiteRtLmRunner* absl_nonnull lm_runner, int decode_start_token_id,
    int decode_stop_token_id, int decode_skip_until_token_id,
    int max_decode_steps) {
  if (max_decode_steps <= 0) {
    return absl::InvalidArgumentError("max_decode_steps must be positive.");
  }
  absl::flat_hash_set<int> stop_tokens;
  if (decode_stop_token_id >= 0) {
    stop_tokens.insert(decode_stop_token_id);
  }
  if (auto* res = lm_runner->mutable_model_resources(); res != nullptr) {
    if (auto metadata = res->GetLlmMetadata();
        metadata.ok() && *metadata != nullptr) {
      std::unique_ptr<::litert::support::Tokenizer> tokenizer;
      for (const auto& stop_token : (*metadata)->stop_tokens()) {
        if (stop_token.has_token_ids()) {
          for (int id : stop_token.token_ids().ids()) {
            stop_tokens.insert(id);
          }
        } else if (!stop_token.token_str().empty()) {
          if (tokenizer == nullptr) {
            auto tok_status = res->GetTokenizer();
            if (tok_status.ok()) {
              tokenizer = std::move(*tok_status);
            }
          }
          if (tokenizer != nullptr) {
            if (auto id = tokenizer->TokenToId(stop_token.token_str());
                id.ok()) {
              stop_tokens.insert(*id);
            }
          }
        }
      }
    }
  }
  return std::unique_ptr<LmDecoder>(new LmDecoder(
      lm_runner, decode_start_token_id, decode_stop_token_id,
      decode_skip_until_token_id, max_decode_steps, std::move(stop_tokens)));
}

LmDecoder::LmDecoder(LiteRtLmRunner* absl_nonnull lm_runner,
                     int decode_start_token_id, int decode_stop_token_id,
                     int decode_skip_until_token_id, int max_decode_steps,
                     absl::flat_hash_set<int> stop_tokens)
    : lm_runner_(lm_runner),
      decode_start_token_id_(decode_start_token_id),
      decode_stop_token_id_(decode_stop_token_id),
      decode_skip_until_token_id_(decode_skip_until_token_id),
      max_decode_steps_(max_decode_steps),
      stop_tokens_(std::move(stop_tokens)) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> LmDecoder::Decode(
    std::vector<::litert::TensorBuffer>& encoder_outputs) {
  if (encoder_outputs.empty()) {
    return absl::InvalidArgumentError("Encoder outputs cannot be empty.");
  }

  ABSL_RETURN_IF_ERROR(lm_runner_->Reset());

  ABSL_ASSIGN_OR_RETURN(auto prefill_inputs,
                        CreateExecutorInputsWithAudio(encoder_outputs[0]));
  ABSL_RETURN_IF_ERROR(lm_runner_->Prefill(prefill_inputs));

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  int32_t current_token =
      decode_start_token_id_ >= 0 ? decode_start_token_id_ : 0;
  bool seen_skip_until_token_id = decode_skip_until_token_id_ < 0;

  LITERT_ASSIGN_OR_RETURN(auto token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));

  for (int step = 0; step < max_decode_steps_; ++step) {
    LITERT_RETURN_IF_ERROR(
        token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
    ABSL_ASSIGN_OR_RETURN(auto decode_inputs,
                          CreateExecutorInputsWithText(token_buf));
    ABSL_ASSIGN_OR_RETURN(auto logits_buf,
                          lm_runner_->Decode(decode_inputs));

    LITERT_ASSIGN_OR_RETURN(auto num_bytes, logits_buf.PackedSize());
    size_t num_logits = num_bytes / sizeof(float);
    if (num_logits == 0) {
      return absl::InternalError("Logits buffer contains 0 elements.");
    }

    std::vector<float> logits(num_logits);
    LITERT_RETURN_IF_ERROR(logits_buf.Read<float>(absl::MakeSpan(logits)));

    auto max_it = std::max_element(logits.begin(), logits.end());
    const int token_id =
        static_cast<int>(std::distance(logits.begin(), max_it));

    if (stop_tokens_.contains(token_id)) {
      break;
    }

    if (seen_skip_until_token_id) {
      decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
          .token_id = token_id, .timestamp_ms = std::nullopt});
    } else if (token_id == decode_skip_until_token_id_) {
      seen_skip_until_token_id = true;
    }

    current_token = token_id;
  }

  return decoded_tokens;
}

}  // namespace litert::omni::asr
