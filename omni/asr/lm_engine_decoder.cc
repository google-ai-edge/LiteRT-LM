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

#include "omni/asr/lm_engine_decoder.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_lm_engine_runner.h"
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/token.pb.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<LmEngineDecoder>> LmEngineDecoder::Create(
    LiteRtLmEngineRunner* absl_nonnull engine_runner, std::string prompt,
    int max_output_tokens, int decode_start_token_id, int decode_stop_token_id,
    int decode_skip_until_token_id) {
  absl::flat_hash_set<int> stop_tokens;
  if (decode_stop_token_id >= 0) {
    stop_tokens.insert(decode_stop_token_id);
  }

  std::unique_ptr<support::Tokenizer> tokenizer;
  if (auto* res = engine_runner->mutable_model_resources(); res != nullptr) {
    auto metadata_status = res->GetLlmMetadata();
    if (metadata_status.ok() && *metadata_status != nullptr) {
      for (const auto& stop_token : (*metadata_status)->stop_tokens()) {
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

  lm::DecodeConfig decode_config = lm::DecodeConfig::CreateDefault();
  if (max_output_tokens > 0) {
    decode_config.SetMaxOutputTokens(max_output_tokens);
  }

  return std::unique_ptr<LmEngineDecoder>(new LmEngineDecoder(
      engine_runner, std::move(prompt), std::move(decode_config),
      decode_start_token_id, decode_stop_token_id, decode_skip_until_token_id,
      std::move(stop_tokens)));
}

LmEngineDecoder::LmEngineDecoder(
    LiteRtLmEngineRunner* absl_nonnull engine_runner, std::string prompt,
    lm::DecodeConfig decode_config, int decode_start_token_id,
    int decode_stop_token_id, int decode_skip_until_token_id,
    absl::flat_hash_set<int> stop_tokens)
    : engine_runner_(engine_runner),
      prompt_(std::move(prompt)),
      decode_config_(std::move(decode_config)),
      decode_start_token_id_(decode_start_token_id),
      decode_stop_token_id_(decode_stop_token_id),
      decode_skip_until_token_id_(decode_skip_until_token_id),
      stop_tokens_(std::move(stop_tokens)) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>>
LmEngineDecoder::Decode(std::vector<::litert::TensorBuffer>& encoder_outputs) {
  if (encoder_outputs.empty()) {
    return absl::InvalidArgumentError("Encoder outputs cannot be empty.");
  }

  ABSL_RETURN_IF_ERROR(engine_runner_->Reset());

  std::vector<lm::InputData> contents;
  if (!prompt_.empty()) {
    contents.emplace_back(lm::InputText(prompt_));
  }

  auto& audio_buf = encoder_outputs[0];
  LITERT_ASSIGN_OR_RETURN(auto type, audio_buf.TensorType());
  TensorBuffer final_audio_buf;
  if (type.Layout().Dimensions().size() < 2) {
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, audio_buf.PackedSize());
    if (packed_size % sizeof(float) != 0) {
      return absl::InvalidArgumentError(
          "Audio buffer packed size must be a multiple of sizeof(float).");
    }
    size_t num_floats = packed_size / sizeof(float);
    constexpr int kDefaultMelBins = 128;
    if (num_floats % kDefaultMelBins != 0) {
      return absl::InvalidArgumentError(
          "Audio buffer float count must be a multiple of mel bins (128).");
    }
    int n_frames = num_floats / kDefaultMelBins;
    LITERT_ASSIGN_OR_RETURN(
        final_audio_buf,
        support::CreateTensorBuffer<float>({1, n_frames, kDefaultMelBins}));
    LITERT_ASSIGN_OR_RETURN(auto in_lock_and_addr,
                            TensorBufferScopedLock::Create<const char>(
                                audio_buf, TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(
        auto out_lock_and_addr,
        TensorBufferScopedLock::Create<char>(final_audio_buf,
                                             TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(size_t out_packed_size,
                            final_audio_buf.PackedSize());
    if (packed_size > out_packed_size) {
      return absl::InternalError(
          "Source audio buffer exceeds allocated destination size.");
    }
    std::memcpy(out_lock_and_addr.second, in_lock_and_addr.second, packed_size);
  } else {
    LITERT_ASSIGN_OR_RETURN(final_audio_buf, audio_buf.Duplicate());
  }
  contents.emplace_back(lm::InputAudio(std::move(final_audio_buf)));
  contents.emplace_back(lm::InputAudioEnd());

  ABSL_RETURN_IF_ERROR(engine_runner_->Prefill(std::move(contents)));
  ABSL_ASSIGN_OR_RETURN(auto responses, engine_runner_->Decode(decode_config_));

  std::vector<int> token_ids;
  if (!responses.GetTokenIds().empty() &&
      !responses.GetTokenIds()[0].empty()) {
    token_ids = responses.GetTokenIds()[0];
  } else if (!responses.GetTexts().empty() &&
             !responses.GetTexts()[0].empty()) {
    if (auto* res = engine_runner_->mutable_model_resources(); res != nullptr) {
      auto tok_status = res->GetTokenizer();
      if (tok_status.ok() && *tok_status != nullptr) {
        auto tok_res = (*tok_status)->TextToTokenIds(responses.GetTexts()[0]);
        if (tok_res.ok()) {
          token_ids = std::move(*tok_res);
        }
      }
    }
  }

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  bool seen_skip_until_token_id = decode_skip_until_token_id_ < 0;
  for (int token_id : token_ids) {
    if (stop_tokens_.contains(token_id)) {
      break;
    }
    if (!seen_skip_until_token_id) {
      if (token_id == decode_skip_until_token_id_) {
        seen_skip_until_token_id = true;
      }
      continue;
    }
    decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
        .token_id = token_id,
        .timestamp_ms = std::nullopt,
    });
  }
  return decoded_tokens;
}

}  // namespace litert::omni::asr
