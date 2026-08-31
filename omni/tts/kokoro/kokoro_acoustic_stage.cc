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

#include "omni/tts/kokoro/kokoro_acoustic_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/common.h"
#include "omni/tts/kokoro/kokoro_io_types.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/kokoro/phonemizer.h"

namespace litert::omni::tts {

namespace {

// Slices long phoneme token sequences into manageable chunks fitting within
// the model's sequence length capacity, breaking at whitespace or punctuation
// boundaries.
std::vector<std::vector<int>> SliceTokenIds(const std::vector<int>& token_ids,
                                            int max_capacity) {
  if (token_ids.size() <= static_cast<size_t>(max_capacity)) {
    return {token_ids};
  }

  std::vector<std::vector<int>> slices;
  size_t start = 0;
  while (start < token_ids.size()) {
    size_t end = std::min(start + max_capacity, token_ids.size());
    if (end < token_ids.size()) {
      size_t split = end;
      // Search backwards for a natural break: whitespace (' ') or punctuation
      // (; : , . ! ?).
      while (split > start + (max_capacity / kokoro::kSplitCapacityDivisor)) {
        int id = token_ids[split - 1];
        if (id == kokoro::kSpaceTokenId ||
            (id >= kokoro::kMinPunctuationTokenId &&
             id <= kokoro::kMaxPunctuationTokenId)) {
          break;
        }
        split--;
      }
      if (split > start + (max_capacity / kokoro::kSplitCapacityDivisor)) {
        end = split;
      }
    }

    std::vector<int> slice;
    // Reserve space for at least 2 tokens (BOS and EOS).
    slice.reserve(std::max(2, static_cast<int>(end - start)));
    // Each slice begins with a BOS token.
    slice.push_back(kokoro::kBosTokenId);
    if (token_ids[start] == kokoro::kBosTokenId) {
      ++start;
    }
    slice.insert(slice.end(), token_ids.begin() + start,
                 token_ids.begin() + end);
    if (slice.back() != kokoro::kEosTokenId) {
      slice.push_back(kokoro::kEosTokenId);
    }
    slices.push_back(std::move(slice));
    start = end;
  }
  return slices;
}

}  // namespace

absl::StatusOr<std::unique_ptr<KokoroAcousticStage>>
KokoroAcousticStage::Create(
    Stage<std::string>* absl_nonnull text_source,
    const KokoroModelConfig& config, absl::string_view model_folder,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = std::unique_ptr<KokoroAcousticStage>(new KokoroAcousticStage(
      text_source, config, model_folder, std::move(resources)));

  // Load the voice pack embedding table (510 token positions x 256 embedding
  // dimensions).
  std::string voice_req = stage->config_.voice_name.empty()
                              ? stage->config_.voice_file
                              : stage->config_.voice_name;
  LITERT_ASSIGN_OR_RETURN(
      stage->voice_pack_,
      kokoro::LoadVoiceEmbedding(stage->model_folder_, voice_req));

  // Retrieve the compiled unified acoustic predictor model.
  LITERT_ASSIGN_OR_RETURN(
      stage->acoustic_model_,
      stage->resources_->GetCompiledModel("kokoro_acoustic"));

  // Allocate reusable input and output tensor buffers for the acoustic model.
  LITERT_ASSIGN_OR_RETURN(stage->acoustic_input_buffers_,
                          stage->acoustic_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->acoustic_output_buffers_,
                          stage->acoustic_model_->CreateOutputBuffers());

  if (stage->acoustic_input_buffers_.size() < 3) {
    return absl::InternalError(absl::StrFormat(
        "kokoro_acoustic expected at least 3 input buffers, but got %d",
        stage->acoustic_input_buffers_.size()));
  }
  if (stage->acoustic_output_buffers_.size() < 4) {
    return absl::InternalError(absl::StrFormat(
        "kokoro_acoustic expected at least 4 output buffers, but got %d",
        stage->acoustic_output_buffers_.size()));
  }

  // Resolve input tensor indices by signature name or buffer sizes.
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.phoneme_ids,
      ResolveInputIndex(*stage->acoustic_model_, "phoneme_ids"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.speaker_style,
      ResolveInputIndex(*stage->acoustic_model_, "speaker_style"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.phoneme_length,
      ResolveInputIndex(*stage->acoustic_model_, "phoneme_length"));

  // Resolve output tensor indices by signature name or buffer sizes.
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.acoustic_features,
      ResolveOutputIndex(*stage->acoustic_model_, "acoustic_features"));
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.pitch_contour,
      ResolveOutputIndex(*stage->acoustic_model_, "pitch_contour"));
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.energy_contour,
      ResolveOutputIndex(*stage->acoustic_model_, "energy_contour"));
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.speech_frame_length,
      ResolveOutputIndex(*stage->acoustic_model_, "speech_frame_length"));

  // Initialize the phonemizer once during acoustic stage creation.
  std::string espeak_dir = ResolveEspeakDataDir(stage->model_folder_);
  LITERT_ASSIGN_OR_RETURN(
      stage->phonemizer_,
      KokoroPhonemizer::Create(espeak_dir, stage->config_.language,
                               stage->config_.custom_lexicon));

  return stage;
}

absl::Status KokoroAcousticStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2) << "[TRACE] Starting KokoroAcousticStage::ScheduleInternal";

  auto text = text_source_.GetOutput();
  if (absl::IsNotFound(text.status())) {
    return absl::OkStatus();
  } else if (!text.ok()) {
    return text.status();
  }
  std::string input_text = std::move(*text);

  // Step 1: Phonemize raw input text using the persistent KokoroPhonemizer into
  // token IDs.
  LITERT_ASSIGN_OR_RETURN(std::vector<int> full_token_ids,
                          phonemizer_->TextToPhonemeIds(input_text));

  // Step 2: Determine bucketing capacity and slice token IDs into sentence
  // chunks.
  LITERT_ASSIGN_OR_RETURN(
      auto ids_buf_bytes,
      acoustic_input_buffers_[input_indices_.phoneme_ids].PackedSize());
  const int model_capacity = static_cast<int>(ids_buf_bytes / sizeof(int64_t));
  const int bucket_size = config_.target_bucket > 0
                              ? std::min(config_.target_bucket, model_capacity)
                              : model_capacity;
  // Reserve 4 tokens because each acoustic slice input:
  // 1) Must start with BOS token and end with EOS token.
  // 2) Reserve headroom for potential punctuation and delimiter spaces, at
  //    beginning and end of slice.
  const int max_capacity = std::max(1, bucket_size - 4);
  std::vector<std::vector<int>> slices =
      SliceTokenIds(full_token_ids, max_capacity);

  // Step 3: Process each phoneme chunk through unified acoustic prediction.
  for (const auto& token_ids : slices) {
    int num_tokens = static_cast<int>(token_ids.size());
    if (num_tokens == 0) num_tokens = 1;

    const int seq_len = std::min(bucket_size, num_tokens);

    // Prepare padded phoneme token IDs buffer of shape [1, bucket_size].
    std::vector<int64_t> ids_i64(bucket_size, 0);
    for (int i = 0; i < seq_len && i < bucket_size &&
                    i < static_cast<int>(token_ids.size());
         ++i) {
      ids_i64[i] = static_cast<int64_t>(token_ids[i]);
    }

    // Extract voice style embedding corresponding to token sequence length.
    // Index is bounded to 0..kMaxVoiceTokens in the 510x256 voice pack
    // embedding table.
    const int ref_idx = std::clamp(num_tokens - 1, 0, kokoro::kMaxVoiceTokens);
    const float* ref_s_ptr = &voice_pack_[ref_idx * kokoro::kVoiceEmbeddingDim];
    // First 128 floats are forwarded to decoder/vocoder, second 128 floats are
    // passed to prosody predictor.
    const float* ref_s_decoder_ptr = ref_s_ptr;
    const float* ref_s_prosody_ptr = ref_s_ptr + kokoro::kStyleSliceDim;

    const int active_len = std::min(seq_len, bucket_size);
    std::vector<int64_t> seq_len_vec = {static_cast<int64_t>(active_len)};

    // Step 4: Write input buffers for acoustic model inference.
    LITERT_RETURN_IF_ERROR(
        acoustic_input_buffers_[input_indices_.phoneme_ids].Write<int64_t>(
            absl::MakeConstSpan(ids_i64)));
    LITERT_RETURN_IF_ERROR(
        acoustic_input_buffers_[input_indices_.speaker_style].Write<float>(
            absl::MakeConstSpan(ref_s_prosody_ptr, kokoro::kStyleSliceDim)));
    LITERT_RETURN_IF_ERROR(
        acoustic_input_buffers_[input_indices_.phoneme_length].Write<int64_t>(
            absl::MakeConstSpan(seq_len_vec)));

    // Step 5: Execute unified acoustic model inference.
    LITERT_RETURN_IF_ERROR(acoustic_model_->Run(acoustic_input_buffers_,
                                                acoustic_output_buffers_));

    // Step 6: Read speech frame length and acoustic output tensors.
    std::vector<int64_t> speech_len_vec(1, 0);
    LITERT_RETURN_IF_ERROR(
        acoustic_output_buffers_[output_indices_.speech_frame_length]
            .Read<int64_t>(absl::MakeSpan(speech_len_vec)));
    const int l_speech =
        std::min<int>(kokoro::kMaxSpeechFrames,
                      std::max<int>(1, static_cast<int>(speech_len_vec[0])));

    LITERT_ASSIGN_OR_RETURN(
        auto asr_packed_size,
        acoustic_output_buffers_[output_indices_.acoustic_features]
            .PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto f0_packed_size,
        acoustic_output_buffers_[output_indices_.pitch_contour].PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto n_packed_size,
        acoustic_output_buffers_[output_indices_.energy_contour].PackedSize());

    std::vector<float> asr_data(asr_packed_size / sizeof(float), 0.0f);
    std::vector<float> f0_n_data(f0_packed_size / sizeof(float), 0.0f);
    std::vector<float> n_aux_data(n_packed_size / sizeof(float), 0.0f);

    LITERT_RETURN_IF_ERROR(
        acoustic_output_buffers_[output_indices_.acoustic_features].Read<float>(
            absl::MakeSpan(asr_data)));
    LITERT_RETURN_IF_ERROR(
        acoustic_output_buffers_[output_indices_.pitch_contour].Read<float>(
            absl::MakeSpan(f0_n_data)));
    LITERT_RETURN_IF_ERROR(
        acoustic_output_buffers_[output_indices_.energy_contour].Read<float>(
            absl::MakeSpan(n_aux_data)));

    // Step 7: Construct KokoroAcousticOutput payload and push downstream.
    KokoroAcousticOutput output;
    output.asr_data = std::move(asr_data);
    output.f0_n_data = std::move(f0_n_data);
    output.n_aux_data = std::move(n_aux_data);
    output.ref_s_decoder.assign(ref_s_decoder_ptr,
                                ref_s_decoder_ptr + kokoro::kStyleSliceDim);
    output.l_speech = l_speech;

    PushOutput(std::move(output));
  }

  return absl::OkStatus();
}

}  // namespace litert::omni::tts
