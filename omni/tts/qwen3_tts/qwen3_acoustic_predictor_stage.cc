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

#include "omni/tts/qwen3_tts/qwen3_acoustic_predictor_stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/qwen3_tts/common.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/text_frontend.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::tts {

namespace {

constexpr absl::string_view kPrefillSignaturePrefix = "prefill";
constexpr absl::string_view kDecodeSignature = "decode";
constexpr absl::string_view kEmbedInputName = "embeddings";
constexpr absl::string_view kPosInputName = "input_pos";
constexpr absl::string_view kMaskInputName = "mask";
constexpr absl::string_view kLogitsOutputName = "logits";
constexpr int kCacheLen = 512;

absl::Status InitializeBuffer(TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(
      auto buffer_lock_and_addr,
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(auto packed_size, buffer.PackedSize());
  std::memset(buffer_lock_and_addr.second, 0, packed_size);
  return absl::OkStatus();
}

absl::Status InitializeBuffers(
    absl::flat_hash_map<std::string, TensorBuffer>& buffers) {
  for (auto& [name, buffer] : buffers) {
    LITERT_RETURN_IF_ERROR(InitializeBuffer(buffer));
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<Qwen3AcousticPredictorStage>>
Qwen3AcousticPredictorStage::Create(
    Stage<FrontendOutput>* absl_nonnull text_frontend,
    Qwen3StageOptions options,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = absl::WrapUnique(new Qwen3AcousticPredictorStage(
      text_frontend, std::move(options), std::move(resources)));

  LITERT_ASSIGN_OR_RETURN(stage->talker_model_,
                          stage->resources_->GetCompiledModel("talker"));
  LITERT_ASSIGN_OR_RETURN(stage->mtp_model_,
                          stage->resources_->GetCompiledModel("mtp"));
  LITERT_ASSIGN_OR_RETURN(
      stage->codec_embedding_model_,
      stage->resources_->GetCompiledModel("codec_embedding"));
  LITERT_ASSIGN_OR_RETURN(stage->mtp_embedding_model_,
                          stage->resources_->GetCompiledModel("mtp_embedding"));

  // Pre-allocate embedding model TensorBuffers
  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_input_buffers_,
                          stage->codec_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_output_buffers_,
                          stage->codec_embedding_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(stage->mtp_emb_input_buffers_,
                          stage->mtp_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->mtp_emb_output_buffers_,
                          stage->mtp_embedding_model_->CreateOutputBuffers());

  // Set up MTP signature and input/output names
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          stage->mtp_model_->GetSignature(0));
  stage->mtp_signature_name_ = std::string(decode_signature.Key());

  ABSL_ASSIGN_OR_RETURN(
      stage->mtp_signatures_,
      lm::GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                                 decode_signature.OutputNames(),
                                                 /*strict=*/false));

  std::string mtp_k_root_name;
  std::string mtp_v_root_name;
  ABSL_RETURN_IF_ERROR(lm::GetKVCacheRootNames(
      decode_signature.InputNames(), decode_signature.OutputNames(),
      mtp_k_root_name, mtp_v_root_name));

  stage->mtp_k_input_names_.clear();
  stage->mtp_v_input_names_.clear();
  stage->mtp_k_output_names_.clear();
  stage->mtp_v_output_names_.clear();

  for (absl::string_view name : decode_signature.InputNames()) {
    if (absl::StartsWith(name, mtp_k_root_name)) {
      stage->mtp_k_input_names_.push_back(std::string(name));
    } else if (absl::StartsWith(name, mtp_v_root_name)) {
      stage->mtp_v_input_names_.push_back(std::string(name));
    }
  }
  for (absl::string_view name : decode_signature.OutputNames()) {
    if (absl::StartsWith(name, mtp_k_root_name)) {
      stage->mtp_k_output_names_.push_back(std::string(name));
    } else if (absl::StartsWith(name, mtp_v_root_name)) {
      stage->mtp_v_output_names_.push_back(std::string(name));
    }
  }

  if (stage->mtp_signatures_.input_embeddings.has_value() &&
      !stage->mtp_signatures_.input_embeddings->empty()) {
    stage->mtp_embed_input_name_ = *stage->mtp_signatures_.input_embeddings;
  } else if (!stage->mtp_signatures_.input_tokens.empty()) {
    stage->mtp_embed_input_name_ = stage->mtp_signatures_.input_tokens;
  } else if (!decode_signature.InputNames().empty()) {
    stage->mtp_embed_input_name_ =
        std::string(decode_signature.InputNames()[0]);
  }

  if (!stage->mtp_signatures_.input_positions.empty()) {
    stage->mtp_pos_input_name_ = stage->mtp_signatures_.input_positions;
  } else if (decode_signature.InputNames().size() > 1) {
    stage->mtp_pos_input_name_ = std::string(decode_signature.InputNames()[1]);
  }

  if (stage->mtp_signatures_.input_attn_mask.has_value() &&
      !stage->mtp_signatures_.input_attn_mask->empty()) {
    stage->mtp_mask_input_name_ = *stage->mtp_signatures_.input_attn_mask;
  } else if (decode_signature.InputNames().size() > 2) {
    stage->mtp_mask_input_name_ = std::string(decode_signature.InputNames()[2]);
  }

  if (!stage->mtp_signatures_.output_logits.empty()) {
    stage->mtp_logits_output_name_ = stage->mtp_signatures_.output_logits;
  } else if (!decode_signature.OutputNames().empty()) {
    stage->mtp_logits_output_name_ =
        std::string(decode_signature.OutputNames()[0]);
  }

  // Pre-allocate MTP TensorBuffers
  LITERT_ASSIGN_OR_RETURN(
      stage->mtp_embed_buf_,
      stage->mtp_model_->CreateInputBuffer(stage->mtp_signature_name_,
                                           stage->mtp_embed_input_name_));
  LITERT_ASSIGN_OR_RETURN(
      stage->mtp_pos_buf_,
      stage->mtp_model_->CreateInputBuffer(stage->mtp_signature_name_,
                                           stage->mtp_pos_input_name_));
  LITERT_ASSIGN_OR_RETURN(
      stage->mtp_mask_buf_,
      stage->mtp_model_->CreateInputBuffer(stage->mtp_signature_name_,
                                           stage->mtp_mask_input_name_));
  LITERT_ASSIGN_OR_RETURN(
      stage->mtp_logits_out_buf_,
      stage->mtp_model_->CreateOutputBuffer(stage->mtp_signature_name_,
                                            stage->mtp_logits_output_name_));

  // Derive MTP cache length from the exported model's mask tensor.
  LITERT_ASSIGN_OR_RETURN(
      auto mtp_mask_type,
      stage->mtp_mask_buf_.TensorType());
  
  auto mtp_mask_dims = mtp_mask_type.Layout().Dimensions();
  
  if (mtp_mask_dims.empty()) {
    return absl::InvalidArgumentError(
        "Invalid MTP attention mask dimensions.");
  }
  
  // Expected: [1, 1, 1, cache_len]
  if (mtp_mask_dims.size() < 1) {
    return absl::InvalidArgumentError(
        "MTP attention mask has no dimensions.");
  }
  
  stage->mtp_cache_len_ = mtp_mask_dims.back();

  for (const auto& k_name : stage->mtp_k_input_names_) {
    LITERT_ASSIGN_OR_RETURN(auto buf0, stage->mtp_model_->CreateInputBuffer(
                                           stage->mtp_signature_name_, k_name));
    LITERT_ASSIGN_OR_RETURN(auto buf1, stage->mtp_model_->CreateInputBuffer(
                                           stage->mtp_signature_name_, k_name));
    stage->mtp_kv_cache_bufs_0_[k_name] = std::move(buf0);
    stage->mtp_kv_cache_bufs_1_[k_name] = std::move(buf1);
  }
  for (const auto& v_name : stage->mtp_v_input_names_) {
    LITERT_ASSIGN_OR_RETURN(auto buf0, stage->mtp_model_->CreateInputBuffer(
                                           stage->mtp_signature_name_, v_name));
    LITERT_ASSIGN_OR_RETURN(auto buf1, stage->mtp_model_->CreateInputBuffer(
                                           stage->mtp_signature_name_, v_name));
    stage->mtp_kv_cache_bufs_0_[v_name] = std::move(buf0);
    stage->mtp_kv_cache_bufs_1_[v_name] = std::move(buf1);
  }

  // Set up Talker model signature and pre-allocate TensorBuffers
  LITERT_ASSIGN_OR_RETURN(
      auto decode_inputs,
      stage->talker_model_->GetSignatureInputNames(kDecodeSignature));
  for (auto input_name : decode_inputs) {
    if (absl::StartsWith(input_name, "kv_cache")) {
      stage->talker_kv_names_.emplace_back(input_name);
    }
  }

  // Find all prefill signatures starting with "prefill" and pre-allocate
  // buffers
  LITERT_ASSIGN_OR_RETURN(auto signature_keys,
                          stage->talker_model_->GetSignatureKeys());
  for (const auto& key : signature_keys) {
    if (absl::StartsWith(key, kPrefillSignaturePrefix)) {
      std::string sig_name = std::string(key);
      LITERT_ASSIGN_OR_RETURN(
          auto pos_tensor_type,
          stage->talker_model_->GetInputTensorType(sig_name, kPosInputName));
      auto pos_dims = pos_tensor_type.Layout().Dimensions();
      if (pos_dims.empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid ", kPosInputName,
                         " dimensions in prefill signature ", sig_name));
      }
      int seq_len = pos_dims.back();

      TalkerPrefillBuffers bufs;
      bufs.seq_len = seq_len;
      bufs.signature_name = sig_name;

      LITERT_ASSIGN_OR_RETURN(
          bufs.emb_buf,
          stage->talker_model_->CreateInputBuffer(sig_name, kEmbedInputName));
      LITERT_ASSIGN_OR_RETURN(
          bufs.pos_buf,
          stage->talker_model_->CreateInputBuffer(sig_name, kPosInputName));
      LITERT_ASSIGN_OR_RETURN(
          bufs.mask_buf,
          stage->talker_model_->CreateInputBuffer(sig_name, kMaskInputName));
      for (const auto& kv_name : stage->talker_kv_names_) {
        LITERT_ASSIGN_OR_RETURN(
            auto out_kv,
            stage->talker_model_->CreateOutputBuffer(sig_name, kv_name));
        bufs.kv_output_bufs[kv_name] = std::move(out_kv);
      }

      stage->talker_prefill_buffers_[seq_len] = std::move(bufs);
    }
  }
  if (stage->talker_prefill_buffers_.empty()) {
    return absl::NotFoundError(absl::StrCat("No signature starting with '",
                                            kPrefillSignaturePrefix,
                                            "' found in talker_model."));
  }

  // Pre-allocate Talker decode buffers
  LITERT_ASSIGN_OR_RETURN(stage->talker_decode_emb_buf_,
                          stage->talker_model_->CreateInputBuffer(
                              kDecodeSignature, kEmbedInputName));
  LITERT_ASSIGN_OR_RETURN(
      stage->talker_decode_pos_buf_,
      stage->talker_model_->CreateInputBuffer(kDecodeSignature, kPosInputName));
  LITERT_ASSIGN_OR_RETURN(stage->talker_decode_mask_buf_,
                          stage->talker_model_->CreateInputBuffer(
                              kDecodeSignature, kMaskInputName));
  LITERT_ASSIGN_OR_RETURN(stage->talker_decode_logits_buf_,
                          stage->talker_model_->CreateOutputBuffer(
                              kDecodeSignature, kLogitsOutputName));
  for (const auto& kv_name : stage->talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(auto in_kv, stage->talker_model_->CreateInputBuffer(
                                            kDecodeSignature, kv_name));
    stage->talker_kv_cache_bufs_0_[kv_name] = std::move(in_kv);
    LITERT_ASSIGN_OR_RETURN(
        auto out_kv,
        stage->talker_model_->CreateOutputBuffer(kDecodeSignature, kv_name));
    stage->talker_kv_cache_bufs_1_[kv_name] = std::move(out_kv);
  }

  stage->talker_cache_len_ = kCacheLen;
  LITERT_ASSIGN_OR_RETURN(auto type,
                          stage->talker_decode_mask_buf_.TensorType());
  auto dims = type.Layout().Dimensions();
  if (dims.size() > 3) {
    stage->talker_cache_len_ = dims[3];
  }

  for (auto& [seq_len, prefill_bufs] : stage->talker_prefill_buffers_) {
    std::vector<int32_t> pos_buf(seq_len, 0);
    for (int i = 0; i < seq_len; ++i) {
      pos_buf[i] = i;
    }
    LITERT_RETURN_IF_ERROR(
        prefill_bufs.pos_buf.Write<int32_t>(absl::MakeConstSpan(pos_buf)));

    std::vector<float> mask_buf(seq_len * stage->talker_cache_len_,
                                qwen3_tts::kNegInf);
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j <= i && j < stage->talker_cache_len_; ++j) {
        mask_buf[i * stage->talker_cache_len_ + j] = 0.0f;
      }
    }
    LITERT_RETURN_IF_ERROR(
        prefill_bufs.mask_buf.Write<float>(absl::MakeConstSpan(mask_buf)));
  }

  return stage;
}

// TODO b/538727793: use cpu/gpu sampler from litert-lm.
int Qwen3AcousticPredictorStage::PickToken(const std::vector<float>& logits,
                                           bool do_sample) {
  if (!do_sample) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
  }

  int n = logits.size();
  std::vector<double> scaled(n);
  double temp = std::max(static_cast<double>(options_.temperature), 1e-6);
  double max_logit = -1e30;

  for (int i = 0; i < n; ++i) {
    scaled[i] = static_cast<double>(logits[i]) / temp;
    if (scaled[i] > max_logit) max_logit = scaled[i];
  }

  int top_k = std::min(options_.top_k, n);
  std::vector<std::pair<double, int>> pairs(n);
  for (int i = 0; i < n; ++i) {
    pairs[i] = {scaled[i], i};
  }
  absl::c_partial_sort(
      pairs, pairs.begin() + top_k,
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<double> probs(top_k);
  double sum_p = 0.0;
  for (int i = 0; i < top_k; ++i) {
    probs[i] = std::exp(pairs[i].first - max_logit);
    sum_p += probs[i];
  }
  for (int i = 0; i < top_k; ++i) {
    probs[i] /= sum_p;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return pairs[dist(rng_)].second;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(qwen3_tts::kHiddenDim, 0.0f);
  LITERT_RETURN_IF_ERROR(codec_emb_input_buffers_[0].Write<int32_t>(
      absl::MakeConstSpan(&code_id, 1)));
  LITERT_RETURN_IF_ERROR(codec_embedding_model_->Run(
      codec_emb_input_buffers_, codec_emb_output_buffers_));
  LITERT_RETURN_IF_ERROR(
      codec_emb_output_buffers_[0].Read<float>(absl::MakeSpan(out)));
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedMtpTokens(
    const std::vector<int>& mtp_codes) {
  std::vector<float> sum_embed(qwen3_tts::kHiddenDim, 0.0f);
  int num_codes = mtp_codes.size();

  for (int g = 0; g < num_codes; ++g) {
    int32_t global_id = g * 2048 + mtp_codes[g];
    LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&global_id, 1)));

    LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(mtp_emb_input_buffers_,
                                                     mtp_emb_output_buffers_));
    LITERT_ASSIGN_OR_RETURN(auto copy_res, support::CopyFromTensorBuffer<float>(
                                               mtp_emb_output_buffers_[0]));
    absl::c_transform(copy_res, sum_embed, sum_embed.begin(),
                      [](float a, float b) { return a + b; });
  }
  return sum_embed;
}

absl::Status Qwen3AcousticPredictorStage::RunPrefill(
    const std::vector<float>& prefill, int p) {
  auto it = talker_prefill_buffers_.lower_bound(p);
  if (it == talker_prefill_buffers_.end()) {
    int max_seq_len = talker_prefill_buffers_.rbegin()->first;
    return absl::InvalidArgumentError(
        absl::StrCat("Prompt length ", p,
                     " exceeds maximum prefill sequence length ", max_seq_len));
  }

  auto& prefill_bufs = it->second;
  const std::string& sig_name = prefill_bufs.signature_name;

  // TODO b/538727793: refactor to use multi chunk prefill, so we don't need to
  // clear out the buffers for each prefill run.
  {
    LITERT_ASSIGN_OR_RETURN(
        auto lock, TensorBufferScopedLock::Create<float>(
                       prefill_bufs.emb_buf, TensorBuffer::LockMode::kWrite));
    std::memcpy(lock.second, prefill.data(),
                p * qwen3_tts::kHiddenDim * sizeof(float));
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  LITERT_ASSIGN_OR_RETURN(auto emb_dup, prefill_bufs.emb_buf.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto pos_dup, prefill_bufs.pos_buf.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto mask_dup, prefill_bufs.mask_buf.Duplicate());
  input_map.insert_or_assign(kEmbedInputName, std::move(emb_dup));
  input_map.insert_or_assign(kPosInputName, std::move(pos_dup));
  input_map.insert_or_assign(kMaskInputName, std::move(mask_dup));

  for (const auto& kv_name : talker_kv_names_) {
    if (talker_kv_cache_bufs_0_.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              talker_kv_cache_bufs_0_[kv_name].Duplicate());
      input_map.insert_or_assign(kv_name, std::move(dup));
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(auto dup,
                            prefill_bufs.kv_output_bufs[kv_name].Duplicate());
    output_map.insert_or_assign(kv_name, std::move(dup));
  }

  LITERT_RETURN_IF_ERROR(talker_model_->Run(sig_name, input_map, output_map));

  for (const auto& kv_name : talker_kv_names_) {
    if (output_map.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         output_map[kv_name], TensorBuffer::LockMode::kRead));
      LITERT_ASSIGN_OR_RETURN(size_t bytes, output_map[kv_name].Size());
      size_t num_floats = bytes / sizeof(float);

      LITERT_ASSIGN_OR_RETURN(size_t dec_bytes,
                              talker_kv_cache_bufs_0_[kv_name].PackedSize());
      size_t total_dec_floats = dec_bytes / sizeof(float);
      std::vector<float> dec_buffer(total_dec_floats, 0.0f);
      std::memcpy(dec_buffer.data(), lock.second,
                  std::min(num_floats, total_dec_floats) * sizeof(float));
      LITERT_RETURN_IF_ERROR(talker_kv_cache_bufs_0_[kv_name].Write<float>(
          absl::MakeConstSpan(dec_buffer)));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Qwen3AcousticPredictorStage::DecodeOutput>
Qwen3AcousticPredictorStage::RunDecode(const float* embed_1024, int pos) {
  LITERT_RETURN_IF_ERROR(talker_decode_emb_buf_.Write<float>(
      absl::MakeConstSpan(embed_1024, qwen3_tts::kHiddenDim)));

  int32_t pos_val = pos;
  LITERT_RETURN_IF_ERROR(
      talker_decode_pos_buf_.Write<int32_t>(absl::MakeConstSpan(&pos_val, 1)));

  std::vector<float> mask(talker_cache_len_, qwen3_tts::kNegInf);
  for (int j = 0; j <= pos && j < talker_cache_len_; ++j) {
    mask[j] = 0.0f;
  }
  LITERT_RETURN_IF_ERROR(
      talker_decode_mask_buf_.Write<float>(absl::MakeConstSpan(mask)));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  LITERT_ASSIGN_OR_RETURN(auto emb_dup, talker_decode_emb_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto pos_dup, talker_decode_pos_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto mask_dup, talker_decode_mask_buf_.Duplicate());
  input_map.insert_or_assign(kEmbedInputName, std::move(emb_dup));
  input_map.insert_or_assign(kPosInputName, std::move(pos_dup));
  input_map.insert_or_assign(kMaskInputName, std::move(mask_dup));

  for (const auto& kv_name : talker_kv_names_) {
    if (talker_kv_cache_bufs_0_.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              talker_kv_cache_bufs_0_[kv_name].Duplicate());
      input_map.insert_or_assign(kv_name, std::move(dup));
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  LITERT_ASSIGN_OR_RETURN(auto logits_dup,
                          talker_decode_logits_buf_.Duplicate());
  output_map.insert_or_assign(kLogitsOutputName, std::move(logits_dup));
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(auto dup,
                            talker_kv_cache_bufs_1_[kv_name].Duplicate());
    output_map.insert_or_assign(kv_name, std::move(dup));
  }

  LITERT_RETURN_IF_ERROR(
      talker_model_->Run(kDecodeSignature, input_map, output_map));

  // Swap talker_kv_cache_bufs_0_ and talker_kv_cache_bufs_1_ for next decode
  // run
  std::swap(talker_kv_cache_bufs_0_, talker_kv_cache_bufs_1_);

  DecodeOutput out;
  out.cb0_logits.resize(qwen3_tts::kCodecVocab);
  out.hidden.resize(qwen3_tts::kHiddenDim);

  {
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        TensorBufferScopedLock::Create<const float>(
            talker_decode_logits_buf_, TensorBuffer::LockMode::kRead));
    const float* raw_logits = lock.second;
    std::memcpy(out.cb0_logits.data(), raw_logits,
                qwen3_tts::kCodecVocab * sizeof(float));
    std::memcpy(out.hidden.data(), raw_logits + qwen3_tts::kCodecVocab,
                qwen3_tts::kHiddenDim * sizeof(float));
  }

  return out;
}

absl::StatusOr<std::vector<int>> Qwen3AcousticPredictorStage::RunMtp(
    const std::vector<float>& hidden, int cb0) {
  LITERT_RETURN_IF_ERROR(InitializeBuffers(mtp_kv_cache_bufs_0_));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(mtp_kv_cache_bufs_1_));

  std::vector<int> codes;
  codes.reserve(qwen3_tts::kNumCodeGroups - 1);

  for (int t = 0; t < qwen3_tts::kNumCodeGroups - 1; ++t) {
    std::vector<float> embed(qwen3_tts::kHiddenDim, 0.0f);
    if (t == 0) {
      embed = hidden;
    } else if (t == 1) {
      ABSL_ASSIGN_OR_RETURN(embed, EmbedCodecToken(cb0));
    } else {
      int head_idx = t - 2;
      int prev_code = codes[head_idx];
      int32_t global_id = head_idx * 2048 + prev_code;
      LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
          absl::MakeConstSpan(&global_id, 1)));
      LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(
          mtp_emb_input_buffers_, mtp_emb_output_buffers_));
      LITERT_RETURN_IF_ERROR(mtp_emb_output_buffers_[0].Read<float>(
          absl::MakeSpan(embed.data(), qwen3_tts::kHiddenDim)));
    }

    LITERT_RETURN_IF_ERROR(
        mtp_embed_buf_.Write<float>(absl::MakeConstSpan(embed)));
    int32_t t_val = t;
    LITERT_RETURN_IF_ERROR(
        mtp_pos_buf_.Write<int32_t>(absl::MakeConstSpan(&t_val, 1)));
    std::vector<float> mask(mtp_cache_len_, qwen3_tts::kNegInf);
    for (int j = 0; j <= t && j < mtp_cache_len_; ++j) mask[j] = 0.0f;
    LITERT_RETURN_IF_ERROR(
        mtp_mask_buf_.Write<float>(absl::MakeConstSpan(mask)));

    absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
    LITERT_ASSIGN_OR_RETURN(auto embed_dup, mtp_embed_buf_.Duplicate());
    LITERT_ASSIGN_OR_RETURN(auto pos_dup, mtp_pos_buf_.Duplicate());
    LITERT_ASSIGN_OR_RETURN(auto mask_dup, mtp_mask_buf_.Duplicate());
    input_map.insert_or_assign(mtp_embed_input_name_, std::move(embed_dup));
    input_map.insert_or_assign(mtp_pos_input_name_, std::move(pos_dup));
    input_map.insert_or_assign(mtp_mask_input_name_, std::move(mask_dup));

    for (const auto& k_name : mtp_k_input_names_) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_cache_bufs_0_[k_name].Duplicate());
      input_map.insert_or_assign(k_name, std::move(dup));
    }
    for (const auto& v_name : mtp_v_input_names_) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_cache_bufs_0_[v_name].Duplicate());
      input_map.insert_or_assign(v_name, std::move(dup));
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer> step_output_map;
    LITERT_ASSIGN_OR_RETURN(auto logits_dup, mtp_logits_out_buf_.Duplicate());
    step_output_map.insert_or_assign(mtp_logits_output_name_,
                                     std::move(logits_dup));

    for (size_t i = 0;
         i < mtp_k_output_names_.size() && i < mtp_k_input_names_.size(); ++i) {
      const auto& out_name = mtp_k_output_names_[i];
      const auto& in_name = mtp_k_input_names_[i];
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_cache_bufs_1_[in_name].Duplicate());
      step_output_map.insert_or_assign(out_name, std::move(dup));
    }
    for (size_t i = 0;
         i < mtp_v_output_names_.size() && i < mtp_v_input_names_.size(); ++i) {
      const auto& out_name = mtp_v_output_names_[i];
      const auto& in_name = mtp_v_input_names_[i];
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_cache_bufs_1_[in_name].Duplicate());
      step_output_map.insert_or_assign(out_name, std::move(dup));
    }

    LITERT_RETURN_IF_ERROR(
        mtp_model_->Run(mtp_signature_name_, input_map, step_output_map));

    std::swap(mtp_kv_cache_bufs_0_, mtp_kv_cache_bufs_1_);

    if (t >= 1) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         mtp_logits_out_buf_, TensorBuffer::LockMode::kRead));
      int head_idx = t - 1;
      const float* logits_ptr = lock.second + head_idx * 2048;
      std::vector<float> logits(logits_ptr, logits_ptr + 2048);
      int picked_code = PickToken(logits, options_.do_sample);
      codes.push_back(picked_code);
    }
  }

  return codes;
}

absl::Status Qwen3AcousticPredictorStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2)
      << "[TRACE] Starting Qwen3AcousticPredictorStage::ScheduleInternal";

  auto frontend_out = text_frontend_.GetOutput();
  if (absl::IsNotFound(frontend_out.status())) {
    return absl::OkStatus();
  } else if (!frontend_out.ok()) {
    return frontend_out.status();
  }
  const auto& frontend = *frontend_out;

  LITERT_RETURN_IF_ERROR(InitializeBuffers(talker_kv_cache_bufs_0_));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(talker_kv_cache_bufs_1_));

  LITERT_RETURN_IF_ERROR(
      RunPrefill(frontend.prompt_embeddings, frontend.prompt_len));

  int pos = frontend.prompt_len - 1;
  const float* last_emb_ptr = frontend.prompt_embeddings.data() + pos * 1024;
  ABSL_ASSIGN_OR_RETURN(Qwen3AcousticPredictorStage::DecodeOutput decode_out,
                        RunDecode(last_emb_ptr, pos));

  std::vector<float> suppress(qwen3_tts::kCodecVocab, 0.0f);
  for (int i = 2048; i < qwen3_tts::kCodecVocab; ++i) {
    suppress[i] = qwen3_tts::kNegInf;
  }
  suppress[qwen3_tts::kCodecEos] = 0.0f;

  std::vector<std::vector<int>> chunk_frames;
  std::vector<float> chunk_codec_features;
  absl::flat_hash_set<int> history;

  int total_generated_frames = 0;

  while (total_generated_frames < options_.max_frames) {
    std::vector<float> scores(qwen3_tts::kCodecVocab);
    for (int i = 0; i < qwen3_tts::kCodecVocab; ++i) {
      scores[i] = decode_out.cb0_logits[i] + suppress[i];
    }

    if (total_generated_frames < 2) {
      scores[qwen3_tts::kCodecEos] = qwen3_tts::kNegInf;
    }

    for (int token : history) {
      if (scores[token] > 0) {
        scores[token] /= options_.repetition_penalty;
      } else {
        scores[token] *= options_.repetition_penalty;
      }
    }

    int cb0 = PickToken(scores, options_.do_sample);
    history.insert(cb0);
    ABSL_VLOG(2) << "[TRACE] Generated frame " << total_generated_frames
                 << " with cb0=" << cb0;
    if (cb0 == qwen3_tts::kCodecEos) break;

    ABSL_ASSIGN_OR_RETURN(auto mtp_codes, RunMtp(decode_out.hidden, cb0));
    std::vector<int> frame;
    frame.reserve(qwen3_tts::kNumCodeGroups);
    frame.push_back(cb0);
    frame.insert(frame.end(), mtp_codes.begin(), mtp_codes.end());
    chunk_frames.push_back(std::move(frame));

    ABSL_ASSIGN_OR_RETURN(auto codec_vec, EmbedCodecToken(cb0));
    ABSL_ASSIGN_OR_RETURN(auto mtp_vec, EmbedMtpTokens(mtp_codes));

    std::vector<float> embed(qwen3_tts::kHiddenDim);
    for (int i = 0; i < qwen3_tts::kHiddenDim; ++i) {
      float val = codec_vec[i] + mtp_vec[i];
      embed[i] = val;
      chunk_codec_features.push_back(val);
    }

    int step = total_generated_frames;
    total_generated_frames++;

    if (step < frontend.trailing_len) {
      const float* tr_ptr =
          frontend.trailing_embeddings.data() + step * qwen3_tts::kHiddenDim;
      for (int i = 0; i < qwen3_tts::kHiddenDim; ++i) embed[i] += tr_ptr[i];
    } else {
      for (int i = 0; i < qwen3_tts::kHiddenDim; ++i)
        embed[i] += frontend.tts_pad_embedding[i];
    }

    if (static_cast<int>(chunk_frames.size()) >= qwen3_tts::kFrameChunkSize) {
      AcousticOutput out;
      out.rvq_frames = std::move(chunk_frames);
      out.codec_features = std::move(chunk_codec_features);
      PushOutput(std::move(out));
      chunk_frames.clear();
      chunk_codec_features.clear();
    }

    pos += 1;
    if (pos >= talker_cache_len_) {
      ABSL_LOG(WARNING) << "Reached talker_cache_len_ (" << talker_cache_len_
                        << "), stopping decode generation.";
      break;
    }
    ABSL_ASSIGN_OR_RETURN(decode_out, RunDecode(embed.data(), pos));
  }

  if (!chunk_frames.empty()) {
    AcousticOutput out;
    out.rvq_frames = std::move(chunk_frames);
    out.codec_features = std::move(chunk_codec_features);
    PushOutput(std::move(out));
  }
  return absl::OkStatus();
}

void Qwen3AcousticPredictorStage::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  ClearOutputsThenSetState(State::kIdle);
}

}  // namespace litert::omni::tts
