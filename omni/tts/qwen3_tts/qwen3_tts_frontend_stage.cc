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

#include "omni/tts/qwen3_tts/qwen3_tts_frontend_stage.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/common.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<Qwen3TtsFrontendStage>>
Qwen3TtsFrontendStage::Create(
    Stage<std::string>* absl_nonnull text_source,
    const std::string& model_dir,
    const Qwen3TtsModelConfig& config,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = absl::WrapUnique(
      new Qwen3TtsFrontendStage(text_source, config, std::move(resources)));

  ABSL_ASSIGN_OR_RETURN(
      std::string tok_json,
      LoadFile(model_dir, stage->config_.tokenizer_file));
  ABSL_ASSIGN_OR_RETURN(
      stage->tokenizer_,
      support::HuggingFaceTokenizer::CreateFromJson(std::move(tok_json)));

  LITERT_ASSIGN_OR_RETURN(
      stage->text_embedding_model_,
      stage->resources_->GetCompiledModel("text_embedding"));
  LITERT_ASSIGN_OR_RETURN(
      stage->text_projection_model_,
      stage->resources_->GetCompiledModel("text_projection"));
  LITERT_ASSIGN_OR_RETURN(
      stage->codec_embedding_model_,
      stage->resources_->GetCompiledModel("codec_embedding"));

  // Pre-allocate input and output TensorBuffers for each model
  LITERT_ASSIGN_OR_RETURN(stage->text_emb_input_buffers_,
                          stage->text_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->text_emb_output_buffers_,
                          stage->text_embedding_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(stage->text_proj_input_buffers_,
                          stage->text_projection_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->text_proj_output_buffers_,
                          stage->text_projection_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_input_buffers_,
                          stage->codec_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_output_buffers_,
                          stage->codec_embedding_model_->CreateOutputBuffers());

  // Precompute system embeddings for fixed tokens (kTtsBos, kTtsEos, kTtsPad)
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<float> sys_embeds,
      stage->EmbedText(
          {qwen3_tts::kTtsBos, qwen3_tts::kTtsEos, qwen3_tts::kTtsPad}));
  stage->tts_bos_.assign(sys_embeds.begin(),
                         sys_embeds.begin() + qwen3_tts::kHiddenDim);
  stage->tts_eos_.assign(sys_embeds.begin() + qwen3_tts::kHiddenDim,
                         sys_embeds.begin() + 2 * qwen3_tts::kHiddenDim);
  stage->tts_pad_.assign(sys_embeds.begin() + 2 * qwen3_tts::kHiddenDim,
                         sys_embeds.end());

  // Load speaker embedding from binary file.
  ABSL_ASSIGN_OR_RETURN(
      std::string spk_buf,
      LoadFile(model_dir, stage->config_.speaker_file));
  if (spk_buf.size() != qwen3_tts::kHiddenDim * sizeof(float)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Speaker embedding size is not correct. Expected ",
                     qwen3_tts::kHiddenDim * sizeof(float), " bytes but got ",
                     spk_buf.size()));
  }
  stage->speaker_emb_.resize(qwen3_tts::kHiddenDim);
  std::memcpy(stage->speaker_emb_.data(), spk_buf.data(), spk_buf.size());

  return stage;
}

absl::StatusOr<std::vector<float>> Qwen3TtsFrontendStage::ProjectText(
    const std::vector<float>& rows, int num_rows) {
  std::vector<float> out(num_rows * qwen3_tts::kHiddenDim);

  for (int r = 0; r < num_rows; ++r) {
    LITERT_RETURN_IF_ERROR(text_proj_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(rows.data() + r * 2048, 2048)));
    LITERT_RETURN_IF_ERROR(text_projection_model_->Run(
        text_proj_input_buffers_, text_proj_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        text_proj_output_buffers_[0].Read<float>(absl::MakeSpan(
            out.data() + r * qwen3_tts::kHiddenDim, qwen3_tts::kHiddenDim)));
  }

  return out;
}

absl::StatusOr<std::vector<float>> Qwen3TtsFrontendStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(qwen3_tts::kHiddenDim, 0.0f);
  if (code_id >= 0 && code_id < qwen3_tts::kCodecVocab) {
    int32_t cid = code_id;
    LITERT_RETURN_IF_ERROR(codec_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&cid, 1)));
    LITERT_RETURN_IF_ERROR(codec_embedding_model_->Run(
        codec_emb_input_buffers_, codec_emb_output_buffers_));
    LITERT_RETURN_IF_ERROR(codec_emb_output_buffers_[0].Read<float>(
        absl::MakeSpan(out.data(), qwen3_tts::kHiddenDim)));
  }
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3TtsFrontendStage::EmbedText(
    const std::vector<int>& ids) {
  int num_ids = ids.size();
  std::vector<float> rows(num_ids * 2048);
  for (int idx = 0; idx < num_ids; ++idx) {
    ABSL_VLOG(2) << "[TRACE] EmbedText: " << ids[idx];
    int32_t tid = ids[idx];
    LITERT_RETURN_IF_ERROR(text_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&tid, 1)));
    LITERT_RETURN_IF_ERROR(text_embedding_model_->Run(
        text_emb_input_buffers_, text_emb_output_buffers_));
    LITERT_RETURN_IF_ERROR(text_emb_output_buffers_[0].Read<float>(
        absl::MakeSpan(rows.data() + idx * 2048, 2048)));
  }
  return ProjectText(rows, num_ids);
}

void Qwen3TtsFrontendStage::AppendSum(absl::Span<const float> a,
                                      absl::Span<const float> b,
                                      std::vector<float>& out) {
  for (size_t i = 0; i < a.size(); ++i) {
    out.push_back(a[i] + b[i]);
  }
}

absl::Status Qwen3TtsFrontendStage::AppendCodecSum(absl::Span<const float> base,
                                                   int code_id,
                                                   std::vector<float>& out) {
  ABSL_ASSIGN_OR_RETURN(const std::vector<float> codec_emb,
                        EmbedCodecToken(code_id));
  AppendSum(base, codec_emb, out);
  return absl::OkStatus();
}

absl::StatusOr<Qwen3TtsFrontendOutput> Qwen3TtsFrontendStage::BuildPrompt(
    const std::string& input_text) {
  const std::string prompt_text =
      absl::StrFormat(qwen3_tts::kPromptTemplate, input_text);
  ABSL_ASSIGN_OR_RETURN(const std::vector<int> ids,
                        tokenizer_->TextToTokenIds(prompt_text));
  if (ids.size() < 8) {
    return absl::InvalidArgumentError(
        "Input text too short for tokenization framing");
  }

  std::vector<int> control;
  if (config_.language == "auto") {
    control = {qwen3_tts::kCodecNoThink, qwen3_tts::kCodecThinkBos,
               qwen3_tts::kCodecThinkEos};
  } else {
    ABSL_ASSIGN_OR_RETURN(int lang_id, GetLanguageId(config_.language));
    control = {qwen3_tts::kCodecThink, qwen3_tts::kCodecThinkBos, lang_id,
               qwen3_tts::kCodecThinkEos};
  }

  Qwen3TtsFrontendOutput out;
  out.token_ids = ids;

  const int dim = qwen3_tts::kHiddenDim;
  const int num_prompt_tokens = control.size() + 6;
  out.prompt_embeddings.reserve(num_prompt_tokens * dim);

  // 1. Role embeddings for prefix tokens (ids[0..2]).
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> role,
      EmbedText(std::vector<int>(ids.begin(), ids.begin() + 3)));
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), role.begin(),
                               role.end());

  // 2. Control tokens (each added to tts_pad_).
  for (int c : control) {
    ABSL_RETURN_IF_ERROR(AppendCodecSum(tts_pad_, c, out.prompt_embeddings));
  }

  // 3. Speaker embedding (added to tts_pad_).
  AppendSum(tts_pad_, speaker_emb_, out.prompt_embeddings);

  // 4. Codec pad token (added to tts_bos_).
  ABSL_RETURN_IF_ERROR(
      AppendCodecSum(tts_bos_, qwen3_tts::kCodecPad, out.prompt_embeddings));

  // 5. First text token (ids[3] embedding added to kCodecBos embedding).
  ABSL_ASSIGN_OR_RETURN(const std::vector<float> first_text,
                        EmbedText({ids[3]}));
  ABSL_RETURN_IF_ERROR(
      AppendCodecSum(first_text, qwen3_tts::kCodecBos, out.prompt_embeddings));

  out.prompt_len = num_prompt_tokens;

  // Trailing text embeddings (ids[4 .. end-5]) followed by tts_eos_.
  if (ids.size() >= 10) {
    ABSL_ASSIGN_OR_RETURN(
        out.trailing_embeddings,
        EmbedText(std::vector<int>(ids.begin() + 4, ids.end() - 5)));
  }
  out.trailing_embeddings.insert(out.trailing_embeddings.end(),
                                 tts_eos_.begin(), tts_eos_.end());
  out.trailing_len = out.trailing_embeddings.size() / dim;
  out.tts_pad_embedding = tts_pad_;

  return out;
}

absl::Status Qwen3TtsFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2) << "[TRACE] Starting Qwen3TtsFrontendStage::ScheduleInternal";

  auto text = text_source_.GetOutput();
  if (absl::IsNotFound(text.status())) {
    return absl::OkStatus();
  } else if (!text.ok()) {
    return text.status();
  }
  ABSL_VLOG(2) << "[TRACE] Running BuildPrompt";
  ABSL_ASSIGN_OR_RETURN(auto prompt, BuildPrompt(*text));
  ABSL_VLOG(2) << "[TRACE] Finished BuildPrompt";
  PushOutput(std::move(prompt));
  return absl::OkStatus();
}

void Qwen3TtsFrontendStage::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  ClearOutputsThenSetState(State::kIdle);
}

}  // namespace litert::omni::tts
