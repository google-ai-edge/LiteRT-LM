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

#include "omni/text2image/flux2/klein_text_encoder_stage.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/base/tensor_utils.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text_encoder_stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {
namespace {

struct FramedPromptIds {
  std::vector<int32_t> input_ids;
  int active_len = 0;
};

absl::Status ValidateFramingConfig(const TextEncoderConfig& config) {
  const int min_framing_tokens = static_cast<int>(
      config.prefix_token_ids.size() + config.suffix_token_ids.size());
  if (config.seq_len <= min_framing_tokens) {
    return absl::InvalidArgumentError(
        absl::StrCat("seq_len (", config.seq_len,
                     ") must be greater than framing token count (",
                     min_framing_tokens, ")."));
  }
  return absl::OkStatus();
}

absl::StatusOr<FramedPromptIds> FramePromptIds(
    absl::Span<const int> user_token_ids, const TextEncoderConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateFramingConfig(config));
  const int min_framing_tokens = static_cast<int>(
      config.prefix_token_ids.size() + config.suffix_token_ids.size());
  const size_t max_user_tokens =
      static_cast<size_t>(config.seq_len - min_framing_tokens);
  const size_t kept_user_tokens =
      std::min(user_token_ids.size(), max_user_tokens);
  if (user_token_ids.size() > max_user_tokens) {
    ABSL_LOG_EVERY_N_SEC(WARNING, 10)
        << "Prompt token count (" << user_token_ids.size()
        << ") exceeds maximum user token capacity (" << max_user_tokens
        << "); truncating to " << kept_user_tokens << " tokens.";
  }
  const size_t active_len = config.prefix_token_ids.size() + kept_user_tokens +
                            config.suffix_token_ids.size();

  FramedPromptIds framed{
      .input_ids = std::vector<int32_t>(config.seq_len, config.pad_token_id),
      .active_len = static_cast<int>(active_len),
  };
  size_t pos = 0;
  for (int32_t id : config.prefix_token_ids) {
    framed.input_ids[pos++] = id;
  }
  for (size_t i = 0; i < kept_user_tokens; ++i) {
    framed.input_ids[pos++] = static_cast<int32_t>(user_token_ids[i]);
  }
  for (int32_t id : config.suffix_token_ids) {
    framed.input_ids[pos++] = id;
  }
  return framed;
}

absl::StatusOr<size_t> GetBufferNumElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  return num_elements;
}

absl::Status ValidateDistinctIndices(absl::Span<const size_t> indices,
                                     size_t num_buffers,
                                     absl::string_view stage_name) {
  if (num_buffers < indices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected at least %d input buffers, got %d.", stage_name,
        indices.size(), num_buffers));
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= num_buffers) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s input index %d out of bounds for %d input buffers.", stage_name,
          indices[i], num_buffers));
    }
    for (size_t j = i + 1; j < indices.size(); ++j) {
      if (indices[i] == indices[j]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s input indices must be distinct, got duplicate %d.", stage_name,
            indices[i]));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<KleinTextEncoderStage>>
KleinTextEncoderStage::Create(
    Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::array<std::unique_ptr<LiteRtRunner>, 3> shard_runners,
    absl::string_view embed_table_fp16, const Config& config) {
  ABSL_RETURN_IF_ERROR(ValidateFramingConfig(config.framing));
  if (config.prompt_dim <= 0 || config.prompt_dim % 3 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "KleinTextEncoderStage prompt_dim (%d) must be a positive multiple of "
        "3.",
        config.prompt_dim));
  }
  const int seq_len = config.framing.seq_len;
  const int hidden_dim = config.prompt_dim / 3;
  const size_t row_bytes = static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
  if (embed_table_fp16.empty() || embed_table_fp16.size() % row_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "KleinTextEncoderStage embed_table_fp16 size (%d) must be a positive "
        "multiple of hidden_dim * 2 (%d).",
        embed_table_fp16.size(), row_bytes));
  }

  std::array<std::vector<TensorBuffer>, 3> shard_input_buffers;
  std::array<std::vector<TensorBuffer>, 3> shard_output_buffers;
  int inferred_num_heads = 0;
  int inferred_head_dim = 0;
  const size_t hidden_elements = static_cast<size_t>(seq_len) * hidden_dim;
  const size_t seq_sq = static_cast<size_t>(seq_len) * seq_len;

  for (size_t s = 0; s < 3; ++s) {
    if (shard_runners[s] == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "KleinTextEncoderStage shard_runners[%d] must not be null.", s));
    }
    const std::string shard_label =
        absl::StrCat("KleinTextEncoderStage shard_", s);
    ABSL_ASSIGN_OR_RETURN(shard_input_buffers[s],
                          shard_runners[s]->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(shard_output_buffers[s],
                          shard_runners[s]->CreateOutputBuffers(""));

    const auto& idx = config.shard_input_indices[s];
    const std::array<size_t, 4> indices = {idx.hidden, idx.attention_mask,
                                           idx.cos, idx.sin};
    ABSL_RETURN_IF_ERROR(ValidateDistinctIndices(
        indices, shard_input_buffers[s].size(), shard_label));
    if (shard_output_buffers[s].size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expected 1 output buffer, got %d.", shard_label,
                          shard_output_buffers[s].size()));
    }

    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        shard_input_buffers[s][idx.hidden], hidden_elements,
        absl::StrCat(shard_label, " hidden")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        shard_output_buffers[s][0], hidden_elements,
        absl::StrCat(shard_label, " output")));

    ABSL_ASSIGN_OR_RETURN(
        const size_t mask_elements,
        GetBufferNumElements(shard_input_buffers[s][idx.attention_mask]));
    if (mask_elements == 0 || mask_elements % seq_sq != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s attention_mask elements (%d) must be a positive multiple of "
          "seq_len^2 (%d).",
          shard_label, mask_elements, seq_sq));
    }
    const int shard_heads = static_cast<int>(mask_elements / seq_sq);
    if (s == 0) {
      inferred_num_heads = shard_heads;
    } else if (shard_heads != inferred_num_heads) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s attention_mask head count (%d) does not match shard_0 (%d).",
          shard_label, shard_heads, inferred_num_heads));
    }
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
        shard_input_buffers[s][idx.attention_mask], mask_elements,
        absl::StrCat(shard_label, " attention_mask")));

    ABSL_ASSIGN_OR_RETURN(
        const size_t cos_elements,
        GetBufferNumElements(shard_input_buffers[s][idx.cos]));
    if (cos_elements == 0 || cos_elements % seq_len != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s cos elements (%d) must be a positive multiple of seq_len (%d).",
          shard_label, cos_elements, seq_len));
    }
    const int shard_head_dim = static_cast<int>(cos_elements / seq_len);
    if (shard_head_dim % 2 != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s head_dim (%d) must be even.", shard_label, shard_head_dim));
    }
    if (s == 0) {
      inferred_head_dim = shard_head_dim;
    } else if (shard_head_dim != inferred_head_dim) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s head_dim (%d) does not match shard_0 (%d).",
                          shard_label, shard_head_dim, inferred_head_dim));
    }
    ABSL_RETURN_IF_ERROR(
        ValidateFloatBuffer(shard_input_buffers[s][idx.cos], cos_elements,
                            absl::StrCat(shard_label, " cos")));
    ABSL_RETURN_IF_ERROR(
        ValidateFloatBuffer(shard_input_buffers[s][idx.sin], cos_elements,
                            absl::StrCat(shard_label, " sin")));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto rope,
      BuildQwen3RotaryPosEmbed(seq_len, inferred_head_dim, config.rope_theta));
  for (size_t s = 0; s < 3; ++s) {
    const auto& idx = config.shard_input_indices[s];
    LITERT_RETURN_IF_ERROR(shard_input_buffers[s][idx.cos].Write<float>(
        absl::MakeConstSpan(rope.first)));
    LITERT_RETURN_IF_ERROR(shard_input_buffers[s][idx.sin].Write<float>(
        absl::MakeConstSpan(rope.second)));
  }

  return absl::WrapUnique(new KleinTextEncoderStage(
      prompt_source, std::move(tokenizer), std::move(shard_runners),
      std::move(shard_input_buffers), std::move(shard_output_buffers),
      embed_table_fp16, config, hidden_dim, inferred_num_heads));
}

KleinTextEncoderStage::KleinTextEncoderStage(
    Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::array<std::unique_ptr<LiteRtRunner>, 3> shard_runners,
    std::array<std::vector<TensorBuffer>, 3> shard_input_buffers,
    std::array<std::vector<TensorBuffer>, 3> shard_output_buffers,
    absl::string_view embed_table_fp16, Config config, int hidden_dim,
    int num_heads)
    : prompt_source_(*prompt_source),
      tokenizer_(std::move(tokenizer)),
      shard_runners_(std::move(shard_runners)),
      shard_input_buffers_(std::move(shard_input_buffers)),
      shard_output_buffers_(std::move(shard_output_buffers)),
      embed_table_fp16_(embed_table_fp16),
      config_(std::move(config)),
      hidden_dim_(hidden_dim),
      num_heads_(num_heads) {}

bool KleinTextEncoderStage::NeedScheduleInternal() const {
  return prompt_source_.HasOutput();
}

absl::Status KleinTextEncoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto prompt = prompt_source_.GetOutput();
  if (absl::IsNotFound(prompt.status())) {
    return absl::OkStatus();
  }
  if (!prompt.ok()) {
    return prompt.status();
  }
  const Text2ImagePrompt& prompt_req = *prompt;

  ABSL_ASSIGN_OR_RETURN(std::vector<int> user_tokens,
                        tokenizer_->TextToTokenIds(absl::StrCat(
                            config_.framing.prompt_prefix, prompt_req.text)));
  ABSL_ASSIGN_OR_RETURN(FramedPromptIds framed,
                        FramePromptIds(user_tokens, config_.framing));

  const int seq_len = config_.framing.seq_len;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> hidden,
      LookupFp16TokenEmbeddings(framed.input_ids, embed_table_fp16_,
                                hidden_dim_));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> attention_mask,
      BuildCausalPaddingAttentionMask(seq_len, framed.active_len, num_heads_));

  const size_t tap_elements = static_cast<size_t>(seq_len) * hidden_dim_;
  std::array<std::vector<float>, 3> taps = {
      std::vector<float>(tap_elements, 0.0f),
      std::vector<float>(tap_elements, 0.0f),
      std::vector<float>(tap_elements, 0.0f),
  };

  for (size_t s = 0; s < 3; ++s) {
    const auto& idx = config_.shard_input_indices[s];
    const absl::Span<const float> shard_in =
        (s == 0) ? absl::MakeConstSpan(hidden)
                 : absl::MakeConstSpan(taps[s - 1]);
    LITERT_RETURN_IF_ERROR(
        shard_input_buffers_[s][idx.hidden].Write<float>(shard_in));
    LITERT_RETURN_IF_ERROR(
        shard_input_buffers_[s][idx.attention_mask].Write<float>(
            absl::MakeConstSpan(attention_mask)));

    ABSL_RETURN_IF_ERROR(shard_runners_[s]->Run("", shard_input_buffers_[s],
                                                shard_output_buffers_[s]));
    LITERT_RETURN_IF_ERROR(
        shard_output_buffers_[s][0].Read<float>(absl::MakeSpan(taps[s])));
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> prompt_embeds,
      InterleaveThreeEncoderTaps(taps[0], taps[1], taps[2], seq_len,
                                 hidden_dim_));

  TextEncoderOutput output;
  output.metadata = prompt_req.metadata;
  output.prompt_embeds = std::move(prompt_embeds);
  PushOutput(std::move(output));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
