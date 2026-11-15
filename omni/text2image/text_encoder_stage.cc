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

#include "omni/text2image/text_encoder_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {
namespace {

struct FramedTokens {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;
};

absl::Status ValidateSeqLen(const TextEncoderStage::Config& config) {
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

absl::StatusOr<FramedTokens> FrameAndPadPromptTokens(
    absl::Span<const int> user_token_ids,
    const TextEncoderStage::Config& config) {
  ABSL_RETURN_IF_ERROR(ValidateSeqLen(config));
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

  FramedTokens framed{
      .input_ids = std::vector<int32_t>(config.seq_len, config.pad_token_id),
      .attention_mask = std::vector<int32_t>(config.seq_len, 0),
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
  for (size_t i = 0; i < active_len; ++i) {
    framed.attention_mask[i] = 1;
  }

  return framed;
}

absl::Status WriteIntTensorBuffer(TensorBuffer& buffer,
                                  absl::Span<const int32_t> values) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  if (tensor_type.ElementType() == ElementType::Int64) {
    std::vector<int64_t> values_i64(values.begin(), values.end());
    LITERT_RETURN_IF_ERROR(
        buffer.Write<int64_t>(absl::MakeConstSpan(values_i64)));
    return absl::OkStatus();
  }
  if (tensor_type.ElementType() == ElementType::Int32) {
    LITERT_RETURN_IF_ERROR(buffer.Write<int32_t>(values));
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "TextEncoderStage input buffer element type must be Int32 or Int64.");
}

absl::Status ValidateTokenInputBuffer(const TensorBuffer& buffer, int seq_len,
                                      absl::string_view buffer_name) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  const ElementType element_type = tensor_type.ElementType();
  size_t elem_size = 0;
  if (element_type == ElementType::Int32) {
    elem_size = sizeof(int32_t);
  } else if (element_type == ElementType::Int64) {
    elem_size = sizeof(int64_t);
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TextEncoderStage %s buffer element type must be Int32 or Int64.",
        buffer_name));
  }
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, buffer.PackedSize());
  const size_t expected_elements = static_cast<size_t>(seq_len);
  if (num_elements != expected_elements ||
      packed_size != expected_elements * elem_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TextEncoderStage %s buffer capacity (%d elements, %d bytes) does not "
        "match seq_len (%d).",
        buffer_name, num_elements, packed_size, seq_len));
  }
  return absl::OkStatus();
}

absl::Status ValidateInputBuffersAndIndices(
    const std::vector<TensorBuffer>& input_buffers,
    const TextEncoderStage::Config& config) {
  const auto& input_indices = config.input_indices;
  if (input_buffers.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TextEncoderStage expected at least 2 input buffers, got %d",
        input_buffers.size()));
  }
  if (input_indices.input_ids >= input_buffers.size() ||
      input_indices.attention_mask >= input_buffers.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TextEncoderStage input indices (%d, %d) out of bounds for %d "
        "input buffers",
        input_indices.input_ids, input_indices.attention_mask,
        input_buffers.size()));
  }
  if (input_indices.input_ids == input_indices.attention_mask) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TextEncoderStage input_ids and attention_mask indices must be "
        "distinct, got %d",
        input_indices.input_ids));
  }
  ABSL_RETURN_IF_ERROR(ValidateTokenInputBuffer(
      input_buffers[input_indices.input_ids], config.seq_len, "input_ids"));
  ABSL_RETURN_IF_ERROR(
      ValidateTokenInputBuffer(input_buffers[input_indices.attention_mask],
                               config.seq_len, "attention_mask"));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<TextEncoderStage>> TextEncoderStage::Create(
    Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner, const Config& config) {
  ABSL_RETURN_IF_ERROR(ValidateSeqLen(config));
  if (runner == nullptr) {
    return absl::InvalidArgumentError(
        "TextEncoderStage runner must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        runner->CreateOutputBuffers(""));

  ABSL_RETURN_IF_ERROR(ValidateInputBuffersAndIndices(input_buffers, config));

  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("TextEncoderStage expected 1 output buffer, got %d.",
                        output_buffers.size()));
  }
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType out_type,
                          output_buffers[0].TensorType());
  if (out_type.ElementType() != ElementType::Float32) {
    return absl::InvalidArgumentError(
        "TextEncoderStage output buffer element type must be Float32.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t out_bytes,
                          output_buffers[0].PackedSize());
  if (out_bytes < sizeof(float) || out_bytes % sizeof(float) != 0) {
    return absl::InvalidArgumentError(
        "TextEncoderStage output buffer is empty or not a multiple of "
        "sizeof(float).");
  }

  return absl::WrapUnique(new TextEncoderStage(
      prompt_source, std::move(tokenizer), std::move(runner),
      std::move(input_buffers), std::move(output_buffers), config));
}

TextEncoderStage::TextEncoderStage(
    Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers, Config config)
    : prompt_source_(*prompt_source),
      tokenizer_(std::move(tokenizer)),
      runner_(std::move(runner)),
      input_buffers_(std::move(input_buffers)),
      output_buffers_(std::move(output_buffers)),
      config_(std::move(config)) {}

bool TextEncoderStage::NeedScheduleInternal() const {
  return prompt_source_.HasOutput();
}

absl::Status TextEncoderStage::ScheduleInternal() {
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
                            config_.prompt_prefix, prompt_req.text)));
  ABSL_ASSIGN_OR_RETURN(FramedTokens framed,
                        FrameAndPadPromptTokens(user_tokens, config_));

  ABSL_RETURN_IF_ERROR(
      WriteIntTensorBuffer(input_buffers_[config_.input_indices.input_ids],
                           absl::MakeConstSpan(framed.input_ids)));
  ABSL_RETURN_IF_ERROR(
      WriteIntTensorBuffer(input_buffers_[config_.input_indices.attention_mask],
                           absl::MakeConstSpan(framed.attention_mask)));

  ABSL_RETURN_IF_ERROR(runner_->Run("", input_buffers_, output_buffers_));

  // Output buffer count, Float32 element type, and non-empty float-aligned size
  // were validated in Create().
  TensorBuffer& final_output = output_buffers_[0];
  LITERT_ASSIGN_OR_RETURN(const size_t out_bytes, final_output.PackedSize());
  ABSL_DCHECK_GE(out_bytes, sizeof(float));
  ABSL_DCHECK_EQ(out_bytes % sizeof(float), 0u);
  std::vector<float> prompt_embeds(out_bytes / sizeof(float), 0.0f);
  LITERT_RETURN_IF_ERROR(
      final_output.Read<float>(absl::MakeSpan(prompt_embeds)));

  TextEncoderOutput output;
  output.metadata = prompt_req.metadata;
  output.prompt_embeds = std::move(prompt_embeds);

  PushOutput(std::move(output));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
