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

#include "omni/any2any/qwen_omni/qwen_omni_text_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

struct SpecialTokenEntry {
  absl::string_view token;
  int default_id = 0;
  int QwenOmniModelConfig::* config_field = nullptr;
};

constexpr SpecialTokenEntry kSpecialTokens[] = {
    {.token = "<|im_start|>", .default_id = qwen_omni::kImStartTokenId},
    {.token = "<|im_end|>", .default_id = qwen_omni::kImEndTokenId},
    {.token = "<|endoftext|>", .default_id = qwen_omni::kEosTokenId},
    {.token = "<|AUDIO|>",
     .config_field = &QwenOmniModelConfig::audio_token_id},
    {.token = "<|audio_bos|>",
     .config_field = &QwenOmniModelConfig::audio_bos_token_id},
    {.token = "<|audio_eos|>",
     .config_field = &QwenOmniModelConfig::audio_eos_token_id},
    {.token = "<|vision_bos|>",
     .config_field = &QwenOmniModelConfig::vision_bos_token_id},
    {.token = "<|vision_eos|>",
     .config_field = &QwenOmniModelConfig::vision_eos_token_id},
    {.token = "<|IMAGE|>",
     .config_field = &QwenOmniModelConfig::image_token_id},
    {.token = "<|VIDEO|>",
     .config_field = &QwenOmniModelConfig::video_token_id},
};

// Splits `text` around Qwen2.5-Omni ChatML and multimodal special tokens
// (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<|AUDIO|>`, `<|audio_bos|>`,
// `<|audio_eos|>`, `<|vision_bos|>`, `<|vision_eos|>`, `<|IMAGE|>`,
// `<|VIDEO|>`) and maps each special token to its token ID while delegating
// normal text spans to `tokenizer.TextToTokenIds()`.
absl::StatusOr<std::vector<int>> TokenizeQwenWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text,
    const QwenOmniModelConfig& config) {
  std::vector<int> all_tokens;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next_special_pos = absl::string_view::npos;
    const SpecialTokenEntry* matched_entry = nullptr;

    for (const SpecialTokenEntry& entry : kSpecialTokens) {
      size_t found = text.find(entry.token, pos);
      if (found != absl::string_view::npos &&
          (next_special_pos == absl::string_view::npos ||
           found < next_special_pos)) {
        next_special_pos = found;
        matched_entry = &entry;
      }
    }

    if (matched_entry == nullptr) {
      ABSL_ASSIGN_OR_RETURN(support::TokenIds part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
      break;
    }

    if (next_special_pos > pos) {
      ABSL_ASSIGN_OR_RETURN(
          support::TokenIds part_ids,
          tokenizer.TextToTokenIds(text.substr(pos, next_special_pos - pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
    }

    absl::StatusOr<int> special_id = tokenizer.TokenToId(matched_entry->token);
    const int fallback_id = matched_entry->config_field != nullptr
                                ? config.*(matched_entry->config_field)
                                : matched_entry->default_id;
    all_tokens.push_back(special_id.ok() ? *special_id : fallback_id);
    pos = next_special_pos + matched_entry->token.size();
  }
  return all_tokens;
}

// Tokenizes untrusted user text without letting `HuggingFaceTokenizer` match
// Qwen `<|...|>` `added_tokens` as raw special control token IDs. Splitting
// between `<` and `|` ensures no piece passed to `TextToTokenIds()` contains
// `<|`, while preserving the literal characters for byte-level BPE.
absl::StatusOr<std::vector<int>> TokenizeUntrustedText(
    support::Tokenizer& tokenizer, absl::string_view text) {
  std::vector<int> token_ids;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t special_delim = text.find("<|", pos);
    if (special_delim == absl::string_view::npos) {
      ABSL_ASSIGN_OR_RETURN(support::TokenIds part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
      break;
    }
    // Include `<` in the current piece and leave `|` at the start of the next
    // piece so neither piece contains `<|`.
    ABSL_ASSIGN_OR_RETURN(
        support::TokenIds part_ids,
        tokenizer.TextToTokenIds(text.substr(pos, special_delim - pos + 1)));
    token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
    pos = special_delim + 1;
  }
  return token_ids;
}

absl::StatusOr<std::vector<float>> EmbedQwenTokenIds(
    LiteRtRunner& text_embedder_runner,
    absl::Span<TensorBuffer> text_emb_input_buffers,
    absl::Span<TensorBuffer> text_emb_output_buffers, int hidden_dim,
    int vocab_size, absl::Span<const int> token_ids) {
  for (int id : token_ids) {
    if (id < 0 || (vocab_size > 0 && id >= vocab_size)) {
      return absl::InvalidArgumentError(
          absl::StrCat("QwenOmniTextStage received out-of-range token ID (", id,
                       "); valid range is [0, ", vocab_size, ")."));
    }
  }

  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, text_emb_input_buffers[0].Size());
  const size_t max_batch = std::max<size_t>(1, in_bytes / sizeof(int32_t));
  const size_t hidden_size = static_cast<size_t>(hidden_dim);
  std::vector<float> embeddings;
  if (hidden_size == 0 ||
      token_ids.size() > std::numeric_limits<size_t>::max() / hidden_size ||
      token_ids.size() > embeddings.max_size() / hidden_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("QwenOmniTextStage token count (", token_ids.size(),
                     ") * llm_hidden_size (", hidden_dim,
                     ") exceeds maximum embedding buffer size."));
  }
  embeddings.resize(token_ids.size() * hidden_size);
  std::vector<int32_t> batch_ids(max_batch, 0);

  size_t offset = 0;
  while (offset < token_ids.size()) {
    const size_t count = std::min(max_batch, token_ids.size() - offset);
    for (size_t i = 0; i < count; ++i) {
      batch_ids[i] = static_cast<int32_t>(token_ids[offset + i]);
    }
    if (count < max_batch) {
      std::fill(batch_ids.begin() + count, batch_ids.end(), 0);
    }
    LITERT_RETURN_IF_ERROR(text_emb_input_buffers[0].Write<int32_t>(
        absl::MakeConstSpan(batch_ids)));
    ABSL_RETURN_IF_ERROR(text_embedder_runner.Run("", text_emb_input_buffers,
                                                  text_emb_output_buffers));

    LITERT_RETURN_IF_ERROR(text_emb_output_buffers[0].Read<float>(
        absl::MakeSpan(embeddings)
            .subspan(offset * hidden_size, count * hidden_size)));
    offset += count;
  }
  return embeddings;
}

}  // namespace

absl::StatusOr<std::unique_ptr<QwenOmniTextStage>> QwenOmniTextStage::Create(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
        end_of_input_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    const QwenOmniModelConfig& config) {
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        text_embedder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        text_embedder_runner->CreateOutputBuffers(""));
  if (input_buffers.empty() || output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "QwenOmniTextStage text_embedder_runner must provide non-empty input "
        "and output buffers.");
  }
  if (config.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "QwenOmniTextStage config.llm_hidden_size must be positive.");
  }
  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, input_buffers[0].Size());
  LITERT_ASSIGN_OR_RETURN(size_t out_bytes, output_buffers[0].Size());
  const size_t input_elements = in_bytes / sizeof(int32_t);
  const size_t expected_out_bytes =
      input_elements * static_cast<size_t>(config.llm_hidden_size) *
      sizeof(float);
  if (input_elements == 0 || out_bytes != expected_out_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("QwenOmniTextStage output buffer size (", out_bytes,
                     " bytes) does not match input elements (", input_elements,
                     ") * llm_hidden_size (", config.llm_hidden_size,
                     ") * sizeof(float) (", expected_out_bytes, " bytes)."));
  }
  return absl::WrapUnique(new QwenOmniTextStage(
      text_queue, end_of_turn_queue, end_of_input_queue, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(input_buffers),
      std::move(output_buffers), config));
}

QwenOmniTextStage::QwenOmniTextStage(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
        end_of_input_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    std::vector<TensorBuffer> text_emb_input_buffers,
    std::vector<TensorBuffer> text_emb_output_buffers,
    const QwenOmniModelConfig& config)
    : text_queue_(*text_queue),
      end_of_turn_queue_(*end_of_turn_queue),
      end_of_input_queue_(*end_of_input_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      text_emb_input_buffers_(std::move(text_emb_input_buffers)),
      text_emb_output_buffers_(std::move(text_emb_output_buffers)) {}

absl::StatusOr<std::vector<float>> QwenOmniTextStage::EmbedTokens(
    absl::Span<const int> token_ids) {
  if (token_ids.empty()) {
    return std::vector<float>{};
  }
  return EmbedQwenTokenIds(*text_embedder_runner_,
                           absl::MakeSpan(text_emb_input_buffers_),
                           absl::MakeSpan(text_emb_output_buffers_),
                           hidden_dim_, config_.llm_vocab_size, token_ids);
}

absl::StatusOr<QwenOmniEmbeddingSegment> QwenOmniTextStage::EmbedControlText(
    absl::string_view text) {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  return EmbedControlTextInternal(text);
}

absl::StatusOr<QwenOmniEmbeddingSegment>
QwenOmniTextStage::EmbedControlTextInternal(absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> token_ids,
      TokenizeQwenWithSpecialTokens(*tokenizer_, text, config_));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings, EmbedTokens(token_ids));
  const int num_tokens = static_cast<int>(token_ids.size());
  return QwenOmniEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .modality = QwenOmniEmbeddingSegment::Modality::kText,
  };
}

bool QwenOmniTextStage::NeedScheduleInternal() const {
  return !text_queue_.Empty() || !end_of_turn_queue_.Empty() ||
         !end_of_input_queue_.Empty();
}

absl::Status QwenOmniTextStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("QwenOmniTextStage has no ready input.");
  }

  enum class Source { kNone, kText, kEndOfTurn, kEndOfInput };
  while (NeedScheduleInternal()) {
    Source min_source = Source::kNone;
    uint64_t min_seqnum = 0;

    if (!text_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* peeked_text,
                            text_queue_.PeekFront());
      min_source = Source::kText;
      min_seqnum = peeked_text->seqnum;
    }
    if (!end_of_turn_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInputTurn>* peeked_eot,
                            end_of_turn_queue_.PeekFront());
      if (min_source == Source::kNone || peeked_eot->seqnum < min_seqnum) {
        min_source = Source::kEndOfTurn;
        min_seqnum = peeked_eot->seqnum;
      }
    }
    if (!end_of_input_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInput>* peeked_eoi,
                            end_of_input_queue_.PeekFront());
      if (min_source == Source::kNone || peeked_eoi->seqnum < min_seqnum) {
        min_source = Source::kEndOfInput;
        min_seqnum = peeked_eoi->seqnum;
      }
    }

    QwenOmniEmbeddingSegment segment;
    switch (min_source) {
      case Source::kText: {
        ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* peeked_text,
                              text_queue_.PeekFront());
        ABSL_ASSIGN_OR_RETURN(segment, ProcessText(peeked_text->item.text));
        ABSL_RETURN_IF_ERROR(text_queue_.PopFront().status());
        break;
      }
      case Source::kEndOfTurn: {
        if (!cached_turn_suffix_.has_value()) {
          ABSL_ASSIGN_OR_RETURN(
              QwenOmniEmbeddingSegment suffix,
              EmbedControlTextInternal("<|im_end|>\n<|im_start|>assistant\n"));
          suffix.is_turn_end = true;
          cached_turn_suffix_ = std::move(suffix);
        }
        segment = *cached_turn_suffix_;
        ABSL_RETURN_IF_ERROR(end_of_turn_queue_.PopFront().status());
        break;
      }
      case Source::kEndOfInput: {
        segment = QwenOmniEmbeddingSegment{.is_turn_end = true};
        ABSL_RETURN_IF_ERROR(end_of_input_queue_.PopFront().status());
        break;
      }
      case Source::kNone:
        return absl::OkStatus();
    }
    this->PushOutput(Sequenced<QwenOmniEmbeddingSegment>{
        .seqnum = min_seqnum,
        .item = std::move(segment),
    });
  }
  return absl::OkStatus();
}

absl::StatusOr<QwenOmniEmbeddingSegment> QwenOmniTextStage::ProcessText(
    absl::string_view text) {
  std::vector<int> token_ids;
  if (!text.empty()) {
    ABSL_ASSIGN_OR_RETURN(token_ids, TokenizeUntrustedText(*tokenizer_, text));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings, EmbedTokens(token_ids));
  const int num_tokens = static_cast<int>(token_ids.size());
  return QwenOmniEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .modality = QwenOmniEmbeddingSegment::Modality::kText,
  };
}

}  // namespace litert::omni::any2any
