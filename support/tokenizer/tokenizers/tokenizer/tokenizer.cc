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
//
// C++ port of `src/tokenizer/mod.rs` (lines 221-end) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/padding.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/truncation.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {

namespace {

// U+FFFD REPLACEMENT CHARACTER, the `'\u{FFFD}'` of the Rust source.
constexpr absl::string_view kReplacementChar = "\xEF\xBF\xBD";

}  // namespace

// ---------------------------------------------------------------------------
// InputSequence

absl::string_view InputSequence::Raw() const {
  if (const auto* s = std::get_if<absl::string_view>(&value_)) return *s;
  if (const auto* s = std::get_if<std::string>(&value_)) return *s;
  return absl::string_view();
}

std::vector<absl::string_view> InputSequence::PreTokenized() const {
  if (const auto* v = std::get_if<std::vector<absl::string_view>>(&value_)) {
    return *v;
  }
  if (const auto* v = std::get_if<std::vector<std::string>>(&value_)) {
    return std::vector<absl::string_view>(v->begin(), v->end());
  }
  return {};
}

// ---------------------------------------------------------------------------
// TokenizerBuilder

absl::StatusOr<Tokenizer> TokenizerBuilder::Build() {
  if (!model_.has_value()) {
    // `BuilderError("Model missing.")`
    return absl::InvalidArgumentError("Model missing.");
  }
  Tokenizer tokenizer(std::move(*model_));
  model_.reset();
  tokenizer.normalizer_ = std::move(normalizer_);
  tokenizer.pre_tokenizer_ = std::move(pre_tokenizer_);
  tokenizer.post_processor_ = std::move(post_processor_);
  tokenizer.decoder_ = std::move(decoder_);
  tokenizer.added_vocabulary_ = std::move(added_vocabulary_);
  tokenizer.truncation_ = std::move(truncation_);
  tokenizer.padding_ = std::move(padding_);
  return tokenizer;
}

// ---------------------------------------------------------------------------
// Tokenizer

Tokenizer::Tokenizer(models::ModelWrapper model) : model_(std::move(model)) {}

absl::StatusOr<Tokenizer> Tokenizer::FromStr(absl::string_view s) {
  ASSIGN_OR_RETURN(Json json, ParseJson(s));
  return FromJson(json);
}

absl::StatusOr<Tokenizer> Tokenizer::FromFile(absl::string_view file) {
  // PORT NOTE: `google_file::read_to_string(file)?`.
  std::ifstream stream{std::string(file), std::ios::in | std::ios::binary};
  if (!stream.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to open file: ", file));
  }
  std::string content((std::istreambuf_iterator<char>(stream)),
                      std::istreambuf_iterator<char>());
  if (stream.bad()) {
    return absl::InternalError(absl::StrCat("Failed to read file: ", file));
  }
  return FromStr(content);
}

absl::StatusOr<Tokenizer> Tokenizer::FromBytes(
    absl::Span<const uint8_t> bytes) {
  return FromStr(absl::string_view(reinterpret_cast<const char*>(bytes.data()),
                                   bytes.size()));
}

absl::StatusOr<std::string> Tokenizer::ToString(bool pretty) const {
  // `serde_json::to_string_pretty(self)` / `serde_json::to_string(self)`.
  return DumpJson(ToJson(), pretty ? 2 : -1);
}

absl::Status Tokenizer::Save(absl::string_view path, bool pretty) const {
  ASSIGN_OR_RETURN(std::string serialized, ToString(pretty));

  std::ofstream file{std::string(path),
                     std::ios::out | std::ios::binary | std::ios::trunc};
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to create file: ", path));
  }
  file.write(serialized.data(), serialized.size());
  file.close();
  if (!file) {
    return absl::InternalError(absl::StrCat("Failed to write file: ", path));
  }
  return absl::OkStatus();
}

absl::Status Tokenizer::WithNormalizer(
    std::optional<normalizers::NormalizerWrapper> normalizer) {
  normalizer_ = std::move(normalizer);
  return added_vocabulary_.RefreshNormalizedTokens(GetNormalizer());
}

absl::Status Tokenizer::WithTruncation(std::optional<TruncationParams> trunc) {
  if (trunc.has_value()) {
    const TruncationParams& trunc_params = *trunc;
    size_t n_added_tokens = GetNAddedTokens(false);
    // PORT NOTE: `usize` subtraction; wraps like a Rust release build when
    // `max_length < n_added_tokens` (a debug build would panic).
    size_t effective_max_length = trunc_params.max_length - n_added_tokens;
    if (effective_max_length < trunc_params.stride) {
      // `TruncationParamError(...)`
      return absl::InvalidArgumentError(absl::StrFormat(
          "tokenizer stride set to %d, which is greater than or equal to its "
          "effective max length of %d (= %d original max length - %d added "
          "special tokens), ",
          trunc_params.stride, effective_max_length, trunc_params.max_length,
          n_added_tokens));
    }
  }
  truncation_ = std::move(trunc);
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, uint32_t> Tokenizer::GetVocab(
    bool with_added_tokens) const {
  absl::flat_hash_map<std::string, uint32_t> final_vocab = model_.GetVocab();

  if (with_added_tokens) {
    const absl::flat_hash_map<std::string, uint32_t>& added_vocab =
        added_vocabulary_.GetVocab();
    if (!added_vocab.empty()) {
      final_vocab.reserve(final_vocab.size() + added_vocab.size());
      for (const auto& [token, id] : added_vocab) {
        final_vocab.insert_or_assign(token, id);
      }
    }
  }

  return final_vocab;
}

size_t Tokenizer::GetVocabSize(bool with_added_tokens) const {
  size_t base = model_.GetVocabSize();
  if (with_added_tokens) {
    const absl::flat_hash_map<std::string, uint32_t>& added =
        added_vocabulary_.GetVocab();
    size_t overlapping = 0;
    for (const auto& [token, id] : added) {
      if (model_.TokenToId(token).has_value()) ++overlapping;
    }
    return base + added.size() - overlapping;
  } else {
    return base;
  }
}

std::optional<std::string> Tokenizer::IdToToken(uint32_t id) const {
  std::optional<std::string> token = added_vocabulary_.SimpleIdToToken(id);
  if (token.has_value()) return token;
  return model_.IdToToken(id);
}

absl::StatusOr<Encoding> Tokenizer::EncodeSingleSequence(
    const InputSequence& sequence, uint32_t type_id,
    OffsetType offsets_type) const {
  auto encode = [&](bool is_pre_tokenized, size_t subseq_idx,
                    absl::string_view subseq) -> absl::StatusOr<Encoding> {
    // PORT NOTE: Rust's `&str` invariant guarantees valid UTF-8 and the C FFI
    // boundary (`from_utf8().unwrap()`) panics on invalid UTF-8; in C++ we
    // validate here at the `Encode*` boundary and return
    // `InvalidArgumentError`.
    if (!IsValidUtf8(subseq)) {
      return absl::InvalidArgumentError("Input sequence is not valid UTF-8");
    }
    ASSIGN_OR_RETURN(
        PreTokenizedString normalized,
        added_vocabulary_.ExtractAndNormalize(GetNormalizer(), subseq));
    ASSIGN_OR_RETURN(PreTokenizedString pre_tokenized,
                     DoPreTokenize(std::move(normalized)));
    ASSIGN_OR_RETURN(
        Encoding subseq_encoding,
        DoTokenize(std::move(pre_tokenized), type_id,
                   is_pre_tokenized ? std::optional<uint32_t>(
                                          static_cast<uint32_t>(subseq_idx))
                                    : std::nullopt,
                   offsets_type));

    return subseq_encoding;
  };

  if (sequence.IsPreTokenized()) {
    // `seq.iter().enumerate().map(|(i, sequence)| encode(true, i,
    // sequence)).collect()` — `impl FromIterator<Encoding> for Encoding` is
    // `Encoding::merge(iter, false)`.
    std::vector<absl::string_view> seq = sequence.PreTokenized();
    std::vector<Encoding> encodings;
    encodings.reserve(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
      ASSIGN_OR_RETURN(Encoding encoding, encode(true, i, seq[i]));
      encodings.push_back(std::move(encoding));
    }
    return Encoding::Merge(std::move(encodings), false);
  }
  return encode(false, 0, sequence.Raw());
}

absl::StatusOr<Encoding> Tokenizer::EncodeFast(const EncodeInput& input,
                                               bool add_special_tokens) const {
  // Extract sequences from the EncodeInput
  const InputSequence& sequence = input.sequence();
  const std::optional<InputSequence>& pair = input.pair();

  // Encode each sequence
  ASSIGN_OR_RETURN(Encoding encoding,
                   EncodeSingleSequence(sequence, 0, OffsetType::kNone));
  std::optional<Encoding> pair_encoding;
  if (pair.has_value()) {
    ASSIGN_OR_RETURN(pair_encoding,
                     EncodeSingleSequence(*pair, 1, OffsetType::kNone));
  }

  // And finally post process
  return PostProcess(std::move(encoding), std::move(pair_encoding),
                     add_special_tokens);
}

absl::StatusOr<Encoding> Tokenizer::Encode(const EncodeInput& input,
                                           bool add_special_tokens) const {
  // Extract sequences from the EncodeInput
  const InputSequence& sequence = input.sequence();
  const std::optional<InputSequence>& pair = input.pair();

  // Encode each sequence
  ASSIGN_OR_RETURN(Encoding encoding,
                   EncodeSingleSequence(sequence, 0, OffsetType::kByte));
  std::optional<Encoding> pair_encoding;
  if (pair.has_value()) {
    ASSIGN_OR_RETURN(pair_encoding,
                     EncodeSingleSequence(*pair, 1, OffsetType::kByte));
  }

  // And finally post process
  return PostProcess(std::move(encoding), std::move(pair_encoding),
                     add_special_tokens);
}

absl::StatusOr<Encoding> Tokenizer::EncodeCharOffsets(
    const EncodeInput& input, bool add_special_tokens) const {
  // Extract sequences from the EncodeInput
  const InputSequence& sequence = input.sequence();
  const std::optional<InputSequence>& pair = input.pair();

  // Encode each sequence
  ASSIGN_OR_RETURN(Encoding encoding,
                   EncodeSingleSequence(sequence, 0, OffsetType::kChar));
  std::optional<Encoding> pair_encoding;
  if (pair.has_value()) {
    ASSIGN_OR_RETURN(pair_encoding,
                     EncodeSingleSequence(*pair, 1, OffsetType::kChar));
  }

  // And finally post process
  return PostProcess(std::move(encoding), std::move(pair_encoding),
                     add_special_tokens);
}

absl::StatusOr<std::string> Tokenizer::Decode(absl::Span<const uint32_t> ids,
                                              bool skip_special_tokens) const {
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());
  for (uint32_t id : ids) {
    std::optional<std::string> token = added_vocabulary_.SimpleIdToToken(id);
    if (!token.has_value()) token = model_.IdToToken(id);
    if (!token.has_value()) continue;
    if (!skip_special_tokens || !added_vocabulary_.IsSpecialToken(*token)) {
      tokens.push_back(*std::move(token));
    }
  }

  if (decoder_.has_value()) {
    return decoder_->Decode(std::move(tokens));
  } else {
    return absl::StrJoin(tokens, " ");
  }
}

::tokenizers::DecodeStream Tokenizer::DecodeStream(
    bool skip_special_tokens) const {
  return ::tokenizers::DecodeStream(this, skip_special_tokens);
}

absl::StatusOr<Encoding> Tokenizer::DoTokenize(PreTokenizedString pretokenized,
                                               uint32_t type_id,
                                               std::optional<uint32_t> word_idx,
                                               OffsetType offsets_type) const {
  std::optional<std::pair<size_t, TruncationDirection>> truncation;
  if (truncation_.has_value() &&
      (truncation_->strategy != TruncationStrategy::kOnlySecond ||
       type_id != 0)) {
    truncation =
        std::make_pair(truncation_->max_length, truncation_->direction);
  }
  RETURN_IF_ERROR(model_.TokenizeInPretokenized(pretokenized, truncation));
  return std::move(pretokenized).IntoEncoding(word_idx, type_id, offsets_type);
}

absl::StatusOr<NormalizedString> Tokenizer::DoNormalize(
    NormalizedString normalized) const {
  if (normalizer_.has_value()) {
    RETURN_IF_ERROR(normalizer_->Normalize(normalized));
  }

  return normalized;
}

absl::StatusOr<PreTokenizedString> Tokenizer::DoPreTokenize(
    PreTokenizedString pretokenized) const {
  if (pre_tokenizer_.has_value()) {
    RETURN_IF_ERROR(pre_tokenizer_->PreTokenize(pretokenized));
  }

  return pretokenized;
}

absl::StatusOr<Encoding> Tokenizer::PostProcess(
    Encoding encoding, std::optional<Encoding> pair_encoding,
    bool add_special_tokens) const {
  // 1. First we truncate if needed
  if (truncation_.has_value()) {
    const TruncationParams& trunc = *truncation_;
    size_t n_added_tokens = GetNAddedTokens(pair_encoding.has_value());

    std::pair<Encoding, std::optional<Encoding>> truncated;
    if (add_special_tokens && n_added_tokens > 0) {
      TruncationParams params = trunc;
      params.max_length = trunc.max_length - n_added_tokens;
      ASSIGN_OR_RETURN(truncated,
                       TruncateEncodings(std::move(encoding),
                                         std::move(pair_encoding), params));
    } else {
      ASSIGN_OR_RETURN(truncated,
                       TruncateEncodings(std::move(encoding),
                                         std::move(pair_encoding), trunc));
    }
    encoding = std::move(truncated.first);
    pair_encoding = std::move(truncated.second);
  }

  // 2. Then We post process
  Encoding final_encoding;
  if (post_processor_.has_value()) {
    ASSIGN_OR_RETURN(
        final_encoding,
        post_processor_->Process(std::move(encoding), std::move(pair_encoding),
                                 add_special_tokens));
  } else {
    std::vector<Encoding> encodings;
    encodings.push_back(std::move(encoding));
    if (pair_encoding.has_value()) {
      encodings.push_back(*std::move(pair_encoding));
    }
    ASSIGN_OR_RETURN(encodings, PostProcessor::DefaultProcess(
                                    std::move(encodings), add_special_tokens));
    if (encodings.size() != 1) {
      // `panic!("We haven't reduced the encodings like we should have")`
      return absl::InternalError(
          "We haven't reduced the encodings like we should have");
    }
    final_encoding = std::move(encodings.back());
  }

  // 3. Then we pad if needed
  if (padding_.has_value()) {
    RETURN_IF_ERROR(
        PadEncodings(absl::MakeSpan(&final_encoding, 1), *padding_));
  }

  return final_encoding;
}

size_t Tokenizer::GetNAddedTokens(bool is_pair) const {
  if (post_processor_.has_value()) {
    return post_processor_->AddedTokens(is_pair);
  } else {
    return 0;
  }
}

absl::StatusOr<std::vector<Encoding>> Tokenizer::EncodeBatch(
    const std::vector<EncodeInput>& inputs, bool add_special_tokens) const {
  std::vector<Encoding> encodings;
  encodings.reserve(inputs.size());
  for (const EncodeInput& input : inputs) {
    ASSIGN_OR_RETURN(Encoding encoding, Encode(input, add_special_tokens));
    encodings.push_back(std::move(encoding));
  }

  if (padding_.has_value()) {
    // We do the padding here to make sure we handle the batch padding
    RETURN_IF_ERROR(PadEncodings(absl::MakeSpan(encodings), *padding_));
  }

  return encodings;
}

absl::StatusOr<std::vector<Encoding>> Tokenizer::EncodeBatchCharOffsets(
    const std::vector<EncodeInput>& inputs, bool add_special_tokens) const {
  std::vector<Encoding> encodings;
  encodings.reserve(inputs.size());
  for (const EncodeInput& input : inputs) {
    ASSIGN_OR_RETURN(Encoding encoding,
                     EncodeCharOffsets(input, add_special_tokens));
    encodings.push_back(std::move(encoding));
  }

  if (padding_.has_value()) {
    // We do the padding here to make sure we handle the batch padding
    RETURN_IF_ERROR(PadEncodings(absl::MakeSpan(encodings), *padding_));
  }

  return encodings;
}

absl::StatusOr<std::vector<Encoding>> Tokenizer::EncodeBatchFast(
    const std::vector<EncodeInput>& inputs, bool add_special_tokens) const {
  std::vector<Encoding> encodings;
  encodings.reserve(inputs.size());
  for (const EncodeInput& input : inputs) {
    ASSIGN_OR_RETURN(Encoding encoding, EncodeFast(input, add_special_tokens));
    encodings.push_back(std::move(encoding));
  }

  if (padding_.has_value()) {
    // We do the padding here to make sure we handle the batch padding
    RETURN_IF_ERROR(PadEncodings(absl::MakeSpan(encodings), *padding_));
  }

  return encodings;
}

absl::StatusOr<std::vector<std::string>> Tokenizer::DecodeBatch(
    absl::Span<const std::vector<uint32_t>> sentences,
    bool skip_special_tokens) const {
  std::vector<std::string> results;
  results.reserve(sentences.size());
  for (const std::vector<uint32_t>& sentence : sentences) {
    ASSIGN_OR_RETURN(std::string decoded,
                     Decode(sentence, skip_special_tokens));
    results.push_back(std::move(decoded));
  }
  return results;
}

// ---------------------------------------------------------------------------
// DecodeStream

absl::Status DecodeStreamError::InvalidPrefix(uint32_t token_id,
                                              absl::string_view expected_prefix,
                                              absl::string_view actual_string) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid prefix encountered while decoding stream. Token ID: %u, "
      "Expected prefix: '%s', Actual string: '%s'",
      token_id, expected_prefix, actual_string));
}

absl::StatusOr<std::optional<std::string>> DecodeStream::Step(uint32_t id) {
  return StepDecodeStream(*tokenizer_, std::vector<uint32_t>{id},
                          skip_special_tokens_, ids_, prefix_, prefix_index_);
}

absl::StatusOr<std::optional<std::string>> StepDecodeStream(
    const Tokenizer& tokenizer, std::vector<uint32_t> token_ids,
    bool skip_special_tokens, std::vector<uint32_t>& ids, std::string& prefix,
    size_t& prefix_index) {
  if (prefix.empty() && !ids.empty()) {
    ASSIGN_OR_RETURN(std::string new_prefix,
                     tokenizer.Decode(ids, skip_special_tokens));
    if (!absl::EndsWith(new_prefix, kReplacementChar)) {
      prefix = std::move(new_prefix);
      prefix_index = ids.size();
    }
  }

  ids.insert(ids.end(), token_ids.begin(), token_ids.end());
  ASSIGN_OR_RETURN(std::string string,
                   tokenizer.Decode(ids, skip_special_tokens));
  if (string.size() > prefix.size() &&
      !absl::EndsWith(string, kReplacementChar)) {
    if (!absl::StartsWith(string, prefix)) {
      return DecodeStreamError::InvalidPrefix(ids.back(), prefix, string);
    }

    std::string new_text = string.substr(prefix.size());
    size_t new_prefix_index = ids.size() - prefix_index;
    ids.erase(ids.begin(), ids.begin() + prefix_index);
    ASSIGN_OR_RETURN(prefix, tokenizer.Decode(ids, skip_special_tokens));
    prefix_index = new_prefix_index;
    return std::optional<std::string>(std::move(new_text));
  } else {
    return std::optional<std::string>(std::nullopt);
  }
}

}  // namespace tokenizers
