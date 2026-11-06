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
// C++ port of `src/tokenizer/pre_tokenizer.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {

namespace {

class BytesToCharOffsetConverter {
 public:
  explicit BytesToCharOffsetConverter(absl::string_view sequence) {
    size_t char_idx = 0;
    size_t pos = 0;
    while (pos < sequence.size()) {
      const size_t b = pos;
      DecodeUtf8(sequence, &pos);
      const size_t len = pos - b;
      for (size_t n = 0; n < len; ++n) {
        map_.insert_or_assign(b + n, char_idx);
      }
      ++char_idx;
    }
  }

  std::optional<Offsets> Convert(Offsets offsets) const {
    auto start_it = map_.find(offsets.first);
    auto end_it = map_.find(offsets.second);
    if (start_it != map_.end() && end_it != map_.end()) {
      return Offsets{start_it->second, end_it->second};
    }
    // If we reached the end, `end` is not in the map
    if (start_it != map_.end() && end_it == map_.end()) {
      // But the one just before should be
      size_t last = start_it->second + 1;
      if (offsets.second > 0) {
        auto prev_it = map_.find(offsets.second - 1);
        if (prev_it != map_.end()) {
          last = prev_it->second;
        }
      }
      return Offsets{start_it->second, last + 1};
    }
    return std::nullopt;
  }

 private:
  absl::flat_hash_map<size_t, size_t> map_;
};

}  // namespace

PreTokenizedString PreTokenizedString::From(absl::string_view s) {
  return PreTokenizedString(NormalizedString::From(s));
}

PreTokenizedString::PreTokenizedString(NormalizedString s)
    : original_(s.GetOriginal()),
      splits_{tokenizers::Split(std::move(s), std::nullopt)} {}

absl::Status PreTokenizedString::Split(
    absl::FunctionRef<absl::StatusOr<std::vector<tokenizers::Split>>(
        size_t, NormalizedString)>
        split_fn) {
  // new_splits is at least as big as self.splits
  std::vector<tokenizers::Split> new_splits;
  new_splits.reserve(splits_.size());
  std::vector<tokenizers::Split> drained = std::move(splits_);
  splits_.clear();
  for (size_t i = 0; i < drained.size(); ++i) {
    tokenizers::Split& original_split = drained[i];
    if (original_split.tokens.has_value()) {
      new_splits.push_back(std::move(original_split));
      continue;
    }

    ASSIGN_OR_RETURN(std::vector<tokenizers::Split> sub_splits,
                     split_fn(i, std::move(original_split.normalized)));
    for (tokenizers::Split& split : sub_splits) {
      if (!split.normalized.IsEmpty()) {
        new_splits.push_back(std::move(split));
      }
    }
  }
  splits_ = std::move(new_splits);

  return absl::OkStatus();
}

absl::Status PreTokenizedString::Split(
    absl::FunctionRef<
        absl::StatusOr<std::vector<NormalizedString>>(size_t, NormalizedString)>
        split_fn) {
  return Split([&split_fn](size_t i, NormalizedString normalized)
                   -> absl::StatusOr<std::vector<tokenizers::Split>> {
    ASSIGN_OR_RETURN(std::vector<NormalizedString> sub_splits,
                     split_fn(i, std::move(normalized)));
    std::vector<tokenizers::Split> out;
    out.reserve(sub_splits.size());
    for (NormalizedString& s : sub_splits) {
      out.emplace_back(std::move(s));
    }
    return out;
  });
}

absl::Status PreTokenizedString::Normalize(
    absl::FunctionRef<absl::Status(NormalizedString&)> normalize) {
  for (tokenizers::Split& split : splits_) {
    if (!split.tokens.has_value()) {
      RETURN_IF_ERROR(normalize(split.normalized));
    }
  }
  return absl::OkStatus();
}

absl::Status PreTokenizedString::Tokenize(
    absl::FunctionRef<
        absl::StatusOr<std::vector<Token>>(const NormalizedString&)>
        tokenize) {
  for (tokenizers::Split& split : splits_) {
    if (!split.tokens.has_value()) {
      ASSIGN_OR_RETURN(std::vector<Token> tokens, tokenize(split.normalized));
      split.tokens = std::move(tokens);
    }
  }

  return absl::OkStatus();
}

absl::Status PreTokenizedString::TokenizeWithLimit(
    absl::FunctionRef<
        absl::StatusOr<std::vector<Token>>(const NormalizedString&)>
        tokenize,
    size_t max_tokens, TruncationDirection direction) {
  size_t total_tokens = 0;
  switch (direction) {
    case TruncationDirection::kLeft: {
      size_t first_tokenized_idx = splits_.size();

      for (size_t idx = splits_.size(); idx > 0; --idx) {
        const size_t i = idx - 1;
        tokenizers::Split& split = splits_[i];
        if (split.tokens.has_value()) {
          total_tokens += split.tokens->size();
          first_tokenized_idx = i;
          continue;
        }

        ASSIGN_OR_RETURN(std::vector<Token> tokens, tokenize(split.normalized));
        total_tokens += tokens.size();
        split.tokens = std::move(tokens);
        first_tokenized_idx = i;

        if (total_tokens >= max_tokens) {
          break;
        }
      }

      splits_.erase(splits_.begin(), splits_.begin() + first_tokenized_idx);
      break;
    }
    case TruncationDirection::kRight: {
      size_t last_tokenized_idx = 0;

      for (size_t i = 0; i < splits_.size(); ++i) {
        tokenizers::Split& split = splits_[i];
        if (split.tokens.has_value()) {
          total_tokens += split.tokens->size();
          last_tokenized_idx = i + 1;
          continue;
        }

        ASSIGN_OR_RETURN(std::vector<Token> tokens, tokenize(split.normalized));
        total_tokens += tokens.size();
        split.tokens = std::move(tokens);
        last_tokenized_idx = i + 1;

        if (total_tokens >= max_tokens) {
          break;
        }
      }

      splits_.erase(splits_.begin() + last_tokenized_idx, splits_.end());
      break;
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<Encoding> PreTokenizedString::IntoEncoding(
    std::optional<uint32_t> word_idx, uint32_t type_id,
    OffsetType offset_type) && {
  if (splits_.empty()) {
    return Encoding();
  }
  const bool all_tokenized = std::all_of(
      splits_.begin(), splits_.end(),
      [](const tokenizers::Split& split) { return split.tokens.has_value(); });
  if (!all_tokenized) {
    return absl::InvalidArgumentError(
        "Split has not been tokenized, call `PreTokenizedString::tokenize` "
        "first");
  }

  std::optional<BytesToCharOffsetConverter> offset_converter;
  switch (offset_type) {
    case OffsetType::kChar:
      offset_converter.emplace(original_);
      break;
    case OffsetType::kByte:
      break;
    case OffsetType::kNone: {
      std::vector<Encoding::Item> items;
      for (tokenizers::Split& split : splits_) {
        for (Token& token : *split.tokens) {
          items.emplace_back(token.id, std::string(), Offsets{0, 0},
                             std::nullopt, 0);
        }
      }
      return Encoding::FromIter(std::move(items));
    }
  }

  std::vector<Encoding::Item> items;
  for (size_t idx = 0; idx < splits_.size(); ++idx) {
    tokenizers::Split& split = splits_[idx];
    const NormalizedString& normalized = split.normalized;
    const Offsets offsets = normalized.OffsetsOriginal();

    for (Token& token : *split.tokens) {
      std::optional<std::pair<size_t, size_t>> converted =
          normalized.ConvertOffsets(
              Range::Normalized(token.offsets.first, token.offsets.second));
      Offsets token_offsets = converted.has_value()
                                  ? Offsets{offsets.first + converted->first,
                                            offsets.first + converted->second}
                                  : token.offsets;

      // Convert to char offsets if relevant
      if (offset_converter.has_value()) {
        token_offsets =
            offset_converter->Convert(token_offsets).value_or(token_offsets);
      }

      items.emplace_back(token.id, std::move(token.value), token_offsets,
                         word_idx.has_value() ? word_idx
                                              : std::optional<uint32_t>(
                                                    static_cast<uint32_t>(idx)),
                         type_id);
    }
  }
  return Encoding::FromIter(std::move(items));
}

std::vector<std::tuple<absl::string_view, Offsets,
                       const std::optional<std::vector<Token>>*>>
PreTokenizedString::GetSplits(OffsetReferential offset_ref,
                              OffsetType offset_type) const {
  std::optional<BytesToCharOffsetConverter> offset_converter;
  if (offset_type == OffsetType::kChar) {
    offset_converter.emplace(original_);
  }

  size_t offset = 0;
  std::vector<std::tuple<absl::string_view, Offsets,
                         const std::optional<std::vector<Token>>*>>
      out;
  out.reserve(splits_.size());
  for (const tokenizers::Split& split : splits_) {
    Offsets offsets;
    switch (offset_ref) {
      case OffsetReferential::kOriginal:
        offsets = split.normalized.OffsetsOriginal();
        break;
      case OffsetReferential::kNormalized: {
        const size_t len = split.normalized.Len();
        offset += len;
        offsets = {offset - len, offset};
        break;
      }
    }

    // Convert to char offsets if relevant
    if (offset_converter.has_value()) {
      offsets = offset_converter->Convert(offsets).value_or(offsets);
    }

    out.emplace_back(split.normalized.Get(), offsets, &split.tokens);
  }
  return out;
}

}  // namespace tokenizers
