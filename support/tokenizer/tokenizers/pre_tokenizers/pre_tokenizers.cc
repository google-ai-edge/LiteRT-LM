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
// C++ port of `src/pre_tokenizers/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/bert.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/delimiter.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/digits.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/fixed_length.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/punctuation.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/sequence.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/unicode_scripts.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

absl::Status PreTokenizerWrapper::PreTokenize(
    PreTokenizedString& normalized) const {
  return std::visit(
      [&normalized](const auto& pretok) {
        return pretok.PreTokenize(normalized);
      },
      variant_);
}

namespace {

// `pub enum EnumType { BertPreTokenizer, ByteLevel, Delimiter, Metaspace,
// Whitespace, Sequence, Split, Punctuation, WhitespaceSplit, Digits,
// UnicodeScripts, FixedLength }`
enum class EnumType {
  kBertPreTokenizer,
  kByteLevel,
  kDelimiter,
  kMetaspace,
  kWhitespace,
  kSequence,
  kSplit,
  kPunctuation,
  kWhitespaceSplit,
  kDigits,
  kUnicodeScripts,
  kFixedLength,
};

std::optional<EnumType> ParseEnumType(absl::string_view type) {
  if (type == "BertPreTokenizer") return EnumType::kBertPreTokenizer;
  if (type == "ByteLevel") return EnumType::kByteLevel;
  if (type == "Delimiter") return EnumType::kDelimiter;
  if (type == "Metaspace") return EnumType::kMetaspace;
  if (type == "Whitespace") return EnumType::kWhitespace;
  if (type == "Sequence") return EnumType::kSequence;
  if (type == "Split") return EnumType::kSplit;
  if (type == "Punctuation") return EnumType::kPunctuation;
  if (type == "WhitespaceSplit") return EnumType::kWhitespaceSplit;
  if (type == "Digits") return EnumType::kDigits;
  if (type == "UnicodeScripts") return EnumType::kUnicodeScripts;
  if (type == "FixedLength") return EnumType::kFixedLength;
  return std::nullopt;
}

template <typename T>
absl::StatusOr<PreTokenizerWrapper> Wrap(absl::StatusOr<T> value) {
  RETURN_IF_ERROR(value.status());
  return PreTokenizerWrapper(*std::move(value));
}

}  // namespace

// `impl<'de> Deserialize<'de> for PreTokenizerWrapper`
absl::StatusOr<PreTokenizerWrapper> PreTokenizerWrapper::FromJson(
    const Json& json) {
  // #[serde(untagged)]
  // pub enum PreTokenizerHelper {
  //     Tagged(Tagged),           // { "type": EnumType, #[serde(flatten)]
  //                               //   rest: serde_json::Value }
  //     Legacy(serde_json::Value),
  // }
  std::optional<EnumType> tagged;
  if (const Json* type = FindField(json, "type");
      type != nullptr && type->is_string()) {
    tagged = ParseEnumType(type->get_ref<const std::string&>());
  }

  if (tagged.has_value()) {
    // PreTokenizerHelper::Tagged(pretok): `values` is `rest` with "type"
    // re-inserted, i.e. the whole object again, handed to the variant named
    // by `type` (errors are propagated as is).
    switch (*tagged) {
      case EnumType::kBertPreTokenizer:
        return Wrap(BertPreTokenizer::FromJson(json));
      case EnumType::kByteLevel:
        return Wrap(ByteLevel::FromJson(json));
      case EnumType::kDelimiter:
        return Wrap(CharDelimiterSplit::FromJson(json));
      case EnumType::kMetaspace:
        return Wrap(Metaspace::FromJson(json));
      case EnumType::kWhitespace:
        return Wrap(Whitespace::FromJson(json));
      case EnumType::kSequence:
        return Wrap(Sequence::FromJson(json));
      case EnumType::kSplit:
        return Wrap(Split::FromJson(json));
      case EnumType::kPunctuation:
        return Wrap(Punctuation::FromJson(json));
      case EnumType::kWhitespaceSplit:
        return Wrap(WhitespaceSplit::FromJson(json));
      case EnumType::kDigits:
        return Wrap(Digits::FromJson(json));
      case EnumType::kUnicodeScripts:
        return Wrap(UnicodeScripts::FromJson(json));
      case EnumType::kFixedLength:
        return Wrap(FixedLength::FromJson(json));
    }
  }

  // PreTokenizerHelper::Legacy(value): `#[serde(untagged)] enum
  // PreTokenizerUntagged`, every variant tried in declaration order. (Each
  // variant still requires its own `type` tag, so this only succeeds for
  // tags that are a struct name but not an `EnumType` name, e.g.
  // `"type": "CharDelimiterSplit"`.)
  if (auto v = BertPreTokenizer::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = ByteLevel::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = CharDelimiterSplit::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Metaspace::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Whitespace::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Sequence::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Split::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Punctuation::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = WhitespaceSplit::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = Digits::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = UnicodeScripts::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  if (auto v = FixedLength::FromJson(json); v.ok()) {
    return PreTokenizerWrapper(*std::move(v));
  }
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum PreTokenizerUntagged");
}

Json PreTokenizerWrapper::ToJson() const {
  return std::visit([](const auto& pretok) { return pretok.ToJson(); },
                    variant_);
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
