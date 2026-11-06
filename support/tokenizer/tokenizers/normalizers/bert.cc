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
// C++ port of `src/normalizers/bert.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/bert.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {
namespace {

// Checks whether a character is whitespace
bool IsWhitespace(char32_t c) {
  // These are technically control characters but we count them as whitespace
  switch (c) {
    case U'\t':
    case U'\n':
    case U'\r':
      return true;
    default:
      return unicode::IsWhitespace(c);
  }
}

// Checks whether a character is a control character
bool IsControl(char32_t c) {
  // These are technically control characters but we count them as whitespace
  switch (c) {
    case U'\t':
    case U'\n':
    case U'\r':
      return false;
    // The definition of `is_control` here is quite large and contains also
    // Cc, Cf, Cn or Co
    // cf. https://unicode.org/reports/tr44/ (Table 12)
    default:
      return unicode::IsOther(c);
  }
}

// Checks whether a character is chinese
// This defines a "chinese character" as anything in the CJK Unicode block:
//   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
//
// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
// despite its name. The modern Korean Hangul alphabet is a different block,
// as is Japanese Hiragana and Katakana. Those alphabets are used to write
// space-separated words, so they are not treated specially and handled
// like for all of the other languages.
bool IsChineseChar(char32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
         (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) ||
         (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

}  // namespace

absl::StatusOr<BertNormalizer> BertNormalizer::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` does not validate the tag on deserialization,
  // which is why `{"type":"BertNormalizer", ...}` as well as
  // `{"type":"Bert", ...}` both parse.
  RETURN_IF_ERROR(RequireObject(json, "struct BertNormalizer"));
  ASSIGN_OR_RETURN(bool clean_text, GetField<bool>(json, "clean_text"));
  ASSIGN_OR_RETURN(bool handle_chinese_chars,
                   GetField<bool>(json, "handle_chinese_chars"));
  ASSIGN_OR_RETURN(std::optional<bool> strip_accents,
                   GetOptionalField<bool>(json, "strip_accents"));
  ASSIGN_OR_RETURN(bool lowercase, GetField<bool>(json, "lowercase"));
  return BertNormalizer(clean_text, handle_chinese_chars, strip_accents,
                        lowercase);
}

Json BertNormalizer::ToJson() const {
  Json json = Json::object();
  json["type"] = "BertNormalizer";
  json["clean_text"] = clean_text_;
  json["handle_chinese_chars"] = handle_chinese_chars_;
  if (strip_accents_.has_value()) {
    json["strip_accents"] = *strip_accents_;
  } else {
    json["strip_accents"] = nullptr;
  }
  json["lowercase"] = lowercase_;
  return json;
}

void BertNormalizer::DoCleanText(NormalizedString& normalized) const {
  normalized
      .Filter(
          [](char32_t c) { return !(c == 0 || c == 0xfffd || IsControl(c)); })
      .Map([](char32_t c) { return IsWhitespace(c) ? U' ' : c; });
}

void BertNormalizer::DoHandleChineseChars(NormalizedString& normalized) const {
  std::vector<std::pair<char32_t, int64_t>> new_chars;
  normalized.ForEach([&new_chars](char32_t c) {
    if (IsChineseChar(c)) {
      new_chars.push_back({U' ', 0});
      new_chars.push_back({c, 1});
      new_chars.push_back({U' ', 1});
    } else {
      new_chars.push_back({c, 0});
    }
  });
  normalized.Transform(new_chars, 0);
}

void BertNormalizer::DoStripAccents(NormalizedString& normalized) const {
  normalized.Nfd().Filter(
      [](char32_t c) { return !unicode::IsMarkNonspacing(c); });
}

void BertNormalizer::DoLowercase(NormalizedString& normalized) const {
  normalized.Lowercase();
}

absl::Status BertNormalizer::Normalize(NormalizedString& normalized) const {
  if (clean_text_) {
    DoCleanText(normalized);
  }
  if (handle_chinese_chars_) {
    DoHandleChineseChars(normalized);
  }
  bool strip_accents = strip_accents_.value_or(lowercase_);
  if (strip_accents) {
    DoStripAccents(normalized);
  }
  if (lowercase_) {
    DoLowercase(normalized);
  }

  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
