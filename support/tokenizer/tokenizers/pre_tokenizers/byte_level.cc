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
// C++ port of `src/pre_tokenizers/byte_level.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

// Converts bytes to unicode characters.
// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
absl::flat_hash_map<uint8_t, char32_t> BytesChar() {
  std::vector<uint8_t> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(static_cast<uint8_t>(b));
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));

  std::vector<uint32_t> cs;
  cs.reserve(bs.size());
  for (uint8_t i : bs) cs.push_back(i);
  uint32_t n = 0;

  for (int b = 0; b <= 255; ++b) {
    const uint8_t byte = static_cast<uint8_t>(b);
    if (std::find(bs.begin(), bs.end(), byte) == bs.end()) {
      bs.push_back(byte);
      cs.push_back((1u << 8) + n);
      n += 1;
    }
  }

  // Safety: cs contains all values from bs (between 0 and 255),
  // and some values of value 2⁸ + n, where n is between 0 and 255. This is
  // between 255 and 512. Both ranges are valid UTF-32 values (which is fully
  // saturated until 0xD000)
  absl::flat_hash_map<uint8_t, char32_t> result;
  result.reserve(bs.size());
  for (size_t i = 0; i < bs.size(); ++i) {
    result.emplace(bs[i], static_cast<char32_t>(cs[i]));
  }
  return result;
}

namespace {

// Regex that matches exactly one token.
// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98
//
// `static RE: LazyLock<SysRegex>`; PORT NOTE: the Rust `unwrap()` of the
// (constant, always valid) pattern becomes a `Status` error returned from
// `PreTokenize` should the regex engine ever reject it.
const absl::StatusOr<SysRegex>& Re() {
  static const absl::StatusOr<SysRegex>* const re = new absl::StatusOr<
      SysRegex>(SysRegex::New(
      R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"));
  return *re;
}

// `static BYTES_CHAR: LazyLock<AHashMap<u8, char>>`
const absl::flat_hash_map<uint8_t, char32_t>& BytesCharMap() {
  static const absl::flat_hash_map<uint8_t, char32_t>* const map =
      new absl::flat_hash_map<uint8_t, char32_t>(BytesChar());
  return *map;
}

// `static CHAR_BYTES: LazyLock<AHashMap<char, u8>>`
const absl::flat_hash_map<char32_t, uint8_t>& CharBytesMap() {
  static const absl::flat_hash_map<char32_t, uint8_t>* const map = [] {
    auto* m = new absl::flat_hash_map<char32_t, uint8_t>();
    for (const auto& [c, b] : BytesChar()) m->emplace(b, c);
    return m;
  }();
  return *map;
}

}  // namespace

absl::flat_hash_set<char32_t> ByteLevel::Alphabet() {
  absl::flat_hash_set<char32_t> alphabet;
  for (const auto& [b, c] : BytesCharMap()) alphabet.insert(c);
  return alphabet;
}

// As a `PreTokenizer`, `ByteLevel` is in charge of transforming all the unicode
// characters into their byte-level counterpart. It also splits the input
// according to the configured regex.
// TODO: Give the ability to modify this regex
absl::Status ByteLevel::PreTokenize(PreTokenizedString& pretokenized) const {
  const absl::StatusOr<SysRegex>& re = Re();
  if (!re.ok()) return re.status();
  const SysRegex* re_ref = &*re;
  RETURN_IF_ERROR(
      pretokenized.Split([&](size_t, NormalizedString normalized)
                             -> absl::StatusOr<std::vector<NormalizedString>> {
        if (add_prefix_space_ && !absl::StartsWith(normalized.Get(), " ")) {
          normalized.Prepend(" ");
        }
        if (use_regex_) {
          return normalized.Split(Pattern(re_ref),
                                  SplitDelimiterBehavior::kIsolated);
        } else {
          std::vector<NormalizedString> result;
          result.push_back(std::move(normalized));
          return result;
        }
      }));
  return pretokenized.Normalize([](NormalizedString& normalized) {
    const absl::string_view s = normalized.Get();
    std::vector<std::pair<char32_t, int64_t>> transformations;
    transformations.reserve(s.size());
    const absl::flat_hash_map<uint8_t, char32_t>& bytes_char = BytesCharMap();
    for (const auto [i, cur_char] : CharIndices(s)) {
      const size_t size = Utf8Len(cur_char);
      for (size_t j = 0; j < size; ++j) {
        transformations.emplace_back(
            bytes_char.at(static_cast<uint8_t>(s[i + j])),
            static_cast<int64_t>(j > 0));
      }
    }
    normalized.Transform(std::move(transformations), 0);
    return absl::OkStatus();
  });
}

// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level
// characters to their unicode counterpart, before merging everything back into
// a single String.
// This decoder will consume the tokens and merge them in one step to alleviate
// the fact that single token decoded might be a byte not representable as
// as String.
absl::StatusOr<std::vector<std::string>> ByteLevel::DecodeChain(
    std::vector<std::string> tokens) const {
  const absl::flat_hash_map<char32_t, uint8_t>& char_bytes = CharBytesMap();
  std::string toks;
  for (const std::string& t : tokens) {
    // `t.chars().try_fold(vec![], ...)`: the bytes of every char of the token
    // if all of them are byte-level chars, else the raw bytes of the token.
    std::optional<std::string> acc = std::string();
    for (const auto [offset, c] : CharIndices(t)) {
      const auto it = char_bytes.find(c);
      if (it == char_bytes.end()) {
        acc = std::nullopt;
        break;
      }
      acc->push_back(static_cast<char>(it->second));
    }
    toks += acc.has_value() ? *acc : t;
  }
  return std::vector<std::string>{FromUtf8Lossy(toks)};
}

// As a `PostProcessor`, `ByteLevel` is in charge of trimming the offsets if
// necessary.
size_t ByteLevel::AddedTokens(bool is_pair) const { return 0; }

absl::StatusOr<std::vector<Encoding>> ByteLevel::ProcessEncodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {
  if (trim_offsets_) {
    for (Encoding& encoding : encodings) {
      ProcessOffsets(encoding, add_prefix_space_);
      for (Encoding& overflowing : encoding.GetOverflowingMut()) {
        ProcessOffsets(overflowing, add_prefix_space_);
      }
    }
  }
  for (size_t i = 0; i < encodings.size(); ++i) {
    encodings[i].SetSequenceId(i);
  }
  return encodings;
  // <dyn PostProcessor>::default_process(encodings, add_special_tokens)
}

void ProcessOffsets(Encoding& encoding, bool add_prefix_space) {
  const char32_t byte_level_space = BytesCharMap().at(' ');
  const auto is_space = [byte_level_space](char32_t c) {
    return c == byte_level_space || unicode::IsWhitespace(c);
  };
  encoding.ProcessTokensWithOffsets(
      [&](size_t i, const std::string& token, Offsets& offsets) {
        // `token.chars().take_while(is_space).count()`
        std::vector<char32_t> chars;
        for (const auto [offset, c] : CharIndices(token)) chars.push_back(c);
        size_t leading_spaces = 0;
        for (char32_t c : chars) {
          if (!is_space(c)) break;
          ++leading_spaces;
        }
        // `token.chars().rev().take_while(is_space).count()`
        size_t trailing_spaces = 0;
        for (auto it = chars.rbegin(); it != chars.rend(); ++it) {
          if (!is_space(*it)) break;
          ++trailing_spaces;
        }

        if (leading_spaces > 0 || trailing_spaces > 0) {
          if (leading_spaces > 0) {
            // If user uses `is_pretokenized=True` we might have
            // offsets that might begin at the start of the string but are
            // NOT the first token.
            const bool is_first = i == 0 || offsets.first == 0;
            if (is_first && add_prefix_space && leading_spaces == 1) {
              // If we are processing the first pair of offsets, with
              // `add_prefix_space`, then we shouldn't remove anything we
              // added. If there are more than one leading spaces though, it
              // means we didn't add them, and they should be removed.
              leading_spaces = 0;
            }
            offsets.first =
                std::min(offsets.first + leading_spaces, offsets.second);
          }
          if (trailing_spaces > 0 && offsets.second >= trailing_spaces) {
            offsets.second =
                std::max(offsets.second - trailing_spaces, offsets.first);
          }
        }
      });
}

absl::StatusOr<ByteLevel> ByteLevel::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "ByteLevel"));
  ASSIGN_OR_RETURN(bool add_prefix_space,
                   GetField<bool>(json, "add_prefix_space"));
  ASSIGN_OR_RETURN(bool trim_offsets, GetField<bool>(json, "trim_offsets"));
  // `#[serde(default = "default_true")]`
  ASSIGN_OR_RETURN(const bool use_regex,
                   GetFieldOr<bool>(json, "use_regex", true));
  return ByteLevel(add_prefix_space, trim_offsets, use_regex);
}

Json ByteLevel::ToJson() const {
  Json json = Json::object();
  json["type"] = "ByteLevel";
  json["add_prefix_space"] = add_prefix_space_;
  json["trim_offsets"] = trim_offsets_;
  json["use_regex"] = use_regex_;
  return json;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
