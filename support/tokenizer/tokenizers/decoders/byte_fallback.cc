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
// C++ port of `src/decoders/byte_fallback.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

namespace {

// `u8::from_str_radix(s, 16).ok()` for the two characters between `<0x` and
// `>`: an optional leading `+` followed by hex digits (Rust accepts the sign,
// so `<0x+a>` is byte 0x0a).
std::optional<uint8_t> ParseHexByte(absl::string_view s) {
  if (!s.empty() && s[0] == '+') s.remove_prefix(1);
  if (s.empty()) return std::nullopt;
  uint32_t value = 0;
  for (char c : s) {
    if (!absl::ascii_isxdigit(static_cast<unsigned char>(c))) {
      return std::nullopt;
    }
    value = value * 16 + (absl::ascii_isdigit(static_cast<unsigned char>(c))
                              ? c - '0'
                              : absl::ascii_tolower(c) - 'a' + 10);
  }
  return static_cast<uint8_t>(value);
}

// `String::from_utf8(previous_byte_tokens.clone())`: either the bytes form a
// valid string, or one U+FFFD per inconvertible byte token.
void FlushByteTokens(std::string& previous_byte_tokens,
                     std::vector<std::string>& new_tokens) {
  if (IsValidUtf8(previous_byte_tokens)) {
    new_tokens.push_back(previous_byte_tokens);
  } else {
    for (size_t i = 0; i < previous_byte_tokens.size(); ++i) {
      new_tokens.push_back("\xEF\xBF\xBD");
    }
  }
  previous_byte_tokens.clear();
}

}  // namespace

absl::StatusOr<std::vector<std::string>> ByteFallback::DecodeChain(
    std::vector<std::string> tokens) const {
  std::vector<std::string> new_tokens;
  // PORT NOTE: `Vec<u8>` kept as a `std::string` of raw bytes.
  std::string previous_byte_tokens;

  for (std::string& token : tokens) {
    std::optional<uint8_t> bytes;
    if (token.size() == 6 && absl::StartsWith(token, "<0x") &&
        absl::EndsWith(token, ">")) {
      bytes = ParseHexByte(absl::string_view(token).substr(3, 2));
    }
    if (bytes.has_value()) {
      previous_byte_tokens.push_back(static_cast<char>(*bytes));
    } else {
      if (!previous_byte_tokens.empty()) {
        FlushByteTokens(previous_byte_tokens, new_tokens);
      }
      new_tokens.push_back(std::move(token));
    }
  }
  if (!previous_byte_tokens.empty()) {
    FlushByteTokens(previous_byte_tokens, new_tokens);
  }

  return new_tokens;
}

absl::StatusOr<ByteFallback> ByteFallback::FromJson(const Json& json) {
  // `type_: MustBe!("ByteFallback")`: the field is mandatory and must hold
  // exactly this string (monostate error texts).
  RETURN_IF_ERROR(RequireObject(json, "struct ByteFallback"));
  auto it = json.find("type");
  if (it == json.end()) return MissingFieldError("type");
  if (!it->is_string()) {
    return InvalidTypeError(*it, "string \"ByteFallback\"");
  }
  if (it->get_ref<const std::string&>() != "ByteFallback") {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid value: string \"", it->get_ref<const std::string&>(),
        "\", expected string \"ByteFallback\""));
  }
  return ByteFallback();
}

Json ByteFallback::ToJson() const {
  Json json = Json::object();
  json["type"] = "ByteFallback";
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
