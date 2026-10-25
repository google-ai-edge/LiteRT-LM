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
// C++ port of `src/normalizers/byte_level.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/byte_level.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {
namespace {

// `static BYTES_CHAR: LazyLock<AHashMap<u8, char>> =
// LazyLock::new(bytes_char);`
const absl::flat_hash_map<uint8_t, char32_t>& BytesChar() {
  static const absl::flat_hash_map<uint8_t, char32_t>* const kBytesChar =
      new absl::flat_hash_map<uint8_t, char32_t>(pre_tokenizers::BytesChar());
  return *kBytesChar;
}

}  // namespace

absl::flat_hash_set<char32_t> ByteLevel::Alphabet() {
  absl::flat_hash_set<char32_t> alphabet;
  for (const auto& [byte, c] : BytesChar()) {
    alphabet.insert(c);
  }
  return alphabet;
}

absl::StatusOr<ByteLevel> ByteLevel::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "ByteLevel"));
  return ByteLevel();
}

Json ByteLevel::ToJson() const { return Json{{"type", "ByteLevel"}}; }

absl::Status ByteLevel::Normalize(NormalizedString& normalized) const {
  if (!normalized.IsEmpty()) {
    absl::string_view s = normalized.Get();
    std::vector<std::pair<char32_t, int64_t>> transformations;
    transformations.reserve(s.size());
    const absl::flat_hash_map<uint8_t, char32_t>& bytes_char = BytesChar();
    for (const auto& [i, cur_char] : CharIndices(s)) {
      size_t size = Utf8Len(cur_char);
      for (size_t k = 0; k < size; ++k) {
        transformations.push_back(
            {bytes_char.at(static_cast<uint8_t>(s[i + k])), k > 0 ? 1 : 0});
      }
    }
    normalized.Transform(transformations, 0);
  }
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
