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
// C++ port of `src/normalizers/unicode.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/unicode.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {

absl::StatusOr<NFD> NFD::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "NFD"));
  return NFD();
}
Json NFD::ToJson() const { return Json{{"type", "NFD"}}; }
absl::Status NFD::Normalize(NormalizedString& normalized) const {
  normalized.Nfd();
  return absl::OkStatus();
}

absl::StatusOr<NFKD> NFKD::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "NFKD"));
  return NFKD();
}
Json NFKD::ToJson() const { return Json{{"type", "NFKD"}}; }
absl::Status NFKD::Normalize(NormalizedString& normalized) const {
  normalized.Nfkd();
  return absl::OkStatus();
}

absl::StatusOr<NFC> NFC::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "NFC"));
  return NFC();
}
Json NFC::ToJson() const { return Json{{"type", "NFC"}}; }
absl::Status NFC::Normalize(NormalizedString& normalized) const {
  normalized.Nfc();
  return absl::OkStatus();
}

absl::StatusOr<NFKC> NFKC::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "NFKC"));
  return NFKC();
}
Json NFKC::ToJson() const { return Json{{"type", "NFKC"}}; }
absl::Status NFKC::Normalize(NormalizedString& normalized) const {
  normalized.Nfkc();
  return absl::OkStatus();
}

namespace {

void DoNmt(NormalizedString& normalized) {
  // Ascii Control characters
  normalized
      .Filter([](char32_t c) {
        return !((c >= 0x0001 && c <= 0x0008) || c == 0x000B ||
                 (c >= 0x000E && c <= 0x001F) || c == 0x007F || c == 0x008F ||
                 c == 0x009F);
      })
      // Other code points considered as whitespace.
      .Map([](char32_t c) -> char32_t {
        switch (c) {
          case 0x0009:
          case 0x000A:
          case 0x000C:
          case 0x000D:
          case 0x1680:
          case 0x200B:
          case 0x200C:
          case 0x200D:
          case 0x200E:
          case 0x200F:
          case 0x2028:
          case 0x2029:
          case 0x2581:
          case 0xFEFF:
          case 0xFFFD:
            return U' ';
          default:
            return c;
        }
      });
}

}  // namespace

absl::StatusOr<Nmt> Nmt::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "Nmt"));
  return Nmt();
}
Json Nmt::ToJson() const { return Json{{"type", "Nmt"}}; }
absl::Status Nmt::Normalize(NormalizedString& normalized) const {
  DoNmt(normalized);
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
