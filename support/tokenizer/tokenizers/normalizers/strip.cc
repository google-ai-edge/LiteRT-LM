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
// C++ port of `src/normalizers/strip.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/strip.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/unicode/normalization.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {

absl::StatusOr<Strip> Strip::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` does not validate the tag on deserialization.
  RETURN_IF_ERROR(RequireObject(json, "struct Strip"));
  ASSIGN_OR_RETURN(bool strip_left, GetField<bool>(json, "strip_left"));
  ASSIGN_OR_RETURN(bool strip_right, GetField<bool>(json, "strip_right"));
  return Strip(strip_left, strip_right);
}

Json Strip::ToJson() const {
  Json json = Json::object();
  json["type"] = "Strip";
  json["strip_left"] = strip_left_;
  json["strip_right"] = strip_right_;
  return json;
}

absl::Status Strip::Normalize(NormalizedString& normalized) const {
  if (strip_left_ && strip_right_) {
    // Fast path
    normalized.Strip();
  } else {
    if (strip_left_) {
      normalized.Lstrip();
    }

    if (strip_right_) {
      normalized.Rstrip();
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<StripAccents> StripAccents::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "StripAccents"));
  return StripAccents();
}

Json StripAccents::ToJson() const { return Json{{"type", "StripAccents"}}; }

absl::Status StripAccents::Normalize(NormalizedString& normalized) const {
  normalized.Filter([](char32_t c) { return !unicode::IsCombiningMark(c); });
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
