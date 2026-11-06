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
// C++ port of `src/normalizers/prepend.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/prepend.h"

#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {

absl::StatusOr<Prepend> Prepend::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` does not validate the tag on deserialization.
  RETURN_IF_ERROR(RequireObject(json, "struct Prepend"));
  ASSIGN_OR_RETURN(std::string prepend, GetField<std::string>(json, "prepend"));
  return Prepend(std::move(prepend));
}

Json Prepend::ToJson() const {
  Json json = Json::object();
  json["type"] = "Prepend";
  json["prepend"] = prepend_;
  return json;
}

absl::Status Prepend::Normalize(NormalizedString& normalized) const {
  if (!normalized.IsEmpty()) {
    normalized.Prepend(prepend_);
  }
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
