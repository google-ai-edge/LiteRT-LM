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
// C++ port of `src/normalizers/utils.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/utils.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {

Sequence::Sequence(std::vector<NormalizerWrapper> normalizers)
    : normalizers_(std::move(normalizers)) {}
Sequence::Sequence(const Sequence& other) = default;
Sequence::Sequence(Sequence&& other) noexcept = default;
Sequence& Sequence::operator=(const Sequence& other) = default;
Sequence& Sequence::operator=(Sequence&& other) noexcept = default;
Sequence::~Sequence() = default;

std::vector<NormalizerWrapper> Sequence::IntoNormalizers() && {
  return std::move(normalizers_);
}

absl::StatusOr<Sequence> Sequence::FromJson(const Json& json) {
  // `#[serde(tag = "type")]` on a struct does not check the tag when
  // deserializing: `type` is simply an ignored field.
  RETURN_IF_ERROR(RequireObject(json, "struct Sequence"));
  ASSIGN_OR_RETURN(const Json* normalizers, GetArrayField(json, "normalizers"));
  std::vector<NormalizerWrapper> result;
  result.reserve(normalizers->size());
  for (const Json& normalizer : *normalizers) {
    ASSIGN_OR_RETURN(NormalizerWrapper wrapper,
                     NormalizerWrapper::FromJson(normalizer));
    result.push_back(std::move(wrapper));
  }
  return Sequence(std::move(result));
}

Json Sequence::ToJson() const {
  Json json = Json::object();
  json["type"] = "Sequence";
  Json normalizers = Json::array();
  for (const NormalizerWrapper& normalizer : normalizers_) {
    normalizers.push_back(normalizer.ToJson());
  }
  json["normalizers"] = std::move(normalizers);
  return json;
}

absl::Status Sequence::Normalize(NormalizedString& normalized) const {
  for (const NormalizerWrapper& normalizer : normalizers_) {
    RETURN_IF_ERROR(normalizer.Normalize(normalized));
  }
  return absl::OkStatus();
}

absl::StatusOr<Lowercase> Lowercase::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "Lowercase"));
  return Lowercase();
}

Json Lowercase::ToJson() const { return Json{{"type", "Lowercase"}}; }

absl::Status Lowercase::Normalize(NormalizedString& normalized) const {
  normalized.Lowercase();
  return absl::OkStatus();
}

}  // namespace tokenizers::normalizers
