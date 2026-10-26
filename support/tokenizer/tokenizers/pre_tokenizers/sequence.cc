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
// C++ port of `src/pre_tokenizers/sequence.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/pre_tokenizers/sequence.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace pre_tokenizers {

Sequence::Sequence(std::vector<PreTokenizerWrapper> pretokenizers)
    : pretokenizers_(std::move(pretokenizers)) {}
Sequence::~Sequence() = default;
Sequence::Sequence(const Sequence& other) = default;
Sequence& Sequence::operator=(const Sequence& other) = default;
Sequence::Sequence(Sequence&& other) noexcept = default;
Sequence& Sequence::operator=(Sequence&& other) noexcept = default;

absl::Status Sequence::PreTokenize(PreTokenizedString& pretokenized) const {
  for (const PreTokenizerWrapper& pretokenizer : pretokenizers_) {
    RETURN_IF_ERROR(pretokenizer.PreTokenize(pretokenized));
  }
  return absl::OkStatus();
}

absl::StatusOr<Sequence> Sequence::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireTypeTag(json, "Sequence"));
  ASSIGN_OR_RETURN(const Json* array, GetArrayField(json, "pretokenizers"));
  std::vector<PreTokenizerWrapper> pretokenizers;
  pretokenizers.reserve(array->size());
  for (const Json& item : *array) {
    ASSIGN_OR_RETURN(PreTokenizerWrapper pretokenizer,
                     PreTokenizerWrapper::FromJson(item));
    pretokenizers.push_back(std::move(pretokenizer));
  }
  return Sequence(std::move(pretokenizers));
}

Json Sequence::ToJson() const {
  Json json = Json::object();
  json["type"] = "Sequence";
  Json array = Json::array();
  for (const PreTokenizerWrapper& pretokenizer : pretokenizers_) {
    array.push_back(pretokenizer.ToJson());
  }
  json["pretokenizers"] = std::move(array);
  return json;
}

bool operator==(const Sequence& a, const Sequence& b) {
  return a.pretokenizers_ == b.pretokenizers_;
}

}  // namespace pre_tokenizers
}  // namespace tokenizers
