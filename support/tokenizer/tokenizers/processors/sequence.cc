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
// C++ port of `src/processors/sequence.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/processors/sequence.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/processors/processors.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace processors {

Sequence::Sequence(std::vector<PostProcessorWrapper> processors)
    : processors_(std::move(processors)) {}
Sequence::Sequence(const Sequence& other) = default;
Sequence::Sequence(Sequence&& other) noexcept = default;
Sequence& Sequence::operator=(const Sequence& other) = default;
Sequence& Sequence::operator=(Sequence&& other) noexcept = default;
Sequence::~Sequence() = default;

const PostProcessorWrapper* Sequence::Get(size_t index) const {
  return index < processors_.size() ? &processors_[index] : nullptr;
}

PostProcessorWrapper* Sequence::GetMut(size_t index) {
  return index < processors_.size() ? &processors_[index] : nullptr;
}

absl::Status Sequence::SetMut(size_t index, PostProcessorWrapper post_proc) {
  if (index >= processors_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("index out of bounds: the len is ", processors_.size(),
                     " but the index is ", index));
  }
  processors_[index] = std::move(post_proc);
  return absl::OkStatus();
}

const std::vector<PostProcessorWrapper>& Sequence::AsRef() const {
  return processors_;
}

std::vector<PostProcessorWrapper>& Sequence::AsMut() { return processors_; }

size_t Sequence::AddedTokens(bool is_pair) const {
  size_t sum = 0;
  for (const PostProcessorWrapper& p : processors_) {
    sum += p.AddedTokens(is_pair);
  }
  return sum;
}

absl::StatusOr<std::vector<Encoding>> Sequence::ProcessEncodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {
  for (const PostProcessorWrapper& processor : processors_) {
    ASSIGN_OR_RETURN(encodings, processor.ProcessEncodings(std::move(encodings),
                                                           add_special_tokens));
  }
  return encodings;
}

absl::StatusOr<Sequence> Sequence::FromJson(const Json& json) {
  // `impl_serde_type!`: `SequenceDeserializer { type: SequenceType,
  // #[serde(flatten, with = "SequenceDef")] struct: Sequence }` -- the `type`
  // tag must be present and equal to "Sequence", then the remaining entries
  // are deserialized as the struct itself.
  RETURN_IF_ERROR(RequireTypeTag(json, "Sequence"));
  ASSIGN_OR_RETURN(const Json* processors_json,
                   GetArrayField(json, "processors"));
  std::vector<PostProcessorWrapper> processors;
  processors.reserve(processors_json->size());
  for (const Json& processor_json : *processors_json) {
    ASSIGN_OR_RETURN(PostProcessorWrapper processor,
                     PostProcessorWrapper::FromJson(processor_json));
    processors.push_back(std::move(processor));
  }
  return Sequence(std::move(processors));
}

Json Sequence::ToJson() const {
  Json json = Json::object();
  json["type"] = "Sequence";
  Json processors = Json::array();
  for (const PostProcessorWrapper& processor : processors_) {
    processors.push_back(processor.ToJson());
  }
  json["processors"] = std::move(processors);
  return json;
}

bool operator==(const Sequence& a, const Sequence& b) {
  return a.processors_ == b.processors_;
}

}  // namespace processors
}  // namespace tokenizers
