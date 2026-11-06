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
// C++ port of `src/decoders/sequence.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/sequence.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/decoders.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

Sequence::Sequence(std::vector<DecoderWrapper> decoders)
    : decoders_(std::move(decoders)) {}
Sequence::Sequence(const Sequence& other) = default;
Sequence::Sequence(Sequence&& other) noexcept = default;
Sequence& Sequence::operator=(const Sequence& other) = default;
Sequence& Sequence::operator=(Sequence&& other) noexcept = default;
Sequence::~Sequence() = default;

const std::vector<DecoderWrapper>& Sequence::GetDecoders() const {
  return decoders_;
}

std::vector<DecoderWrapper>& Sequence::GetDecodersMut() { return decoders_; }

absl::StatusOr<std::vector<std::string>> Sequence::DecodeChain(
    std::vector<std::string> tokens) const {
  for (const DecoderWrapper& decoder : decoders_) {
    ASSIGN_OR_RETURN(tokens, decoder.DecodeChain(std::move(tokens)));
  }
  return tokens;
}

absl::StatusOr<Sequence> Sequence::FromJson(const Json& json) {
  // `impl_serde_type!`: `SequenceDeserializer { type: SequenceType,
  // #[serde(flatten, with = "SequenceDef")] struct: Sequence }` -- the `type`
  // tag must be present and equal to "Sequence", then the remaining entries
  // are deserialized as the struct itself.
  RETURN_IF_ERROR(RequireTypeTag(json, "Sequence"));
  ASSIGN_OR_RETURN(const Json* decoders_json, GetArrayField(json, "decoders"));
  std::vector<DecoderWrapper> decoders;
  decoders.reserve(decoders_json->size());
  for (const Json& decoder_json : *decoders_json) {
    ASSIGN_OR_RETURN(DecoderWrapper decoder,
                     DecoderWrapper::FromJson(decoder_json));
    decoders.push_back(std::move(decoder));
  }
  return Sequence(std::move(decoders));
}

Json Sequence::ToJson() const {
  Json json = Json::object();
  json["type"] = "Sequence";
  Json decoders = Json::array();
  for (const DecoderWrapper& decoder : decoders_) {
    decoders.push_back(decoder.ToJson());
  }
  json["decoders"] = std::move(decoders);
  return json;
}

bool operator==(const Sequence& a, const Sequence& b) {
  return a.decoders_ == b.decoders_;
}

}  // namespace decoders
}  // namespace tokenizers
