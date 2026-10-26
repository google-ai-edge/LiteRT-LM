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
// C++ port of `src/decoders/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/decoders.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/bpe.h"
#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"
#include "support/tokenizer/tokenizers/decoders/ctc.h"
#include "support/tokenizer/tokenizers/decoders/fuse.h"
#include "support/tokenizer/tokenizers/decoders/sequence.h"
#include "support/tokenizer/tokenizers/decoders/strip.h"
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

namespace {

// `pub enum EnumType { BPEDecoder, ByteLevel, WordPiece, Metaspace, CTC,
// Sequence, Replace, Fuse, Strip, ByteFallback }`.
enum class EnumType {
  kBPEDecoder,
  kByteLevel,
  kWordPiece,
  kMetaspace,
  kCTC,
  kSequence,
  kReplace,
  kFuse,
  kStrip,
  kByteFallback,
};

constexpr struct {
  absl::string_view name;
  EnumType value;
} kEnumTypes[] = {
    {"BPEDecoder", EnumType::kBPEDecoder},
    {"ByteLevel", EnumType::kByteLevel},
    {"WordPiece", EnumType::kWordPiece},
    {"Metaspace", EnumType::kMetaspace},
    {"CTC", EnumType::kCTC},
    {"Sequence", EnumType::kSequence},
    {"Replace", EnumType::kReplace},
    {"Fuse", EnumType::kFuse},
    {"Strip", EnumType::kStrip},
    {"ByteFallback", EnumType::kByteFallback},
};

// `Tagged { #[serde(rename = "type")] variant: EnumType, #[serde(flatten)]
// rest: serde_json::Value }`: succeeds iff the value is an object whose
// `type` entry is a string naming one of the variants.
std::optional<EnumType> MatchTagged(const Json& json) {
  if (!json.is_object()) return std::nullopt;
  auto it = json.find("type");
  if (it == json.end() || !it->is_string()) return std::nullopt;
  const std::string& name = it->get_ref<const std::string&>();
  for (const auto& entry : kEnumTypes) {
    if (entry.name == name) return entry.value;
  }
  return std::nullopt;
}

template <typename T>
absl::StatusOr<DecoderWrapper> Wrap(absl::StatusOr<T> decoder) {
  RETURN_IF_ERROR(decoder.status());
  return DecoderWrapper(*std::move(decoder));
}

}  // namespace

absl::StatusOr<DecoderWrapper> DecoderWrapper::FromJson(const Json& json) {
  // `let helper = DecoderHelper::deserialize(deserializer).expect("Helper");`
  // (`DecoderHelper` is untagged: `Tagged` first, then `Legacy(Value)` which
  // accepts anything, so this cannot fail.)
  const std::optional<EnumType> variant = MatchTagged(json);
  if (variant.has_value()) {
    // `DecoderHelper::Tagged(model)`: the flattened `rest` plus the `type`
    // entry is exactly the input object again, deserialized as the variant
    // named by the tag.
    switch (*variant) {
      case EnumType::kBPEDecoder:
        return Wrap(BPEDecoder::FromJson(json));
      case EnumType::kByteLevel:
        return Wrap(ByteLevel::FromJson(json));
      case EnumType::kWordPiece:
        return Wrap(WordPiece::FromJson(json));
      case EnumType::kMetaspace:
        return Wrap(Metaspace::FromJson(json));
      case EnumType::kCTC:
        return Wrap(CTC::FromJson(json));
      case EnumType::kSequence:
        return Wrap(Sequence::FromJson(json));
      case EnumType::kReplace:
        return Wrap(Replace::FromJson(json));
      case EnumType::kFuse:
        return Wrap(Fuse::FromJson(json));
      case EnumType::kStrip:
        return Wrap(Strip::FromJson(json));
      case EnumType::kByteFallback:
        return Wrap(ByteFallback::FromJson(json));
    }
  }
  // `DecoderHelper::Legacy(value)`: `#[serde(untagged)] enum DecoderUntagged`,
  // every variant is tried in declaration order and the first success wins
  // (serde does not validate the `type` tag of plain `#[serde(tag = "type")]`
  // structs, so e.g. `{"suffix":"</w>"}` is a `BPEDecoder`).
  if (auto d = BPEDecoder::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = ByteLevel::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = WordPiece::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = Metaspace::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = CTC::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = Sequence::FromJson(json); d.ok()) {
    return DecoderWrapper(*std::move(d));
  }
  if (auto d = Replace::FromJson(json); d.ok()) {
    return DecoderWrapper(*std::move(d));
  }
  if (auto d = Fuse::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = Strip::FromJson(json); d.ok()) return DecoderWrapper(*d);
  if (auto d = ByteFallback::FromJson(json); d.ok()) return DecoderWrapper(*d);
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum DecoderUntagged");
}

Json DecoderWrapper::ToJson() const {
  return std::visit([](const auto& decoder) { return decoder.ToJson(); },
                    variant_);
}

absl::StatusOr<std::vector<std::string>> DecoderWrapper::DecodeChain(
    std::vector<std::string> tokens) const {
  return std::visit(
      [&tokens](const auto& decoder) {
        return decoder.DecodeChain(std::move(tokens));
      },
      variant_);
}

}  // namespace decoders
}  // namespace tokenizers
