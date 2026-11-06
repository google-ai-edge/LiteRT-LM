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
// C++ port of `src/normalizers/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/normalizers/normalizers.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/bert.h"
#include "support/tokenizer/tokenizers/normalizers/byte_level.h"
#include "support/tokenizer/tokenizers/normalizers/prepend.h"
#include "support/tokenizer/tokenizers/normalizers/replace.h"
#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"
#include "support/tokenizer/tokenizers/normalizers/strip.h"
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/normalizers/utils.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::normalizers {
namespace {

// `#[derive(Debug, Serialize, Deserialize)] pub enum EnumType { ... }`
// (the accepted values of the `type` tag; note `Bert`, not `BertNormalizer`).
enum class EnumType {
  kBert,
  kStrip,
  kStripAccents,
  kNFC,
  kNFD,
  kNFKC,
  kNFKD,
  kSequence,
  kLowercase,
  kNmt,
  kPrecompiled,
  kReplace,
  kPrepend,
  kByteLevel,
};

std::optional<EnumType> ParseEnumType(absl::string_view name) {
  if (name == "Bert") return EnumType::kBert;
  if (name == "Strip") return EnumType::kStrip;
  if (name == "StripAccents") return EnumType::kStripAccents;
  if (name == "NFC") return EnumType::kNFC;
  if (name == "NFD") return EnumType::kNFD;
  if (name == "NFKC") return EnumType::kNFKC;
  if (name == "NFKD") return EnumType::kNFKD;
  if (name == "Sequence") return EnumType::kSequence;
  if (name == "Lowercase") return EnumType::kLowercase;
  if (name == "Nmt") return EnumType::kNmt;
  if (name == "Precompiled") return EnumType::kPrecompiled;
  if (name == "Replace") return EnumType::kReplace;
  if (name == "Prepend") return EnumType::kPrepend;
  if (name == "ByteLevel") return EnumType::kByteLevel;
  return std::nullopt;
}

// `NormalizerWrapper::X(serde_json::from_value(values).map_err(
//      serde::de::Error::custom)?)`
template <typename T>
absl::StatusOr<NormalizerWrapper> FromValue(const Json& values) {
  ASSIGN_OR_RETURN(T normalizer, T::FromJson(values));
  return NormalizerWrapper(std::move(normalizer));
}

// One attempt of the `#[serde(untagged)] enum NormalizerUntagged`.
template <typename T>
bool TryUntagged(const Json& value, std::optional<NormalizerWrapper>& result) {
  absl::StatusOr<T> normalizer = T::FromJson(value);
  if (!normalizer.ok()) return false;
  result.emplace(*std::move(normalizer));
  return true;
}

}  // namespace

absl::StatusOr<NormalizerWrapper> NormalizerWrapper::FromJson(
    const Json& json) {
  // `NormalizerHelper` is `#[serde(untagged)]`: `Tagged { type: EnumType,
  // #[serde(flatten)] rest }` is tried first. It matches iff the value is a
  // map whose `type` field is one of the `EnumType` names; any other value
  // (no `type`, an unknown or non-string `type`, a non-map) is `Legacy`.
  std::optional<EnumType> variant;
  if (json.is_object()) {
    auto it = json.find("type");
    if (it != json.end() && it->is_string()) {
      variant = ParseEnumType(it->get_ref<const std::string&>());
    }
  }

  if (variant.has_value()) {
    // `values` is `rest` with `"type": variant` re-inserted, which is the
    // input map itself (the tag round-trips unchanged).
    const Json& values = json;
    switch (*variant) {
      case EnumType::kBert:
        return FromValue<BertNormalizer>(values);
      case EnumType::kStrip:
        return FromValue<Strip>(values);
      case EnumType::kStripAccents:
        return FromValue<StripAccents>(values);
      case EnumType::kNFC:
        return FromValue<NFC>(values);
      case EnumType::kNFD:
        return FromValue<NFD>(values);
      case EnumType::kNFKC:
        return FromValue<NFKC>(values);
      case EnumType::kNFKD:
        return FromValue<NFKD>(values);
      case EnumType::kSequence:
        return FromValue<Sequence>(values);
      case EnumType::kLowercase:
        return FromValue<Lowercase>(values);
      case EnumType::kNmt:
        return FromValue<Nmt>(values);
      case EnumType::kPrecompiled:
        // Rust: `serde_json::from_str(&serde_json::to_string(&values)
        //            .expect("Can reserialize
        //            precompiled")).expect("Precompiled")`
        // (the string round-trip exists because `from_base64` wants a borrowed
        // `&str`, which a `serde_json::Value` cannot provide).
        // PORT NOTE: a malformed `Precompiled` panics in Rust (`expect`); the
        // error is returned instead.
        return FromValue<Precompiled>(values);
      case EnumType::kReplace:
        return FromValue<Replace>(values);
      case EnumType::kPrepend:
        return FromValue<Prepend>(values);
      case EnumType::kByteLevel:
        return FromValue<ByteLevel>(values);
    }
  }

  // `NormalizerHelper::Legacy(value)`: `serde_json::from_value::<
  // NormalizerUntagged>(value)` tries every variant in declaration order and
  // keeps the first that deserializes.
  std::optional<NormalizerWrapper> untagged;
  if (TryUntagged<BertNormalizer>(json, untagged) ||
      TryUntagged<Strip>(json, untagged) ||
      TryUntagged<StripAccents>(json, untagged) ||
      TryUntagged<NFC>(json, untagged) || TryUntagged<NFD>(json, untagged) ||
      TryUntagged<NFKC>(json, untagged) || TryUntagged<NFKD>(json, untagged) ||
      TryUntagged<Sequence>(json, untagged) ||
      TryUntagged<Lowercase>(json, untagged) ||
      TryUntagged<Nmt>(json, untagged) ||
      // PORT NOTE: `NormalizerUntagged::Precompiled` never matches in Rust:
      // `from_base64` deserializes a borrowed `&str`, which fails with
      // "expected a borrowed string" when going through
      // `serde_json::from_value` (the reason for the tagged round-trip above).
      // The variant is therefore skipped here to keep the same behaviour.
      TryUntagged<Replace>(json, untagged) ||
      TryUntagged<Prepend>(json, untagged) ||
      TryUntagged<ByteLevel>(json, untagged)) {
    return *std::move(untagged);
  }
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum NormalizerUntagged");
}

Json NormalizerWrapper::ToJson() const {
  return std::visit([](const auto& normalizer) { return normalizer.ToJson(); },
                    variant_);
}

absl::Status NormalizerWrapper::Normalize(NormalizedString& normalized) const {
  return std::visit(
      [&normalized](const auto& normalizer) {
        return normalizer.Normalize(normalized);
      },
      variant_);
}

}  // namespace tokenizers::normalizers
