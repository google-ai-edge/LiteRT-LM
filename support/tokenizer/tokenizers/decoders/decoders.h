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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_DECODERS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_DECODERS_H_

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/bpe.h"
#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"
#include "support/tokenizer/tokenizers/decoders/ctc.h"
#include "support/tokenizer/tokenizers/decoders/fuse.h"
#include "support/tokenizer/tokenizers/decoders/sequence.h"
#include "support/tokenizer/tokenizers/decoders/strip.h"
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/normalizers/replace.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace decoders {

// Re-export these as decoders (`pub use super::pre_tokenizers::byte_level;`
// and `metaspace`).
using ::tokenizers::normalizers::Replace;
using ::tokenizers::pre_tokenizers::ByteLevel;
using ::tokenizers::pre_tokenizers::Metaspace;

// `#[derive(Serialize)] #[serde(untagged)] pub enum DecoderWrapper`, with the
// custom `Deserialize` of `decoders/mod.rs` (see `FromJson`).
class DecoderWrapper : public Decoder {
 public:
  // The variants, in Rust declaration order.
  using Variant = std::variant<BPEDecoder, ByteLevel, WordPiece, Metaspace, CTC,
                               Sequence, Replace, Fuse, Strip, ByteFallback>;

  // `impl_enum_from!(X, DecoderWrapper, Variant)` for every variant.
  DecoderWrapper(BPEDecoder decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(ByteLevel decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(WordPiece decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(Metaspace decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(CTC decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(Sequence decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(Replace decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(Fuse decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(Strip decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}
  DecoderWrapper(ByteFallback decoder)  // NOLINT(google-explicit-constructor)
      : variant_(std::move(decoder)) {}

  // The wrapped decoder (`match self { Self::BPE(bpe) => ... }`).
  const Variant& variant() const { return variant_; }
  Variant& variant() { return variant_; }
  template <typename T>
  bool Is() const {
    return std::holds_alternative<T>(variant_);
  }
  template <typename T>
  const T* GetIf() const {
    return std::get_if<T>(&variant_);
  }
  template <typename T>
  T* GetIf() {
    return std::get_if<T>(&variant_);
  }

  // `impl Decoder for DecoderWrapper`.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  // `impl<'de> Deserialize<'de> for DecoderWrapper`: an object whose `type`
  // is one of the variant names is deserialized as that variant (`Tagged`);
  // anything else (`Legacy`) is tried against every variant in declaration
  // order (`DecoderUntagged`) and fails with "data did not match any variant
  // of untagged enum DecoderUntagged".
  static absl::StatusOr<DecoderWrapper> FromJson(const Json& json);
  // `#[serde(untagged)]` serialization: the wrapped decoder's own JSON.
  Json ToJson() const;

  friend bool operator==(const DecoderWrapper& a, const DecoderWrapper& b) {
    return a.variant_ == b.variant_;
  }
  friend bool operator!=(const DecoderWrapper& a, const DecoderWrapper& b) {
    return !(a == b);
  }

 private:
  Variant variant_;
};

}  // namespace decoders
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_DECODERS_DECODERS_H_
