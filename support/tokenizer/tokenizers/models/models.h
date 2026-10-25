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
// C++ port of `src/models/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_MODELS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_MODELS_H_

// Popular tokenizer models.
//
// PORT NOTE: `TrainerWrapper` and the `Trainer` plumbing are not ported
// (training is out of scope), nor are `Model::save` / `Model::get_trainer`.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/model.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/models/unigram/unigram.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"  // IWYU pragma: export
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::models {

// --- std / serde emulation shared by the model implementations -------------
//
// PORT NOTE: these helpers stand in for the Rust standard library and serde
// machinery used by the model submodules (`BufRead::lines`,
// `fs::read_to_string`, `#[derive(Deserialize)]` of primitives and maps). They
// reproduce the serde_json error texts.

// `File::open` + `read_to_string` without the UTF-8 check. The error message
// is the OS error text, e.g. "No such file or directory (os error 2)".
absl::StatusOr<std::string> ReadToString(absl::string_view path);

// `BufRead::lines()` over `content`: splits on '\n' and strips one trailing
// '\r' per line; invalid UTF-8 is an error like in Rust.
absl::StatusOr<std::vector<std::string>> BufReadLines(
    absl::string_view content);

// `serde::de::Unexpected` for a JSON value: "boolean `true`", "integer `5`",
// "floating point `1.5`", "string \"x\"", "sequence", "map" or "null".
std::string SerdeUnexpected(const Json& value);

// `serde::de::Error::invalid_type(unexp, &exp)`:
// "invalid type: {unexp}, expected {exp}".
absl::Status SerdeInvalidType(const Json& value, absl::string_view expected);

// `serde::de::Error::invalid_value(unexp, &exp)`:
// "invalid value: {unexp}, expected {exp}".
absl::Status SerdeInvalidValue(const Json& value, absl::string_view expected);

// `serde::de::Error::missing_field(field)`: "missing field `{field}`".
absl::Status SerdeMissingField(absl::string_view field);

// `Deserialize` for the primitive types used by the models.
absl::StatusOr<std::string> StringFromJson(const Json& value);  // String
absl::StatusOr<bool> BoolFromJson(const Json& value);           // bool
absl::StatusOr<uint32_t> U32FromJson(const Json& value);        // u32
absl::StatusOr<size_t> UsizeFromJson(const Json& value);        // usize
absl::StatusOr<float> F32FromJson(const Json& value);           // f32
absl::StatusOr<double> F64FromJson(const Json& value);          // f64
// `AHashMap<String, u32>`
absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> VocabFromJson(
    const Json& value);

// `Serialize` for an `f32`: serde_json emits the shortest decimal that
// round-trips through `f32`, which is what this returns (as a JSON number).
Json F32ToJson(float value);

// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in
// order of token ID, smallest to largest.
class OrderedVocabIter {
 public:
  explicit OrderedVocabIter(
      const absl::flat_hash_map<uint32_t, std::string>& vocab_r)
      : vocab_r_(&vocab_r) {}

  // `impl Serialize for OrderedVocabIter`
  Json Serialize() const;

 private:
  const absl::flat_hash_map<uint32_t, std::string>* vocab_r_;
};

class ModelWrapper : public Model {
 public:
  // Variants in the Rust declaration order.
  // WordPiece must stay before WordLevel here for deserialization (for
  // retrocompatibility with the versions not including the "type"), since
  // WordLevel is a subset of WordPiece
  using Variant = std::variant<bpe::BPE, wordpiece::WordPiece,
                               wordlevel::WordLevel, unigram::Unigram>;

  // PORT NOTE: Rust's enum has no `Default`; a default-constructed wrapper
  // holds an empty `BPE` (the first variant).
  ModelWrapper() = default;
  ~ModelWrapper() override = default;

  // `impl_enum_from!`
  ModelWrapper(bpe::BPE model) : model_(std::move(model)) {}  // NOLINT
  ModelWrapper(wordpiece::WordPiece model)                    // NOLINT
      : model_(std::move(model)) {}
  ModelWrapper(wordlevel::WordLevel model)  // NOLINT
      : model_(std::move(model)) {}
  ModelWrapper(unigram::Unigram model) : model_(std::move(model)) {}  // NOLINT

  // `impl Deserialize for ModelWrapper` (tagged by `type`, with the legacy
  // untagged fallback) and `#[derive(Serialize)] #[serde(untagged)]`.
  static absl::StatusOr<ModelWrapper> FromJson(const Json& json);
  Json ToJson() const;

  // `impl Model for ModelWrapper`
  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view tokens) const override;
  std::optional<uint32_t> TokenToId(absl::string_view token) const override;
  std::optional<std::string> IdToToken(uint32_t id) const override;
  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override;
  size_t GetVocabSize() const override;

  void ClearCache();
  void ResizeCache(size_t capacity);

  // `PartialEq`
  bool operator==(const ModelWrapper& other) const;
  bool operator!=(const ModelWrapper& other) const { return !(*this == other); }

  // Access to the wrapped variant (the Rust `match`).
  const Variant& variant() const { return model_; }
  Variant& variant() { return model_; }

  template <typename T>
  bool Is() const {
    return std::holds_alternative<T>(model_);
  }
  template <typename T>
  const T* GetIf() const {
    return std::get_if<T>(&model_);
  }
  template <typename T>
  T* GetIf() {
    return std::get_if<T>(&model_);
  }

 private:
  Variant model_;
};

}  // namespace tokenizers::models

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_MODELS_H_
