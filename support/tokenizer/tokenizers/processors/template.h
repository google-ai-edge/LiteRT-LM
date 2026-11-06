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
// C++ port of `src/processors/template.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// # Template Processing
//
// Provides a way to specify templates in order to add the special tokens to
// each input sequence as relevant.
//
// ## Example
//
// Let's take `BERT` tokenizer as an example. It uses two special tokens, used
// to delimitate each sequence. `[CLS]` is always used at the beginning of the
// first sequence, and `[SEP]` is added at the end of both the first, and the
// pair sequences. The final result looks like this:
// - Single sequence: `[CLS] Hello there [SEP]`
// - Pair sequences: `[CLS] My name is Anthony [SEP] What is my name? [SEP]`
//
// With the type ids as following:
// ```markdown
// [CLS]   ...   [SEP]   ...   [SEP]
//   0      0      0      1      1
// ```
//
// So, we can define a `TemplateProcessing` that will achieve this result:
// ```
// TemplateProcessingBuilder builder = TemplateProcessing::Builder();
// // The template when we only have a single sequence:
// RETURN_IF_ERROR(builder.TrySingle({"[CLS]", "$0", "[SEP]"}));
// // Same as:
// RETURN_IF_ERROR(builder.TrySingle("[CLS] $0 [SEP]"));
//
// // The template when we have both sequences:
// RETURN_IF_ERROR(builder.TryPair({"[CLS]:0", "$A:0", "[SEP]:0", "$B:1",
//                                  "[SEP]:1"}));
// // Same as:
// RETURN_IF_ERROR(builder.TryPair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1"));
// // Or:
// RETURN_IF_ERROR(builder.TryPair("[CLS] $0 [SEP] $B:1 [SEP]:1"));
//
// // The list of special tokens used by each sequences
// builder.SpecialTokens({{"[CLS]", 1}, {"[SEP]", 0}});
// ASSIGN_OR_RETURN(TemplateProcessing template, builder.Build());
// ```
//
// In this example, each input sequence is identified using a `$` construct.
// This identifier lets us specify each input sequence, and the type_id to use.
// When nothing is specified, it uses the default values. Here are the
// different ways to specify it:
// - Specifying the sequence, with default `type_id == 0`: `$A` or `$B`
// - Specifying the `type_id` with default `sequence == A`: `$0`, `$1`, `$2`,
//   ...
// - Specifying both: `$A:0`, `$B:1`, ...
//
// The same construct is used for special tokens: `<identifier>(:<type_id>)?`.
//
// **Warning**: You must ensure that you are giving the correct tokens/ids as
// these will be added to the `Encoding` without any further check. If the
// given ids correspond to something totally different in a `Tokenizer` using
// this `PostProcessor`, it might lead to unexpected results.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_TEMPLATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_TEMPLATE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

// Represents any sequences received as input of the PostProcessor
//
// PORT NOTE: this is `template::Sequence` of the Rust crate. The C++ port
// keeps one namespace per directory, where `Sequence` is already the
// post-processor of `sequence.rs`, hence the name `SequenceId`.
// serde: `"A"` / `"B"`.
enum class SequenceId {
  // This is the first sequence, the one that is always specified
  kA,
  // This is the pair sequence, that is optional
  kB,
};

absl::StatusOr<SequenceId> SequenceIdFromJson(const Json& json);
Json SequenceIdToJson(SequenceId id);

// Represents the different kind of pieces that constitute a template.
// It can be either the input sequence or a `SpecialToken`:
//
// - The `Sequence` has an associated `type_id` which is used by default
//   for any token inside this sequence. The `Sequence` corresponds to one
//   of the input sequence given as input of the `PostProcessor`.
//
// - The `SpecialToken` has an associated `id`. It corresponds to a
//   `SpecialToken`.
//
// The easiest way to build a `Piece` is actually by converting it from a
// string:
// ```
// ASSIGN_OR_RETURN(Piece sequence_with_type_id_0, Piece::TryFrom("$0"));
// ASSIGN_OR_RETURN(Piece sequence_with_type_id_1, Piece::TryFrom("$1"));
// ASSIGN_OR_RETURN(Piece special_token_cls, Piece::TryFrom("[CLS]"));
// ```
//
// serde (externally tagged): `{"Sequence":{"id":"A","type_id":0}}` or
// `{"SpecialToken":{"id":"[CLS]","type_id":0}}`.
class Piece {
 public:
  enum class Kind { kSequence, kSpecialToken };

  // `Piece::Sequence { id, type_id }`.
  static Piece Sequence(SequenceId id, uint32_t type_id) {
    return Piece(Kind::kSequence, id, "", type_id);
  }
  // `Piece::SpecialToken { id, type_id }`.
  static Piece SpecialToken(std::string id, uint32_t type_id) {
    return Piece(Kind::kSpecialToken, SequenceId::kA, std::move(id), type_id);
  }

  Kind kind() const { return kind_; }
  bool IsSequence() const { return kind_ == Kind::kSequence; }
  bool IsSpecialToken() const { return kind_ == Kind::kSpecialToken; }
  // The `id` of a `Piece::Sequence` (only meaningful when `IsSequence()`).
  SequenceId sequence_id() const { return sequence_id_; }
  // The `id` of a `Piece::SpecialToken` (only meaningful when
  // `IsSpecialToken()`).
  const std::string& special_token_id() const { return special_token_id_; }
  uint32_t type_id() const { return type_id_; }

  // `impl TryFrom<String> for Piece` / `impl TryFrom<&str> for Piece`
  // (`Error = String` -> `InvalidArgumentError`).
  static absl::StatusOr<Piece> TryFrom(absl::string_view s);

  static absl::StatusOr<Piece> FromJson(const Json& json);
  Json ToJson() const;

  // Rust `{:?}` of the piece, e.g. `Sequence { id: A, type_id: 0 }` or
  // `SpecialToken { id: "[CLS]", type_id: 0 }`.
  std::string DebugString() const;

  friend bool operator==(const Piece& a, const Piece& b) {
    if (a.kind_ != b.kind_ || a.type_id_ != b.type_id_) return false;
    return a.kind_ == Kind::kSequence
               ? a.sequence_id_ == b.sequence_id_
               : a.special_token_id_ == b.special_token_id_;
  }
  friend bool operator!=(const Piece& a, const Piece& b) { return !(a == b); }

 private:
  Piece(Kind kind, SequenceId sequence_id, std::string special_token_id,
        uint32_t type_id)
      : kind_(kind),
        sequence_id_(sequence_id),
        special_token_id_(std::move(special_token_id)),
        type_id_(type_id) {}

  // `fn extract_id(s: &str) -> Option<Self>`.
  static std::optional<Piece> ExtractId(absl::string_view s);
  // `fn with_type_id(self, type_id: u32) -> Self`.
  Piece WithTypeId(uint32_t type_id) const;

  Kind kind_;
  SequenceId sequence_id_;
  std::string special_token_id_;
  uint32_t type_id_;
};

// Represents a bunch of tokens to be used in a template.
// Usually, special tokens have only one associated id/token but in
// some cases, it might be interesting to have multiple ids/tokens.
//
// # Examples
// ```
// // Simple cases, where a single id/token is necessary:
// SpecialToken cls("[CLS]", 1);
// SpecialToken sep(0, "[SEP]");  // The order in the tuple is not important
//
// // More complex case with multiple values:
// ASSIGN_OR_RETURN(SpecialToken complex,
//                  SpecialToken::New("A complex special token:",
//                                    {0, 1, 2, 3, 4},
//                                    {"A", "complex", "special", "token",
//                                     ":"}));
// ```
//
// serde: `{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]}`.
class SpecialToken {
 public:
  // `impl From<(String, u32)>` / `impl From<(&str, u32)>`.
  SpecialToken(absl::string_view token, uint32_t id)
      : id_(token), ids_{id}, tokens_{std::string(token)} {}
  // `impl From<(u32, String)>` / `impl From<(u32, &str)>`.
  SpecialToken(uint32_t id, absl::string_view token)
      : SpecialToken(token, id) {}
  SpecialToken(  // NOLINT(google-explicit-constructor)
      const std::pair<std::string, uint32_t>& v)
      : SpecialToken(v.first, v.second) {}
  SpecialToken(  // NOLINT(google-explicit-constructor)
      const std::pair<uint32_t, std::string>& v)
      : SpecialToken(v.second, v.first) {}

  // `SpecialToken::new(id, ids, tokens)`: `ids` and `tokens` must have the
  // same length.
  static absl::StatusOr<SpecialToken> New(std::string id,
                                          std::vector<uint32_t> ids,
                                          std::vector<std::string> tokens);

  // A unique id used to identify this SpecialToken in the template
  const std::string& id() const { return id_; }
  // The list of associated ids
  const std::vector<uint32_t>& ids() const { return ids_; }
  // The list of associated tokens
  const std::vector<std::string>& tokens() const { return tokens_; }

  static absl::StatusOr<SpecialToken> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const SpecialToken& a, const SpecialToken& b) {
    return a.id_ == b.id_ && a.ids_ == b.ids_ && a.tokens_ == b.tokens_;
  }
  friend bool operator!=(const SpecialToken& a, const SpecialToken& b) {
    return !(a == b);
  }

 private:
  SpecialToken(std::string id, std::vector<uint32_t> ids,
               std::vector<std::string> tokens)
      : id_(std::move(id)), ids_(std::move(ids)), tokens_(std::move(tokens)) {}

  std::string id_;
  std::vector<uint32_t> ids_;
  std::vector<std::string> tokens_;
};

// A Template represents a Vec<`Piece`>.
//
// We can easily build one as follows
// ```
// // By providing a `String` or `&str`, we just split on whitespaces:
// ASSIGN_OR_RETURN(Template template, Template::TryFrom("[CLS] $0 [SEP]"));
//
// // By providing pieces directly:
// ASSIGN_OR_RETURN(Template template,
//                  Template::TryFrom({"[CLS]", "$0", "[SEP]"}));
// ```
// Both of these methods give the same result.
//
// serde: `#[serde(transparent)]` -> the JSON array of the pieces.
class Template {
 public:
  Template() = default;
  // `Template(pieces)`.
  explicit Template(std::vector<Piece> pieces) : pieces_(std::move(pieces)) {}

  // `impl TryFrom<String>` / `impl TryFrom<&str>`: split on ' '.
  static absl::StatusOr<Template> TryFrom(absl::string_view s);
  // `impl<T: TryInto<Piece>> TryFrom<Vec<T>>`.
  static absl::StatusOr<Template> TryFrom(const std::vector<std::string>& v);

  // `.0`
  const std::vector<Piece>& pieces() const { return pieces_; }

  static absl::StatusOr<Template> FromJson(const Json& json);
  Json ToJson() const;

  // Rust `{:?}` of the template, e.g.
  // `Template([Sequence { id: A, type_id: 0 }])`.
  std::string DebugString() const;

  friend bool operator==(const Template& a, const Template& b) {
    return a.pieces_ == b.pieces_;
  }
  friend bool operator!=(const Template& a, const Template& b) {
    return !(a == b);
  }

 private:
  std::vector<Piece> pieces_;
};

// A bunch of `SpecialToken` represented by their ID.
// Internally, `Tokens` is a `HashMap<String, SpecialToken>` and can be built
// from a HashMap or a Vec<`SpecialToken`>.
//
// serde: `#[serde(transparent)]`; serialized with `ordered_map` (keys sorted).
class Tokens {
 public:
  Tokens() = default;
  // `impl<T: Into<SpecialToken>> From<Vec<T>> for Tokens`.
  Tokens(std::vector<SpecialToken> v);  // NOLINT(google-explicit-constructor)
  // `impl From<AHashMap<String, SpecialToken>> for Tokens`.
  Tokens(  // NOLINT(google-explicit-constructor)
      absl::flat_hash_map<std::string, SpecialToken> v)
      : map_(std::move(v)) {}

  // `.0`
  const absl::flat_hash_map<std::string, SpecialToken>& map() const {
    return map_;
  }
  absl::flat_hash_map<std::string, SpecialToken>& map() { return map_; }

  static absl::StatusOr<Tokens> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Tokens& a, const Tokens& b) {
    return a.map_ == b.map_;
  }
  friend bool operator!=(const Tokens& a, const Tokens& b) { return !(a == b); }

 private:
  absl::flat_hash_map<std::string, SpecialToken> map_;
};

class TemplateProcessingBuilder;

// This PostProcessor takes care of processing each input `Encoding` by
// applying the corresponding template, before merging them in the final
// Encoding.
//
// A `Template` is actually a sequence of `Piece` that will be
// concatenated together in the given order. Each `Piece` represents either
// one of the input `Encoding` or a `SpecialToken`.
//
// ## Example
// ```
// TemplateProcessingBuilder builder = TemplateProcessing::Builder();
// RETURN_IF_ERROR(builder.TrySingle("[CLS] $A [SEP]"));
// RETURN_IF_ERROR(builder.TryPair("[CLS] $A [SEP] $B:1 [SEP]:1"));
// builder.SpecialTokens({{"[CLS]", 1}, {"[SEP]", 0}});
// ASSIGN_OR_RETURN(TemplateProcessing template, builder.Build());
// ```
//
// serde: `#[serde(tag = "type", from = "TemplateProcessingDeserializer")]` ->
// `{"type":"TemplateProcessing","single":[...],"pair":[...],
// "special_tokens":{...}}`; `added_single` / `added_pair` are `#[serde(skip)]`
// and recomputed when deserializing.
class TemplateProcessing : public PostProcessor {
 public:
  // `impl Default for TemplateProcessing`: `single = "$0"`, `pair = "$1"`,
  // no special tokens.
  TemplateProcessing();

  // `TemplateProcessing::builder()`.
  static TemplateProcessingBuilder Builder();

  // Getter for `single`: `format!("{:?}", self.single)`.
  std::string GetSingle() const { return single_.DebugString(); }
  // Setter for `single`
  void SetSingle(Template single) { single_ = std::move(single); }
  // Getter for `pair`
  const Template& GetPair() const { return pair_; }
  // Setter for `pair`
  void SetPair(Template pair) { pair_ = std::move(pair); }
  // Getter for `added_single`
  size_t GetAddedSingle() const { return added_single_; }
  // Setter for `added_single`
  void SetAddedSingle(size_t added_single) { added_single_ = added_single; }
  // Getter for `added_pair`
  size_t GetAddedPair() const { return added_pair_; }
  // Setter for `added_pair`
  void SetAddedPair(size_t added_pair) { added_pair_ = added_pair; }
  // Getter for `special_tokens`
  const Tokens& GetSpecialTokens() const { return special_tokens_; }
  // Setter for `special_tokens`
  void SetSpecialTokens(Tokens special_tokens) {
    special_tokens_ = std::move(special_tokens);
  }
  // The `single` template itself (`pub single: Template`).
  const Template& single() const { return single_; }

  // `impl PostProcessor for TemplateProcessing`.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  static absl::StatusOr<TemplateProcessing> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const TemplateProcessing& a,
                         const TemplateProcessing& b) {
    return a.single_ == b.single_ && a.pair_ == b.pair_ &&
           a.added_single_ == b.added_single_ &&
           a.added_pair_ == b.added_pair_ &&
           a.special_tokens_ == b.special_tokens_;
  }
  friend bool operator!=(const TemplateProcessing& a,
                         const TemplateProcessing& b) {
    return !(a == b);
  }

 private:
  friend class TemplateProcessingBuilder;

  TemplateProcessing(Template single, Template pair, size_t added_single,
                     size_t added_pair, Tokens special_tokens)
      : single_(std::move(single)),
        pair_(std::move(pair)),
        added_single_(added_single),
        added_pair_(added_pair),
        special_tokens_(std::move(special_tokens)) {}

  // `fn apply_template(&self, template, encodings, add_special_tokens)`.
  absl::StatusOr<std::vector<Encoding>> ApplyTemplate(
      const std::vector<Piece>& pieces, std::vector<Encoding> encodings,
      bool add_special_tokens) const;

  Template single_;
  Template pair_;
  size_t added_single_;
  size_t added_pair_;
  Tokens special_tokens_;
};

// Count the number of added tokens in the given template
size_t CountAdded(const Template& container, const Tokens* special_tokens);

// `#[derive(Builder)]` for `TemplateProcessing` (derive_builder), with
// `build_fn(validate = "Self::validate")`.
class TemplateProcessingBuilder {
 public:
  TemplateProcessingBuilder() = default;

  // `single(Template)` / `try_single(value: TryInto<Template>)`. Default:
  // `"$0"`.
  TemplateProcessingBuilder& Single(Template single) {
    single_ = std::move(single);
    return *this;
  }
  absl::Status TrySingle(absl::string_view single);
  absl::Status TrySingle(const std::vector<std::string>& single);

  // `pair(Template)` / `try_pair(value: TryInto<Template>)`. Default:
  // `"$A:0 $B:1"`.
  TemplateProcessingBuilder& Pair(Template pair) {
    pair_ = std::move(pair);
    return *this;
  }
  absl::Status TryPair(absl::string_view pair);
  absl::Status TryPair(const std::vector<std::string>& pair);

  // `special_tokens(value: Into<Tokens>)`. Default: no special tokens.
  TemplateProcessingBuilder& SpecialTokens(Tokens special_tokens) {
    special_tokens_ = std::move(special_tokens);
    return *this;
  }

  // `build()`: runs `validate` and fills in the defaults
  // (`TemplateProcessingBuilderError::ValidationError` ->
  // `InvalidArgumentError` with the same message).
  absl::StatusOr<TemplateProcessing> Build() const;

 private:
  // `fn default_added(&self, is_single: bool) -> usize`.
  size_t DefaultAdded(bool is_single) const;
  // `fn validate(&self) -> Result<(), String>`.
  absl::Status Validate() const;

  std::optional<Template> single_;
  std::optional<Template> pair_;
  std::optional<Tokens> special_tokens_;
};

}  // namespace processors
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_TEMPLATE_H_
