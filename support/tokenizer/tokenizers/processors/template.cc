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

#include "support/tokenizer/tokenizers/processors/template.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace processors {

namespace {

// `str::parse::<u32>()` (`u32::from_str_radix(s, 10)`): an optional leading
// `+` followed by one or more ASCII decimal digits, in range for `u32`.
std::optional<uint32_t> ParseU32(absl::string_view s) {
  if (!s.empty() && s[0] == '+') s.remove_prefix(1);
  if (s.empty()) return std::nullopt;
  uint64_t value = 0;
  for (char c : s) {
    if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
      return std::nullopt;
    }
    value = value * 10 + static_cast<uint64_t>(c - '0');
    if (value > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
  }
  return static_cast<uint32_t>(value);
}

}  // namespace

absl::StatusOr<SequenceId> SequenceIdFromJson(const Json& json) {
  static constexpr absl::string_view kVariants[] = {"A", "B"};
  if (!json.is_string()) {
    return InvalidTypeError(json, "variant identifier");
  }
  const std::string& s = json.get_ref<const std::string&>();
  if (s == "A") return SequenceId::kA;
  if (s == "B") return SequenceId::kB;
  return UnknownVariantError(s, kVariants);
}

Json SequenceIdToJson(SequenceId id) {
  return Json(id == SequenceId::kA ? "A" : "B");
}

std::optional<Piece> Piece::ExtractId(absl::string_view s) {
  if (absl::StartsWith(s, "$")) {
    const absl::string_view rest = s.substr(1);

    // If the id is just `$`, we use 0 as type_id, and Sequence A
    if (rest.empty()) {
      return Piece::Sequence(SequenceId::kA, 0);
    }
    if (rest == "A" || rest == "a") {
      return Piece::Sequence(SequenceId::kA, 0);
    }
    if (rest == "B" || rest == "b") {
      return Piece::Sequence(SequenceId::kB, 0);
    }
    if (std::optional<uint32_t> type_id = ParseU32(rest); type_id.has_value()) {
      return Piece::Sequence(SequenceId::kA, *type_id);
    }
    return std::nullopt;
  } else {
    return Piece::SpecialToken(std::string(s), 0);
  }
}

Piece Piece::WithTypeId(uint32_t type_id) const {
  switch (kind_) {
    case Kind::kSequence:
      return Piece::Sequence(sequence_id_, type_id);
    case Kind::kSpecialToken:
      return Piece::SpecialToken(special_token_id_, type_id);
  }
  return *this;
}

absl::StatusOr<Piece> Piece::TryFrom(absl::string_view s) {
  const std::vector<absl::string_view> parts = absl::StrSplit(s, ':');

  auto err = [s]() {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot build Piece from string \"", s, "\""));
  };
  if (parts.size() == 2) {
    std::optional<uint32_t> type_id = ParseU32(parts[1]);
    if (!type_id.has_value()) return err();
    std::optional<Piece> piece = ExtractId(parts[0]);
    if (!piece.has_value()) return err();
    return piece->WithTypeId(*type_id);
  }
  if (parts.size() == 1) {
    std::optional<Piece> piece = ExtractId(parts[0]);
    if (!piece.has_value()) return err();
    return *std::move(piece);
  }
  return err();
}

absl::StatusOr<Piece> Piece::FromJson(const Json& json) {
  static constexpr absl::string_view kVariants[] = {"Sequence", "SpecialToken"};
  if (json.is_object()) {
    if (json.size() != 1) {
      return InvalidValueError(json, "map with a single key");
    }
    const std::string& variant = json.begin().key();
    const Json& value = json.begin().value();
    if (variant == "Sequence") {
      RETURN_IF_ERROR(RequireObject(value, "struct variant Piece::Sequence"));
      const Json* id_json = FindField(value, "id");
      if (id_json == nullptr) return MissingFieldError("id");
      ASSIGN_OR_RETURN(SequenceId id, SequenceIdFromJson(*id_json));
      ASSIGN_OR_RETURN(uint32_t type_id, GetField<uint32_t>(value, "type_id"));
      return Piece::Sequence(id, type_id);
    }
    if (variant == "SpecialToken") {
      RETURN_IF_ERROR(
          RequireObject(value, "struct variant Piece::SpecialToken"));
      ASSIGN_OR_RETURN(std::string id, GetField<std::string>(value, "id"));
      ASSIGN_OR_RETURN(uint32_t type_id, GetField<uint32_t>(value, "type_id"));
      return Piece::SpecialToken(std::move(id), type_id);
    }
    return UnknownVariantError(variant, kVariants);
  }
  if (json.is_string()) {
    const std::string& variant = json.get_ref<const std::string&>();
    if (variant != "Sequence" && variant != "SpecialToken") {
      return UnknownVariantError(variant, kVariants);
    }
    return absl::InvalidArgumentError(
        "invalid type: unit variant, expected struct variant");
  }
  return InvalidTypeError(json, "enum Piece");
}

Json Piece::ToJson() const {
  Json inner = Json::object();
  if (kind_ == Kind::kSequence) {
    inner["id"] = SequenceIdToJson(sequence_id_);
    inner["type_id"] = type_id_;
    Json json = Json::object();
    json["Sequence"] = std::move(inner);
    return json;
  } else {
    inner["id"] = special_token_id_;
    inner["type_id"] = type_id_;
    Json json = Json::object();
    json["SpecialToken"] = std::move(inner);
    return json;
  }
}

std::string Piece::DebugString() const {
  if (kind_ == Kind::kSequence) {
    return absl::StrCat(
        "Sequence { id: ", sequence_id_ == SequenceId::kA ? "A" : "B",
        ", type_id: ", type_id_, " }");
  } else {
    return absl::StrCat(
        "SpecialToken { id: ", DumpJson(Json(special_token_id_)),
        ", type_id: ", type_id_, " }");
  }
}

absl::StatusOr<SpecialToken> SpecialToken::New(
    std::string id, std::vector<uint32_t> ids,
    std::vector<std::string> tokens) {
  if (ids.size() != tokens.size()) {
    return absl::InvalidArgumentError(
        "SpecialToken: ids and tokens must be of the same length");
  } else {
    return SpecialToken(std::move(id), std::move(ids), std::move(tokens));
  }
}

absl::StatusOr<SpecialToken> SpecialToken::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireObject(json, "struct SpecialToken"));
  ASSIGN_OR_RETURN(std::string id, GetField<std::string>(json, "id"));
  ASSIGN_OR_RETURN(std::vector<uint32_t> ids,
                   GetField<std::vector<uint32_t>>(json, "ids"));
  ASSIGN_OR_RETURN(std::vector<std::string> tokens,
                   GetField<std::vector<std::string>>(json, "tokens"));
  return SpecialToken(std::move(id), std::move(ids), std::move(tokens));
}

Json SpecialToken::ToJson() const {
  Json json = Json::object();
  json["id"] = id_;
  json["ids"] = ids_;
  json["tokens"] = tokens_;
  return json;
}

absl::StatusOr<Template> Template::TryFrom(const std::vector<std::string>& v) {
  std::vector<Piece> pieces;
  pieces.reserve(v.size());
  for (const std::string& p : v) {
    ASSIGN_OR_RETURN(Piece piece, Piece::TryFrom(p));
    pieces.push_back(std::move(piece));
  }
  return Template(std::move(pieces));
}

absl::StatusOr<Template> Template::TryFrom(absl::string_view s) {
  std::vector<Piece> pieces;
  for (absl::string_view p : absl::StrSplit(s, ' ')) {
    ASSIGN_OR_RETURN(Piece piece, Piece::TryFrom(p));
    pieces.push_back(std::move(piece));
  }
  return Template(std::move(pieces));
}

absl::StatusOr<Template> Template::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireArray(json, "a sequence"));
  std::vector<Piece> pieces;
  pieces.reserve(json.size());
  for (const Json& elem : json) {
    ASSIGN_OR_RETURN(Piece piece, Piece::FromJson(elem));
    pieces.push_back(std::move(piece));
  }
  return Template(std::move(pieces));
}

Json Template::ToJson() const {
  Json json = Json::array();
  for (const Piece& piece : pieces_) {
    json.push_back(piece.ToJson());
  }
  return json;
}

std::string Template::DebugString() const {
  std::vector<std::string> formatted;
  formatted.reserve(pieces_.size());
  for (const Piece& piece : pieces_) {
    formatted.push_back(piece.DebugString());
  }
  return absl::StrCat("Template([", absl::StrJoin(formatted, ", "), "])");
}

Tokens::Tokens(std::vector<SpecialToken> v) {
  map_.reserve(v.size());
  for (SpecialToken& token : v) {
    std::string key = token.id();
    map_.insert_or_assign(std::move(key), std::move(token));
  }
}

absl::StatusOr<Tokens> Tokens::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireObject(json, "a map"));
  absl::flat_hash_map<std::string, SpecialToken> map;
  map.reserve(json.size());
  for (const auto& [key, val] : json.items()) {
    ASSIGN_OR_RETURN(SpecialToken token, SpecialToken::FromJson(val));
    map.insert_or_assign(key, std::move(token));
  }
  return Tokens(std::move(map));
}

Json Tokens::ToJson() const {
  // `#[serde(serialize_with = "crate::utils::ordered_map")]`: `BTreeMap` key
  // order (lexicographical).
  std::vector<const std::pair<const std::string, SpecialToken>*> entries;
  entries.reserve(map_.size());
  for (const auto& entry : map_) {
    entries.push_back(&entry);
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto* a, const auto* b) { return a->first < b->first; });
  Json json = Json::object();
  for (const auto* entry : entries) {
    json[entry->first] = entry->second.ToJson();
  }
  return json;
}

size_t CountAdded(const Template& container, const Tokens* special_tokens) {
  size_t sum = 0;
  for (const Piece& p : container.pieces()) {
    if (p.IsSpecialToken() && special_tokens != nullptr) {
      auto it = special_tokens->map().find(p.special_token_id());
      if (it != special_tokens->map().end()) {
        sum += it->second.ids().size();
      }
    }
  }
  return sum;
}

absl::Status TemplateProcessingBuilder::TrySingle(absl::string_view single) {
  ASSIGN_OR_RETURN(Template t, Template::TryFrom(single));
  single_ = std::move(t);
  return absl::OkStatus();
}

absl::Status TemplateProcessingBuilder::TrySingle(
    const std::vector<std::string>& single) {
  ASSIGN_OR_RETURN(Template t, Template::TryFrom(single));
  single_ = std::move(t);
  return absl::OkStatus();
}

absl::Status TemplateProcessingBuilder::TryPair(absl::string_view pair) {
  ASSIGN_OR_RETURN(Template t, Template::TryFrom(pair));
  pair_ = std::move(t);
  return absl::OkStatus();
}

absl::Status TemplateProcessingBuilder::TryPair(
    const std::vector<std::string>& pair) {
  ASSIGN_OR_RETURN(Template t, Template::TryFrom(pair));
  pair_ = std::move(t);
  return absl::OkStatus();
}

size_t TemplateProcessingBuilder::DefaultAdded(bool is_single) const {
  const std::optional<Template>& container = is_single ? single_ : pair_;
  if (!container.has_value()) return 0;
  const Tokens* spt = special_tokens_.has_value() ? &*special_tokens_ : nullptr;
  return CountAdded(*container, spt);
}

absl::Status TemplateProcessingBuilder::Validate() const {
  bool pair_has_both = true;
  if (pair_.has_value()) {
    bool has_a = false;
    bool has_b = false;
    for (const Piece& piece : pair_->pieces()) {
      if (piece.IsSequence() && piece.sequence_id() == SequenceId::kA) {
        has_a = true;
      }
      if (piece.IsSequence() && piece.sequence_id() == SequenceId::kB) {
        has_b = true;
      }
    }
    pair_has_both = has_a && has_b;
  }
  if (!pair_has_both) {
    return absl::InvalidArgumentError(
        "Template for `pair` must use both sequences");
  }

  auto check =
      [this](absl::string_view sp) -> std::optional<absl::string_view> {
    const bool exist =
        special_tokens_.has_value() && special_tokens_->map().contains(sp);
    if (!exist) return sp;
    return std::nullopt;
  };

  // PORT NOTE: Rust collects missing special token IDs into an `AHashSet<&str>`
  // and formats `missing.iter().join(", ")`; deduplicating in encounter order
  // produces one of the valid `AHashSet` iteration orders deterministically.
  std::vector<absl::string_view> missing;
  absl::flat_hash_set<absl::string_view> seen;
  auto collect_missing = [&](const std::optional<Template>& tmpl) {
    if (!tmpl.has_value()) return;
    for (const Piece& piece : tmpl->pieces()) {
      if (piece.IsSpecialToken()) {
        if (std::optional<absl::string_view> sp =
                check(piece.special_token_id());
            sp.has_value() && seen.insert(*sp).second) {
          missing.push_back(*sp);
        }
      }
    }
  };
  collect_missing(single_);
  collect_missing(pair_);

  if (missing.empty()) {
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing SpecialToken(s) with id(s) `",
                     absl::StrJoin(missing, ", "), "`"));
  }
}

absl::StatusOr<TemplateProcessing> TemplateProcessingBuilder::Build() const {
  RETURN_IF_ERROR(Validate());
  Template single = single_.has_value()
                        ? *single_
                        : Template({Piece::Sequence(SequenceId::kA, 0)});
  Template pair = pair_.has_value()
                      ? *pair_
                      : Template({Piece::Sequence(SequenceId::kA, 0),
                                  Piece::Sequence(SequenceId::kB, 1)});
  const size_t added_single = DefaultAdded(true);
  const size_t added_pair = DefaultAdded(false);
  Tokens special_tokens =
      special_tokens_.has_value() ? *special_tokens_ : Tokens();
  return TemplateProcessing(std::move(single), std::move(pair), added_single,
                            added_pair, std::move(special_tokens));
}

TemplateProcessing::TemplateProcessing()
    : single_({Piece::Sequence(SequenceId::kA, 0)}),
      pair_({Piece::Sequence(SequenceId::kA, 1)}),
      added_single_(0),
      added_pair_(0),
      special_tokens_() {}

TemplateProcessingBuilder TemplateProcessing::Builder() {
  return TemplateProcessingBuilder();
}

absl::StatusOr<std::vector<Encoding>> TemplateProcessing::ApplyTemplate(
    const std::vector<Piece>& pieces, std::vector<Encoding> encodings,
    bool add_special_tokens) const {
  std::vector<Encoding> final_encodings;
  for (const Piece& piece : pieces) {
    if (piece.IsSequence()) {
      const size_t i = piece.sequence_id() != SequenceId::kA ? 1 : 0;
      // PORT NOTE: `&mut encodings[i]` panics in Rust if a single-sequence
      // template references sequence B (`i == 1`); return an error instead.
      if (i >= encodings.size()) {
        return ProcessorError::InvalidEncodingsVecLength();
      }
      Encoding& encoding = encodings[i];
      encoding.SetTypeIds(
          std::vector<uint32_t>(encoding.Len(), piece.type_id()));
      encoding.SetSequenceId(i);
      final_encodings.push_back(encoding);
    } else {
      if (add_special_tokens) {
        // PORT NOTE: `&self.special_tokens.0[id]` panics in Rust if `id` is
        // absent (possible when constructed via `FromJson` or `SetSingle`
        // without builder validation); return an error instead.
        auto it = special_tokens_.map().find(piece.special_token_id());
        if (it == special_tokens_.map().end()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Missing SpecialToken(s) with id(s) `",
                           piece.special_token_id(), "`"));
        }
        const SpecialToken& tok = it->second;
        const size_t len = tok.ids().size();

        Encoding encoding(
            tok.ids(), std::vector<uint32_t>(len, piece.type_id()),
            tok.tokens(),
            // words
            std::vector<std::optional<uint32_t>>(len, std::nullopt),
            // offsets
            std::vector<Offsets>(len, Offsets(0, 0)),
            // special_tokens_mask
            std::vector<uint32_t>(len, 1),
            // attention_mask
            std::vector<uint32_t>(len, 1),
            // overflowing
            {},
            // sequence_range
            {});
        final_encodings.push_back(std::move(encoding));
      }
    }
  }
  return final_encodings;
}

size_t TemplateProcessing::AddedTokens(bool is_pair) const {
  if (is_pair) {
    return added_pair_;
  } else {
    return added_single_;
  }
}

absl::StatusOr<std::vector<Encoding>> TemplateProcessing::ProcessEncodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {
  // PORT NOTE: the `_ => todo!()` arm in Rust (and the commented-out match
  // immediately above it in `template.rs`) is replaced by returning
  // `ProcessorError::InvalidEncodingsVecLength`.
  const std::vector<Piece>* tmpl = nullptr;
  switch (encodings.size()) {
    case 2:
      tmpl = &pair_.pieces();
      break;
    case 1:
      tmpl = &single_.pieces();
      break;
    default:
      return ProcessorError::InvalidEncodingsVecLength();
  }
  return ApplyTemplate(*tmpl, std::move(encodings), add_special_tokens);
}

absl::StatusOr<TemplateProcessing> TemplateProcessing::FromJson(
    const Json& json) {
  // `#[serde(tag = "type", from = "TemplateProcessingDeserializer")]`:
  // `TemplateProcessingDeserializer` is a plain `#[serde(tag = "type")]`
  // struct (serde does not validate the `type` tag when deserializing), and
  // `From<TemplateProcessingDeserializer>` recomputes `added_single` and
  // `added_pair`.
  RETURN_IF_ERROR(RequireObject(json, "struct TemplateProcessingDeserializer"));
  const Json* single_json = FindField(json, "single");
  if (single_json == nullptr) return MissingFieldError("single");
  ASSIGN_OR_RETURN(Template single, Template::FromJson(*single_json));
  const Json* pair_json = FindField(json, "pair");
  if (pair_json == nullptr) return MissingFieldError("pair");
  ASSIGN_OR_RETURN(Template pair, Template::FromJson(*pair_json));
  const Json* special_tokens_json = FindField(json, "special_tokens");
  if (special_tokens_json == nullptr) {
    return MissingFieldError("special_tokens");
  }
  ASSIGN_OR_RETURN(Tokens special_tokens,
                   Tokens::FromJson(*special_tokens_json));
  const size_t added_single = CountAdded(single, &special_tokens);
  const size_t added_pair = CountAdded(pair, &special_tokens);
  return TemplateProcessing(std::move(single), std::move(pair), added_single,
                            added_pair, std::move(special_tokens));
}

Json TemplateProcessing::ToJson() const {
  Json json = Json::object();
  json["type"] = "TemplateProcessing";
  json["single"] = single_.ToJson();
  json["pair"] = pair_.ToJson();
  json["special_tokens"] = special_tokens_.ToJson();
  return json;
}

}  // namespace processors
}  // namespace tokenizers
