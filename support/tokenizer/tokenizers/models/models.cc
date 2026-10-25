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

#include "support/tokenizer/tokenizers/models/models.h"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/models/unigram/model.h"
#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::models {

// --- std / serde emulation -------------------------------------------------

absl::StatusOr<std::string> ReadToString(absl::string_view path) {
  std::ifstream file(std::string(path), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("failed to open `", path, "` (os error ", errno, ")"));
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (file.bad()) {
    return absl::InternalError(absl::StrCat("failed to read `", path, "`"));
  }
  return content;
}

absl::StatusOr<std::vector<std::string>> BufReadLines(
    absl::string_view content) {
  if (!IsValidUtf8(content)) {
    return absl::InvalidArgumentError("stream did not contain valid UTF-8");
  }
  std::vector<std::string> lines;
  size_t start = 0;
  while (start < content.size()) {
    const size_t newline = content.find('\n', start);
    absl::string_view line;
    if (newline == absl::string_view::npos) {
      line = content.substr(start);
      start = content.size();
    } else {
      line = content.substr(start, newline - start);
      start = newline + 1;
      // `lines()` only strips a '\r' that precedes the '\n'.
      if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
      }
    }
    lines.emplace_back(line);
  }
  return lines;
}

std::string SerdeUnexpected(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null:
      // serde_json prints `Unexpected::Unit` as "null".
      return "null";
    case Json::value_t::boolean:
      return absl::StrCat("boolean `", value.get<bool>() ? "true" : "false",
                          "`");
    case Json::value_t::number_unsigned:
      return absl::StrCat("integer `", value.get<uint64_t>(), "`");
    case Json::value_t::number_integer:
      return absl::StrCat("integer `", value.get<int64_t>(), "`");
    case Json::value_t::number_float:
      return absl::StrCat("floating point `", value.get<double>(), "`");
    case Json::value_t::string:
      return absl::StrCat(
          "string \"",
          absl::StrReplaceAll(value.get_ref<const std::string&>(),
                              {{"\\", "\\\\"}, {"\"", "\\\""}}),
          "\"");
    case Json::value_t::array:
      return "sequence";
    case Json::value_t::object:
      return "map";
    default:
      return "unknown";
  }
}

absl::Status SerdeInvalidType(const Json& value, absl::string_view expected) {
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid type: ", SerdeUnexpected(value), ", expected ", expected));
}

absl::Status SerdeInvalidValue(const Json& value, absl::string_view expected) {
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid value: ", SerdeUnexpected(value), ", expected ", expected));
}

absl::Status SerdeMissingField(absl::string_view field) {
  return absl::InvalidArgumentError(
      absl::StrCat("missing field `", field, "`"));
}

absl::StatusOr<std::string> StringFromJson(const Json& value) {
  if (!value.is_string()) {
    return SerdeInvalidType(value, "a string");
  }
  return value.get<std::string>();
}

absl::StatusOr<bool> BoolFromJson(const Json& value) {
  if (!value.is_boolean()) {
    return SerdeInvalidType(value, "a boolean");
  }
  return value.get<bool>();
}

namespace {

// Integer deserialization shared by `u32` / `usize`: serde rejects floats
// with `invalid_type` and out-of-range integers with `invalid_value`.
template <typename T>
absl::StatusOr<T> UnsignedFromJson(const Json& value,
                                   absl::string_view expected) {
  if (value.is_number_unsigned()) {
    const uint64_t v = value.get<uint64_t>();
    if (v > std::numeric_limits<T>::max()) {
      return SerdeInvalidValue(value, expected);
    }
    return static_cast<T>(v);
  }
  if (value.is_number_integer()) {
    return SerdeInvalidValue(value, expected);
  }
  return SerdeInvalidType(value, expected);
}

}  // namespace

absl::StatusOr<uint32_t> U32FromJson(const Json& value) {
  return UnsignedFromJson<uint32_t>(value, "u32");
}

absl::StatusOr<size_t> UsizeFromJson(const Json& value) {
  return UnsignedFromJson<size_t>(value, "usize");
}

absl::StatusOr<float> F32FromJson(const Json& value) {
  if (!value.is_number()) {
    return SerdeInvalidType(value, "f32");
  }
  return static_cast<float>(value.get<double>());
}

absl::StatusOr<double> F64FromJson(const Json& value) {
  if (!value.is_number()) {
    return SerdeInvalidType(value, "f64");
  }
  return value.get<double>();
}

absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> VocabFromJson(
    const Json& value) {
  if (!value.is_object()) {
    return SerdeInvalidType(value, "a map");
  }
  absl::flat_hash_map<std::string, uint32_t> vocab;
  vocab.reserve(value.size());
  for (const auto& [token, id] : value.items()) {
    ASSIGN_OR_RETURN(const uint32_t id_value, U32FromJson(id));
    vocab[token] = id_value;
  }
  return vocab;
}

Json F32ToJson(float value) {
  if (!std::isfinite(value)) {
    // serde_json serializes non-finite floats as `null`.
    return Json(nullptr);
  }
  // Shortest decimal representation that round-trips through `f32`.
  std::string repr;
  for (int precision = 1; precision <= 9; ++precision) {
    repr = absl::StrFormat("%.*g", precision, value);
    if (std::strtof(repr.c_str(), nullptr) == value) {
      break;
    }
  }
  double as_double = value;
  (void)absl::SimpleAtod(repr, &as_double);
  return Json(as_double);
}

// --- OrderedVocabIter -------------------------------------------------------

Json OrderedVocabIter::Serialize() const {
  // There could be holes so max + 1 is more correct than vocab_r.len()
  std::vector<uint32_t> holes;
  Json result = Json::object();
  if (!vocab_r_->empty()) {
    uint32_t max = 0;
    for (const auto& [id, token] : *vocab_r_) {
      if (id > max) {
        max = id;
      }
    }
    for (uint64_t i = 0; i < static_cast<uint64_t>(max) + 1; ++i) {
      auto token = vocab_r_->find(static_cast<uint32_t>(i));
      if (token != vocab_r_->end()) {
        result[token->second] = static_cast<uint32_t>(i);
      } else {
        holes.push_back(static_cast<uint32_t>(i));
      }
    }
  }

  if (!holes.empty()) {
    ABSL_LOG(WARNING) << "The OrderedVocab you are attempting to serialize "
                         "contains holes for indices ["
                      << absl::StrJoin(holes, ", ")
                      << "], your vocabulary could be corrupted!";
  }
  return result;
}

// --- ModelWrapper -----------------------------------------------------------

absl::StatusOr<ModelWrapper> ModelWrapper::FromJson(const Json& json) {
  // `ModelHelper::Tagged`: an object whose "type" is one of the `EnumType`
  // variants. The `#[serde(flatten)] rest` is the whole object minus "type";
  // every model deserializer accepts (and checks) its own "type" field, so
  // the object is forwarded as is. Errors of the tagged variant are final.
  if (json.is_object()) {
    auto type = json.find("type");
    if (type != json.end() && type->is_string()) {
      const std::string& variant = type->get_ref<const std::string&>();
      if (variant == "BPE") {
        ASSIGN_OR_RETURN(bpe::BPE model, bpe::BPE::FromJson(json));
        return ModelWrapper(std::move(model));
      }
      if (variant == "WordPiece") {
        ASSIGN_OR_RETURN(wordpiece::WordPiece model,
                         wordpiece::WordPiece::FromJson(json));
        return ModelWrapper(std::move(model));
      }
      if (variant == "WordLevel") {
        ASSIGN_OR_RETURN(wordlevel::WordLevel model,
                         wordlevel::WordLevel::FromJson(json));
        return ModelWrapper(std::move(model));
      }
      if (variant == "Unigram") {
        ASSIGN_OR_RETURN(unigram::Unigram model,
                         unigram::Unigram::FromJson(json));
        return ModelWrapper(std::move(model));
      }
    }
  }
  // `ModelHelper::Legacy(value)` -> `ModelUntagged`: try every variant in
  // declaration order.
  if (absl::StatusOr<bpe::BPE> model = bpe::BPE::FromJson(json); model.ok()) {
    return ModelWrapper(*std::move(model));
  }
  // WordPiece must stay before WordLevel here for deserialization (for
  // retrocompatibility with the versions not including the "type"), since
  // WordLevel is a subset of WordPiece
  if (absl::StatusOr<wordpiece::WordPiece> model =
          wordpiece::WordPiece::FromJson(json);
      model.ok()) {
    return ModelWrapper(*std::move(model));
  }
  if (absl::StatusOr<wordlevel::WordLevel> model =
          wordlevel::WordLevel::FromJson(json);
      model.ok()) {
    return ModelWrapper(*std::move(model));
  }
  if (absl::StatusOr<unigram::Unigram> model = unigram::Unigram::FromJson(json);
      model.ok()) {
    return ModelWrapper(*std::move(model));
  }
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum ModelUntagged");
}

Json ModelWrapper::ToJson() const {
  return std::visit([](const auto& model) { return model.ToJson(); }, model_);
}

absl::StatusOr<std::vector<Token>> ModelWrapper::Tokenize(
    absl::string_view tokens) const {
  return std::visit([tokens](const auto& t) { return t.Tokenize(tokens); },
                    model_);
}

std::optional<uint32_t> ModelWrapper::TokenToId(absl::string_view token) const {
  return std::visit([token](const auto& t) { return t.TokenToId(token); },
                    model_);
}

std::optional<std::string> ModelWrapper::IdToToken(uint32_t id) const {
  return std::visit([id](const auto& t) { return t.IdToToken(id); }, model_);
}

absl::flat_hash_map<std::string, uint32_t> ModelWrapper::GetVocab() const {
  return std::visit([](const auto& t) { return t.GetVocab(); }, model_);
}

size_t ModelWrapper::GetVocabSize() const {
  return std::visit([](const auto& t) { return t.GetVocabSize(); }, model_);
}

void ModelWrapper::ClearCache() {
  if (auto* model = std::get_if<unigram::Unigram>(&model_)) {
    model->ClearCache();
  } else if (auto* model = std::get_if<bpe::BPE>(&model_)) {
    model->ClearCache();
  }
}

void ModelWrapper::ResizeCache(size_t capacity) {
  if (auto* model = std::get_if<unigram::Unigram>(&model_)) {
    model->ResizeCache(capacity);
  } else if (auto* model = std::get_if<bpe::BPE>(&model_)) {
    model->ResizeCache(capacity);
  }
}

bool ModelWrapper::operator==(const ModelWrapper& other) const {
  return model_ == other.model_;
}

}  // namespace tokenizers::models
