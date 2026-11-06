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
// Support code for the C++ port of the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team); see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/utils/json.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace tokenizers {
namespace {

// A SAX handler that builds the document like `Json::parse` does but records
// the message of a syntax error instead of throwing (or silently returning a
// discarded value).
class ParseHandler : public nlohmann::json_sax<Json> {
 public:
  explicit ParseHandler(Json* result)
      : dom_(*result, /*allow_exceptions_=*/false) {}

  bool null() override { return dom_.null(); }
  bool boolean(bool val) override { return dom_.boolean(val); }
  bool number_integer(number_integer_t val) override {
    return dom_.number_integer(val);
  }
  bool number_unsigned(number_unsigned_t val) override {
    return dom_.number_unsigned(val);
  }
  bool number_float(number_float_t val, const string_t& s) override {
    return dom_.number_float(val, s);
  }
  bool string(string_t& val) override { return dom_.string(val); }
  bool binary(binary_t& val) override { return dom_.binary(val); }
  bool start_object(std::size_t elements) override {
    return dom_.start_object(elements);
  }
  bool key(string_t& val) override { return dom_.key(val); }
  bool end_object() override { return dom_.end_object(); }
  bool start_array(std::size_t elements) override {
    return dom_.start_array(elements);
  }
  bool end_array() override { return dom_.end_array(); }
  bool parse_error(std::size_t /*position*/, const std::string& /*last_token*/,
                   const nlohmann::detail::exception& ex) override {
    error_ = ex.what();
    return false;
  }

  const std::string& error() const { return error_; }

 private:
  nlohmann::detail::json_sax_dom_parser<
      Json, nlohmann::detail::string_input_adapter_type>
      dom_;
  std::string error_;
};

// `{:?}` of a Rust `&str`: double-quoted, with `"`, `\` and control
// characters escaped.
std::string RustDebugString(absl::string_view s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\0':
        out += "\\0";
        break;
      default:
        if (c < 0x20 || c == 0x7F) {
          absl::StrAppendFormat(&out, "\\u{%x}", c);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
  return out;
}

// `OneOf { names }` of serde: "`a`" / "`a` or `b`" / "one of `a`, `b`, `c`".
std::string OneOf(absl::Span<const absl::string_view> names) {
  if (names.size() == 1) return absl::StrCat("`", names[0], "`");
  if (names.size() == 2) {
    return absl::StrCat("`", names[0], "` or `", names[1], "`");
  }
  std::string out = "one of ";
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out += ", ";
    absl::StrAppend(&out, "`", names[i], "`");
  }
  return out;
}

}  // namespace

absl::StatusOr<Json> ParseJson(absl::string_view text) {
  Json result;
  ParseHandler handler(&result);
  // Pointers are iterators; `strict` rejects trailing non-whitespace like
  // serde_json does.
  if (!Json::sax_parse(text.data(), text.data() + text.size(), &handler,
                       nlohmann::detail::input_format_t::json,
                       /*strict=*/true) ||
      result.is_discarded()) {
    return absl::InvalidArgumentError(
        handler.error().empty() ? "invalid JSON" : handler.error());
  }
  return result;
}

std::string DumpJson(const Json& value, int indent) {
  return value.dump(indent, ' ', /*ensure_ascii=*/false,
                    Json::error_handler_t::replace);
}

bool JsonEqualIgnoringKeyOrder(const Json& a, const Json& b) {
  if (a.is_object() && b.is_object()) {
    if (a.size() != b.size()) return false;
    for (const auto& [key, value] : a.items()) {
      const Json* other = FindField(b, key);
      if (other == nullptr || !JsonEqualIgnoringKeyOrder(value, *other)) {
        return false;
      }
    }
    return true;
  }
  if (a.is_array() && b.is_array()) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!JsonEqualIgnoringKeyOrder(a[i], b[i])) return false;
    }
    return true;
  }
  return a == b;
}

std::string JsonUnexpected(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null:
      return "null";
    case Json::value_t::boolean:
      return absl::StrCat("boolean `", value.get<bool>() ? "true" : "false",
                          "`");
    case Json::value_t::number_integer:
      return absl::StrCat("integer `", value.get<int64_t>(), "`");
    case Json::value_t::number_unsigned:
      return absl::StrCat("integer `", value.get<uint64_t>(), "`");
    case Json::value_t::number_float: {
      // serde_json prints floats with ryu ("1.0", "1.5", "1e20").
      // PORT NOTE: approximated with the shortest round-trip representation
      // plus a ".0" for integral values; exponents may be spelled differently.
      const double d = value.get<double>();
      std::string text = absl::StrCat(d);
      if (std::isfinite(d) && text.find_first_of(".e") == std::string::npos) {
        text += ".0";
      }
      return absl::StrCat("floating point `", text, "`");
    }
    case Json::value_t::string:
      return absl::StrCat("string ",
                          RustDebugString(value.get_ref<const std::string&>()));
    case Json::value_t::array:
      return "sequence";
    case Json::value_t::object:
      return "map";
    case Json::value_t::binary:
      return "byte array";
    case Json::value_t::discarded:
      break;
  }
  return "unit value";
}

absl::Status JsonError(absl::string_view message) {
  return absl::InvalidArgumentError(message);
}

absl::Status InvalidTypeError(const Json& unexpected,
                              absl::string_view expected) {
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid type: ", JsonUnexpected(unexpected), ", expected ", expected));
}

absl::Status InvalidValueError(const Json& unexpected,
                               absl::string_view expected) {
  return absl::InvalidArgumentError(absl::StrCat(
      "invalid value: ", JsonUnexpected(unexpected), ", expected ", expected));
}

absl::Status InvalidValueError(absl::string_view unexpected,
                               absl::string_view expected) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid value: ", unexpected, ", expected ", expected));
}

absl::Status InvalidLengthError(size_t length, absl::string_view expected) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid length ", length, ", expected ", expected));
}

absl::Status MissingFieldError(absl::string_view field) {
  return absl::InvalidArgumentError(
      absl::StrCat("missing field `", field, "`"));
}

absl::Status DuplicateFieldError(absl::string_view field) {
  return absl::InvalidArgumentError(
      absl::StrCat("duplicate field `", field, "`"));
}

absl::Status UnknownFieldError(absl::string_view field,
                               absl::Span<const absl::string_view> expected) {
  if (expected.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("unknown field `", field, "`, there are no fields"));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown field `", field, "`, expected ", OneOf(expected)));
}

absl::Status UnknownVariantError(absl::string_view variant,
                                 absl::Span<const absl::string_view> expected) {
  if (expected.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("unknown variant `", variant, "`, there are no variants"));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "unknown variant `", variant, "`, expected ", OneOf(expected)));
}

absl::Status RequireObject(const Json& value, absl::string_view expected) {
  if (!value.is_object()) return InvalidTypeError(value, expected);
  return absl::OkStatus();
}

absl::Status RequireArray(const Json& value, absl::string_view expected) {
  if (!value.is_array()) return InvalidTypeError(value, expected);
  return absl::OkStatus();
}

absl::Status DenyUnknownFields(
    const Json& object, absl::Span<const absl::string_view> known_fields) {
  if (!object.is_object()) return absl::OkStatus();
  for (const auto& [key, value] : object.items()) {
    bool known = false;
    for (absl::string_view field : known_fields) {
      if (field == key) {
        known = true;
        break;
      }
    }
    if (!known) return UnknownFieldError(key, known_fields);
  }
  return absl::OkStatus();
}

absl::Status RequireTypeTag(const Json& json, absl::string_view expected) {
  if (!json.is_object()) {
    return InvalidTypeError(json, absl::StrCat("struct ", expected));
  }
  const Json* type = FindField(json, "type");
  if (type == nullptr) return MissingFieldError("type");
  if (!type->is_string()) return InvalidTypeError(*type, "variant identifier");
  const std::string& got = type->get_ref<const std::string&>();
  if (got != expected) return UnknownVariantError(got, {expected});
  return absl::OkStatus();
}

const Json* FindField(const Json& object, absl::string_view name) {
  if (!object.is_object()) return nullptr;
  auto it = object.find(name);
  return it == object.end() ? nullptr : &*it;
}

absl::StatusOr<const Json*> GetArrayField(const Json& object,
                                          absl::string_view name) {
  const Json* field = FindField(object, name);
  if (field == nullptr) return MissingFieldError(name);
  if (!field->is_array()) return InvalidTypeError(*field, "a sequence");
  return field;
}

absl::StatusOr<const Json*> GetObjectField(const Json& object,
                                           absl::string_view name) {
  const Json* field = FindField(object, name);
  if (field == nullptr) return MissingFieldError(name);
  if (!field->is_object()) return InvalidTypeError(*field, "a map");
  return field;
}

}  // namespace tokenizers
