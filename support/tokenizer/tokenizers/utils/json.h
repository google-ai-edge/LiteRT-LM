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
//
// The crate (de)serializes everything with `serde` / `serde_json`. The port
// uses nlohmann/json for the JSON document model and emulates the behaviour of
// the serde derives by hand in each type's `FromJson` / `ToJson`. This header
// provides:
//   * `Json`: the document type. An insertion-ordered object type (like
//     `nlohmann::ordered_json`, so that `ToJson` reproduces serde's field
//     order) with a hash index, because `ordered_json` looks keys up linearly
//     and parsing a 100k-entry vocabulary object with it is quadratic.
//   * `ParseJson` / `DumpJson`: non-throwing parse and serialization
//     (nlohmann is built without exceptions here; a thrown exception aborts).
//   * The `serde::de::Error` constructors with serde's exact messages
//     (`MissingFieldError`, `UnknownFieldError`, `InvalidTypeError`, ...).
//   * Typed accessors (`JsonAs<T>`, `GetField<T>`, `GetOptionalField<T>`,
//     ...) that apply serde's conversion rules (`u32` rejects negative and
//     floating point numbers, `f64` accepts integers, `Option<T>` treats `null`
//     and a missing field as `None`, ...).
//
// Never index a `Json` with `operator[]`/`at()` on a key that may be missing
// and never call `get<T>()` without a type check: both abort on error.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_JSON_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_JSON_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "support/tokenizer/tokenizers/utils/utf8.h"

namespace tokenizers {

namespace json_internal {

// The object type of `Json`: `nlohmann::ordered_map` semantics (a vector of
// `(key, value)` pairs in insertion order; duplicate insertions keep the first
// entry) plus a hash index from key to position, so that lookups and inserts
// are O(1) instead of O(n). Only the operations `basic_json` uses are
// provided; the mutating `std::vector` operations that would desynchronize
// the index are hidden.
template <class Key, class T, class IgnoredLess = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
class IndexedOrderedMap
    : public std::vector<std::pair<const Key, T>, Allocator> {
 public:
  using key_type = Key;
  using mapped_type = T;
  using Container = std::vector<std::pair<const Key, T>, Allocator>;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;
  using size_type = typename Container::size_type;
  using value_type = typename Container::value_type;
  using key_compare = std::equal_to<>;

  IndexedOrderedMap() = default;
  explicit IndexedOrderedMap(const Allocator& alloc) : Container(alloc) {}
  template <class It>
  IndexedOrderedMap(It first, It last, const Allocator& alloc = Allocator())
      : Container(alloc) {
    insert(first, last);
  }
  IndexedOrderedMap(std::initializer_list<value_type> init,
                    const Allocator& alloc = Allocator())
      : Container(alloc) {
    insert(init.begin(), init.end());
  }
  IndexedOrderedMap(const IndexedOrderedMap&) = default;
  IndexedOrderedMap(IndexedOrderedMap&&) = default;
  IndexedOrderedMap& operator=(const IndexedOrderedMap&) = default;
  IndexedOrderedMap& operator=(IndexedOrderedMap&&) = default;

  template <class KeyType>
  using enable_if_key =
      nlohmann::detail::enable_if_t<nlohmann::detail::is_usable_as_key_type<
                                        key_compare, key_type, KeyType>::value,
                                    int>;

  // --- Lookup -------------------------------------------------------------

  template <class KeyType, enable_if_key<KeyType> = 0>
  iterator find(KeyType&& key) {
    auto it = index_.find(key);
    return it == index_.end() ? Container::end()
                              : Container::begin() + it->second;
  }
  template <class KeyType, enable_if_key<KeyType> = 0>
  const_iterator find(KeyType&& key) const {
    auto it = index_.find(key);
    return it == index_.end() ? Container::end()
                              : Container::begin() + it->second;
  }
  iterator find(const key_type& key) {
    auto it = index_.find(key);
    return it == index_.end() ? Container::end()
                              : Container::begin() + it->second;
  }
  const_iterator find(const key_type& key) const {
    auto it = index_.find(key);
    return it == index_.end() ? Container::end()
                              : Container::begin() + it->second;
  }

  template <class KeyType, enable_if_key<KeyType> = 0>
  size_type count(KeyType&& key) const {
    return index_.contains(key) ? 1 : 0;
  }
  size_type count(const key_type& key) const {
    return index_.contains(key) ? 1 : 0;
  }

  // Like `nlohmann::ordered_map::at`, a missing key is a `std::out_of_range`
  // (which, without exceptions, aborts). `basic_json::at` does its own lookup
  // and never calls this; prefer `find`.
  template <class KeyType, enable_if_key<KeyType> = 0>
  T& at(KeyType&& key) {
    auto it = find(std::forward<KeyType>(key));
    if (it == Container::end()) KeyNotFound();
    return it->second;
  }
  template <class KeyType, enable_if_key<KeyType> = 0>
  const T& at(KeyType&& key) const {
    auto it = find(std::forward<KeyType>(key));
    if (it == Container::end()) KeyNotFound();
    return it->second;
  }
  T& at(const key_type& key) {
    auto it = find(key);
    if (it == Container::end()) KeyNotFound();
    return it->second;
  }
  const T& at(const key_type& key) const {
    auto it = find(key);
    if (it == Container::end()) KeyNotFound();
    return it->second;
  }

  template <class KeyType, enable_if_key<KeyType> = 0>
  T& operator[](KeyType&& key) {
    return emplace(std::forward<KeyType>(key), T{}).first->second;
  }
  T& operator[](const key_type& key) { return emplace(key, T{}).first->second; }
  template <class KeyType, enable_if_key<KeyType> = 0>
  const T& operator[](KeyType&& key) const {
    return at(std::forward<KeyType>(key));
  }
  const T& operator[](const key_type& key) const { return at(key); }

  // --- Modifiers ----------------------------------------------------------

  template <class KeyType, enable_if_key<KeyType> = 0>
  std::pair<iterator, bool> emplace(KeyType&& key, T&& t) {
    auto it = index_.find(key);
    if (it != index_.end()) {
      return {Container::begin() + it->second, false};
    }
    Container::emplace_back(std::forward<KeyType>(key), std::forward<T>(t));
    index_.emplace(Container::back().first, Container::size() - 1);
    return {std::prev(Container::end()), true};
  }
  std::pair<iterator, bool> emplace(const key_type& key, T&& t) {
    auto it = index_.find(key);
    if (it != index_.end()) {
      return {Container::begin() + it->second, false};
    }
    Container::emplace_back(key, std::forward<T>(t));
    index_.emplace(key, Container::size() - 1);
    return {std::prev(Container::end()), true};
  }

  std::pair<iterator, bool> insert(value_type&& value) {
    return emplace(value.first, std::move(value.second));
  }
  std::pair<iterator, bool> insert(const value_type& value) {
    auto it = index_.find(value.first);
    if (it != index_.end()) {
      return {Container::begin() + it->second, false};
    }
    Container::push_back(value);
    index_.emplace(value.first, Container::size() - 1);
    return {std::prev(Container::end()), true};
  }
  template <class InputIt,
            class = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<InputIt>::iterator_category,
                std::input_iterator_tag>::value>>
  void insert(InputIt first, InputIt last) {
    for (auto it = first; it != last; ++it) {
      insert(*it);
    }
  }

  template <class KeyType, enable_if_key<KeyType> = 0>
  size_type erase(KeyType&& key) {
    auto it = find(std::forward<KeyType>(key));
    if (it == Container::end()) return 0;
    erase(it, std::next(it));
    return 1;
  }
  size_type erase(const key_type& key) {
    auto it = find(key);
    if (it == Container::end()) return 0;
    erase(it, std::next(it));
    return 1;
  }
  iterator erase(iterator pos) { return erase(pos, std::next(pos)); }
  iterator erase(iterator first, iterator last) {
    if (first == last) return first;
    const auto elements_affected = std::distance(first, last);
    const auto offset = std::distance(Container::begin(), first);
    for (auto it = first; it != last; ++it) {
      index_.erase(it->first);
    }
    // The keys are const, so the elements after the erased range cannot be
    // move-assigned over them: re-construct them in place instead.
    for (auto it = first; std::next(it, elements_affected) != Container::end();
         ++it) {
      it->~value_type();
      new (&*it) value_type{std::move(*std::next(it, elements_affected))};
    }
    Container::resize(Container::size() -
                      static_cast<size_type>(elements_affected));
    for (size_type i = static_cast<size_type>(offset); i < Container::size();
         ++i) {
      index_[Container::operator[](i).first] = i;
    }
    return Container::begin() + offset;
  }

  void clear() {
    Container::clear();
    index_.clear();
  }

  // `std::vector` operations that would bypass the index.
  void push_back(const value_type&) = delete;
  void emplace_back() = delete;
  void pop_back() = delete;
  void resize(size_type) = delete;
  void assign() = delete;

  friend bool operator==(const IndexedOrderedMap& a,
                         const IndexedOrderedMap& b) {
    return static_cast<const Container&>(a) == static_cast<const Container&>(b);
  }
  friend bool operator!=(const IndexedOrderedMap& a,
                         const IndexedOrderedMap& b) {
    return !(a == b);
  }

 private:
  // `std::out_of_range("key not found")` of `nlohmann::ordered_map::at`:
  // thrown when exceptions are enabled, otherwise an abort, exactly like
  // nlohmann's `JSON_THROW`. Unreachable through the helpers of this header.
  [[noreturn]] static void KeyNotFound() {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    throw std::out_of_range("key not found");
#else
    std::abort();
#endif
  }

  absl::flat_hash_map<Key, size_type> index_;
};

}  // namespace json_internal

// The JSON document type of the port. API-identical to `nlohmann::json` /
// `nlohmann::ordered_json` (objects keep insertion order; `operator==` on
// objects is therefore order-sensitive, see `JsonEqualIgnoringKeyOrder`).
using Json = nlohmann::basic_json<json_internal::IndexedOrderedMap>;

// --- Parsing and serialization ----------------------------------------------

// `serde_json::from_str::<Value>(text)`: parses `text`, or returns an
// `absl::InvalidArgumentError` describing the syntax error. Never throws.
absl::StatusOr<Json> ParseJson(absl::string_view text);

// `serde_json::to_string(&value)` (`indent < 0`, compact: no spaces, like
// serde_json) / `serde_json::to_string_pretty` (`indent == 2`). Never throws:
// invalid UTF-8 in strings is replaced by U+FFFD.
std::string DumpJson(const Json& value, int indent = -1);

// Structural equality that ignores the order of object keys (like comparing
// two `serde_json::Value`s). For tests.
bool JsonEqualIgnoringKeyOrder(const Json& a, const Json& b);

// --- serde::de::Error constructors -----------------------------------------
// All deserialization errors are `absl::InvalidArgumentError`s whose message
// is what serde / serde_json would produce (minus the "at line X column Y"
// suffix serde_json adds when deserializing straight from text).

// How serde_json describes an unexpected value (`JsonUnexpected`): "null",
// "boolean `true`", "integer `42`", "floating point `1.5`", "string \"abc\"",
// "sequence", "map".
std::string JsonUnexpected(const Json& value);

// `Error::custom(message)`.
absl::Status JsonError(absl::string_view message);
// `Error::invalid_type(unexpected, &expected)`:
// "invalid type: <unexpected>, expected <expected>".
absl::Status InvalidTypeError(const Json& unexpected,
                              absl::string_view expected);
// `Error::invalid_value(unexpected, &expected)`:
// "invalid value: <unexpected>, expected <expected>".
absl::Status InvalidValueError(const Json& unexpected,
                               absl::string_view expected);
absl::Status InvalidValueError(absl::string_view unexpected,
                               absl::string_view expected);
// `Error::invalid_length(len, &expected)`:
// "invalid length <len>, expected <expected>".
absl::Status InvalidLengthError(size_t length, absl::string_view expected);
// `Error::missing_field(field)`: "missing field `<field>`".
absl::Status MissingFieldError(absl::string_view field);
// `Error::duplicate_field(field)`: "duplicate field `<field>`".
absl::Status DuplicateFieldError(absl::string_view field);
// `Error::unknown_field(field, expected)`: "unknown field `<field>`, expected
// `a`" / "... expected `a` or `b`" / "... expected one of `a`, `b`, `c`" /
// "... there are no fields".
absl::Status UnknownFieldError(absl::string_view field,
                               absl::Span<const absl::string_view> expected);
// `Error::unknown_variant(variant, expected)`: "unknown variant `<variant>`,
// expected ..." with the same `expected` formatting as `UnknownFieldError`.
absl::Status UnknownVariantError(absl::string_view variant,
                                 absl::Span<const absl::string_view> expected);

// --- Structural checks ------------------------------------------------------

// Ok iff `value` is an object; otherwise `InvalidTypeError(value, expected)`
// with e.g. `expected = "struct BPE"` (what a derived struct visitor reports).
absl::Status RequireObject(const Json& value, absl::string_view expected);
// Ok iff `value` is an array; otherwise `InvalidTypeError(value, expected)`
// with e.g. `expected = "a sequence"`.
absl::Status RequireArray(const Json& value, absl::string_view expected);
// `#[serde(deny_unknown_fields)]`: the first key of `object` (in document
// order) that is not in `known_fields` is an `UnknownFieldError`.
absl::Status DenyUnknownFields(
    const Json& object, absl::Span<const absl::string_view> known_fields);
// The `impl_serde_type!` / `#[serde(tag = "type")]` + `MustBe!` check: Ok iff
// `json` is an object whose "type" field is the string `expected`. Errors:
// not an object -> `InvalidTypeError(json, "struct <expected>")`; no "type"
// -> "missing field `type`"; another string -> "unknown variant `<got>`,
// expected `<expected>`"; not a string -> "invalid type: ..., expected variant
// identifier".
absl::Status RequireTypeTag(const Json& json, absl::string_view expected);

// --- Field access -----------------------------------------------------------

// The value of field `name` of `object`, or nullptr if absent. `object` must
// be an object (checked: returns nullptr otherwise).
const Json* FindField(const Json& object, absl::string_view name);

namespace json_internal {

// Conversion rules for `JsonAs<T>`: `Convert` applies serde's visitor logic
// for the Rust type `T` stands for and `kExpected` is that visitor's
// `Expected` text.
template <typename T, typename Enable = void>
struct Converter;  // Unsupported type.

template <>
struct Converter<bool> {
  static constexpr absl::string_view kExpected = "a boolean";
  static absl::StatusOr<bool> Convert(const Json& value) {
    if (!value.is_boolean()) return InvalidTypeError(value, kExpected);
    return value.get<bool>();
  }
};

template <>
struct Converter<std::string> {
  static constexpr absl::string_view kExpected = "a string";
  static absl::StatusOr<std::string> Convert(const Json& value) {
    if (!value.is_string()) return InvalidTypeError(value, kExpected);
    return value.get_ref<const std::string&>();
  }
};

// Rust `char`: a string holding exactly one character.
template <>
struct Converter<char32_t> {
  static constexpr absl::string_view kExpected = "a character";
  static absl::StatusOr<char32_t> Convert(const Json& value) {
    if (!value.is_string()) return InvalidTypeError(value, kExpected);
    const std::string& s = value.get_ref<const std::string&>();
    size_t pos = 0;
    if (s.empty()) return InvalidValueError(value, kExpected);
    const char32_t c = DecodeUtf8(s, &pos);
    if (pos != s.size()) return InvalidValueError(value, kExpected);
    return c;
  }
};

// Any JSON value (a `serde_json::Value` field).
template <>
struct Converter<Json> {
  static constexpr absl::string_view kExpected = "any valid JSON value";
  static absl::StatusOr<Json> Convert(const Json& value) { return value; }
};

template <size_t kSize, bool kSigned>
struct IntegerName;
template <>
struct IntegerName<1, false> {
  static constexpr absl::string_view kName = "u8";
};
template <>
struct IntegerName<2, false> {
  static constexpr absl::string_view kName = "u16";
};
template <>
struct IntegerName<4, false> {
  static constexpr absl::string_view kName = "u32";
};
// PORT NOTE: `u64` and `usize` are the same C++ type; the crate only
// deserializes `usize` fields, so that is the name used in messages.
template <>
struct IntegerName<8, false> {
  static constexpr absl::string_view kName = "usize";
};
template <>
struct IntegerName<1, true> {
  static constexpr absl::string_view kName = "i8";
};
template <>
struct IntegerName<2, true> {
  static constexpr absl::string_view kName = "i16";
};
template <>
struct IntegerName<4, true> {
  static constexpr absl::string_view kName = "i32";
};
template <>
struct IntegerName<8, true> {
  static constexpr absl::string_view kName = "i64";
};

// Rust integers: an integral JSON number in range. A floating point number
// is a type error, an out-of-range (e.g. negative for `u32`) integer a value
// error, exactly like serde's primitive visitors.
template <typename T>
struct Converter<
    T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                        !std::is_same_v<T, char32_t>>> {
  static constexpr absl::string_view kExpected =
      IntegerName<sizeof(T), std::is_signed_v<T>>::kName;
  static absl::StatusOr<T> Convert(const Json& value) {
    if (value.is_number_unsigned()) {
      const uint64_t u = value.get<uint64_t>();
      if constexpr (static_cast<uint64_t>(std::numeric_limits<T>::max()) <
                    std::numeric_limits<uint64_t>::max()) {
        if (u > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
          return InvalidValueError(value, kExpected);
        }
      }
      return static_cast<T>(u);
    }
    if (value.is_number_integer()) {
      // nlohmann stores every non-negative integer as unsigned, so this one
      // is negative.
      const int64_t i = value.get<int64_t>();
      if constexpr (std::is_signed_v<T>) {
        if (i < static_cast<int64_t>(std::numeric_limits<T>::min())) {
          return InvalidValueError(value, kExpected);
        }
        return static_cast<T>(i);
      } else {
        return InvalidValueError(value, kExpected);
      }
    }
    return InvalidTypeError(value, kExpected);
  }
};

// Rust `f32` / `f64`: any JSON number (integers are accepted).
template <typename T>
struct Converter<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  static constexpr absl::string_view kExpected =
      sizeof(T) == 4 ? absl::string_view("f32") : absl::string_view("f64");
  static absl::StatusOr<T> Convert(const Json& value) {
    if (!value.is_number()) return InvalidTypeError(value, kExpected);
    return static_cast<T>(value.get<double>());
  }
};

// `Option<T>`: `null` is `None`.
template <typename T>
struct Converter<std::optional<T>> {
  static constexpr absl::string_view kExpected = "option";
  static absl::StatusOr<std::optional<T>> Convert(const Json& value) {
    if (value.is_null()) return std::optional<T>();
    absl::StatusOr<T> inner = Converter<T>::Convert(value);
    if (!inner.ok()) return inner.status();
    return std::optional<T>(*std::move(inner));
  }
};

// `Vec<T>`.
template <typename T>
struct Converter<std::vector<T>> {
  static constexpr absl::string_view kExpected = "a sequence";
  static absl::StatusOr<std::vector<T>> Convert(const Json& value) {
    if (!value.is_array()) return InvalidTypeError(value, kExpected);
    std::vector<T> result;
    result.reserve(value.size());
    for (const Json& element : value) {
      absl::StatusOr<T> converted = Converter<T>::Convert(element);
      if (!converted.ok()) return converted.status();
      result.push_back(*std::move(converted));
    }
    return result;
  }
};

// `(A, B)`: an array of exactly two elements.
template <typename A, typename B>
struct Converter<std::pair<A, B>> {
  static constexpr absl::string_view kExpected = "a tuple of size 2";
  static absl::StatusOr<std::pair<A, B>> Convert(const Json& value) {
    if (!value.is_array()) return InvalidTypeError(value, kExpected);
    if (value.size() != 2) return InvalidLengthError(value.size(), kExpected);
    absl::StatusOr<A> a = Converter<A>::Convert(value[0]);
    if (!a.ok()) return a.status();
    absl::StatusOr<B> b = Converter<B>::Convert(value[1]);
    if (!b.ok()) return b.status();
    return std::pair<A, B>(*std::move(a), *std::move(b));
  }
};

// `HashMap<String, V>`.
template <typename V>
struct Converter<absl::flat_hash_map<std::string, V>> {
  static constexpr absl::string_view kExpected = "a map";
  static absl::StatusOr<absl::flat_hash_map<std::string, V>> Convert(
      const Json& value) {
    if (!value.is_object()) return InvalidTypeError(value, kExpected);
    absl::flat_hash_map<std::string, V> result;
    result.reserve(value.size());
    for (const auto& [key, element] : value.items()) {
      absl::StatusOr<V> converted = Converter<V>::Convert(element);
      if (!converted.ok()) return converted.status();
      result.emplace(key, *std::move(converted));
    }
    return result;
  }
};

}  // namespace json_internal

// Deserializes `value` as the Rust type `T` stands for. Supported `T`:
// `bool`, `std::string`, `char32_t` (`char`), integers (`uint32_t` = `u32`,
// `size_t`/`uint64_t` = `usize`, `int64_t` = `i64`, ...), `float` (`f32`),
// `double` (`f64`), `Json` (any value), `std::optional<T>` (`Option<T>`),
// `std::vector<T>` (`Vec<T>`), `std::pair<A, B>` (`(A, B)`) and
// `absl::flat_hash_map<std::string, V>` (`HashMap<String, V>`), nested freely.
template <typename T>
absl::StatusOr<T> JsonAs(const Json& value) {
  return json_internal::Converter<T>::Convert(value);
}

// The `Expected` text serde uses for `T` (e.g. "u32", "a string").
template <typename T>
constexpr absl::string_view JsonExpected() {
  return json_internal::Converter<T>::kExpected;
}

// A required field: `object[name]` deserialized as `T`; a missing field is a
// `MissingFieldError`, a `null` or wrong-typed value an `InvalidTypeError`.
template <typename T>
absl::StatusOr<T> GetField(const Json& object, absl::string_view name) {
  const Json* field = FindField(object, name);
  if (field == nullptr) return MissingFieldError(name);
  return json_internal::Converter<T>::Convert(*field);
}

// An `Option<T>` field: missing or `null` is `std::nullopt`.
template <typename T>
absl::StatusOr<std::optional<T>> GetOptionalField(const Json& object,
                                                  absl::string_view name) {
  const Json* field = FindField(object, name);
  if (field == nullptr || field->is_null()) return std::optional<T>();
  absl::StatusOr<T> value = json_internal::Converter<T>::Convert(*field);
  if (!value.ok()) return value.status();
  return std::optional<T>(*std::move(value));
}

// A `#[serde(default)]` field: `default_value` when missing (a present `null`
// is still an error for non-`Option` `T`).
template <typename T>
absl::StatusOr<T> GetFieldOr(const Json& object, absl::string_view name,
                             T default_value) {
  const Json* field = FindField(object, name);
  if (field == nullptr) return std::move(default_value);
  return json_internal::Converter<T>::Convert(*field);
}

// Required array / object fields, accessed in place (no copy of possibly
// large sub-documents). Errors as `GetField`, with "a sequence" / "a map".
absl::StatusOr<const Json*> GetArrayField(const Json& object,
                                          absl::string_view name);
absl::StatusOr<const Json*> GetObjectField(const Json& object,
                                           absl::string_view name);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_JSON_H_
