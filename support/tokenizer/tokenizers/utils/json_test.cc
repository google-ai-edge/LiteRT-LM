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
// Tests for `utils/json.h` (including the `impl_serde_type!` doctests of
// `src/utils/mod.rs`).

#include "support/tokenizer/tokenizers/utils/json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(JsonTest, ParseAndDumpPreserveInsertionOrder) {
  ASSERT_OK_AND_ASSIGN(Json j,
                       ParseJson(R"({"z":1,"a":2,"m":[true,null,"x"]})"));
  EXPECT_EQ(DumpJson(j), R"({"z":1,"a":2,"m":[true,null,"x"]})");

  // Invalid JSON returns InvalidArgumentError instead of throwing/aborting.
  EXPECT_THAT(ParseJson(""), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseJson("{unclosed"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseJson("123 trailing"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonTest, JsonEqualIgnoringKeyOrder) {
  ASSERT_OK_AND_ASSIGN(Json a, ParseJson(R"({"b":1,"a":{"y":2,"x":3}})"));
  ASSERT_OK_AND_ASSIGN(Json b, ParseJson(R"({"a":{"x":3,"y":2},"b":1})"));
  // Ordered equality distinguishes key order; unordered helper ignores it.
  EXPECT_NE(a, b);
  EXPECT_TRUE(JsonEqualIgnoringKeyOrder(a, b));

  ASSERT_OK_AND_ASSIGN(Json c, ParseJson(R"({"a":{"x":3,"y":99},"b":1})"));
  EXPECT_FALSE(JsonEqualIgnoringKeyOrder(a, c));
}

TEST(JsonTest, IndexedOrderedMapOperationsAndScale) {
  Json obj = Json::object();
  constexpr int kNumKeys = 20000;
  for (int i = 0; i < kNumKeys; ++i) {
    obj[absl::StrCat("tok_", i)] = i;
  }
  EXPECT_EQ(obj.size(), static_cast<size_t>(kNumKeys));
  for (int i = 0; i < kNumKeys; i += 137) {
    const std::string key = absl::StrCat("tok_", i);
    ASSERT_THAT(GetField<int>(obj, key), IsOkAndHolds(i));
  }
  EXPECT_EQ(obj.erase("tok_0"), 1u);
  EXPECT_EQ(obj.erase("tok_missing"), 0u);
  EXPECT_EQ(FindField(obj, "tok_0"), nullptr);
  ASSERT_THAT(GetField<int>(obj, "tok_1"), IsOkAndHolds(1));
  ASSERT_THAT(GetField<int>(obj, "tok_19999"), IsOkAndHolds(19999));
}

// Mirrors the `impl_serde_type!` doctests in `src/utils/mod.rs`.
TEST(JsonTest, ImplSerdeTypeExamples) {
  // Struct Point { x: i32, #[serde(default = "default_y")] y: i32 } (default 5)
  Json point = {{"type", "Point"}, {"x", 1}, {"y", 2}};
  EXPECT_EQ(DumpJson(point), R"({"type":"Point","x":1,"y":2})");

  ASSERT_OK_AND_ASSIGN(Json point_default_y,
                       ParseJson(R"({"type":"Point","x":1})"));
  EXPECT_OK(RequireTypeTag(point_default_y, "Point"));
  EXPECT_THAT(GetField<int32_t>(point_default_y, "x"), IsOkAndHolds(1));
  EXPECT_THAT(GetFieldOr<int32_t>(point_default_y, "y", 5), IsOkAndHolds(5));

  // Point1D missing field `type`.
  ASSERT_OK_AND_ASSIGN(Json missing_type, ParseJson(R"({"x":1})"));
  EXPECT_THAT(
      RequireTypeTag(missing_type, "Point1D"),
      StatusIs(absl::StatusCode::kInvalidArgument, "missing field `type`"));

  // Unit struct serialization & rejection when `type` is missing.
  Json unit = {{"type", "Unit"}};
  EXPECT_EQ(DumpJson(unit), R"({"type":"Unit"})");
  ASSERT_OK_AND_ASSIGN(Json bad_unit, ParseJson(R"({"some_field":1})"));
  EXPECT_THAT(
      RequireTypeTag(bad_unit, "Unit"),
      StatusIs(absl::StatusCode::kInvalidArgument, "missing field `type`"));
}

TEST(JsonTest, SerdeErrorMessages) {
  EXPECT_EQ(MissingFieldError("foo").message(), "missing field `foo`");
  EXPECT_EQ(DuplicateFieldError("foo").message(), "duplicate field `foo`");
  EXPECT_EQ(UnknownFieldError("x", {}).message(),
            "unknown field `x`, there are no fields");
  EXPECT_EQ(UnknownFieldError("x", {"a"}).message(),
            "unknown field `x`, expected `a`");
  EXPECT_EQ(UnknownFieldError("x", {"a", "b"}).message(),
            "unknown field `x`, expected `a` or `b`");
  EXPECT_EQ(UnknownFieldError("x", {"a", "b", "c"}).message(),
            "unknown field `x`, expected one of `a`, `b`, `c`");
  EXPECT_EQ(UnknownVariantError("Bad", {"Good", "Other"}).message(),
            "unknown variant `Bad`, expected `Good` or `Other`");
  EXPECT_EQ(InvalidLengthError(3, "a tuple of size 2").message(),
            "invalid length 3, expected a tuple of size 2");

  ASSERT_OK_AND_ASSIGN(
      Json obj,
      ParseJson(
          R"({"n":null,"b":true,"i":-7,"u":42,"f":1.5,"s":"hi","a":[],"o":{}})"));
  EXPECT_EQ(InvalidTypeError(obj["n"], "a string").message(),
            "invalid type: null, expected a string");
  EXPECT_EQ(InvalidTypeError(obj["b"], "u32").message(),
            "invalid type: boolean `true`, expected u32");
  EXPECT_EQ(InvalidValueError(obj["i"], "u32").message(),
            "invalid value: integer `-7`, expected u32");
  EXPECT_EQ(InvalidTypeError(obj["u"], "a boolean").message(),
            "invalid type: integer `42`, expected a boolean");
  EXPECT_EQ(InvalidTypeError(obj["f"], "u32").message(),
            "invalid type: floating point `1.5`, expected u32");
  EXPECT_EQ(InvalidTypeError(obj["s"], "struct BPE").message(),
            "invalid type: string \"hi\", expected struct BPE");
  EXPECT_EQ(InvalidTypeError(obj["a"], "a map").message(),
            "invalid type: sequence, expected a map");
  EXPECT_EQ(InvalidTypeError(obj["o"], "a sequence").message(),
            "invalid type: map, expected a sequence");
}

TEST(JsonTest, TypedFieldAccessorsAndStrictNumericRules) {
  ASSERT_OK_AND_ASSIGN(Json obj, ParseJson(R"({
        "s": "hello",
        "c": "▁",
        "empty_c": "",
        "two_c": "ab",
        "b": false,
        "u": 42,
        "neg": -1,
        "big": 5000000000,
        "flt": 2.5,
        "null_v": null,
        "arr": [1, 2, 3],
        "pair": ["a", 7],
        "bad_pair": ["only_one"],
        "map": {"k1": 10, "k2": 20}
      })"));

  EXPECT_THAT(GetField<std::string>(obj, "s"), IsOkAndHolds("hello"));
  EXPECT_THAT(GetField<char32_t>(obj, "c"), IsOkAndHolds(U'▁'));
  EXPECT_THAT(GetField<char32_t>(obj, "empty_c"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid value: string \"\", expected a character"));
  EXPECT_THAT(GetField<char32_t>(obj, "two_c"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid value: string \"ab\", expected a character"));
  EXPECT_THAT(GetField<bool>(obj, "b"), IsOkAndHolds(false));

  // u32 accepts in-range unsigned ints, rejects negative, > UINT32_MAX, and
  // floats.
  EXPECT_THAT(GetField<uint32_t>(obj, "u"), IsOkAndHolds(42u));
  EXPECT_THAT(GetField<uint32_t>(obj, "neg"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid value: integer `-1`, expected u32"));
  EXPECT_THAT(GetField<uint32_t>(obj, "big"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid value: integer `5000000000`, expected u32"));
  EXPECT_THAT(GetField<uint32_t>(obj, "flt"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid type: floating point `2.5`, expected u32"));
  EXPECT_THAT(GetField<size_t>(obj, "big"),
              IsOkAndHolds(size_t{5000000000ULL}));

  // f64 accepts both floats and integers.
  EXPECT_THAT(GetField<double>(obj, "flt"), IsOkAndHolds(2.5));
  EXPECT_THAT(GetField<double>(obj, "u"), IsOkAndHolds(42.0));
  EXPECT_THAT(GetField<double>(obj, "neg"), IsOkAndHolds(-1.0));

  // Option<T>: both missing and null return std::nullopt.
  EXPECT_THAT(GetOptionalField<std::string>(obj, "missing"),
              IsOkAndHolds(std::nullopt));
  EXPECT_THAT(GetOptionalField<std::string>(obj, "null_v"),
              IsOkAndHolds(std::nullopt));
  EXPECT_THAT(GetOptionalField<std::string>(obj, "s"),
              IsOkAndHolds(std::optional<std::string>("hello")));
  EXPECT_THAT(GetOptionalField<uint32_t>(obj, "s"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Containers and tuples.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> arr_val,
                       GetField<std::vector<uint32_t>>(obj, "arr"));
  EXPECT_THAT(arr_val, ElementsAre(1u, 2u, 3u));
  using StrU32 = std::pair<std::string, uint32_t>;
  EXPECT_THAT(GetField<StrU32>(obj, "pair"), IsOkAndHolds(StrU32("a", 7u)));
  EXPECT_THAT(GetField<StrU32>(obj, "bad_pair"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid length 1, expected a tuple of size 2"));
  using MapU32 = absl::flat_hash_map<std::string, uint32_t>;
  ASSERT_OK_AND_ASSIGN(MapU32 map_val, GetField<MapU32>(obj, "map"));
  EXPECT_THAT(map_val, UnorderedElementsAre(Pair("k1", 10u), Pair("k2", 20u)));

  // In-place array/object getters.
  ASSERT_OK_AND_ASSIGN(const Json* arr_ptr, GetArrayField(obj, "arr"));
  EXPECT_EQ(arr_ptr->size(), 3u);
  ASSERT_OK_AND_ASSIGN(const Json* map_ptr, GetObjectField(obj, "map"));
  EXPECT_EQ(map_ptr->size(), 2u);
  EXPECT_THAT(GetArrayField(obj, "map"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid type: map, expected a sequence"));
  EXPECT_THAT(GetObjectField(obj, "arr"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid type: sequence, expected a map"));
}

TEST(JsonTest, DenyUnknownFieldsAndRequireTypeTag) {
  ASSERT_OK_AND_ASSIGN(
      Json obj, ParseJson(R"({"type":"ByteLevel","add_prefix_space":true})"));
  EXPECT_OK(RequireTypeTag(obj, "ByteLevel"));
  EXPECT_OK(DenyUnknownFields(obj, {"type", "add_prefix_space"}));
  EXPECT_THAT(DenyUnknownFields(obj, {"type"}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "unknown field `add_prefix_space`, expected `type`"));
  EXPECT_THAT(RequireTypeTag(obj, "Metaspace"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "unknown variant `ByteLevel`, expected `Metaspace`"));
  EXPECT_THAT(
      RequireTypeTag(Json("not_an_obj"), "ByteLevel"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "invalid type: string \"not_an_obj\", expected struct ByteLevel"));
}

}  // namespace
}  // namespace tokenizers
