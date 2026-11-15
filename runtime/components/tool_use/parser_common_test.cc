// Copyright 2025 The Google AI Edge Authors.
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

#include "runtime/components/tool_use/parser_common.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/rust/parsers.rs.h"

namespace litert::lm {
namespace {

TEST(ConvertJsonValueTest, HandlesComplexStructure) {
  // We use parse_json_expression to create a JsonValue because JsonValue
  // is an opaque CXX type and cannot be constructed from C++.
  // parse_json_expression expects a "Tool Call" structure (name + arguments).
  const std::string kJsonInput = R"json({
    "name": "test_tool",
    "arguments": {
      "null_field": null,
      "bool_true": true,
      "bool_false": false,
      "number_int": 42,
      "number_float": 3.14,
      "string_val": "hello",
      "array_mixed": [1, "two", false],
      "object_nested": {
        "inner_key": "inner_val"
      }
    }
  })json";

  const auto result = parse_json_expression(kJsonInput);
  ASSERT_TRUE(result.is_ok)
      << "Failed to parse setup JSON: " << std::string(result.error);
  ASSERT_EQ(result.tool_calls.size(), 1);

  // Extract the "arguments" field which contains our test data.
  const auto& tool_call = result.tool_calls[0];
  const auto arguments = tool_call.object_get("arguments");

  // Convert from JsonValue to nlohmann::ordered_json.
  nlohmann::ordered_json converted = ConvertJsonValue(*arguments);

  // Verify primitive types.
  EXPECT_TRUE(converted["null_field"].is_null());
  EXPECT_EQ(converted["bool_true"], true);
  EXPECT_EQ(converted["bool_false"], false);
  EXPECT_EQ(converted["number_int"], 42);
  EXPECT_NEAR(converted["number_float"].get<double>(), 3.14, 0.0001);
  EXPECT_EQ(converted["string_val"], "hello");

  // Numeric *type* must survive the conversion, not just the numeric value.
  // The EXPECT_EQ above cannot catch a regression here: nlohmann's operator==
  // compares numbers across representations, so 42 == 42.0 holds.
  EXPECT_TRUE(converted["number_int"].is_number_integer());
  EXPECT_TRUE(converted["number_float"].is_number_float());

  // Verify array.
  ASSERT_TRUE(converted["array_mixed"].is_array());
  ASSERT_EQ(converted["array_mixed"].size(), 3);
  EXPECT_EQ(converted["array_mixed"][0], 1);
  EXPECT_TRUE(converted["array_mixed"][0].is_number_integer());
  EXPECT_EQ(converted["array_mixed"][1], "two");
  EXPECT_EQ(converted["array_mixed"][2], false);

  // Verify nested object.
  ASSERT_TRUE(converted["object_nested"].is_object());
  EXPECT_EQ(converted["object_nested"]["inner_key"], "inner_val");
}

// Asserts on the serialized form, which -- unlike operator== -- distinguishes
// an integer from a float. Downstream consumers see exactly this text, and a
// strict JSON decoder rejects "1000.0" for a field declared as an integer.
TEST(ConvertJsonValueTest, PreservesNumericRepresentation) {
  struct TestCase {
    std::string name;
    std::string literal;
    std::string expected_dump;
    bool expect_integer;
  };
  const TestCase kTestCases[] = {
      {"positive_int", "1000", "1000", true},
      {"negative_int", "-678", "-678", true},
      {"zero", "0", "0", true},
      {"float_with_zero_fraction", "1000.0", "1000.0", false},
      {"float", "3.14", "3.14", false},
      // Exponent notation stays a float: the writer chose that form, and there
      // is no way to tell an intended integer from an intended float.
      {"exponent", "1e3", "1000.0", false},
      // Beyond 2^53 an f64 round trip would silently change the digits.
      {"large_int_above_2_53", "9007199254740993", "9007199254740993", true},
  };

  for (const TestCase& test_case : kTestCases) {
    SCOPED_TRACE(test_case.name);
    const std::string input =
        absl::StrCat(R"({"name": "test_tool", "arguments": {"value": )",
                     test_case.literal, "}}");

    const auto result = parse_json_expression(input);
    ASSERT_TRUE(result.is_ok)
        << "Failed to parse setup JSON: " << std::string(result.error);
    ASSERT_EQ(result.tool_calls.size(), 1);

    const auto arguments = result.tool_calls[0].object_get("arguments");
    const nlohmann::ordered_json converted = ConvertJsonValue(*arguments);

    EXPECT_EQ(converted["value"].dump(), test_case.expected_dump);
    EXPECT_EQ(converted["value"].is_number_integer(), test_case.expect_integer);
  }
}

}  // namespace
}  // namespace litert::lm
