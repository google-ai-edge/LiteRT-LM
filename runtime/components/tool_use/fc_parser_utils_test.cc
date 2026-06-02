// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/tool_use/fc_parser_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace {

using ::litert::lm::ParseFcExpression;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(FcParserUtilsTest, ParseToolCall) {
  EXPECT_THAT(ParseFcExpression(R"(call:tool_name{x:1})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "x": 1
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithStringArgument) {
  EXPECT_THAT(
      ParseFcExpression(R"(call:tool_name{text:<escape>hello world<escape>})"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "hello world"
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallNoArgumentsEmptyBraces) {
  EXPECT_THAT(ParseFcExpression(R"(call:tool_name{})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {}
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallNoArgumentsNoBraces) {
  EXPECT_THAT(ParseFcExpression(R"(call:tool_name)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {}
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithStringArgumentDoubleQuoteEscape) {
  EXPECT_THAT(
      ParseFcExpression(R"(call:tool_name{text:<|"|>hello world<|"|>})"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "hello world"
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithSingleQuotesInStringArgument) {
  EXPECT_THAT(ParseFcExpression(
                  R"(call:tool_name{text:<escape>foo 'bar' baz<escape>})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "foo 'bar' baz"
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithDoubleQuotesInStringArgument) {
  EXPECT_THAT(ParseFcExpression(
                  R"(call:tool_name{text:<escape>foo "bar" baz<escape>})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "foo \"bar\" baz"
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithBackslashInStringArgument) {
  EXPECT_THAT(ParseFcExpression(
                  R"(call:tool_name{text:<escape>foo \b\a\r baz<escape>})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "foo \\b\\a\\r baz"
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseToolCallWithNestedQuotesInStringArgument) {
  EXPECT_THAT(ParseFcExpression(
                  R"(call:tool_name{text:<escape>"foo \"bar\" baz"<escape>})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "text": "\"foo \\\"bar\\\" baz\""
                }
              }])json")));
}

TEST(FcParserUtilsTest, ParseEmptyArguments) {
  EXPECT_THAT(ParseFcExpression(R"(call:empty_args_func{})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "empty_args_func",
                "arguments": {}
              }])json")));
}

TEST(FcParserUtilsTest, ParseComplexArguments) {
  EXPECT_THAT(ParseFcExpression(
                  R"(call:complex_func{
  str_arg:<escape>value<escape>,
  int_arg:123,
  float_arg:-4.5,
  bool_arg:false,
  null_arg:null,
  list_arg:[1,<escape>two<escape>,true,null,{nested_key:<escape>nested_val<escape>}],
  obj_arg:{
    nested_str:<escape>val1<escape>,
    nested_int:-678,
    nested_float:9.1,
    nested_bool:true,
    nested_null:null,
    nested_list:[10,20],
    nested_obj: {
      key:<escape>val<escape>
    }
  }
})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "complex_func",
                "arguments": {
                  "str_arg": "value",
                  "int_arg": 123,
                  "float_arg": -4.5,
                  "bool_arg": false,
                  "null_arg": null,
                  "list_arg": [
                    1,
                    "two",
                    true,
                    null,
                    {
                      "nested_key": "nested_val"
                    }
                  ],
                  "obj_arg": {
                    "nested_str": "val1",
                    "nested_int": -678,
                    "nested_float": 9.1,
                    "nested_bool": true,
                    "nested_null": null,
                    "nested_list": [
                      10,
                      20
                    ],
                    "nested_obj": {
                      "key": "val"
                    }
                  }
                }
              }])json")));
}

TEST(FcParserUtilsTest, InvalidToolCallSyntax) {
  // Extra comma in the argument list.
  EXPECT_THAT(ParseFcExpression(R"(call:bad_tool_call{x:1,})"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, MissingToolName) {
  EXPECT_THAT(ParseFcExpression(R"(call:{x:1})"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, ArgumentsNotObject) {
  EXPECT_THAT(ParseFcExpression(R"(call:wrong_args[1,2,3])"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, MissingParameterKey) {
  EXPECT_THAT(ParseFcExpression(R"(call:{1})"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, EmptyInputString) {
  EXPECT_THAT(ParseFcExpression(""),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([])json")));
}

TEST(FcParserUtilsTest, NoCallPrefix) {
  EXPECT_THAT(ParseFcExpression(R"(print{x:1})"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, UnescapedString) {
  EXPECT_THAT(ParseFcExpression(R"(call:print{text:"hello"})"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FcParserUtilsTest, DuplicateKeysInObject) {
  EXPECT_THAT(ParseFcExpression(R"(call:foo{a:1,b:2,a:3})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "foo",
                "arguments": {
                  "a": 1,
                  "b": 2
                }
              }])json")));
}
TEST(FcParserUtilsTest, ToolNameWithDotsAndDashes) {
  EXPECT_THAT(ParseFcExpression(R"(call:tool.name-123{})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool.name-123",
                "arguments": {}
              }])json")));
}

TEST(FcParserUtilsTest, ParameterNameWithDotsAndDashes) {
  EXPECT_THAT(ParseFcExpression(R"(call:tool_name{param.name-123:1})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "tool_name",
                "arguments": {
                  "param.name-123": 1
                }
              }])json")));
}

}  // namespace
