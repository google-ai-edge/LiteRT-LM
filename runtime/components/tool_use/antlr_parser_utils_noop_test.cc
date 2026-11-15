// Copyright 2026 The Google AI Edge Authors.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/tool_use/fc_parser_utils.h"
#include "runtime/components/tool_use/json_parser_utils.h"
#include "runtime/components/tool_use/python_parser_utils.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(AntlrParserUtilsNoopTest, ParsePythonExpressionReturnsUnimplemented) {
  EXPECT_THAT(ParsePythonExpression("tool(x=1)"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(AntlrParserUtilsNoopTest, ParseJsonExpressionReturnsUnimplemented) {
  EXPECT_THAT(ParseJsonExpression(R"([{"name": "tool"}])"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(AntlrParserUtilsNoopTest, ParseFcExpressionReturnsUnimplemented) {
  EXPECT_THAT(ParseFcExpression("call:tool{x:1}"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace litert::lm
