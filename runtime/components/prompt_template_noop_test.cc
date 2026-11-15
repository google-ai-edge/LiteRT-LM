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

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(PromptTemplateNoopTest, ApplyReturnsUnimplemented) {
  PromptTemplate prompt_template("{{ messages[0].content }}");
  EXPECT_EQ(prompt_template.GetTemplateSource(), "{{ messages[0].content }}");
  EXPECT_FALSE(prompt_template.GetCapabilities().supports_single_turn);

  PromptTemplate copy = prompt_template;
  EXPECT_EQ(copy.GetTemplateSource(), "{{ messages[0].content }}");

  PromptTemplate moved = std::move(copy);
  EXPECT_EQ(moved.GetTemplateSource(), "{{ messages[0].content }}");

  // Copying or copy-assigning a moved-from PromptTemplate must not crash.
  PromptTemplate copy_of_moved(copy);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(copy_of_moved.GetTemplateSource(), "");
  moved = copy;
  EXPECT_EQ(moved.GetTemplateSource(), "");

  PromptTemplateInput input;
  EXPECT_THAT(prompt_template.Apply(input),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace litert::lm
