// Copyright 2025 The ODML Authors.
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

#include "runtime/framework/resource_management/utils/resource_manager_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {

TEST(LlmResourceManagerUtilsTest, RemoveAllMatchingTokens) {
  std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> processed_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int time_step = 0;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({}));
  EXPECT_EQ(time_step, 10);

  input_ids = {3, 4, 5, 6, 7, 8};
  processed_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({}));
  EXPECT_EQ(time_step, 8);
}

TEST(LlmResourceManagerUtilsTest, RemoveAllMatchingPrefixTokens) {
  std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> processed_tokens = {1, 2, 3, 4, 5, 6};
  int time_step = 0;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({7, 8, 9, 10}));
  EXPECT_EQ(time_step, 6);

  input_ids = {3, 4, 5, 6, 7, 8, 9, 10};
  processed_tokens = {1, 2, 3, 4, 5, 6, 7};
  time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({8, 9, 10}));
  EXPECT_EQ(time_step, 7);
}

TEST(LlmResourceManagerUtilsTest, RemovePartialMatchingTokens) {
  std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> processed_tokens = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  int time_step = 0;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({6, 7, 8, 9, 10}));
  EXPECT_EQ(time_step, 5);

  input_ids = {3, 4, 5, 6, 7, 8, 9, 10};
  processed_tokens = {1, 2, 3, 4, 5, 6, 0, 0, 0, 0};
  time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({7, 8, 9, 10}));
  EXPECT_EQ(time_step, 6);
}

TEST(LlmResourceManagerUtilsTest, RemoveNoMatchingTokens) {
  std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> processed_tokens = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int time_step = 0;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  EXPECT_EQ(time_step, 0);

  input_ids = {3, 4, 5, 6, 7, 8, 9, 10};
  processed_tokens = {3, 2, 1, 4, 5, 6, 7, 8, 9, 10};
  time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({3, 4, 5, 6, 7, 8, 9, 10}));
  EXPECT_EQ(time_step, 2);
}

TEST(LlmResourceManagerUtilsTest, RemoveNegativeTokens) {
  std::vector<int> input_ids = {3, 4, 5, -1, -1, -1, -3};
  std::vector<int> processed_tokens = {1, 2, 3, 4, 5, -1, -1, -1, -1, -3};
  int time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({-1, -1, -1, -3}));
  EXPECT_EQ(time_step, 5);

  input_ids = {-2, -2, -2, -4};
  processed_tokens = {-2, -2, -2, -2, -2, -4};
  time_step = 2;
  EXPECT_OK(RemoveMatchingTokens(processed_tokens, &input_ids, &time_step));
  EXPECT_EQ(input_ids, std::vector<int>({-2, -2, -2, -4}));
  EXPECT_EQ(time_step, 2);
}

}  // namespace litert::lm
