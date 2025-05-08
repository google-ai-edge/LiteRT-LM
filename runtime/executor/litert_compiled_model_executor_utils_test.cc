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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, JoinPathBasicTest) {
  std::string path1 = "";
  std::string path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path1."));

  path1 = "path1";
  path2 = "";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path2."));

  path1 = "path1";
  path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2), "path1/path2");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, BasenameBasicTest) {
  std::string model_path = "/path/to/model.tflite";
  EXPECT_THAT(Basename(model_path), "model.tflite");
}

}  // namespace
}  // namespace litert::lm
