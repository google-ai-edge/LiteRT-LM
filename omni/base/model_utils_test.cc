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

#include "omni/base/model_utils.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;

TEST(ModelUtilsTest, CheckFileReadableNonExistent) {
  EXPECT_THAT(CheckFileReadable("/non/existent/file.bin"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, LoadFileNonExistent) {
  EXPECT_THAT(LoadFile("/non/existent/dir", "model.tflite"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, CreateCompiledModelNonExistent) {
  auto env_expected = Environment::Create({});
  if (env_expected.HasValue()) {
    ModelOptions options;
    options.model_dir = "/non/existent/dir";
    EXPECT_THAT(CreateCompiledModel(*env_expected, options, "model.tflite"),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

}  // namespace
}  // namespace litert::omni
