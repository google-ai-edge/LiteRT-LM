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

#include "c/error_reporter.h"

#include <string>
#include <thread>  // NOLINT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/error_reporter_internal.h"

namespace {

using ::testing::HasSubstr;

TEST(ErrorReporterTest, InitialStateOrClearedState) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, SetAndGetErrorInternal) {
  litert_lm_clear_last_error();
  litert::lm::c::SetLastError(
      absl::InvalidArgumentError("test argument error"));
  EXPECT_EQ(litert_lm_get_last_error_code(),
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("test argument error"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, SetLastErrorOkClearsError) {
  litert::lm::c::SetLastError(absl::InternalError("internal failure"));
  EXPECT_NE(litert_lm_get_last_error_code(), 0);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);

  litert::lm::c::SetLastError(absl::OkStatus());
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, ApiCallSetsAndClearsError) {
  litert_lm_clear_last_error();

  // Passing null model_path to litert_lm_engine_settings_create should trigger
  // an error.
  LiteRtLmEngineSettings* invalid_settings = litert_lm_engine_settings_create(
      /*model_path=*/nullptr, "cpu", nullptr, nullptr);
  EXPECT_EQ(invalid_settings, nullptr);
  EXPECT_EQ(litert_lm_get_last_error_code(),
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("model_path cannot be null"));

  // A successful call should clear the error.
  LiteRtLmEngineSettings* valid_settings = litert_lm_engine_settings_create(
      "dummy_model_path", "cpu", nullptr, nullptr);
  ASSERT_NE(valid_settings, nullptr);
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  litert_lm_engine_settings_delete(valid_settings);
}

TEST(ErrorReporterTest, ThreadIsolation) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  std::thread worker([]() {
    EXPECT_EQ(litert_lm_get_last_error_code(), 0);
    EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

    litert::lm::c::SetLastError(absl::NotFoundError("error on worker thread"));
    EXPECT_EQ(litert_lm_get_last_error_code(),
              static_cast<int>(absl::StatusCode::kNotFound));
    ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                HasSubstr("error on worker thread"));
  });

  worker.join();

  // Main thread's error state should remain clean / unaffected by worker
  // thread.
  EXPECT_EQ(litert_lm_get_last_error_code(), 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

}  // namespace
