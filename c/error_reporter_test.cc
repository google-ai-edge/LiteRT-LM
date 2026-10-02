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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/error_reporter_internal.h"

namespace {

using ::testing::HasSubstr;

TEST(ErrorReporterTest, InitialStateOrClearedState) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, SetAndGetErrorInternal) {
  litert_lm_clear_last_error();
  litert::lm::c::SetLastError(
      absl::InvalidArgumentError("test argument error"));
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("test argument error"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, SetLastErrorOkClearsError) {
  litert::lm::c::SetLastError(absl::InternalError("internal failure"));
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);

  litert::lm::c::SetLastError(absl::OkStatus());
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, ApiCallSetsErrorAndManualClear) {
  litert_lm_clear_last_error();

  // Passing null model_path to litert_lm_engine_settings_create should trigger
  // an error.
  LiteRtLmEngineSettings* invalid_settings = nullptr;
  EXPECT_EQ(
      litert_lm_engine_settings_create(/*model_path=*/nullptr, "cpu", nullptr,
                                       nullptr, &invalid_settings),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(invalid_settings, nullptr);
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("model_path must not be NULL"));

  // A successful call does not clear the error (following standard C
  // conventions).
  LiteRtLmEngineSettings* valid_settings = nullptr;
  ASSERT_EQ(litert_lm_engine_settings_create("dummy_model_path", "cpu", nullptr,
                                             nullptr, &valid_settings),
            kLiteRtLmStatusOk);
  ASSERT_NE(valid_settings, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("model_path must not be NULL"));

  // litert_lm_clear_last_error explicitly clears the error.
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  litert_lm_engine_settings_delete(valid_settings);
}

TEST(ErrorReporterTest, ThreadIsolation) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  std::thread worker([]() {
    EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

    litert::lm::c::SetLastError(absl::NotFoundError("error on worker thread"));
    ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                HasSubstr("error on worker thread"));
  });

  worker.join();

  // Main thread's error state should remain clean / unaffected by worker
  // thread.
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);
}

TEST(ErrorReporterTest, ToCStatusRecordsErrorAndReturnsCode) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert::lm::c::ToCStatus(absl::NotFoundError("missing file")),
            kLiteRtLmStatusNotFound);
  EXPECT_THAT(litert_lm_get_last_error_message(), HasSubstr("missing file"));
}

TEST(ErrorReporterTest, ToCStatusOkDoesNotClearPriorError) {
  litert::lm::c::SetLastError(absl::InternalError("earlier failure"));
  EXPECT_EQ(litert::lm::c::ToCStatus(absl::OkStatus()), kLiteRtLmStatusOk);
  EXPECT_THAT(litert_lm_get_last_error_message(), HasSubstr("earlier failure"));
}

TEST(ErrorReporterTest, ReturnErrorRecordsErrorAndReturnsCode) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert::lm::c::ReturnError(absl::StatusCode::kOutOfRange,
                                       "index 3 out of range"),
            kLiteRtLmStatusOutOfRange);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("index 3 out of range"));
}

LiteRtLmStatusCode ReturnIfNullEntryPoint(const int* arg) {
  LITERT_LM_C_RETURN_IF_NULL(arg);
  return kLiteRtLmStatusOk;
}

TEST(ErrorReporterTest, ReturnIfNullMacro) {
  litert_lm_clear_last_error();
  int value = 0;
  EXPECT_EQ(ReturnIfNullEntryPoint(&value), kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  EXPECT_EQ(ReturnIfNullEntryPoint(nullptr), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("arg must not be NULL"));
}

LiteRtLmStatusCode ReturnIfErrorEntryPoint(const absl::Status& status,
                                           bool* reached_end) {
  LITERT_LM_C_RETURN_IF_ERROR(status);
  *reached_end = true;
  return kLiteRtLmStatusOk;
}

TEST(ErrorReporterTest, ReturnIfErrorMacro) {
  litert_lm_clear_last_error();
  bool reached_end = false;
  EXPECT_EQ(ReturnIfErrorEntryPoint(absl::OkStatus(), &reached_end),
            kLiteRtLmStatusOk);
  EXPECT_TRUE(reached_end);

  reached_end = false;
  EXPECT_EQ(
      ReturnIfErrorEntryPoint(absl::UnavailableError("busy"), &reached_end),
      kLiteRtLmStatusUnavailable);
  EXPECT_FALSE(reached_end);
  EXPECT_THAT(litert_lm_get_last_error_message(), HasSubstr("busy"));
}

LiteRtLmStatusCode AssignOrReturnEntryPoint(absl::StatusOr<std::string> input,
                                            std::string* out) {
  LITERT_LM_C_ASSIGN_OR_RETURN(std::string value, std::move(input));
  LITERT_LM_C_ASSIGN_OR_RETURN(*out, absl::StatusOr<std::string>(value + "!"));
  return kLiteRtLmStatusOk;
}

TEST(ErrorReporterTest, AssignOrReturnMacro) {
  litert_lm_clear_last_error();
  std::string out;
  EXPECT_EQ(AssignOrReturnEntryPoint(std::string("hi"), &out),
            kLiteRtLmStatusOk);
  EXPECT_EQ(out, "hi!");

  out.clear();
  EXPECT_EQ(AssignOrReturnEntryPoint(absl::DataLossError("corrupt"), &out),
            kLiteRtLmStatusDataLoss);
  EXPECT_TRUE(out.empty());
  EXPECT_THAT(litert_lm_get_last_error_message(), HasSubstr("corrupt"));
}

}  // namespace
