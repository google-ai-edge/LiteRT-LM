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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TEST_UTILS_H_

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl                   // NOLINT
#include "absl/status/statusor.h"  // from @com_google_absl                 // NOLINT
#include "litert/cc/litert_macros.h"  // from @litert  // NOLINT

#if !defined(EXPECT_OK)
#define EXPECT_OK(status) EXPECT_TRUE(status.ok())
#endif  // defined(EXPECT_OK)

#if !defined(ASSERT_OK)
#define ASSERT_OK(status) ASSERT_TRUE(status.ok())
#endif  // defined(ASSERT_OK)

#if !defined(ASSERT_OK_AND_ASSIGN)
#define ASSERT_OK_AND_ASSIGN(DECL, EXPR) \
  _ASSERT_OK_AND_ASSIGN_IMPL(_CONCAT_NAME(_statusor_, __LINE__), DECL, EXPR)
#define _ASSERT_OK_AND_ASSIGN_IMPL(TMP_VAR, DECL, EXPR) \
  auto&& TMP_VAR = (EXPR);                              \
  ASSERT_TRUE(TMP_VAR.ok()) << TMP_VAR.status();        \
  DECL = std::move(*TMP_VAR)
#endif  // !defined(ASSERT_OK_AND_ASSIGN)

namespace testing::status {
namespace {

// Helper functions and templates to get absl::Status from arbitrary class.
const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}

template <class T>
const absl::Status& GetStatus(const absl::StatusOr<T>& statusor) {
  return statusor.status();
}

}  // namespace

MATCHER_P(StatusIs, code, "") {
  return GetStatus(arg).code() == code;
}

MATCHER_P2(StatusIs, code, msg, "") {
  const auto& status = GetStatus(arg);
  return status.code() == code &&
    testing::ExplainMatchResult(msg, status.message(), result_listener);
}

MATCHER_P(IsOkAndHolds, value, "") {
  return arg.ok() && *arg == static_cast<decltype(*arg)>(value);
}

}  // namespapce testing::status

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TEST_UTILS_H_
