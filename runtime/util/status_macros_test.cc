#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

// Re-define status macros.
#undef ASSIGN_OR_RETURN  // NOLINT: for testing
#undef RETURN_IF_ERROR   // NOLINT: for testing
#undef RET_CHECK         // NOLINT: for testing
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(StatusMacrosTest, AssignOrReturn_Success) {
  auto status_or = []() -> absl::StatusOr<int> {
    ASSIGN_OR_RETURN(int x, absl::StatusOr<int>(1));
    return x;
  }();
  EXPECT_OK(status_or);
  EXPECT_EQ(*status_or, 1);
}

TEST(StatusMacrosTest, AssignOrReturn_Failure) {
  auto status_or = []() -> absl::StatusOr<int> {
    ASSIGN_OR_RETURN(
        int x,
        absl::StatusOr<int>(absl::InternalError("It's an internal error.")));
    return x;
  }();
  EXPECT_THAT(status_or,
              StatusIs(absl::StatusCode::kInternal, "It's an internal error."));
}

TEST(StatusMacrosTest, ReturnIfError_Success) {
  auto status = []() -> absl::Status {
    RETURN_IF_ERROR(absl::OkStatus());
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, ReturnIfError_Failure) {
  auto status = []() -> absl::Status {
    RETURN_IF_ERROR(absl::InternalError("It's an internal error."));
    return absl::OkStatus();
  }();
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInternal, "It's an internal error."));
}

TEST(StatusMacrosTest, RetCheck_Success) {
  auto status = []() -> absl::Status {
    RET_CHECK(true) << "It's a RET_CHECK failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_Failure) {
  auto status = []() -> absl::Status {
    RET_CHECK(false) << "It's a RET_CHECK failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "false: It's a RET_CHECK failure."));
}

TEST(StatusMacrosTest, RetCheck_EQ_Success) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_EQ(x, 1) << "It's a RET_CHECK_EQ failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_EQ_Failure) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_EQ(x, 1) << "It's a RET_CHECK_EQ failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) == (1): It's a RET_CHECK_EQ failure."));
}

TEST(StatusMacrosTest, RetCheck_EQ_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_EQ(x, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_EQ failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) == (1): It's a RET_CHECK_EQ failure."));
}

TEST(StatusMacrosTest, RetCheck_NE_Success) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_NE(x, 1) << "It's a RET_CHECK_NE failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_NE_Failure) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_NE(x, 1) << "It's a RET_CHECK_NE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) != (1): It's a RET_CHECK_NE failure."));
}

TEST(StatusMacrosTest, RetCheck_NE_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_NE(x, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_NE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) != (1): It's a RET_CHECK_NE failure."));
}

TEST(StatusMacrosTest, RetCheck_GT_Success) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_GT(x, 1) << "It's a RET_CHECK_GT failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_GT_Failure) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_GT(x, 1) << "It's a RET_CHECK_GT failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) > (1): It's a RET_CHECK_GT failure."));
}

TEST(StatusMacrosTest, RetCheck_GT_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_GT(x, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_GT failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) > (1): It's a RET_CHECK_GT failure."));
}

TEST(StatusMacrosTest, RetCheck_LT_Success) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_LT(x, 2) << "It's a RET_CHECK_LT failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_LT_Failure) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_LT(x, 1) << "It's a RET_CHECK_LT failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) < (1): It's a RET_CHECK_LT failure."));
}

TEST(StatusMacrosTest, RetCheck_LT_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_LT(x, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_LT failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) < (1): It's a RET_CHECK_LT failure."));
}

TEST(StatusMacrosTest, RetCheck_GE_Success) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_GE(x, 1) << "It's a RET_CHECK_GE failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_GE_Failure) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_GE(x, 2) << "It's a RET_CHECK_GE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) >= (2): It's a RET_CHECK_GE failure."));
}

TEST(StatusMacrosTest, RetCheck_GE_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_GE(x, 2).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_GE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) >= (2): It's a RET_CHECK_GE failure."));
}

TEST(StatusMacrosTest, RetCheck_LE_Success) {
  auto status = []() -> absl::Status {
    int x = 1;
    RET_CHECK_LE(x, 2) << "It's a RET_CHECK_LE failure.";
    return absl::OkStatus();
  }();
  EXPECT_OK(status);
}

TEST(StatusMacrosTest, RetCheck_LE_Failure) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_LE(x, 1) << "It's a RET_CHECK_LE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               "(x) <= (1): It's a RET_CHECK_LE failure."));
}

TEST(StatusMacrosTest, RetCheck_LE_Failure_SetCode) {
  auto status = []() -> absl::Status {
    int x = 2;
    RET_CHECK_LE(x, 1).SetCode(absl::StatusCode::kInvalidArgument)
        << "It's a RET_CHECK_LE failure.";
    return absl::OkStatus();
  }();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               "(x) <= (1): It's a RET_CHECK_LE failure."));
}

}  // namespace
}  // namespace litert::lm
