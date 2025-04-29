#include "runtime/util/litert_status_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

absl::StatusOr<bool> ReturnHelper(litert::Expected<bool> expected) {
  LITERT_ASSIGN_OR_RETURN_ABSL(bool result, expected);
  return result;
}

TEST(TfliteGpuExecutorUtilsTest, LitertAssignOrReturnAbsl) {
  EXPECT_TRUE(ReturnHelper(true).ok());
  EXPECT_TRUE(ReturnHelper(false).ok());

  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidArgument)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorMemoryAllocationFailure)),
      StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorRuntimeFailure)),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorMissingInputTensor)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorUnsupported)),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorNotFound)),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorTimeoutExpired)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorWrongVersion)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorUnknown)),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorFileIO)),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidFlatbuffer)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorDynamicLoading)),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorSerialization)),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorCompilation)),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorIndexOOB)),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidIrType)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidGraphInvariant)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorGraphModification)),
      StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidToolConfig)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ReturnHelper(litert::Error(kLiteRtStatusLegalizeNoMatch)),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      ReturnHelper(litert::Error(kLiteRtStatusErrorInvalidLegalization)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
