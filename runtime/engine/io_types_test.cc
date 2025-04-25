#include "runtime/engine/io_types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @abseil-cpp

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;
using ::testing::status::IsOkAndHolds;

TEST(ResponsesTest, GetResponseTextAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetResponseTextAt(0), IsOkAndHolds("Hello World!"));
  EXPECT_THAT(responses.GetResponseTextAt(1), IsOkAndHolds("How's it going?"));
  EXPECT_THAT(responses.GetResponseTextAt(2),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetScoreAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetMutableScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, HasScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
}

TEST(ResponsesTest, GetMutableResponseTexts) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetMutableResponseTexts()[0], "Hello World!");
  EXPECT_THAT(responses.GetMutableResponseTexts()[1], "How's it going?");
}

}  // namespace
}  // namespace litert::lm
