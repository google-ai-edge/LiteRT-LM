#include "runtime/engine/io_types.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

std::string FloatToString(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

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

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";
  responses.GetMutableScores()[0] = 0.9f;
  responses.GetMutableScores()[1] = -0.5f;  // Test with a negative score

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: " +
      FloatToString(0.9f) +
      "):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: " +
      FloatToString(-0.5f) +
      "):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndNoScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: N/A):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: N/A):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

}  // namespace
}  // namespace litert::lm
