#include "runtime/components/token_id_util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(TokenIdUtilTest, PreprocessTokenIds) {
  std::vector<int> token_ids = {1, 2, 3, 4, 5};
  EXPECT_OK(PreprocessTokenIds(token_ids, /*start_token_id=*/0,
                               /*max_num_tokens=*/10,
                               /*context_length_ratio_threhold=*/0.9f));
  EXPECT_THAT(token_ids, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TokenIdUtilTest, PreprocessTokenIdsExceeedThreshold) {
  std::vector<int> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_THAT(PreprocessTokenIds(token_ids, /*start_token_id=*/0,
                                 /*max_num_tokens=*/10,
                                 /*context_length_ratio_threhold=*/0.9f),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TokenIdUtilTest, StopTokenFoundTrue) {
  std::vector<int> decoded_token_ids = {0, 2, 0, 4, 5};
  std::vector<bool> stop_token_found = {false, true, false, true, true};
  EXPECT_THAT(StopTokenFound(decoded_token_ids, /*stop_token_ids=*/{0},
                             stop_token_found),
              IsOkAndHolds(true));
}

TEST(TokenIdUtilTest, MultiStopTokenFoundTrue) {
  std::vector<int> decoded_token_ids = {0, 2, 1, 4, 5};
  std::vector<bool> stop_token_found = {false, true, false, true, true};
  EXPECT_THAT(StopTokenFound(decoded_token_ids, /*stop_token_ids=*/{0, 1},
                             stop_token_found),
              IsOkAndHolds(true));
}

TEST(TokenIdUtilTest, StopTokenFoundFalse) {
  std::vector<int> decoded_token_ids = {0, 2, 0, 4, 5};
  std::vector<bool> stop_token_found = {false, false, false, false, false};
  EXPECT_THAT(StopTokenFound(decoded_token_ids, /*stop_token_ids=*/{0},
                             stop_token_found),
              IsOkAndHolds(false));
  EXPECT_THAT(stop_token_found, ElementsAre(true, false, true, false, false));
}

TEST(TokenIdUtilTest, StopTokenFoundInvalidInput) {
  std::vector<int> decoded_token_ids = {1, 2, 3, 4, 5};
  std::vector<bool> stop_token_found = {false, false, false, false};
  EXPECT_THAT(StopTokenFound(decoded_token_ids, /*stop_token_ids=*/{4},
                             stop_token_found),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
