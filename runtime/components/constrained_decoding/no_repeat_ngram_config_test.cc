#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(NoRepeatNgramConfigTest, ClampsValuesToLowerBounds) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/-5, /*window_size=*/-3);

  // `no_repeat_ngram_size` should be clamped to 0.
  EXPECT_EQ(config.no_repeat_ngram_size(), 0);
  // `window_size` should be clamped to 0.
  EXPECT_EQ(config.window_size(), 0);
  EXPECT_FALSE(config.enabled());
}

TEST(NoRepeatNgramConfigTest, RetainsValidValues) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/10);

  EXPECT_EQ(config.no_repeat_ngram_size(), 3);
  EXPECT_EQ(config.window_size(), 10);
  EXPECT_TRUE(config.enabled());
}

TEST(NoRepeatNgramConfigTest, WindowSizeLessThanNgramSize) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/5, /*window_size=*/3);

  EXPECT_EQ(config.no_repeat_ngram_size(), 5);
  EXPECT_EQ(config.window_size(), 5);
  EXPECT_TRUE(config.enabled());
}

TEST(NoRepeatNgramConfigTest, DefaultIsDisabled) {
  NoRepeatNgramConfig config = NoRepeatNgramConfig::Default();

  EXPECT_EQ(config.no_repeat_ngram_size(), 0);
  EXPECT_EQ(config.window_size(), 0);
  EXPECT_FALSE(config.enabled());
}

}  // namespace
}  // namespace litert::lm
