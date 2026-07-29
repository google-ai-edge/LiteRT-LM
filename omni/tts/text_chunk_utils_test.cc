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

#include "omni/tts/text_chunk_utils.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {
namespace {

TEST(TextChunkUtilsTest, FindFirstDelimiterDefaultConfig) {
  TextChunkConfig config;

  auto match1 = FindFirstDelimiter("Hello world. How are you?", config);
  EXPECT_TRUE(match1.found());
  EXPECT_EQ(match1.pos, 11);  // index of '.' after world
  EXPECT_EQ(match1.length, 1);

  auto match2 = FindFirstDelimiter("No delimiters here", config);
  EXPECT_FALSE(match2.found());
  EXPECT_EQ(match2.pos, absl::string_view::npos);
}

TEST(TextChunkUtilsTest, FindFirstDelimiterEarliestAndLongestMatch) {
  TextChunkConfig config;
  config.delimiters = {".", "..."};

  // When '...' matches at pos 5, both '.' and '...' match at pos 5.
  // FindFirstDelimiter should prefer the longer delimiter match.
  auto match = FindFirstDelimiter("hello...world", config);
  EXPECT_TRUE(match.found());
  EXPECT_EQ(match.pos, 5);
  EXPECT_EQ(match.length, 3);
}

TEST(TextChunkUtilsTest, ShouldScheduleConditions) {
  TextChunkConfig config;
  config.max_buffer_size = 10;

  // 1. Buffer has delimiter -> true
  EXPECT_TRUE(ShouldSchedule("hello.", /*is_finished=*/false, config));

  // 2. Buffer has no delimiter, not finished, under max size -> false
  EXPECT_FALSE(ShouldSchedule("hello", /*is_finished=*/false, config));

  // 3. Buffer has no delimiter, finished, non-empty -> true
  EXPECT_TRUE(ShouldSchedule("hello", /*is_finished=*/true, config));

  // 4. Buffer is empty, finished -> false
  EXPECT_FALSE(ShouldSchedule("", /*is_finished=*/true, config));

  // 5. Buffer length >= max_buffer_size -> true
  EXPECT_TRUE(ShouldSchedule("12345678901", /*is_finished=*/false, config));
}

TEST(TextChunkUtilsTest, ExtractNextChunkIncludeDelimiter) {
  TextChunkConfig config;
  config.include_delimiter = true;
  std::string buffer = "First sentence. Second sentence!";

  auto chunk1 = ExtractNextChunk(buffer, /*start_index=*/0,
                                 /*is_finished=*/false, config);
  ASSERT_TRUE(chunk1.has_value());
  EXPECT_EQ(chunk1->chunk, "First sentence.");
  EXPECT_EQ(chunk1->next_start_index, 15);

  auto chunk2 =
      ExtractNextChunk(buffer, /*start_index=*/chunk1->next_start_index,
                       /*is_finished=*/false, config);
  ASSERT_TRUE(chunk2.has_value());
  EXPECT_EQ(chunk2->chunk, " Second sentence!");
  EXPECT_EQ(chunk2->next_start_index, buffer.size());
}

TEST(TextChunkUtilsTest, ExtractNextChunkExcludeDelimiter) {
  TextChunkConfig config;
  config.delimiters = {","};
  config.include_delimiter = false;
  std::string buffer = "apple,banana,cherry";

  auto chunk1 = ExtractNextChunk(buffer, /*start_index=*/0,
                                 /*is_finished=*/false, config);
  ASSERT_TRUE(chunk1.has_value());
  EXPECT_EQ(chunk1->chunk, "apple");
  EXPECT_EQ(chunk1->next_start_index, 6);

  auto chunk2 =
      ExtractNextChunk(buffer, /*start_index=*/chunk1->next_start_index,
                       /*is_finished=*/false, config);
  ASSERT_TRUE(chunk2.has_value());
  EXPECT_EQ(chunk2->chunk, "banana");
  EXPECT_EQ(chunk2->next_start_index, 13);

  auto chunk3 =
      ExtractNextChunk(buffer, /*start_index=*/chunk2->next_start_index,
                       /*is_finished=*/true, config);
  ASSERT_TRUE(chunk3.has_value());
  EXPECT_EQ(chunk3->chunk, "cherry");
  EXPECT_EQ(chunk3->next_start_index, buffer.size());
}

TEST(TextChunkUtilsTest, ExtractNextChunkMaxBufferSize) {
  TextChunkConfig config;
  config.delimiters = {};
  config.max_buffer_size = 5;
  std::string buffer = "123456789012";

  auto chunk1 = ExtractNextChunk(buffer, /*start_index=*/0,
                                 /*is_finished=*/false, config);
  ASSERT_TRUE(chunk1.has_value());
  EXPECT_EQ(chunk1->chunk, "12345");
  EXPECT_EQ(chunk1->next_start_index, 5);

  auto chunk2 =
      ExtractNextChunk(buffer, /*start_index=*/chunk1->next_start_index,
                       /*is_finished=*/false, config);
  ASSERT_TRUE(chunk2.has_value());
  EXPECT_EQ(chunk2->chunk, "67890");
  EXPECT_EQ(chunk2->next_start_index, 10);

  auto chunk3 =
      ExtractNextChunk(buffer, /*start_index=*/chunk2->next_start_index,
                       /*is_finished=*/true, config);
  ASSERT_TRUE(chunk3.has_value());
  EXPECT_EQ(chunk3->chunk, "12");
  EXPECT_EQ(chunk3->next_start_index, buffer.size());
}

TEST(TextChunkUtilsTest, ExtractNextChunkLeadingDelimiterExcluded) {
  TextChunkConfig config;
  config.delimiters = {","};
  config.include_delimiter = false;
  std::string buffer = ",world,";

  auto chunk1 = ExtractNextChunk(buffer, /*start_index=*/0,
                                 /*is_finished=*/false, config);
  ASSERT_TRUE(chunk1.has_value());
  EXPECT_EQ(chunk1->chunk, "world");
  EXPECT_EQ(chunk1->next_start_index, 7);
}

}  // namespace
}  // namespace litert::omni::tts
