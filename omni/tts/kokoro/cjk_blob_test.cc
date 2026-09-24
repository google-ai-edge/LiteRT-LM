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

#include "omni/tts/kokoro/cjk_blob.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/tts/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::testing::status::IsOkAndHolds;

// Packs `keys` into the (offset index, NUL-separated blob) pair the readers
// expect. Callers pass them already sorted, as the packer emits them.
std::pair<std::string, std::string> PackStrings(
    const std::vector<std::string>& keys) {
  std::string index;
  std::string blob;
  for (const std::string& key : keys) {
    AppendUint32(blob.size(), index);
    blob += key;
    blob.push_back('\0');
  }
  AppendUint32(blob.size(), index);
  return {index, blob};
}

TEST(CjkBlobTest, ReadsSectionsByName) {
  const std::string blob =
      BuildCjkBlob({{"w-key", "hello"}, {"p-val", "world!!"}});
  ASSERT_OK_AND_ASSIGN(const CjkBlob parsed, CjkBlob::Create(blob));

  EXPECT_TRUE(parsed.Contains("w-key"));
  EXPECT_TRUE(parsed.Contains("p-val"));
  EXPECT_FALSE(parsed.Contains("missing"));
  EXPECT_THAT(parsed.Section("w-key"), IsOkAndHolds("hello"));
  EXPECT_THAT(parsed.Section("p-val"), IsOkAndHolds("world!!"));
  EXPECT_THAT(
      parsed.Section("missing").status(),
      ::testing::Property(&absl::Status::code, absl::StatusCode::kNotFound));
}

TEST(CjkBlobTest, RejectsMalformedBlobs) {
  EXPECT_THAT(CjkBlob::Create("").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CjkBlob::Create(std::string(64, '\0')).status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  // A section that runs off the end of the blob must be caught rather than
  // handed out as a view into whatever follows in memory.
  std::string truncated = BuildCjkBlob({{"w-key", "hello"}});
  truncated.resize(truncated.size() - 4);
  EXPECT_THAT(CjkBlob::Create(truncated).status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
}

TEST(SortedStringIndexTest, FindsExactKeys) {
  const auto [index, keys] = PackStrings({"一点", "中国", "你好", "语音"});
  ASSERT_OK_AND_ASSIGN(const SortedStringIndex sorted,
                       SortedStringIndex::Create(index, keys));

  EXPECT_EQ(sorted.Size(), 4);
  EXPECT_EQ(sorted.Key(0), "一点");
  EXPECT_EQ(sorted.Find("你好"), 2);
  EXPECT_EQ(sorted.Find("语音"), 3);
  EXPECT_EQ(sorted.Find("世界"), -1);
  EXPECT_EQ(sorted.Find(""), -1);
}

TEST(SortedStringIndexTest, AnswersThePrefixQuestionJiebaAsks) {
  // Building the DAG walks a fragment out one character at a time and stops as
  // soon as no entry starts with it, so the prefix test has to be exact about
  // partial multi-byte characters too.
  const auto [index, keys] = PackStrings({"中国", "中国人", "中心"});
  ASSERT_OK_AND_ASSIGN(const SortedStringIndex sorted,
                       SortedStringIndex::Create(index, keys));

  EXPECT_TRUE(sorted.HasPrefix("中"));
  EXPECT_TRUE(sorted.HasPrefix("中国"));
  EXPECT_TRUE(sorted.HasPrefix("中国人"));
  EXPECT_TRUE(sorted.HasPrefix(""));
  EXPECT_FALSE(sorted.HasPrefix("中国人民"));
  EXPECT_FALSE(sorted.HasPrefix("人"));

  // A key is a prefix of itself, but is not necessarily an entry.
  EXPECT_EQ(sorted.Find("中"), -1);
}

TEST(SortedStringIndexTest, RejectsMalformedIndexes) {
  EXPECT_THAT(SortedStringIndex::Create("", "").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  // Offsets must strictly ascend and stay inside the payload.
  std::string index;
  AppendUint32(0, index);
  AppendUint32(99, index);
  EXPECT_THAT(SortedStringIndex::Create(index, "ab").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  std::string duplicate_offsets;
  AppendUint32(0, duplicate_offsets);
  AppendUint32(2, duplicate_offsets);
  AppendUint32(2, duplicate_offsets);
  EXPECT_THAT(SortedStringIndex::Create(duplicate_offsets,
                                        absl::string_view("a\0b\0", 4))
                  .status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
}

TEST(StringValueTableTest, ReadsValuesByPosition) {
  const auto [index, values] = PackStrings({"ni↓xau↓", "ꭧʊ→ŋkwo↗"});
  ASSERT_OK_AND_ASSIGN(const StringValueTable table,
                       StringValueTable::Create(index, values));

  EXPECT_EQ(table.Size(), 2);
  EXPECT_EQ(table.Value(0), "ni↓xau↓");
  EXPECT_EQ(table.Value(1), "ꭧʊ→ŋkwo↗");
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
