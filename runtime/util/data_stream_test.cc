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

#include "runtime/util/data_stream.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/file_data_stream.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

std::string GetTestModelPath() {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  return model_path.string();
}

TEST(DataStreamTest, SubStreamReadAndPreserve) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));

  std::vector<char> buffer(8);
  ASSERT_OK(sub_stream->ReadAndPreserve(buffer.data(), 0, 8));

  std::string magic_number(
      absl::string_view(buffer.data(), buffer.size()).substr(0, 8));
  EXPECT_EQ(magic_number, "LITERTLM");
}

TEST(DataStreamTest, SubStreamReadAndDiscard) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));

  std::vector<char> buffer(8);
  ASSERT_OK(sub_stream->ReadAndDiscard(buffer.data(), 0, 8));

  std::string magic_number(
      absl::string_view(buffer.data(), buffer.size()).substr(0, 8));
  EXPECT_EQ(magic_number, "LITERTLM");
}

TEST(DataStreamTest, SubStreamReadOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));

  std::vector<char> buffer(9);
  absl::Status status = sub_stream->ReadAndPreserve(buffer.data(), 8, 1);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
  status = sub_stream->ReadAndPreserve(buffer.data(), 0, 9);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
}

TEST(DataStreamTest, SubStreamDiscard) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));

  ASSERT_OK(sub_stream->Discard(0, 8));
}

TEST(DataStreamTest, SubStreamDiscardOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));

  absl::Status status = sub_stream->Discard(8, 1);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
  status = sub_stream->Discard(0, 9);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
}

TEST(DataStreamTest, SubStreamOfSubStream) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));
  ASSERT_OK_AND_ASSIGN(auto sub_sub_stream, sub_stream->OpenSubStream(1, 7));

  std::vector<char> buffer(7);
  ASSERT_OK(sub_sub_stream->ReadAndPreserve(buffer.data(), 0, 7));

  std::string magic_number(
      absl::string_view(buffer.data(), buffer.size()).substr(0, 7));
  EXPECT_EQ(magic_number, "ITERTLM");
}

TEST(DataStreamTest, SubStreamCannotOverlap) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream1, stream->OpenSubStream(0, 10));
  absl::Status status = stream->OpenSubStream(5, 10).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(),
              HasSubstr("overlaps with an existing locked region"));
  status = stream->OpenSubStream(0, 1).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(),
              HasSubstr("overlaps with an existing locked region"));
  status = stream->OpenSubStream(9, 10).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(),
              HasSubstr("overlaps with an existing locked region"));
  // Non-overlapping should succeed.
  ASSERT_OK(stream->OpenSubStream(10, 10));
}

TEST(DataStreamTest, SubStreamCannotBeCreatedAfterPreviousIsDestroyed) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  {
    ASSERT_OK_AND_ASSIGN(auto sub_stream1, stream->OpenSubStream(0, 10));
    EXPECT_THAT(stream->OpenSubStream(0, 10),
                StatusIs(absl::StatusCode::kAlreadyExists));
  }
  // Even though sub_stream1 is out of scope, we should not be able to create a
  // new stream on the same region.
  EXPECT_THAT(stream->OpenSubStream(0, 10),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(DataStreamTest, SubStreamOfSubStreamOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 8));
  absl::Status status = sub_stream->OpenSubStream(0, 9).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
  status = sub_stream->OpenSubStream(1, 8).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
  status = sub_stream->OpenSubStream(8, 1).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Exceeded bounds of substream"));
}

TEST(DataStreamTest, SubSubStreamCannotOverlap) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK_AND_ASSIGN(auto sub_stream, stream->OpenSubStream(0, 100));
  ASSERT_OK(sub_stream->OpenSubStream(10, 20));
  absl::Status status = sub_stream->OpenSubStream(15, 20).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(),
              HasSubstr("overlaps with an existing locked region"));
  status = sub_stream->OpenSubStream(0, 11).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(),
              HasSubstr("overlaps with an existing locked region"));
  // Non-overlapping should succeed.
  ASSERT_OK(sub_stream->OpenSubStream(30, 10));
}

}  // namespace
}  // namespace litert::lm
