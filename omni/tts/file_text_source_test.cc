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

#include "omni/tts/file_text_source.h"

#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "omni/tts/text_chunk_utils.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

using ::absl_testing::IsOk;

std::string CreateTempFile(const std::string& filename,
                           const std::string& content) {
  std::string file_path = absl::StrCat(testing::TempDir(), "/", filename);
  std::ofstream ofs(file_path);
  ofs << content;
  ofs.close();
  return file_path;
}

TEST(FileTextSourceTest, NonExistentFileFails) {
  auto source = FileTextSource::Create(
      absl::StrCat(testing::TempDir(), "/non_existent_file_12345.txt"));
  EXPECT_FALSE(source.ok());
}

TEST(FileTextSourceTest, ReadFileWithDefaultDelimiters) {
  std::string file_path =
      CreateTempFile("test1.txt", "Hello world. How are you?\nFine!");
  ASSERT_OK_AND_ASSIGN(auto source, FileTextSource::Create(file_path));

  std::vector<std::string> chunks;
  while (source->NeedSchedule()) {
    ASSERT_THAT(source->Schedule(), IsOk());
    auto output = source->GetOutput();
    if (output.ok()) {
      chunks.push_back(*output);
    }
  }

  EXPECT_THAT(chunks, ::testing::ElementsAre("Hello world.", " How are you?",
                                             "\n", "Fine!"));
}

TEST(FileTextSourceTest, ReadFileWithCustomDelimiterAndNoInclude) {
  std::string file_path = CreateTempFile("test2.txt", "apple,banana,cherry");
  TextChunkConfig config;
  config.delimiters = {","};
  config.include_delimiter = false;

  ASSERT_OK_AND_ASSIGN(auto source, FileTextSource::Create(file_path, config));

  std::vector<std::string> chunks;
  while (source->NeedSchedule()) {
    ASSERT_THAT(source->Schedule(), IsOk());
    auto output = source->GetOutput();
    if (output.ok()) {
      chunks.push_back(*output);
    }
  }

  EXPECT_THAT(chunks, ::testing::ElementsAre("apple", "banana", "cherry"));
}

TEST(FileTextSourceTest, ReadFileWithMaxBufferSize) {
  std::string file_path = CreateTempFile("test3.txt", "1234567890");
  TextChunkConfig config;
  config.delimiters = {};
  config.max_buffer_size = 4;

  ASSERT_OK_AND_ASSIGN(auto source, FileTextSource::Create(file_path, config));

  std::vector<std::string> chunks;
  while (source->NeedSchedule()) {
    ASSERT_THAT(source->Schedule(), IsOk());
    auto output = source->GetOutput();
    if (output.ok()) {
      chunks.push_back(*output);
    }
  }

  EXPECT_THAT(chunks, ::testing::ElementsAre("1234", "5678", "90"));
}

TEST(FileTextSourceTest, ResetReReadsFile) {
  std::string file_path =
      CreateTempFile("test4.txt", "First line.\nSecond line.");
  ASSERT_OK_AND_ASSIGN(auto source, FileTextSource::Create(file_path));

  // First pass
  std::vector<std::string> pass1;
  while (source->NeedSchedule()) {
    ASSERT_THAT(source->Schedule(), IsOk());
    auto output = source->GetOutput();
    if (output.ok()) {
      pass1.push_back(*output);
    }
  }
  EXPECT_THAT(pass1,
              ::testing::ElementsAre("First line.", "\n", "Second line."));

  // Reset
  source->Reset();

  // Second pass
  std::vector<std::string> pass2;
  while (source->NeedSchedule()) {
    ASSERT_THAT(source->Schedule(), IsOk());
    auto output = source->GetOutput();
    if (output.ok()) {
      pass2.push_back(*output);
    }
  }
  EXPECT_THAT(pass2,
              ::testing::ElementsAre("First line.", "\n", "Second line."));
}

}  // namespace
}  // namespace litert::omni::tts
