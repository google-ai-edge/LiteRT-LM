// Copyright 2025 The ODML Authors.
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

#include "runtime/util/lora_util.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

void WriteFile(absl::string_view path, absl::string_view contents) {
  std::ofstream ofstr(std::string(path), std::ios::out);
  ofstr << contents;
}

void CheckContents(MemoryMappedFileWithAutoAlignment& file,
                   absl::string_view expected) {
  EXPECT_EQ(file.length(), expected.size());

  absl::string_view contents(static_cast<const char*>(file.data()),
                             file.length());
  EXPECT_EQ(contents, expected);
}

TEST(MemoryMappedFileWithAutoAlignment, SucceedsMapping) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto handle = ScopedFile::Open(path.string());
  ASSERT_OK(handle);

  auto file = MemoryMappedFileWithAutoAlignment::Create(handle->file());
  ASSERT_OK(file);
  CheckContents(**file, "foo bar");
}

TEST(MemoryMappedFileWithAutoAlignment, SucceedsMappingLengthAndOffset) {
  size_t alignment = MemoryMappedFile::GetOffsetAlignment();
  ASSERT_GT(alignment, 10);  // Ensure alignment is large enough for test

  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  std::string file_contents = "BEGIN_";
  file_contents += std::string(alignment - 10, 'A');
  file_contents += "ALIGN1_";  // "A" is at offset alignment - 4
  file_contents += std::string(alignment - 10, 'B');
  file_contents += "ALIGN2_";  // "A" is at offset 2 * alignment - 7
  file_contents += "END";
  WriteFile(path.string(), file_contents);

  auto scoped_file = *ScopedFile::Open(path.string());

  // Case 1: offset = 0, size = 0 (whole file)
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(scoped_file.file());
    ASSERT_OK(file);
    CheckContents(**file, file_contents);
  }

  // Case 2: offset = alignment, size = 0 (from offset to end)
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(),
                                                          alignment);
    ASSERT_OK(file);
    CheckContents(**file, file_contents.substr(alignment));
  }

  // Case 3: offset = 0, size = 3
  {
    auto file =
        MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(), 0, 3);
    ASSERT_OK(file);
    CheckContents(**file, "BEG");
  }

  // Case 4: offset = alignment - 2, size = 5
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(),
                                                          alignment - 2, 5);
    ASSERT_OK(file);
    CheckContents(**file, "IGN1_");  // file_contents.substr(alignment - 2, 5))
  }

  // Case 5: offset = alignment + 1, size = 3 (unaligned offset)
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(),
                                                          alignment + 1, 3);
    ASSERT_OK(file);
    CheckContents(**file, "1_B");
  }

  // Case 6: offset = alignment - 1, size = 0 (unaligned offset to end)
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(),
                                                          alignment - 1);
    ASSERT_OK(file);
    CheckContents(**file, file_contents.substr(alignment - 1));
  }

  // Case 7: offset = 1, size = 5 (unaligned offset)
  {
    auto file =
        MemoryMappedFileWithAutoAlignment::Create(scoped_file.file(), 1, 5);
    ASSERT_OK(file);
    CheckContents(**file, "EGIN_");
  }

  // Case 8: offset = 2 * alignment - 10, size = 6
  {
    auto file = MemoryMappedFileWithAutoAlignment::Create(
        scoped_file.file(), 2 * alignment - 10, 6);
    ASSERT_OK(file);
    CheckContents(**file, "BBBALI");
  }
}

TEST(MemoryMappedFileWithAutoAlignment, FailsMappingNonExistentFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "bad.txt";
  auto handle = ScopedFile::Open(path.string());
  EXPECT_FALSE(handle.ok());
}

// Test fixture for IsLoRAInputName tests.
class IsLoRAInputNameTest : public ::testing::Test {};

TEST_F(IsLoRAInputNameTest, MatchesValidPattern1) {
  // Test various valid combinations for the first pattern.
  EXPECT_TRUE(IsLoRAInputName("query_w_prime_left_0"));
  EXPECT_TRUE(IsLoRAInputName("key_w_prime_right_34"));
  EXPECT_TRUE(IsLoRAInputName("value_w_prime_left_9"));
  EXPECT_TRUE(IsLoRAInputName("post_w_prime_right_123"));
}

TEST_F(IsLoRAInputNameTest, MatchesValidPattern2) {
  // Test various valid combinations for the second pattern.
  EXPECT_TRUE(IsLoRAInputName("lora_atten_q_a_prime_weight_0"));
  EXPECT_TRUE(IsLoRAInputName("lora_atten_k_b_prime_weight_34"));
  EXPECT_TRUE(IsLoRAInputName("lora_atten_v_a_prime_weight_9"));
  EXPECT_TRUE(IsLoRAInputName("lora_atten_o_b_prime_weight_123"));
}

TEST_F(IsLoRAInputNameTest, RejectsIncorrectComponentCount) {
  // Too few parts.
  EXPECT_FALSE(IsLoRAInputName("query_w_prime_left"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_a_prime_weight"));
  // Too many parts.
  EXPECT_FALSE(IsLoRAInputName("query_w_prime_left_0_extra"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_a_prime_weight_0_extra"));
}

TEST_F(IsLoRAInputNameTest, RejectsInvalidPrefixOrKeywords) {
  // Pattern 1 with incorrect keywords.
  EXPECT_FALSE(IsLoRAInputName("badprefix_w_prime_left_0"));
  EXPECT_FALSE(IsLoRAInputName("query_x_prime_left_0"));
  EXPECT_FALSE(IsLoRAInputName("query_w_bad_left_0"));
  EXPECT_FALSE(IsLoRAInputName("query_w_prime_badside_0"));

  // Pattern 2 with incorrect keywords.
  EXPECT_FALSE(IsLoRAInputName("bad_atten_q_a_prime_weight_0"));
  EXPECT_FALSE(IsLoRAInputName("lora_bad_q_a_prime_weight_0"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_x_a_prime_weight_0"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_x_prime_weight_0"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_a_bad_weight_0"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_a_prime_bad_0"));
}

TEST_F(IsLoRAInputNameTest, RejectsNonNumericLayerNumber) {
  EXPECT_FALSE(IsLoRAInputName("query_w_prime_left_ten"));
  EXPECT_FALSE(IsLoRAInputName("lora_atten_q_a_prime_weight_one"));
  EXPECT_FALSE(IsLoRAInputName("key_w_prime_right_"));
  EXPECT_FALSE(IsLoRAInputName("key_w_prime_right_1a"));
}

TEST_F(IsLoRAInputNameTest, RejectsPartialOrIncompleteMatches) {
  // Looks like the start of a pattern but isn't a full match.
  EXPECT_FALSE(IsLoRAInputName("query_w_prime_left_0_but_theres_more"));
  EXPECT_FALSE(IsLoRAInputName("not_the_start_query_w_prime_left_0"));
}

TEST_F(IsLoRAInputNameTest, RejectsEmptyAndMalformedStrings) {
  EXPECT_FALSE(IsLoRAInputName(""));
  EXPECT_FALSE(IsLoRAInputName("____"));
  EXPECT_FALSE(IsLoRAInputName("just_a_string"));
}

}  // namespace
}  // namespace litert::lm
