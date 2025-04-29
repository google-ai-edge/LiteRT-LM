// Copyright 2024 The ODML Authors.
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

#include "runtime/util/memory_mapped_file.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

void WriteFile(absl::string_view path, absl::string_view contents) {
  std::ofstream ofstr(std::string(path), std::ios::out);
  ofstr << contents;
}

std::string ReadFile(absl::string_view path) {
  auto ifstr = std::ifstream(std::string(path));
  std::stringstream contents;
  contents << ifstr.rdbuf();
  return contents.str();
}

void CheckContents(MemoryMappedFile& file, absl::string_view expected) {
  EXPECT_EQ(file.length(), expected.size());

  absl::string_view contents(static_cast<const char*>(file.data()),
                             file.length());
  EXPECT_EQ(contents, expected);
}

TEST(MemoryMappedFile, SucceedsMapping) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = MemoryMappedFile::Create(path.string());
  ASSERT_OK(file);
  CheckContents(**file, "foo bar");
}

TEST(MemoryMappedFile, SucceedsMappingOpenFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  absl::StatusOr<std::unique_ptr<MemoryMappedFile>> file;
  {
    auto handle = ScopedFile::Open(path.string());
    ASSERT_OK(handle);
    file = MemoryMappedFile::Create(handle->file());
  }

  ASSERT_OK(file);
  CheckContents(**file, "foo bar");
}

TEST(MemoryMappedFile, MapsValidScopedFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto scoped_file = ScopedFile::Open(path.string());
  ASSERT_OK(scoped_file);
  {
    auto file = MemoryMappedFile::Create(scoped_file->file());
    ASSERT_OK(file);
    CheckContents(**file, "foo bar");
  }
  // Save handle to make sure it gets closed.
  auto handle = scoped_file->file();
  {
    ScopedFile other_file = std::move(*scoped_file);
    EXPECT_FALSE(MemoryMappedFile::Create(scoped_file->file()).ok());
    auto file = MemoryMappedFile::Create(other_file.file());
    ASSERT_OK(file);
    CheckContents(**file, "foo bar");
  }
  EXPECT_FALSE(MemoryMappedFile::Create(handle).ok());
}

TEST(MemoryMappedFile, SucceedsMappingLengthAndOffset) {
  size_t offset = MemoryMappedFile::GetOffsetAlignment();
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  std::string file_contents(offset, ' ');
  file_contents += "foo bar";
  WriteFile(path.string(), file_contents);

  auto scoped_file = *ScopedFile::Open(path.string());
  {
    auto file = MemoryMappedFile::Create(scoped_file.file(), offset);
    ASSERT_OK(file);
    CheckContents(**file, "foo bar");
  }
  {
    auto file = MemoryMappedFile::Create(scoped_file.file(), offset, 3);
    ASSERT_OK(file);
    CheckContents(**file, "foo");
  }
  {
    auto file = MemoryMappedFile::Create(scoped_file.file());
    ASSERT_OK(file);
    CheckContents(**file, file_contents);
  }
  {
    auto file1 = MemoryMappedFile::Create(scoped_file.file(), offset, 3, "key");
    ASSERT_OK(file1);
    CheckContents(**file1, "foo");

    auto file2 = MemoryMappedFile::Create(scoped_file.file(), offset, 5, "key");
    ASSERT_OK(file2);
    CheckContents(**file2, "foo b");
  }
}

TEST(MemoryMappedFile, FailsMappingNonExistentFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "bad.txt";
  ASSERT_FALSE(MemoryMappedFile::Create(path.string()).ok());
}

TEST(MemoryMappedFile, ModifiesDataButNotFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = MemoryMappedFile::Create(path.string());
  ASSERT_OK(file);
  EXPECT_EQ((*file)->length(), 7);
  char* data = static_cast<char*>((*file)->data());
  data[0] = 'x';

  CheckContents(**file, "xoo bar");
  EXPECT_EQ(ReadFile(path.string()), "foo bar");
}

TEST(MemoryMappedFile, ModifiesFileWhenMutable) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = MemoryMappedFile::CreateMutable(path.string());
  ASSERT_OK(file);
  EXPECT_EQ((*file)->length(), 7);
  char* data = static_cast<char*>((*file)->data());
  data[0] = 'x';

  CheckContents(**file, "xoo bar");
  EXPECT_EQ(ReadFile(path.string()), "xoo bar");
}

}  // namespace
}  // namespace litert::lm
