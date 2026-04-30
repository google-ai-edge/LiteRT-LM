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

#include "runtime/util/file_util.h"

#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

#if defined(_WIN32)
constexpr absl::string_view kPathSeparator = "\\";
#else
constexpr absl::string_view kPathSeparator = "/";
#endif

TEST(FileUtilTest, JoinPath) {
  std::string path1 = "";
  std::string path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path1."));

  path1 = "path1";
  path2 = "";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path2."));

  path1 = "path1";
  path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::StrCat("path1", kPathSeparator, "path2"));
}

TEST(FileUtilTest, Basename) {
  std::string model_path = absl::StrCat(kPathSeparator, "path", kPathSeparator,
                                        "to", kPathSeparator, "model.tflite");
  EXPECT_THAT(Basename(model_path), "model.tflite");
}

TEST(FileUtilTest, Dirname) {
  std::string model_path = absl::StrCat(kPathSeparator, "path", kPathSeparator,
                                        "to", kPathSeparator, "model.tflite");
  EXPECT_THAT(Dirname(model_path),
              absl::StrCat(kPathSeparator, "path", kPathSeparator, "to",
                           kPathSeparator));
}

TEST(FileUtilTest, GetFileCacheIdentifier) {
  ASSERT_OK_AND_ASSIGN(auto temp_file,
                       JoinPath(testing::TempDir(), "test_file.txt"));
  std::ofstream ofs(temp_file);
  ofs << "test data";
  ofs.close();

  ASSERT_OK_AND_ASSIGN(auto id, GetFileCacheIdentifier(temp_file));
  // Split the ID into {timestamp}_{filesize}. We avoid using MatchesRegex
  // because gtest's simplified regex engine on Windows doesn't support
  // character classes.
  std::vector<std::string> parts = absl::StrSplit(id, '_');
  ASSERT_EQ(parts.size(), 2);

  // The first part is the last modified timestamp, which can be negative in
  // some environments.
  absl::string_view ts = parts[0];
  if (ts.starts_with('-')) {
    ts.remove_prefix(1);
  }
  EXPECT_FALSE(ts.empty());
  for (char c : ts) {
    EXPECT_TRUE(c >= '0' && c <= '9');
  }

  // The second part is the file size. "test data" is exactly 9 bytes.
  EXPECT_EQ(parts[1], "9");

  EXPECT_FALSE(GetFileCacheIdentifier("non_existent_file").ok());
}

TEST(FileUtilTest, FileExists) {
  ASSERT_OK_AND_ASSIGN(auto temp_file,
                       JoinPath(testing::TempDir(), "exists_test.txt"));
  EXPECT_FALSE(FileExists(temp_file));

  std::ofstream ofs(temp_file);
  ofs << "data";
  ofs.close();

  EXPECT_TRUE(FileExists(temp_file));
}

TEST(FileUtilTest, DeleteStaleCaches) {
  std::string temp_dir = testing::TempDir();
  std::string model_name = "model.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_path, JoinPath(temp_dir, model_name));

  std::ofstream model_ofs(model_path);
  model_ofs << "model data";
  model_ofs.close();

  ASSERT_OK_AND_ASSIGN(auto id, GetFileCacheIdentifier(model_path));
  std::string identifier = absl::StrCat("_", id);

  // Generate file: model.litertlm.xnnpack_cache_identifier
  ASSERT_OK_AND_ASSIGN(
      auto tail_identifier_cache,
      JoinPath(temp_dir,
               absl::StrCat(model_name, ".xnnpack_cache", identifier)));
  std::ofstream tail_ofs(tail_identifier_cache);
  tail_ofs << "tail identifier cache";
  tail_ofs.close();

  // Generate file: model.litertlm.identifier_mldrift_program_cache.bin
  ASSERT_OK_AND_ASSIGN(
      auto middle_identifier_cache,
      JoinPath(temp_dir, absl::StrCat(model_name, identifier,
                                      "_mldrift_program_cache.bin")));
  std::ofstream middle_ofs(middle_identifier_cache);
  middle_ofs << "middle identifier cache";
  middle_ofs.close();

  // Generate file: model.litertlm.xnnpack_cache (existing xnnpack cache)
  ASSERT_OK_AND_ASSIGN(
      auto existing_cpu_cache,
      JoinPath(temp_dir, absl::StrCat(model_name, ".xnnpack_cache")));
  std::ofstream exact_ofs(existing_cpu_cache);
  exact_ofs << "exact cache";
  exact_ofs.close();

  // Generate file: model.litertlm_mldrift_program_cache.bin
  ASSERT_OK_AND_ASSIGN(
      auto existing_gpu_cache,
      JoinPath(temp_dir,
               absl::StrCat(model_name, "_mldrift_program_cache.bin")));
  std::ofstream gpu_ofs(existing_gpu_cache);
  gpu_ofs << "existing gpu cache";
  gpu_ofs.close();

  ASSERT_OK_AND_ASSIGN(auto unrelated, JoinPath(temp_dir, "unrelated.txt"));
  std::ofstream unrelated_ofs(unrelated);
  unrelated_ofs << "unrelated data";
  unrelated_ofs.close();

  // Delete CPU caches.
  auto cpu_deleted = DeleteStaleCaches(temp_dir, model_name, ".xnnpack_cache");
  ASSERT_OK(cpu_deleted);
  EXPECT_EQ(*cpu_deleted, 2);

  EXPECT_TRUE(FileExists(model_path));
  // CPU caches should be removed.
  EXPECT_FALSE(FileExists(tail_identifier_cache));
  EXPECT_FALSE(FileExists(existing_cpu_cache));
  // GPU caches should exist.
  EXPECT_TRUE(FileExists(middle_identifier_cache));
  EXPECT_TRUE(FileExists(existing_gpu_cache));
  EXPECT_TRUE(FileExists(unrelated));

  // Delete GPU caches.
  auto gpu_deleted = DeleteStaleCaches(
      temp_dir, model_name, "_mldrift_program_cache.bin");
  ASSERT_OK(gpu_deleted);
  EXPECT_EQ(*gpu_deleted, 2);

  EXPECT_TRUE(FileExists(model_path));
  // GPU caches should be removed.
  EXPECT_FALSE(FileExists(middle_identifier_cache));
  EXPECT_FALSE(FileExists(existing_gpu_cache));
  EXPECT_TRUE(FileExists(unrelated));
}

}  // namespace
}  // namespace litert::lm
