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

#include "runtime/util/nvidia_dispatch_library.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <initializer_list>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

// Creates an empty directory `name` under the test's temporary directory, puts
// an empty file in it for each of `file_names`, and returns its path.
std::string CreateDirWithFiles(
    absl::string_view name,
    std::initializer_list<absl::string_view> file_names) {
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / std::string(name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  for (absl::string_view file_name : file_names) {
    std::ofstream(dir / std::string(file_name)).close();
  }
  return dir.string();
}

TEST(NvidiaDispatchLibraryTest, FindsTheLinuxFileName) {
  const std::string dir =
      CreateDirWithFiles("linux", {"libLiteRtDispatch_Nvidia.so"});

  EXPECT_TRUE(HasNvidiaDispatchLibrary(dir));
}

TEST(NvidiaDispatchLibraryTest, FindsTheWindowsFileName) {
  const std::string dir =
      CreateDirWithFiles("windows", {"libLiteRtDispatch_Nvidia.dll"});

  EXPECT_TRUE(HasNvidiaDispatchLibrary(dir));
}

TEST(NvidiaDispatchLibraryTest, IgnoresOtherLibraries) {
  const std::string dir = CreateDirWithFiles(
      "other_libraries",
      {"libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Nvidia.so"});

  EXPECT_FALSE(HasNvidiaDispatchLibrary(dir));
}

TEST(NvidiaDispatchLibraryTest, IgnoresADirectoryWithTheFileName) {
  const std::string dir = CreateDirWithFiles("subdirectory", {});
  std::filesystem::create_directories(std::filesystem::path(dir) /
                                      "libLiteRtDispatch_Nvidia.so");

  EXPECT_FALSE(HasNvidiaDispatchLibrary(dir));
}

TEST(NvidiaDispatchLibraryTest, FindsNothingWithoutADirectory) {
  EXPECT_FALSE(HasNvidiaDispatchLibrary(""));
  EXPECT_FALSE(HasNvidiaDispatchLibrary(
      (std::filesystem::path(::testing::TempDir()) / "missing").string()));
}

}  // namespace
}  // namespace litert::lm
