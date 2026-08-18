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

#include "c/capabilities.h"

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

namespace {

std::string GetRunfilePath(const std::string& relative_path) {
  std::string srcdir = ::testing::SrcDir();
  // On Windows, SrcDir() may return paths with backslashes. The LiteRT LM C API
  // expects forward slashes.
  std::replace(srcdir.begin(), srcdir.end(), '\\', '/');
  return srcdir + "/" + relative_path;
}

TEST(CapabilitiesCTest, HasSpeculativeDecodingSupport) {
  std::string model_path = GetRunfilePath(
      "litert_lm/runtime/testdata/test_lm.litertlm");
  LiteRtLmLoadedFile* file = litert_lm_loaded_file_create(model_path.c_str());
  ASSERT_NE(file, nullptr);
  EXPECT_FALSE(litert_lm_loaded_file_has_speculative_decoding_support(file));
  litert_lm_loaded_file_delete(file);
}

TEST(CapabilitiesCTest, CreateInvalidPathReturnsNull) {
  LiteRtLmLoadedFile* file =
      litert_lm_loaded_file_create("/invalid/path/that/does/not/exist");
  EXPECT_EQ(file, nullptr);
}

}  // namespace
