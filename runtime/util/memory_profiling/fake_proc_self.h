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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FAKE_PROC_SELF_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FAKE_PROC_SELF_H_

// Test helper: a fake /proc/self directory for the readers built on fdinfo.h.

#include <sys/stat.h>

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// A fake /proc/self directory, private to the current test: `fdinfo/<fd>`
// files with the given contents and no `fd/` directory, so inode lookups
// through `fd/` fail the way they do on a kernel that does not expose them.
class FakeProcSelf {
 public:
  FakeProcSelf()
      : dir_(absl::StrCat(
            testing::TempDir(), "/proc_self_",
            testing::UnitTest::GetInstance()->current_test_info()->name())) {
    mkdir(dir_.c_str(), 0755);
    mkdir(absl::StrCat(dir_, "/fdinfo").c_str(), 0755);
  }

  void AddFd(int fd, absl::string_view fdinfo) {
    std::ofstream file(absl::StrCat(dir_, "/fdinfo/", fd));
    file << fdinfo;
  }

  // Writes a top-level file of the fake directory, e.g. "maps", "status" or
  // "smaps_rollup".
  void WriteFile(absl::string_view name, absl::string_view contents) {
    std::ofstream file(absl::StrCat(dir_, "/", name));
    file << contents;
  }

  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FAKE_PROC_SELF_H_
