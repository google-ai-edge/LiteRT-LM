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

#include "runtime/util/memory_profiling/fdinfo.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/fake_proc_self.h"

namespace litert::lm {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(ParseFdinfoTest, SplitsKeysAndValuesAndStripsWhitespace) {
  // Tab separated (current kernels), space separated (older ones), a value
  // with internal spaces (DRM memory keys), and a line without a colon.
  EXPECT_THAT(ParseFdinfo("pos:\t0\n"
                          "flags:\t02100002\n"
                          "exp_name: system\n"
                          "drm-total-vram:\t1024 KiB\n"
                          "garbage line\n"),
              UnorderedElementsAre(Pair("pos", "0"), Pair("flags", "02100002"),
                                   Pair("exp_name", "system"),
                                   Pair("drm-total-vram", "1024 KiB")));
}

TEST(ParseFdinfoTest, LastRepeatedKeyWins) {
  EXPECT_THAT(ParseFdinfo("size: 1\nsize: 2\n"),
              UnorderedElementsAre(Pair("size", "2")));
}

TEST(ParseFdinfoTest, EmptyInputHasNoFields) {
  EXPECT_THAT(ParseFdinfo(""), IsEmpty());
}

TEST(ForEachFdinfoTest, MissingDirectoryVisitsNothing) {
  int visits = 0;
  ForEachFdinfo(absl::StrCat(testing::TempDir(), "/does_not_exist"),
                [&](absl::string_view, const FdinfoFields&) { ++visits; });
  EXPECT_EQ(visits, 0);
}

TEST(ForEachFdinfoTest, VisitsEveryDescriptorWithItsFields) {
  FakeProcSelf proc;
  proc.AddFd(3, "pos:\t0\nino:\t77\n");
  proc.AddFd(12, "pos:\t0\nsize:\t4096\nexp_name:\tsystem\n");

  std::vector<std::string> seen;
  ForEachFdinfo(
      proc.dir(), [&](absl::string_view fd_name, const FdinfoFields& fields) {
        const auto exp_name = fields.find("exp_name");
        seen.push_back(absl::StrCat(
            fd_name, ":", exp_name == fields.end() ? "-" : exp_name->second));
      });
  // Directory order is unspecified, so compare as a set.
  EXPECT_THAT(seen, UnorderedElementsAre("3:-", "12:system"));
}

}  // namespace
}  // namespace litert::lm
