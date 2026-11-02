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

#include <dirent.h>

#include <fstream>
#include <sstream>
#include <string>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

FdinfoFields ParseFdinfo(absl::string_view contents) {
  FdinfoFields fields;
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    const absl::string_view::size_type colon = line.find(':');
    if (colon == absl::string_view::npos) continue;
    fields[std::string(absl::StripAsciiWhitespace(line.substr(0, colon)))] =
        std::string(absl::StripAsciiWhitespace(line.substr(colon + 1)));
  }
  return fields;
}

void ForEachFdinfo(
    absl::string_view proc_self_dir,
    absl::FunctionRef<void(absl::string_view fd_name, const FdinfoFields&)>
        fn) {
  const std::string fdinfo_dir = absl::StrCat(proc_self_dir, "/fdinfo");
  DIR* dir = opendir(fdinfo_dir.c_str());
  if (dir == nullptr) return;
  while (const struct dirent* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') continue;
    std::ifstream file(absl::StrCat(fdinfo_dir, "/", entry->d_name));
    if (!file.is_open()) continue;  // Closed between readdir and open.
    std::stringstream contents;
    contents << file.rdbuf();
    fn(entry->d_name, ParseFdinfo(contents.str()));
  }
  closedir(dir);
}

}  // namespace litert::lm
