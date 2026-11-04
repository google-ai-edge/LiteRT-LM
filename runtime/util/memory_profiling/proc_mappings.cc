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

#include "runtime/util/memory_profiling/proc_mappings.h"

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl

namespace litert::lm {

namespace {

// Whole contents of `path`, or std::nullopt if it cannot be read. /proc files
// report a size of zero, so they are read to the end rather than sized.
std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return std::nullopt;
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// The value of a "<key>:\t<n> kB" line in bytes, or std::nullopt if `line` is
// not for `key` or does not parse.
std::optional<uint64_t> KbField(absl::string_view line, absl::string_view key) {
  if (!absl::ConsumePrefix(&line, key) || !absl::ConsumePrefix(&line, ":")) {
    return std::nullopt;
  }
  line = absl::StripAsciiWhitespace(line);
  absl::ConsumeSuffix(&line, "kB");
  uint64_t kb = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(line), &kb)) {
    return std::nullopt;
  }
  return kb * 1024;
}

}  // namespace

std::vector<ProcMapping> ParseProcMaps(absl::string_view contents) {
  std::vector<ProcMapping> mappings;
  // Each line: "start-end perms offset dev inode [path]", where the path may
  // contain spaces and is absent for anonymous memory.
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (line.empty()) continue;
    std::vector<absl::string_view> fields =
        absl::StrSplit(line, absl::MaxSplits(' ', 5), absl::SkipEmpty());
    if (fields.size() < 5) continue;
    std::vector<absl::string_view> range = absl::StrSplit(fields[0], '-');
    ProcMapping mapping;
    if (range.size() != 2 || !absl::SimpleHexAtoi(range[0], &mapping.start) ||
        !absl::SimpleHexAtoi(range[1], &mapping.end) ||
        mapping.end < mapping.start) {
      continue;
    }
    if (fields.size() == 6) {
      mapping.path = std::string(absl::StripAsciiWhitespace(fields[5]));
    }
    mappings.push_back(std::move(mapping));
  }
  return mappings;
}

std::vector<ProcMapping> ReadProcMaps(absl::string_view proc_self_dir) {
  const std::optional<std::string> contents =
      ReadFile(absl::StrCat(proc_self_dir, "/maps"));
  if (!contents.has_value()) return {};
  return ParseProcMaps(*contents);
}

std::optional<ProcResidentCounters> ParseProcStatusResident(
    absl::string_view contents) {
  std::optional<uint64_t> vm_rss;
  uint64_t rss_shmem = 0;
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (const auto value = KbField(line, "VmRSS")) vm_rss = *value;
    if (const auto value = KbField(line, "RssShmem")) rss_shmem = *value;
  }
  if (!vm_rss.has_value()) return std::nullopt;
  return ProcResidentCounters{.vm_rss_bytes = *vm_rss,
                              .rss_shmem_bytes = rss_shmem};
}

std::optional<ProcResidentCounters> ReadProcStatusResident(
    absl::string_view proc_self_dir) {
  const std::optional<std::string> contents =
      ReadFile(absl::StrCat(proc_self_dir, "/status"));
  if (!contents.has_value()) return std::nullopt;
  return ParseProcStatusResident(*contents);
}

std::optional<uint64_t> ParseSmapsRollupRss(absl::string_view contents) {
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (const auto value = KbField(line, "Rss")) return *value;
  }
  return std::nullopt;
}

std::optional<uint64_t> ReadSmapsRollupRss(absl::string_view proc_self_dir) {
  const std::optional<std::string> contents =
      ReadFile(absl::StrCat(proc_self_dir, "/smaps_rollup"));
  if (!contents.has_value()) return std::nullopt;
  return ParseSmapsRollupRss(*contents);
}

}  // namespace litert::lm
