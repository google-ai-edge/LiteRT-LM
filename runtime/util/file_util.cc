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

#include <chrono>      // NOLINT: Required for file metadata retrieval.
#include <filesystem>  // NOLINT: Required for file metadata retrieval.
#include <string>
#include <system_error>  // NOLINT: Required for file metadata retrieval.
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

namespace {

#if defined(_WIN32)
constexpr char kPathSeparator = '\\';
#else
constexpr char kPathSeparator = '/';
#endif

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
#if defined(_WIN32)
  absl::string_view::size_type pos = path.find_last_of("\\/");
#else
  absl::string_view::size_type pos = path.find_last_of(kPathSeparator);
#endif

  // Handle the case with no '/' or '\' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(path.substr(0, 0), path);

  // Handle the case with a single leading '/' or '\' in 'path'.
  if (pos == 0)
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));

  return std::make_pair(path.substr(0, pos + 1),
                        absl::ClippedSubstr(path, pos + 1));
}

}  // namespace

// 40% of the time in JoinPath() is from calls with 2 arguments, so we
// specialize that case.
absl::StatusOr<std::string> JoinPath(absl::string_view path1,
                                     absl::string_view path2) {
  if (path1.empty()) return absl::InvalidArgumentError("Empty path1.");
  if (path2.empty()) return absl::InvalidArgumentError("Empty path2.");
  if (path1.back() == kPathSeparator) {
    if (path2.front() == kPathSeparator)
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else {
    if (path2.front() != kPathSeparator)
      return absl::StrCat(path1, std::string(1, kPathSeparator), path2);
  }
  return absl::StrCat(path1, path2);
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

absl::StatusOr<std::string> GetFileCacheIdentifier(absl::string_view path) {
  std::error_code ec;
  std::filesystem::path p{std::string(path)};

  if (!std::filesystem::exists(p, ec) || ec) {
    return absl::InternalError(absl::StrCat("File does not exist: ", path));
  }

  auto size = std::filesystem::file_size(p, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to get file size: ", ec.message()));
  }

  auto mtime = std::filesystem::last_write_time(p, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to get last write time: ", ec.message()));
  }

  auto duration = mtime.time_since_epoch();
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  return absl::StrCat(seconds, "_", size);
}

bool FileExists(absl::string_view path) {
  std::error_code ec;
  std::filesystem::path p{std::string(path)};
  return std::filesystem::exists(p, ec) &&
         std::filesystem::is_regular_file(p, ec);
}

absl::StatusOr<int> DeleteStaleCaches(absl::string_view cache_dir,
                                      absl::string_view model_basename,
                                      absl::string_view suffix) {
  std::error_code ec;
  std::filesystem::path dir_path{std::string(cache_dir)};

  if (!std::filesystem::exists(dir_path, ec) ||
      !std::filesystem::is_directory(dir_path, ec)) {
    // If directory doesn't exist, nothing to delete.
    return 0;
  }

  std::string target_prefix = absl::StrCat(model_basename, suffix);

  int deleted_count = 0;
  absl::Status status = absl::OkStatus();

  std::filesystem::directory_iterator it(dir_path, ec);
  std::filesystem::directory_iterator end;

  while (it != end && !ec) {
    const auto& entry = *it;
    std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, target_prefix) ||
        (absl::StartsWith(name, model_basename) &&
         absl::EndsWith(name, suffix))) {
      std::error_code remove_ec;
      if (std::filesystem::remove(entry.path(), remove_ec)) {
        deleted_count++;
      } else if (remove_ec) {
        status = absl::InternalError(
            absl::StrCat("Failed to delete cache file: ", entry.path().string(),
                         ", error: ", remove_ec.message()));
      }
    }
    it.increment(ec);
  }
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Directory iteration failed: ", ec.message()));
  }

  if (!status.ok()) {
    return status;
  }

  return deleted_count;
}

}  // namespace litert::lm
