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

#include <cerrno>
#include <chrono>      // NOLINT: Required for file metadata retrieval.
#include <filesystem>  // NOLINT: Required for file metadata retrieval.
#include <string>
#include <system_error>  // NOLINT: Required for file metadata retrieval.
#include <unordered_map>
#include <utility>

#include "runtime/util/scoped_file.h"

#if defined(_WIN32)
#include <Windows.h>
#else
#include <sys/stat.h>
#endif

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

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
  static auto* cached_identifiers =
      new std::unordered_map<std::string, std::string>();
  static absl::NoDestructor<absl::Mutex> cached_identifiers_mutex;
  std::string path_str(path);
  {
    absl::MutexLock lock(cached_identifiers_mutex.get());
    if (auto it = cached_identifiers->find(path_str);
        it != cached_identifiers->end()) {
      return it->second;
    }
  }

  std::error_code ec;
  std::filesystem::path p{path_str};

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

  auto current_file_time = std::filesystem::file_time_type::clock::now();
  auto current_sys_time = std::chrono::system_clock::now();
  auto sys_time =
      std::chrono::time_point_cast<std::chrono::system_clock::duration>(
          mtime - current_file_time + current_sys_time);
  auto duration = sys_time.time_since_epoch();
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  std::string identifier = absl::StrCat(seconds, "_", size);
  {
    absl::MutexLock lock(cached_identifiers_mutex.get());
    (*cached_identifiers)[path_str] = identifier;
  }
  return identifier;
}

#if defined(_WIN32)
absl::StatusOr<std::string> GetFileCacheIdentifier(
    const ScopedFile& scoped_file) {
  if (!scoped_file.IsValid()) {
    return absl::InvalidArgumentError("ScopedFile is invalid");
  }
  HANDLE hFile = scoped_file.file();
  FILETIME ftWrite;
  if (!GetFileTime(hFile, NULL, NULL, &ftWrite)) {
    return absl::InternalError("Failed to get file time by handle on Windows");
  }
  ULARGE_INTEGER ull;
  ull.LowPart = ftWrite.dwLowDateTime;
  ull.HighPart = ftWrite.dwHighDateTime;
  int64_t seconds =
      static_cast<int64_t>(ull.QuadPart / 10000000ULL) - 11644473600LL;

  LARGE_INTEGER size;
  if (!GetFileSizeEx(hFile, &size)) {
    return absl::InternalError("Failed to get file size by handle on Windows");
  }

  return absl::StrCat(seconds, "_", size.QuadPart);
}
#else
absl::StatusOr<std::string> GetFileCacheIdentifier(
    const ScopedFile& scoped_file) {
  if (!scoped_file.IsValid()) {
    return absl::InvalidArgumentError("ScopedFile is invalid");
  }
  int fd = scoped_file.file();
  struct stat info;
  if (fstat(fd, &info) < 0) {
    return absl::ErrnoToStatus(errno, "fstat() failed");
  }
  return absl::StrCat(info.st_mtime, "_", info.st_size);
}
#endif

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
