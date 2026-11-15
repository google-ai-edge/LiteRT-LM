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
#include <filesystem>  // NOLINT: Required for file metadata retrieval.
#include <string>
#include <system_error>  // NOLINT: Required for file metadata retrieval.
#include <unordered_map>
#include <utility>

#include "runtime/util/scoped_file.h"

#if defined(_WIN32)
#include <Windows.h>
#include <io.h>

#include <cstdint>
#else
#include <sys/stat.h>
#include <unistd.h>
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

#if defined(_WIN32)
int64_t FileTimeToUnixSeconds(const FILETIME& ft) {
  ULARGE_INTEGER ull;
  ull.LowPart = ft.dwLowDateTime;
  ull.HighPart = ft.dwHighDateTime;
  return static_cast<int64_t>(ull.QuadPart / 10000000ULL) - 11644473600LL;
}
#endif

// Computes the `<mtime_sec>_<size>` cache identifier for `path` without
// consulting the in-process path cache.
//
// Uses `stat()` on POSIX and `GetFileAttributesExW()` on Windows directly
// rather than opening a file descriptor/handle for `fstat()` / `GetFileTime()`:
// querying metadata by path takes a single syscall without `open()` + `close()`
// file-descriptor overhead or requiring read permissions on the file itself,
// while returning the exact same timestamp and size fields as
// `GetFileCacheIdentifier(const ScopedFile&)`.
absl::StatusOr<std::string> ComputeFileCacheIdentifier(absl::string_view path) {
  std::string path_str(path);
#if defined(_WIN32)
  std::filesystem::path p{path_str};
  WIN32_FILE_ATTRIBUTE_DATA data;
  if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &data)) {
    return absl::InternalError(
        absl::StrCat("GetFileAttributesExW() failed for: ", path));
  }
  if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    return absl::InternalError(absl::StrCat("Path is a directory: ", path));
  }
  int64_t seconds = FileTimeToUnixSeconds(data.ftLastWriteTime);
  uint64_t size = (static_cast<uint64_t>(data.nFileSizeHigh) << 32) |
                  static_cast<uint64_t>(data.nFileSizeLow);
  return absl::StrCat(seconds, "_", size);
#else
  struct stat info;
  if (stat(path_str.c_str(), &info) < 0) {
    return absl::InternalError(absl::StrCat("stat() failed for: ", path));
  }
  if (S_ISDIR(info.st_mode)) {
    return absl::InternalError(absl::StrCat("Path is a directory: ", path));
  }
  return absl::StrCat(info.st_mtime, "_", info.st_size);
#endif
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
    absl::MutexLock lock(*cached_identifiers_mutex);
    if (auto it = cached_identifiers->find(path_str);
        it != cached_identifiers->end()) {
      return it->second;
    }
  }

  absl::StatusOr<std::string> identifier = ComputeFileCacheIdentifier(path);
  if (!identifier.ok()) {
    return identifier.status();
  }

  absl::MutexLock lock(*cached_identifiers_mutex);
  (*cached_identifiers)[path_str] = *identifier;
  return *identifier;
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
  int64_t seconds = FileTimeToUnixSeconds(ftWrite);

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

bool IsDirectoryWritable(absl::string_view path) {
  if (path.empty()) {
    return false;
  }
  std::string path_str(path);
  std::error_code ec;
  std::filesystem::path p{path_str};
  if (!std::filesystem::exists(p, ec) ||
      !std::filesystem::is_directory(p, ec)) {
    return false;
  }
#if defined(_WIN32)
  return _waccess(p.c_str(), 2) == 0;
#else
  return access(path_str.c_str(), W_OK) == 0;
#endif
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
