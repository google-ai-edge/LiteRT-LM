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

#include "omni/tts/kokoro/espeak_assets.h"

#include <atomic>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <string>
#include <system_error>  // NOLINT: Required by std::filesystem.

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/zip_utils.h"

namespace litert::omni::tts::kokoro {

namespace {

// Table file every valid espeak-ng data directory contains. Its presence marks
// a completed extraction.
constexpr absl::string_view kEspeakPhonemeTableFile = "phontab";

// Returns true if `entry_name` is safe to write below the destination
// directory, i.e. it is relative and does not escape via "..".
bool IsSafeEntryName(const std::filesystem::path& entry_name) {
  if (entry_name.empty() || entry_name.is_absolute()) return false;
  for (const auto& part : entry_name) {
    if (part == "..") return false;
  }
  return true;
}

// Drops a leading "espeak-ng-data/" component so that archives rooted at the
// data directory and archives containing it both unpack to the same layout.
std::filesystem::path StripDataDirPrefix(const std::filesystem::path& name) {
  auto it = name.begin();
  if (it == name.end() || it->string() != kEspeakDataDirName) return name;
  std::filesystem::path stripped;
  for (++it; it != name.end(); ++it) {
    stripped /= *it;
  }
  return stripped;
}

// Writes the zip entries of `archive` below `dest_dir`.
absl::Status WriteZipEntries(absl::string_view archive,
                             const std::filesystem::path& dest_dir) {
  ABSL_ASSIGN_OR_RETURN(auto entries, lm::ExtractFilesfromZipFile(archive));

  for (const auto& [name, location] : entries) {
    // Directory entries are implied by the files they contain.
    if (name.empty() || name.back() == '/') continue;

    const std::filesystem::path entry_name(name);
    if (!IsSafeEntryName(entry_name)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "espeak-ng archive contains an unsafe entry name: ", name));
    }
    const std::filesystem::path stripped_name = StripDataDirPrefix(entry_name);
    if (stripped_name.empty()) continue;
    const std::filesystem::path entry_path = dest_dir / stripped_name;

    std::error_code ec;
    std::filesystem::create_directories(entry_path.parent_path(), ec);
    if (ec) {
      return absl::InternalError(
          absl::StrCat("Failed to create espeak-ng data directory '",
                       entry_path.parent_path().string(), "': ", ec.message()));
    }

    std::ofstream file(entry_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return absl::InternalError(absl::StrCat(
          "Failed to write espeak-ng data file: ", entry_path.string()));
    }
    file.write(archive.data() + location.offset,
               static_cast<std::streamsize>(location.size));
    if (!file.good()) {
      return absl::InternalError(absl::StrCat(
          "Failed to write espeak-ng data file: ", entry_path.string()));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> UnpackEspeakDataFromLitertLm(
    lm::ModelResources& lm_resources, absl::string_view cache_dir) {
  ABSL_ASSIGN_OR_RETURN(
      absl::string_view archive,
      lm_resources.GetGenericBinaryDataBuffer(kEspeakNgSectionName));

  std::error_code ec;
  const std::filesystem::path base_dir =
      cache_dir.empty() ? std::filesystem::temp_directory_path(ec)
                        : std::filesystem::path(std::string(cache_dir));
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "Failed to resolve a directory for espeak-ng data: ", ec.message()));
  }
  const std::filesystem::path dest_dir = base_dir / kEspeakDataDirName;

  const std::string phoneme_table_file = std::string(kEspeakPhonemeTableFile);

  // Reuse a previous extraction.
  if (std::filesystem::exists(dest_dir / phoneme_table_file, ec)) {
    ABSL_LOG(INFO) << "Reusing unpacked espeak-ng data in: "
                   << dest_dir.string();
    return dest_dir.string();
  }

  // Unpack into a uniquely named temporary directory and move it into place,
  // so that concurrent extractions do not collide and a partial extraction is
  // never mistaken for a complete one.
  static std::atomic<uint64_t> staging_counter{0};
  const uint64_t unique_id =
      staging_counter.fetch_add(1, std::memory_order_relaxed);
  const std::filesystem::path staging_dir =
      base_dir / absl::StrCat(kEspeakDataDirName, ".unpacking.",
                              reinterpret_cast<uintptr_t>(&unique_id), ".",
                              unique_id);
  std::filesystem::remove_all(staging_dir, ec);
  std::filesystem::create_directories(staging_dir, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to create espeak-ng staging directory '",
                     staging_dir.string(), "': ", ec.message()));
  }

  // Clean up the staging directory after the function returns.
  absl::Cleanup cleanup_staging_dir = [&staging_dir, &ec] {
    std::filesystem::remove_all(staging_dir, ec);
  };
  ABSL_RETURN_IF_ERROR(WriteZipEntries(archive, staging_dir));
  if (!std::filesystem::exists(staging_dir / phoneme_table_file, ec)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "espeak-ng archive does not contain ", kEspeakPhonemeTableFile));
  }

  if (std::filesystem::exists(dest_dir / phoneme_table_file, ec)) {
    return dest_dir.string();
  }

  std::filesystem::remove_all(dest_dir, ec);
  std::filesystem::rename(staging_dir, dest_dir, ec);
  if (ec) {
    if (std::filesystem::exists(dest_dir / phoneme_table_file, ec)) {
      return dest_dir.string();
    }
    return absl::InternalError(
        absl::StrCat("Failed to move unpacked espeak-ng data to '",
                     dest_dir.string(), "': ", ec.message()));
  }

  ABSL_LOG(INFO) << "Unpacked espeak-ng data from the model container to: "
                 << dest_dir.string();
  return dest_dir.string();
}

}  // namespace litert::omni::tts::kokoro
