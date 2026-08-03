/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/util/zip_utils.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "minizip/ioapi.h"  // from @minizip
#include "minizip/unzip.h"  // from @minizip
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/zip_readonly_mem_file.h"
#include "zlib.h"  // from @zlib

namespace litert::lm {

namespace {

// Wrapper function around calls to unzip to avoid repeating conversion logic
// from error code to Status.
absl::Status UnzipErrorToStatus(int error) {
  if (error != UNZ_OK) {
    return absl::UnknownError("Unable to read the file in zip archive.");
  }
  return absl::OkStatus();
}

// Stores a file name, position in zip buffer and size.
struct ZipFileInfo {
  std::string name;
  ZPOS64_T position;
  ZPOS64_T size;
};

// Returns the ZipFileInfo corresponding to the current file in the provided
// unzFile object.
absl::StatusOr<ZipFileInfo> GetCurrentZipFileInfo(const unzFile& zf) {
  // Open file in raw mode, as data is expected to be uncompressed.
  int method;
  RETURN_IF_ERROR(UnzipErrorToStatus(
      unzOpenCurrentFile2(zf, &method, /*level=*/nullptr, /*raw=*/1)));
  absl::Cleanup unzipper_closer = [zf]() {
    auto status = UnzipErrorToStatus(unzCloseCurrentFile(zf));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to close the current zip file: " << status;
    }
  };
  if (method != Z_NO_COMPRESSION) {
    return absl::UnknownError("Expected uncompressed zip archive.");
  }

  // Get file info a first time to get filename size.
  unz_file_info64 file_info;
  RETURN_IF_ERROR(UnzipErrorToStatus(unzGetCurrentFileInfo64(
      zf, &file_info, /*szFileName=*/nullptr, /*szFileNameBufferSize=*/0,
      /*extraField=*/nullptr, /*extraFieldBufferSize=*/0,
      /*szComment=*/nullptr, /*szCommentBufferSize=*/0)));

  // Second call to get file name.
  auto file_name_size = file_info.size_filename;
  char* c_file_name = (char*)malloc(file_name_size);
  RETURN_IF_ERROR(UnzipErrorToStatus(unzGetCurrentFileInfo64(
      zf, &file_info, c_file_name, file_name_size,
      /*extraField=*/nullptr, /*extraFieldBufferSize=*/0,
      /*szComment=*/nullptr, /*szCommentBufferSize=*/0)));
  std::string file_name = std::string(c_file_name, file_name_size);
  free(c_file_name);

  // Get position in file.
  auto position = unzGetCurrentFileZStreamPos64(zf);
  if (position == 0) {
    return absl::UnknownError("Unable to get position in zip archive.");
  }

  // Perform the cleanup manually for error propagation.
  std::move(unzipper_closer).Cancel();
  // Close file and return.
  RETURN_IF_ERROR(UnzipErrorToStatus(unzCloseCurrentFile(zf)));

  return ZipFileInfo{
      .name = std::move(file_name),
      .position = position,
      .size = file_info.uncompressed_size};
}

}  // namespace

absl::StatusOr<absl::flat_hash_map<std::string, OffsetAndSize>>
ExtractFilesfromZipFile(absl::string_view data) {
  // Create in-memory read-only zip file.
  ZipReadOnlyMemFile mem_file = ZipReadOnlyMemFile(data);
  // Open zip.
  unzFile zf = unzOpen2_64(/*path=*/nullptr, &mem_file.GetFileFunc64Def());
  if (zf == nullptr) {
    return absl::UnknownError("Unable to open zip archive.");
  }
  absl::Cleanup unzipper_closer = [zf]() {
    if (unzClose(zf) != UNZ_OK) {
      ABSL_LOG(ERROR) << "Unable to close zip archive.";
    }
  };
  // Get number of files.
  unz_global_info global_info;
  if (unzGetGlobalInfo(zf, &global_info) != UNZ_OK) {
    return absl::UnknownError("Unable to get zip archive info.");
  }

  // Browse through files in archive.
  absl::flat_hash_map<std::string, OffsetAndSize> files;
  if (global_info.number_entry > 0) {
    int error = unzGoToFirstFile(zf);
    while (error == UNZ_OK) {
      ASSIGN_OR_RETURN(auto zip_file_info, GetCurrentZipFileInfo(zf));
      // Store result in map.
      files[zip_file_info.name] = OffsetAndSize{
          .offset = static_cast<size_t>(zip_file_info.position),
          .size = static_cast<size_t>(zip_file_info.size),
      };
      error = unzGoToNextFile(zf);
    }
    if (error != UNZ_END_OF_LIST_OF_FILE) {
      return absl::UnknownError(
          "Unable to read associated file in zip archive.");
    }
  }
  // Perform the cleanup manually for error propagation.
  std::move(unzipper_closer).Cancel();
  // Close zip.
  if (unzClose(zf) != UNZ_OK) {
    return absl::UnknownError("Unable to close zip archive.");
  }
  return files;
}

void SetExternalFile(absl::string_view file_content,
                     proto::ExternalFile* model_file, bool is_copy) {
  if (is_copy) {
    std::string str_content{file_content};
    model_file->set_file_content(str_content);
  } else {
    auto pointer = reinterpret_cast<uint64_t>(file_content.data());
    model_file->mutable_file_pointer_meta()->set_pointer(pointer);
    model_file->mutable_file_pointer_meta()->set_length(file_content.length());
  }
}

}  // namespace litert::lm
