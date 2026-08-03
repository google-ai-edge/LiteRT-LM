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

#include "runtime/util/model_asset_bundle_resources.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/zip_utils.h"

namespace litert::lm {

/* static */
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::shared_ptr<ScopedFile> model_asset_bundle_file) {
  if (!model_asset_bundle_file || !model_asset_bundle_file->IsValid()) {
    return absl::InvalidArgumentError(
        "The model asset bundle file is not valid.");
  }

  ASSIGN_OR_RETURN(auto mapped_model_asset_bundle_file,
                   MemoryMappedFile::Create(model_asset_bundle_file->file()));

  absl::string_view data(
      reinterpret_cast<const char*>(mapped_model_asset_bundle_file->data()),
      mapped_model_asset_bundle_file->length());

  ASSIGN_OR_RETURN(auto files, ExtractFilesfromZipFile(data));

  return absl::WrapUnique(new ModelAssetBundleResources(
      tag, std::move(mapped_model_asset_bundle_file), std::move(files)));
}

// static
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::shared_ptr<MemoryMappedFile> model_asset_bundle_file) {
  if (!model_asset_bundle_file) {
    return absl::InvalidArgumentError(
        "The model asset bundle file is not valid.");
  }

  absl::string_view data(
      reinterpret_cast<const char*>(model_asset_bundle_file->data()),
      model_asset_bundle_file->length());

  ASSIGN_OR_RETURN(auto files, ExtractFilesfromZipFile(data));

  return absl::WrapUnique(new ModelAssetBundleResources(
      tag, std::move(model_asset_bundle_file), std::move(files)));
}

// static
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(const std::string& tag,
                                  ScopedFile&& model_asset_bundle_file) {
  return Create(
      tag, std::make_shared<ScopedFile>(std::move(model_asset_bundle_file)));
}

ModelAssetBundleResources::ModelAssetBundleResources(
    std::string tag,
    std::shared_ptr<MemoryMappedFile> mapped_model_asset_bundle_file,
    absl::flat_hash_map<std::string, OffsetAndSize> files)
    : tag_(std::move(tag)),
      mapped_model_asset_bundle_file_(
          std::move(mapped_model_asset_bundle_file)),
      files_(std::move(files)) {}

absl::StatusOr<absl::string_view> ModelAssetBundleResources::GetFile(
    absl::string_view filename) const {
  auto it = files_.find(filename);
  if (it != files_.end()) {
    return absl::string_view(
        reinterpret_cast<const char*>(mapped_model_asset_bundle_file_->data()) +
            it->second.offset,
        it->second.size);
  }
  return absl::NotFoundError(
      absl::StrCat("No file with name: ", filename,
                   ". All files in the model asset bundle are: ",
                   absl::StrJoin(ListFiles(), ", ")));
}

std::vector<absl::string_view> ModelAssetBundleResources::ListFiles() const {
  std::vector<absl::string_view> file_names;
  file_names.reserve(files_.size());
  for (const auto& [file_name, _] : files_) {
    file_names.push_back(file_name);
  }
  return file_names;
}
}  // namespace litert::lm
