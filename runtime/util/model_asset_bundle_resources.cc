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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/external_file_handler.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/zip_utils.h"

namespace litert::lm {

ModelAssetBundleResources::ModelAssetBundleResources(
    const std::string& tag,
    std::unique_ptr<proto::ExternalFile> model_asset_bundle_file)
    : tag_(tag), model_asset_bundle_file_(std::move(model_asset_bundle_file)) {}

/* static */
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::unique_ptr<proto::ExternalFile> model_asset_bundle_file) {
  if (model_asset_bundle_file == nullptr) {
    return absl::InvalidArgumentError(

        "The model asset bundle file proto cannot be nullptr.");
  }
  auto model_bundle_resources = absl::WrapUnique(
      new ModelAssetBundleResources(tag, std::move(model_asset_bundle_file)));
  RETURN_IF_ERROR(model_bundle_resources->ExtractFilesFromExternalFileProto());
  return model_bundle_resources;
}

absl::Status ModelAssetBundleResources::ExtractFilesFromExternalFileProto() {
  if (model_asset_bundle_file_->has_file_name()) {
    std::string path_to_resource = model_asset_bundle_file_->file_name();
    model_asset_bundle_file_->set_file_name(path_to_resource);
  }
  ASSIGN_OR_RETURN(model_asset_bundle_file_handler_,
                   ExternalFileHandler::CreateFromExternalFile(
                       model_asset_bundle_file_.get()));
  const char* buffer_data =
      model_asset_bundle_file_handler_->GetFileContent().data();
  size_t buffer_size =
      model_asset_bundle_file_handler_->GetFileContent().size();
  return ExtractFilesfromZipFile(buffer_data, buffer_size, &files_);
}

absl::StatusOr<absl::string_view> ModelAssetBundleResources::GetFile(
    const std::string& filename) const {
  auto it = files_.find(filename);
  if (it == files_.end()) {
    auto files = ListFiles();
    std::string all_files = absl::StrJoin(files.begin(), files.end(), ", ");

    return absl::NotFoundError(
        absl::StrFormat("No file with name: %s. All files in the model asset "
                        "bundle are: %s.",
                        filename, all_files));
  }
  return it->second;
}

std::vector<std::string> ModelAssetBundleResources::ListFiles() const {
  std::vector<std::string> file_names;
  for (const auto& [file_name, _] : files_) {
    file_names.push_back(file_name);
  }
  return file_names;
}

}  // namespace litert::lm
