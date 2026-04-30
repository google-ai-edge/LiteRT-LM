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

#include "runtime/executor/vision_executor_settings.h"

#include <ostream>
#include <string>
#include <variant>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<VisionExecutorSettings> VisionExecutorSettings::CreateDefault(
    const ModelAssets& model_assets, Backend encoder_backend,
    Backend adapter_backend) {
  VisionExecutorSettings settings(model_assets);
  RETURN_IF_ERROR(settings.SetEncoderBackend(encoder_backend));
  RETURN_IF_ERROR(settings.SetAdapterBackend(adapter_backend));
  return settings;
}

Backend VisionExecutorSettings::GetEncoderBackend() const {
  return encoder_backend_;
}

absl::Status VisionExecutorSettings::SetEncoderBackend(Backend backend) {
  if (backend != Backend::GPU && backend != Backend::CPU &&
      backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported encoder backend: ", backend));
  }
  encoder_backend_ = backend;
  RETURN_IF_ERROR(SetBackend(encoder_backend_));
  return absl::OkStatus();
}

Backend VisionExecutorSettings::GetAdapterBackend() const {
  return adapter_backend_;
}

absl::Status VisionExecutorSettings::SetAdapterBackend(Backend backend) {
  if (backend != Backend::GPU && backend != Backend::CPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported adapter backend: ", backend));
  }
  adapter_backend_ = backend;
  return absl::OkStatus();
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
VisionExecutorSettings::GetWeightCacheFile(absl::string_view suffix,
                                           bool check_and_clean) const {
  if (absl::StrContains(suffix, kAdapterName) && GetScopedAdapterCacheFile()) {
    return GetScopedAdapterCacheFile();
  } else if (absl::StrContains(suffix, kEncoderName) &&
             GetScopedEncoderCacheFile()) {
    return GetScopedEncoderCacheFile();
  }

  return ExecutorSettingsBase::GetWeightCacheFile(suffix, check_and_clean);
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
VisionExecutorSettings::GetProgramCacheFile(absl::string_view suffix,
                                            bool check_and_clean) const {
  if (absl::StrContains(suffix, kAdapterName) &&
      GetScopedAdapterProgramCacheFile()) {
    return GetScopedAdapterProgramCacheFile();
  } else if (absl::StrContains(suffix, kEncoderName) &&
             GetScopedEncoderProgramCacheFile()) {
    return GetScopedEncoderProgramCacheFile();
  }

  return ExecutorSettingsBase::GetProgramCacheFile(suffix, check_and_clean);
}

std::ostream& operator<<(std::ostream& os,
                         const VisionExecutorSettings& settings) {
  os << "VisionExecutorSettings: " << std::endl;
  os << "  ModelAssets: " << settings.GetModelAssets() << std::endl;
  os << "  EncoderBackend: " << settings.GetEncoderBackend() << std::endl;
  os << "  AdapterBackend: " << settings.GetAdapterBackend() << std::endl;
  return os;
}

}  // namespace litert::lm
