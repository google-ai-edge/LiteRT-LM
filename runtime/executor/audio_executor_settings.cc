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

#include "runtime/executor/audio_executor_settings.h"

#include <ostream>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorSettings& settings) {
  os << "AudioExecutorSettings: " << std::endl;
  os << "ModelAssets: " << settings.GetModelAssets() << std::endl;
  os << "MaxSequenceLength: " << settings.GetMaxSequenceLength() << std::endl;
  os << "Backend: " << settings.GetBackend() << std::endl;
  os << "BundledWithMainModel: " << settings.GetBundledWithMainModel()
     << std::endl;
  os << "NumThreads(CPU only): " << settings.GetNumThreads() << std::endl;
  return os;
}

absl::StatusOr<AudioExecutorSettings> AudioExecutorSettings::CreateDefault(
    const ModelAssets& model_assets, int max_sequence_length, Backend backend,
    bool bundled_with_main_model) {
  AudioExecutorSettings settings(model_assets, max_sequence_length,
                                 /*num_threads=*/4);
  RETURN_IF_ERROR(settings.SetBackend(backend));
  settings.SetBundledWithMainModel(bundled_with_main_model);
  return settings;
}

int AudioExecutorSettings::GetMaxSequenceLength() const {
  return max_sequence_length_;
}

void AudioExecutorSettings::SetMaxSequenceLength(int max_sequence_length) {
  max_sequence_length_ = max_sequence_length;
}

absl::Status AudioExecutorSettings::SetBackend(const Backend& backend) {
  if (backend != Backend::CPU && backend != Backend::GPU &&
      backend != Backend::GPU_ARTISAN) {
    return absl::InvalidArgumentError(
        "Currently AudioExecutor only supports CPU, GPU and GPU_ARTISAN.");
  }
  backend_ = backend;
  return absl::OkStatus();
}

bool AudioExecutorSettings::GetBundledWithMainModel() const {
  return bundled_with_main_model_;
}

void AudioExecutorSettings::SetBundledWithMainModel(
    bool bundled_with_main_model) {
  bundled_with_main_model_ = bundled_with_main_model;
}

}  // namespace litert::lm
