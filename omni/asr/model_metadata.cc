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

#include "omni/asr/model_metadata.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/omni_registry.h"

namespace litert::omni::asr {
namespace {

constexpr int kDefaultSampleRateHz = 16000;
constexpr int kDefaultInputMilliseconds = 5000;
constexpr int kWhisperInputMilliseconds = 30000;

}  // namespace

absl::Status PopulateConfigFromMetadataJson(absl::string_view model_name,
                                            absl::string_view /*json_str*/,
                                            AsrEngineConfig& config) {
  auto reg_model = OmniRegistry::Get().GetRegisteredModel(model_name);
  if (!reg_model.ok() ||
      reg_model->metadata_proto_type_id != "litert.lm.proto.AsrMetadata") {
    return absl::NotFoundError(
        absl::StrCat("Unsupported or unknown ASR model: ", model_name));
  }

  if (config.model_url.empty()) {
    config.model_url = reg_model->model_remote_url;
  }
  // NOTE: Hardcoding these default metadata values is acceptable for now
  // because `PopulateConfigFromMetadataJson` is transitional and will be
  // removed soon once callers migrate to `OmniEngine` / `OmniRegistry`.
  if (!config.metadata.has_sample_rate_hz()) {
    config.metadata.set_sample_rate_hz(kDefaultSampleRateHz);
  }
  if (!config.metadata.has_input_milliseconds()) {
    config.metadata.set_input_milliseconds(reg_model->canonical_model_name ==
                                                   "whisper-tiny"
                                               ? kWhisperInputMilliseconds
                                               : kDefaultInputMilliseconds);
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::asr
