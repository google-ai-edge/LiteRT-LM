// Copyright 2024 The MediaPipe Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_NAME_CHECK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_NAME_CHECK_H_

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace odml::infra::gpu {

inline bool IsLoRATensorName(absl::string_view name) {
  return absl::EndsWith(name, "_prime_left") ||
         absl::EndsWith(name, "_prime_right");
}

inline bool IsFeedForwardTensorName(absl::string_view name) {
  return absl::StrContains(name, ".ffn_layer") ||
         absl::StrContains(name, ".mlp");
}

inline bool IsAttentionTensorName(absl::string_view name) {
  return absl::StrContains(name, ".self_attention") ||
         absl::StrContains(name, ".attn");
}

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_NAME_CHECK_H_
