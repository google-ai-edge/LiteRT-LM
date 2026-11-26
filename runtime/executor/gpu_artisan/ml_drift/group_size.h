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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GROUP_SIZE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GROUP_SIZE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift

namespace odml::infra::gpu {

// Stores the `cost` of processing a batch of `sequence_size` tokens.
struct GroupSize {
  int sequence_size;
  float cost;
};

// Gets the group sizes to use if no explicit sequence_size is specified. Group
// sizes are in descending order.
inline std::vector<GroupSize> GetRecommendedGroupSizes(
    const ml_drift::GpuInfo& gpu_info) {
  // These values were chosen with local experimentation. The relative cost of
  // each group size should be relatively consistent across devices.
  if (gpu_info.IsIntel()) {
    return {
        GroupSize{.sequence_size = 512, .cost = 5.3 * 512},
        GroupSize{.sequence_size = 128, .cost = 6.5 * 128},
        GroupSize{.sequence_size = 64, .cost = 12.5 * 64},
        GroupSize{.sequence_size = 8, .cost = 19.7f * 8},
        GroupSize{.sequence_size = 1, .cost = 77.4f},
    };
  } else if (gpu_info.IsApple()) {
    return {
        GroupSize{.sequence_size = 1024, .cost = 2.2f * 1024},
        GroupSize{.sequence_size = 512, .cost = 2.6f * 512},
        GroupSize{.sequence_size = 128, .cost = 3.3f * 128},
        GroupSize{.sequence_size = 64, .cost = 4.5f * 64},
        GroupSize{.sequence_size = 8, .cost = 5.6f * 8},
        GroupSize{.sequence_size = 1, .cost = 27.6f},
    };
  } else {
    return {
        GroupSize{.sequence_size = 512, .cost = 1.1 * 512},
        GroupSize{.sequence_size = 128, .cost = 1.6 * 128},
        GroupSize{.sequence_size = 64, .cost = 2.2 * 64},
        GroupSize{.sequence_size = 8, .cost = 5.1f * 8},
        GroupSize{.sequence_size = 1, .cost = 21.8f},
    };
  }
}

// Given `input_length` returns the optimal sequence of group sizes to use to
// fully process the input.
absl::StatusOr<std::vector<int>> GetGroupSizesForInput(
    int input_length, const std::vector<GroupSize>& group_sizes);

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GROUP_SIZE_H_
