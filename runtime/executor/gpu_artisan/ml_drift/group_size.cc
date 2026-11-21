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

#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/util.h"  // from @ml_drift

namespace odml::infra::gpu {
namespace {

// TODO: b/317226672 - Rewrite this as iterative instead of recursive.
std::vector<int> GetGroupSizes(int length, absl::Span<const GroupSize> sizes,
                               float* cost) {
  if (sizes.empty()) return {};
  int cur_size = sizes[0].sequence_size;
  float cur_cost = sizes[0].cost;
  // The largest group is first in the list and has the lowest cost per token.
  // Output as many full large groups as possible.
  std::vector<int> ret(length / cur_size, cur_size);
  *cost += cur_cost * ret.size();
  length = length % cur_size;
  if (length == 0) {
    return ret;
  }

  // For the remainder of the length, decide between a single large group or the
  // best configuration of smaller groups.
  float alt_cost = 0.0f;
  auto alt_sizes = GetGroupSizes(length, sizes.subspan(1), &alt_cost);
  if (cur_cost < alt_cost) {
    *cost += cur_cost;
    ret.push_back(cur_size);
  } else {
    *cost += alt_cost;
    ret.insert(ret.end(), alt_sizes.begin(), alt_sizes.end());
  }
  return ret;
}

}  // namespace

absl::StatusOr<std::vector<int>> GetGroupSizesForInput(
    int input_length, const std::vector<GroupSize>& group_sizes) {
  for (int i = 1; i < group_sizes.size(); ++i) {
    GroupSize group_a = group_sizes[i - 1];
    GroupSize group_b = group_sizes[i];
    // Sequence size must be descending, cost per token must be ascending.
    if (group_b.sequence_size >= group_a.sequence_size ||
        group_b.cost / group_b.sequence_size <
            group_a.cost / group_a.sequence_size) {
      return absl::FailedPreconditionError(
          "Group sizes must be in descending order, and cost per token must be "
          "in ascending order.");
    }
  }
  if (group_sizes.size() == 1) {
    int size = group_sizes[0].sequence_size;
    return std::vector<int>(::ml_drift::DivideRoundUp(input_length, size),
                            size);
  }
  float cost = 0.0f;
  return GetGroupSizes(input_length, absl::MakeSpan(group_sizes), &cost);
}

}  // namespace odml::infra::gpu
