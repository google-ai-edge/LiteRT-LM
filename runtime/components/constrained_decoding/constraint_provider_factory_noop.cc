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

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateConstraintProvider(
    const ConstraintProviderConfig& constraint_provider_config,
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids) {
  return absl::UnimplementedError(
      "Constrained decoding is disabled in this build.");
}

}  // namespace litert::lm
