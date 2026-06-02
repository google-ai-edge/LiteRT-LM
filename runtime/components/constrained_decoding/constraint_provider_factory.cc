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

#include "runtime/components/constrained_decoding/constraint_provider_factory.h"

#include <memory>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/external_constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/tokenizer.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateConstraintProvider(
    const ConstraintProviderConfig& constraint_provider_config,
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids) {
  if (std::holds_alternative<ExternalConstraintConfig>(
          constraint_provider_config)) {
    return std::make_unique<ExternalConstraintProvider>();
  } else if (std::holds_alternative<LlGuidanceConfig>(
                 constraint_provider_config)) {
    auto llg_guidance_config =
        std::get<LlGuidanceConfig>(constraint_provider_config);
    if (!llg_guidance_config.eos_id.has_value()) {
      // If eos_id is not provided in the config, use the first valid stop token
      // as the eos_id.
      for (const auto& stop_sequence : stop_token_ids) {
        if (stop_sequence.size() == 1) {
          llg_guidance_config.eos_id = stop_sequence[0];
          break;
        }
      }
      if (!llg_guidance_config.eos_id.has_value()) {
        return absl::InvalidArgumentError(
            "LlGuidanceConfig::eos_id wasn't set and no valid stop token was "
            "found in SessionConfig.");
      }
    }
    return LlgConstraintProvider::Create(tokenizer, llg_guidance_config);
  }

  return absl::UnimplementedError("Unknown ConstraintProviderConfig type.");
}

}  // namespace litert::lm
