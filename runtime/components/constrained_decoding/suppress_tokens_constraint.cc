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

#include "runtime/components/constrained_decoding/suppress_tokens_constraint.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"

namespace litert::lm {

SuppressTokensConstraint::SuppressTokensConstraint(int vocab_size,
                                                   SuppressTokensConfig config)
    : vocab_size_(vocab_size), config_(std::move(config)) {
  if (!config_.enabled()) {
    mask_ = BitmapLogitMask::CreateAllAllowed(vocab_size_);
  } else {
    std::vector<int> disallowed;
    for (int token_id : config_.suppress_tokens()) {
      if (token_id >= 0 && token_id < vocab_size_) {
        disallowed.push_back(token_id);
      }
    }
    mask_ = BitmapLogitMask::CreateFromDisallowedTokens(
        vocab_size_, absl::MakeConstSpan(disallowed));
  }
}

SuppressTokensConstraint::SuppressTokensConstraint(
    int vocab_size, absl::flat_hash_set<int> suppress_tokens)
    : SuppressTokensConstraint(
          vocab_size, SuppressTokensConfig(std::move(suppress_tokens))) {}

std::unique_ptr<Constraint::State> SuppressTokensConstraint::Start() const {
  return std::make_unique<SuppressTokensState>();
}

bool SuppressTokensConstraint::IsEnded(const State& state) const {
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
SuppressTokensConstraint::ComputeNext(const State& state, int token) const {
  if (token < 0 || token >= vocab_size_) {
    return absl::InvalidArgumentError("Invalid token id.");
  }
  return std::make_unique<SuppressTokensState>();
}

absl::StatusOr<std::unique_ptr<LogitMask>>
SuppressTokensConstraint::ComputeMask(const State& state) const {
  return std::make_unique<BitmapLogitMask>(*mask_);
}

}  // namespace litert::lm
