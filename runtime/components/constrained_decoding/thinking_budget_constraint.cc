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

#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

// Wraps a Bitmap and additionally allows `extra_token_id_`.
class OverlayTokenBitmap : public Bitmap {
 public:
  OverlayTokenBitmap(std::unique_ptr<Bitmap> base, int extra_token_id,
                     int vocab_size)
      : base_(std::move(base)),
        extra_token_id_(extra_token_id),
        vocab_size_(vocab_size) {}
  bool Get(int index) const override {
    if (index < 0 || index >= vocab_size_) return false;
    return index == extra_token_id_ || base_ == nullptr || base_->Get(index);
  }

 private:
  const std::unique_ptr<Bitmap> base_;
  const int extra_token_id_;
  const int vocab_size_;
};

// Wraps a non-bitmap LogitMask and restores the original logit for
// `extra_token_id_` so that token stays allowed even if `base_` masks it.
class OverlayTokenLogitMask : public LogitMask {
 public:
  OverlayTokenLogitMask(std::unique_ptr<LogitMask> base, int extra_token_id)
      : base_(std::move(base)), extra_token_id_(extra_token_id) {}
  absl::Status Apply(absl::Span<float> logits) const override {
    return ApplyImpl(logits);
  }
  absl::Status Apply(absl::Span<tflite::half> logits) const override {
    return ApplyImpl(logits);
  }

 private:
  template <typename T>
  absl::Status ApplyImpl(absl::Span<T> logits) const {
    if (extra_token_id_ < 0 ||
        extra_token_id_ >= static_cast<int>(logits.size())) {
      return base_->Apply(logits);
    }
    const T saved = logits[extra_token_id_];
    ABSL_RETURN_IF_ERROR(base_->Apply(logits));
    logits[extra_token_id_] = saved;
    return absl::OkStatus();
  }

  const std::unique_ptr<LogitMask> base_;
  const int extra_token_id_;
};

}  // namespace

std::unique_ptr<Constraint::State> ThinkingBudgetConstraint::Start() const {
  auto state = std::make_unique<ThinkingState>();
  state->thinking_token_count = 0;
  // If no start token IDs are provided, it means the start-of-thought token
  // was prefilled in the prompt. We bypass the start matching phase and
  // transition immediately to the active thinking phase.
  if (start_token_ids_.empty()) {
    state->in_thinking = true;
    state->matching_start_index = -1;
  } else {
    state->in_thinking = false;
    state->matching_start_index = 0;
    // Initialize user_state so step-0 masks and skipped-thinking commits share
    // the same state.
    if (user_constraint_ != nullptr) {
      state->user_state = user_constraint_->Start();
    }
  }
  state->natural_end_match_index = 0;
  state->forced_end_token_index = -1;
  return state;
}

bool ThinkingBudgetConstraint::IsEnded(const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);
  // We are not ended if we are still matching start tokens or actively
  // thinking.
  if (s.in_thinking || s.matching_start_index >= 0) {
    return false;
  }

  // We are not in thinking, delegate to the user_constraint_ if there is any.
  if (user_constraint_ != nullptr) {
    return user_constraint_->IsEnded(*s.user_state);
  }

  // Returns false to avoid restarting the thinking constraint.
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
ThinkingBudgetConstraint::ComputeNext(const Constraint::State& state,
                                      int token) const {
  const auto& s = static_cast<const ThinkingState&>(state);
  auto next_s = std::make_unique<ThinkingState>();
  next_s->thinking_token_count = s.thinking_token_count;
  next_s->in_thinking = s.in_thinking;
  next_s->forced_end_token_index = s.forced_end_token_index;
  next_s->matching_start_index = s.matching_start_index;

  // We are in content phase if we are not thinking and not matching start
  // tokens.
  const bool in_content_phase = !s.in_thinking && s.matching_start_index == -1;

  if (in_content_phase && user_constraint_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(next_s->user_state,
                          user_constraint_->ComputeNext(*s.user_state, token));
  }

  int natural_end_match_index = s.natural_end_match_index;

  // Phase A: Matching start tokens
  if (next_s->matching_start_index >= 0) {
    if (ProcessStartMatching(*next_s, token)) {
      // Mismatch: skipped thinking. Transition to content phase.
      RET_CHECK_EQ(s.matching_start_index, 0)
          << "Start-of-thought mismatch after partial match at index "
          << s.matching_start_index;
      next_s->in_thinking = false;
      next_s->matching_start_index = -1;
      if (user_constraint_ != nullptr) {
        RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
        ABSL_ASSIGN_OR_RETURN(next_s->user_state, user_constraint_->ComputeNext(
                                                      *s.user_state, token));
      }
    }
  } else if (next_s->in_thinking) {
    // Phase B: Actively thinking
    if (next_s->forced_end_token_index >= 0) {
      ProcessForcedEnd(*next_s, token);
    } else {
      next_s->thinking_token_count++;
      ProcessNaturalEnd(*next_s, natural_end_match_index, token);
      CheckBudget(*next_s);
    }
  }

  // If we just transitioned out of thinking to content phase.
  const bool transitioned_to_content = s.in_thinking && !next_s->in_thinking;
  if (transitioned_to_content && user_constraint_ != nullptr &&
      next_s->user_state == nullptr) {
    next_s->user_state = user_constraint_->Start();
  }

  next_s->natural_end_match_index = natural_end_match_index;
  return next_s;
}

absl::StatusOr<std::unique_ptr<LogitMask>>
ThinkingBudgetConstraint::ComputeMask(const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);

  // Force the rest of the start sequence after a partial match.
  if (s.matching_start_index > 0) {
    return BitmapLogitMask::CreateSingleAllowedToken(
        vocab_size_, start_token_ids_[s.matching_start_index]);
  }

  // On step 0, allow either start_token_ids_[0] or valid user_constraint_
  // tokens.
  if (s.matching_start_index == 0) {
    if (user_constraint_ == nullptr) {
      return BitmapLogitMask::CreateAllAllowed(vocab_size_);
    }
    RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
    const int start_token = start_token_ids_[0];
    RET_CHECK(start_token >= 0 && start_token < vocab_size_)
        << "Start token out of vocabulary range: " << start_token;
    ABSL_ASSIGN_OR_RETURN(auto user_mask,
                          user_constraint_->ComputeMask(*s.user_state));
    if (user_mask->GetType() != MaskType::kBitmap) {
      return std::make_unique<OverlayTokenLogitMask>(std::move(user_mask),
                                                     start_token);
    }
    const auto& bitmap = static_cast<const BitmapLogitMask&>(*user_mask);
    RET_CHECK_EQ(bitmap.vocab_size(), vocab_size_);
    std::vector<uint64_t> words(bitmap.words().begin(), bitmap.words().end());
    words[start_token / 64] |= (uint64_t{1} << (start_token % 64));
    return std::make_unique<BitmapLogitMask>(vocab_size_, std::move(words));
  }

  // Suspend user constraint during thinking.
  if (s.in_thinking) {
    if (s.forced_end_token_index >= 0) {
      return BitmapLogitMask::CreateSingleAllowedToken(
          vocab_size_, end_token_ids_[s.forced_end_token_index]);
    }
    return BitmapLogitMask::CreateAllAllowed(vocab_size_);
  }

  if (user_constraint_ != nullptr) {
    RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
    return user_constraint_->ComputeMask(*s.user_state);
  }

  return BitmapLogitMask::CreateAllAllowed(vocab_size_);
}

absl::StatusOr<std::unique_ptr<Bitmap>> ThinkingBudgetConstraint::ComputeBitmap(
    const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);

  if (s.matching_start_index > 0) {
    return std::make_unique<SingleAllowedTokenBitmap>(
        start_token_ids_[s.matching_start_index]);
  }

  if (s.matching_start_index == 0) {
    if (user_constraint_ == nullptr) {
      return std::make_unique<AllAllowedBitmap>();
    }
    RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
    const int start_token = start_token_ids_[0];
    RET_CHECK(start_token >= 0 && start_token < vocab_size_)
        << "Start token out of vocabulary range: " << start_token;
    ABSL_ASSIGN_OR_RETURN(auto user_bitmap,
                          user_constraint_->ComputeBitmap(*s.user_state));
    return std::make_unique<OverlayTokenBitmap>(std::move(user_bitmap),
                                                start_token, vocab_size_);
  }

  // Suspend user constraint during thinking.
  if (s.in_thinking) {
    if (s.forced_end_token_index >= 0) {
      return std::make_unique<SingleAllowedTokenBitmap>(
          end_token_ids_[s.forced_end_token_index]);
    }
    return std::make_unique<AllAllowedBitmap>();
  }

  if (user_constraint_ != nullptr) {
    RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
    return user_constraint_->ComputeBitmap(*s.user_state);
  }

  return std::make_unique<AllAllowedBitmap>();
}

void ThinkingBudgetConstraint::ProcessForcedEnd(ThinkingState& state,
                                                int token) const {
  // Verify that the model generated the expected forced end token.
  // We only allow these tokens in ComputeMask, so it should match.
  if (token == end_token_ids_[state.forced_end_token_index]) {
    state.forced_end_token_index++;
    // If we have generated all forced end tokens, we are done thinking.
    if (state.forced_end_token_index >= end_token_ids_.size()) {
      state.in_thinking = false;
      state.forced_end_token_index = -1;
    }
  } else {
    // Should not happen under normal circumstances if ComputeMask restricts
    // vocabulary.
    state.forced_end_token_index = -1;
  }
}

bool ThinkingBudgetConstraint::ProcessStartMatching(ThinkingState& state,
                                                    int token) const {
  if (token == start_token_ids_[state.matching_start_index]) {
    state.matching_start_index++;
    // If we fully matched the start sequence, transition to thinking.
    if (state.matching_start_index >= start_token_ids_.size()) {
      state.matching_start_index = -1;
      state.in_thinking = true;
    }
  } else {
    // Mismatch: The model generated a token that is not part of the start
    // sequence.
    return true;  // Indicates thinking was skipped.
  }
  return false;
}

void ThinkingBudgetConstraint::ProcessNaturalEnd(ThinkingState& state,
                                                 int& natural_end_match_index,
                                                 int token) const {
  // Track if the model naturally generates the end-of-thinking sequence.
  if (token == end_token_ids_[natural_end_match_index]) {
    natural_end_match_index++;
    if (natural_end_match_index >= end_token_ids_.size()) {
      // Naturally reached the end of thinking.
      state.in_thinking = false;
      natural_end_match_index = 0;
    }
  } else if (token == end_token_ids_[0]) {
    // Restart matching from the first end token if we got a partial match
    // reset.
    natural_end_match_index = 1;
  } else {
    natural_end_match_index = 0;
  }
}

void ThinkingBudgetConstraint::CheckBudget(ThinkingState& state) const {
  // If we exceeded the budget, trigger the forced end sequence.
  if (budget_ >= 0 && state.in_thinking &&
      state.thinking_token_count >= budget_) {
    state.forced_end_token_index = 0;
  }
}

}  // namespace litert::lm
