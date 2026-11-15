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

#include "runtime/components/constrained_decoding/constraint.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/logit_mask.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Bitmap>> Constraint::ComputeBitmap(
    const State& state) const {
  return absl::UnimplementedError(
      "ComputeBitmap is not implemented. Use ComputeMask instead.");
}

absl::StatusOr<std::unique_ptr<LogitMask>> Constraint::ComputeMask(
    const State& state) const {
  ABSL_ASSIGN_OR_RETURN(auto bitmap, ComputeBitmap(state));
  const int vocab_size = GetVocabularySize();
  if (!bitmap) {
    return BitmapLogitMask::CreateAllAllowed(vocab_size);
  }
  std::vector<bool> allowed(vocab_size, false);
  for (int i = 0; i < vocab_size; ++i) {
    allowed[i] = bitmap->Get(i);
  }
  return std::make_unique<BitmapLogitMask>(vocab_size, std::move(allowed));
}

}  // namespace litert::lm
