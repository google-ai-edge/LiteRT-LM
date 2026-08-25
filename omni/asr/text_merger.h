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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Represents the merged text result from processing an audio chunk.
struct MergeResult {
  // Finalized text that will not change in subsequent chunks.
  std::string confirmed_text;
  // Pending text subject to alignment and change in the next chunk.
  std::string unconfirmed_text;
};

// Abstract interface for aligning and merging overlapping word streams.
class TextMerger : public SingleThreadedStageWithDeque<MergeResult> {
 public:
  using MergeResult = ::litert::omni::asr::MergeResult;

  explicit TextMerger(
      Stage<std::vector<Detokenizer::Word>>* absl_nonnull detokenizer)
      : detokenizer_(*detokenizer) {}

  ~TextMerger() override = default;

  // Flushes remaining unconfirmed text into the output queue at end of stream.
  virtual absl::Status Flush() = 0;

 protected:
  bool NeedScheduleInternal() const override {
    return detokenizer_.HasOutput();
  }

  Stage<std::vector<Detokenizer::Word>>& detokenizer_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_
