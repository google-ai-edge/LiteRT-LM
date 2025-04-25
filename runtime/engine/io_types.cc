#include "runtime/engine/io_types.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace litert::lm {

// A container to host the model responses.
Responses::Responses(int num_output_candidates)
    : num_output_candidates_(num_output_candidates) {
  response_texts_ = std::vector<std::string>(num_output_candidates_);
}

absl::StatusOr<absl::string_view> Responses::GetResponseTextAt(
    int index) const {
  if (index < 0 || index >= num_output_candidates_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Index ", index, " is out of range [0, ",
                     num_output_candidates_, ")."));
  }
  return response_texts_[index];
}

absl::StatusOr<float> Responses::GetScoreAt(int index) const {
  if (index < 0 || index >= scores_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Index ", index, " is out of range [0, ", scores_.size(), ")."));
  }
  return scores_[index];
}

std::vector<std::string>& Responses::GetMutableResponseTexts() {
  return response_texts_;
}

std::vector<float>& Responses::GetMutableScores() {
  if (scores_.empty()) {
    scores_ = std::vector<float>(num_output_candidates_,
                                 -std::numeric_limits<float>::infinity());
  }
  return scores_;
}

}  // namespace litert::lm
