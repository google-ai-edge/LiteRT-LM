#include "runtime/engine/io_types.h"

#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

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
  if (scores_.empty()) {
    return absl::InvalidArgumentError("Scores are not set.");
  }
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

std::ostream& operator<<(std::ostream& os, const Responses& responses) {
  if (responses.GetNumOutputCandidates() == 0) {
    os << " No reponses." << std::endl;
    return os;
  }
  os << "Total candidates: " << responses.GetNumOutputCandidates() << ":"
     << std::endl;

  for (int i = 0; i < responses.GetNumOutputCandidates(); ++i) {
    absl::StatusOr<float> score_status = responses.GetScoreAt(i);
    if (score_status.ok()) {
      os << "  Candidate " << i << " (score: " << *score_status
         << "):" << std::endl;
    } else {
      os << "  Candidate " << i << " (score: N/A):" << std::endl;
    }

    absl::StatusOr<absl::string_view> text_status =
        responses.GetResponseTextAt(i);
    if (text_status.ok()) {
      os << "    Text: \"" << *text_status << "\"" << std::endl;
    } else {
      os << "    Text: Error - " << text_status.status().message() << std::endl;
    }
  }
  return os;  // Return the ostream to allow chaining
}

}  // namespace litert::lm
