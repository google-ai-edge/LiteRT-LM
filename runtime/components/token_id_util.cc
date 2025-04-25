#include "runtime/components/token_id_util.h"

#include <algorithm>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

absl::Status PreprocessTokenIds(std::vector<int>& token_ids, int start_token_id,
                                int max_num_tokens,
                                float context_length_ratio_threhold) {
  if (token_ids.size() + 1 > max_num_tokens * context_length_ratio_threhold) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The input context length is too long. The input token length is %d "
        "and the max_num_tokens is %d.",
        token_ids.size() + 1, max_num_tokens));
  }
  // Prepend the start token id to the token ids.
  token_ids.insert(token_ids.begin(), start_token_id);
  return absl::OkStatus();
}

absl::StatusOr<bool> StopTokenFound(absl::Span<const int> decoded_token_ids,
                                    const std::vector<int>& stop_token_ids,
                                    std::vector<bool>& stop_token_found) {
  if (decoded_token_ids.size() != stop_token_found.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The size of the decoded token ids is %d and the size of the stop "
        "token found vector is %d. They should be the same.",
        decoded_token_ids.size(), stop_token_found.size()));
  }
  for (int i = 0; i < decoded_token_ids.size(); ++i) {
    if (stop_token_found[i]) {
      continue;
    }
    for (const auto& stop_token_id : stop_token_ids) {
      if (decoded_token_ids[i] == stop_token_id) {
        stop_token_found[i] = true;
        break;
      }
    }
  }
  return std::all_of(
      stop_token_found.begin(), stop_token_found.end(),
      [](int stop_token_found) { return stop_token_found == true; });
}

}  // namespace litert::lm
