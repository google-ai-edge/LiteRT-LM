#include "third_party/odml/litert_lm/runtime/executor/fake_llm_executor.h"

#include <limits>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "third_party/odml/litert_lm/runtime/executor/llm_executor.h"
#include "third_party/odml/litert_lm/runtime/util/convert_tensor_buffer.h"
#include "third_party/odml/litert_lm/runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// Converts the given ids to logits TensorBuffer in the shape of [batch_size,
// vocab_size].
void DecodeIdsToLogits(const std::vector<int>& ids, int vocab_size,
                       ::litert::TensorBuffer& output_logits) {
  auto logits_span = ReferTensorBufferAsSpan<float>(output_logits);
  for (int i = 0; i < ids.size(); ++i) {
    for (int j = 0; j < vocab_size; ++j) {
      int index = i * vocab_size + j;
      if (ids[i] == j) {
        (*logits_span)[index] = std::numeric_limits<float>::max();
      } else {
        (*logits_span)[index] = std::numeric_limits<float>::lowest();
      }
    }
  }
}

// Checks if the given expected and actual spans are equivalent in terms of the
// size and values.
absl::Status CheckEquivalent(absl::Span<int> expected, absl::Span<int> actual) {
  if (expected.size() != actual.size()) {
    return absl::InvalidArgumentError(absl::StrCat("Expected token size is ",
                                                   expected.size(), " but got ",
                                                   actual.size()));
  }
  for (int i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      return absl::InvalidArgumentError(absl::StrCat("Expected token at index ",
                                                     i, " is ", expected[i],
                                                     " but got ", actual[i]));
    }
  }
  return absl::OkStatus();
}

}  // namespace

FakeLlmExecutor::FakeLlmExecutor(
    int vocab_size, const std::vector<std::vector<int>>& prefill_tokens_set,
    const std::vector<std::vector<int>>& decode_tokens_set)
    : vocab_size_(vocab_size),
      prefill_tokens_set_(prefill_tokens_set),
      decode_tokens_set_(decode_tokens_set),
      prefill_times_(0),
      decode_times_(0) {}

absl::Status FakeLlmExecutor::Prefill(const Inputs& inputs) {
  if (prefill_times_ >= prefill_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Prefill function has been called more times than the number of "
        "expected prefill tokens.",
        prefill_times_));
  }
  auto input_span = ReferTensorBufferAsSpan<int>(inputs.text_input.token_ids);
  RETURN_IF_ERROR(CheckEquivalent(
      absl::MakeSpan(prefill_tokens_set_[prefill_times_]), *input_span));
  prefill_times_++;
  return absl::OkStatus();
}

absl::Status FakeLlmExecutor::Prefill(
    const Inputs& inputs, const PrefillQueryParams& prefill_query_params) {
  return Prefill(inputs);
}

absl::Status FakeLlmExecutor::Decode(::litert::TensorBuffer& output_tokens) {
  if (decode_times_ >= decode_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode function has been called more times than the number of "
        "expected decode tokens.",
        decode_times_));
  }
  auto tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
  for (int i = 0; i < decode_tokens_set_[decode_times_].size(); ++i) {
    (*tokens_span)[i] = decode_tokens_set_[decode_times_][i];
  }
  decode_times_++;
  return absl::OkStatus();
}

absl::Status FakeLlmExecutor::Decode(const Inputs& inputs,
                                     ::litert::TensorBuffer& output_logits) {
  if (decode_times_ >= decode_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode function has been called more times than the number of "
        "expected decode tokens.",
        decode_times_));
  }
  if (decode_times_ > 0) {
    // Check that the input tokens match the decode tokens from the last call.
    auto input_span = ReferTensorBufferAsSpan<int>(inputs.text_input.token_ids);
    RETURN_IF_ERROR(CheckEquivalent(
        absl::MakeSpan(decode_tokens_set_[decode_times_ - 1]), *input_span));
  }
  DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                    output_logits);
  decode_times_++;
  return absl::OkStatus();
}

}  // namespace litert::lm
