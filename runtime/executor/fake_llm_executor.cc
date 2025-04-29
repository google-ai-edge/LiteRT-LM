#include "runtime/executor/fake_llm_executor.h"

#include <limits>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

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

absl::Status FakeLlmExecutor::Prefill(const ExecutorInputs& inputs) {
  if (prefill_times_ >= prefill_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Prefill function has been called more times than the number of "
        "expected prefill tokens.",
        prefill_times_));
  }
  auto input_span =
      ReferTensorBufferAsSpan<int>(*(*inputs.GetTextTokenIdsPtr()));
  RETURN_IF_ERROR(CheckEquivalent(
      absl::MakeSpan(prefill_tokens_set_[prefill_times_]), *input_span));
  prefill_times_++;
  return absl::OkStatus();
}

absl::Status FakeLlmExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& prefill_params) {
  if (prefill_params.GetWaitForCompletion()) {
    // Sleep some time here to simulate a synchronous prefill.
    // We can time the function time in test to make sure the code calls prefill
    // with a correct wait_for_completion flag.
    absl::SleepFor(absl::Milliseconds(100));
  }
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

absl::Status FakeLlmExecutor::Decode(const ExecutorInputs& inputs,
                                     ::litert::TensorBuffer& output_logits) {
  if (decode_times_ >= decode_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode function has been called more times than the number of "
        "expected decode tokens.",
        decode_times_));
  }
  if (decode_times_ > 0) {
    // Check that the input tokens match the decode tokens from the last call.
    auto input_span = ReferTensorBufferAsSpan<int>(
        *(*inputs.GetTextTokenIdsPtr()));
    RETURN_IF_ERROR(CheckEquivalent(
        absl::MakeSpan(decode_tokens_set_[decode_times_ - 1]), *input_span));
  }
  DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                    output_logits);
  decode_times_++;
  return absl::OkStatus();
}

}  // namespace litert::lm
