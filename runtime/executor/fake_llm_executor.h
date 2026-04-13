// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MOCK_LLM_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MOCK_LLM_EXECUTOR_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Fake LLM executor for testing.
class FakeLlmExecutor : public LlmExecutor {
 public:
  // Creates a fake LLM executor with the given prefill and decode tokens.
  // - vocab_size: The vocabulary size of the LLM. It is used to determine the
  //   shape of the output logits TensorBuffer.
  // - prefill_tokens_set:The prefill tokens ([num_calls, num_tokens]) are the
  //   tokens that are expected to be passed in at each time. The Prefill
  //   function will only return OkStatus if the input tokens match the expected
  //   tokens.
  // - decode_tokens_set: The decode tokens ([num_calls, batch_size]) are the
  //   tokens that will be returned at each time the Decode function is called.
  // - batch_size: The batch size of the LLM. It is used to determine the shape
  //   of the output logits TensorBuffer.
  // - audio_embedding: The audio embedding ([num_calls, num_tokens,
  //   embedding_dim]) is the expected audio embedding that will be passed in
  //   at each time the Prefill function is called. The Prefill function will
  //   only return OkStatus if the input audio embedding matches the expected
  //   audio embedding.
  FakeLlmExecutor(
      int vocab_size, const std::vector<std::vector<int>>& prefill_tokens_set,
      const std::vector<std::vector<int>>& decode_tokens_set,
      int batch_size = 1,
      std::optional<std::vector<float>> audio_embedding = std::nullopt);

  absl::Status Prefill(const ExecutorInputs& inputs) override;
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override;

  absl::StatusOr<std::vector<std::vector<int>>> Decode() override;

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override;

  absl::Status Decode(const ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override;

  absl::string_view ExecutorBackendName() const override {
    return "FakeLlmExecutorBackend";
  };

  absl::StatusOr<int> GetVocabSize() override { return vocab_size_; }

  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return executor_settings_;
  };
  absl::StatusOr<LlmExecutorSettings*> GetMutableExecutorSettings() {
    return &executor_settings_;
  };
  absl::StatusOr<int> GetCurrentStep() const override { return current_step_; }
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return runtime_config_;
  };
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return runtime_state_;
  };

  absl::Status SetCurrentStep(int current_step) override {
    current_step_ = current_step;
    if (current_step >= prefill_tokens_total_) {
      decode_times_ = current_step - prefill_tokens_total_;
    } else {
      decode_times_ = 0;
    }
    return absl::OkStatus();
  }

  // Sets the status to be returned by the Prefill function.
  void SetPrefillStatus(const absl::Status& status) {
    prefill_status_ = status;
  }

  // Sets the status to be returned by the Decode function.
  void SetDecodeStatus(const absl::Status& status) { decode_status_ = status; }

  // Sets the delay before decoding. Useful for testing the cancellation
  // logic. The default value is 0, which means no delay.
  void SetDecodeDelay(absl::Duration delay) { decode_delay_ = delay; }

  absl::Status Reset() override;

 private:
  // Util function to try to sleep for the decode delay duration (if set). This
  // is used to simulate a long-running task.
  void TryDecodeDelay();

  int vocab_size_;
  std::vector<std::vector<int>> prefill_tokens_set_;
  std::vector<std::vector<int>> decode_tokens_set_;
  std::optional<std::vector<float>> audio_embedding_set_;
  int batch_size_;

  // The number of times the Prefill function has been called.
  int prefill_times_;
  // The number of times the Decode function has been called.
  int decode_times_;

  // The executor settings.
  LlmExecutorSettings executor_settings_;

  // The current step of the executor.
  int current_step_;

  // The total number of prefill tokens processed.
  int prefill_tokens_total_ = 0;

  // The processed tokens of the executor.
  ProcessedTokens processed_tokens_;

  // The status to be returned by the Prefill function.
  absl::Status prefill_status_ = absl::OkStatus();
  // The status to be returned by the Decode function.
  absl::Status decode_status_ = absl::OkStatus();

  // The delay before decoding. Useful for testing the cancellation logic.
  // The default value is 0, which means no delay.
  absl::Duration decode_delay_;

  enum class LastOp {
    kNone,
    kPrefill,
    kDecode,
  };
  LastOp last_op_ = LastOp::kNone;

  RuntimeConfig runtime_config_;
  RuntimeState runtime_state_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MOCK_LLM_EXECUTOR_H_
