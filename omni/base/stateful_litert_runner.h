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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STATEFUL_LITERT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STATEFUL_LITERT_RUNNER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"

namespace litert::omni {

// Interface for stateful LiteRT runners that maintain recurrent state tensors
// across inference steps for a single stateful signature.
class StatefulLiteRtRunner {
 public:
  virtual ~StatefulLiteRtRunner() = default;

  // Resets internal state buffers (zeroes them out).
  virtual absl::Status Reset() = 0;

  // Runs a single recurrent step with non-state input buffers.
  // Returns non-state output buffers.
  // If `auto_commit_state` is true, automatically swaps the output states into
  // the active input states.
  virtual absl::StatusOr<absl::Span<const TensorBuffer>> Step(
      absl::Span<const TensorBuffer> non_state_inputs,
      bool auto_commit_state) = 0;

  // Commits the output states produced by the last Step() call as the active
  // input states for subsequent steps.
  virtual absl::Status CommitState() = 0;

  // Returns the active input state buffers.
  virtual absl::Span<const TensorBuffer> GetActiveInputStates() const = 0;

  // Returns mutable non-state input buffers for writing inputs directly.
  virtual absl::Span<TensorBuffer> GetNonStateInputBuffers() = 0;

  // Returns non-state output buffers from the latest step.
  virtual absl::Span<const TensorBuffer> GetNonStateOutputBuffers() const = 0;
};

// Implementation of StatefulLiteRtRunner wrapping a LiteRtRunner.
class StatefulLiteRtRunnerImpl : public StatefulLiteRtRunner {
 public:
  // Creates a StatefulLiteRtRunner for the given stateful signature.
  static absl::StatusOr<std::unique_ptr<StatefulLiteRtRunnerImpl>> Create(
      LiteRtRunner* absl_nonnull runner, absl::string_view signature_name,
      size_t num_non_state_inputs, size_t num_non_state_outputs);

  // Creates a StatefulLiteRtRunner taking ownership of the LiteRtRunner.
  static absl::StatusOr<std::unique_ptr<StatefulLiteRtRunnerImpl>> Create(
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      absl::string_view signature_name, size_t num_non_state_inputs,
      size_t num_non_state_outputs);

  ~StatefulLiteRtRunnerImpl() override = default;

  absl::Status Reset() override;

  absl::StatusOr<absl::Span<const TensorBuffer>> Step(
      absl::Span<const TensorBuffer> non_state_inputs,
      bool auto_commit_state) override;

  absl::Status CommitState() override;

  absl::Span<const TensorBuffer> GetActiveInputStates() const override;

  absl::Span<TensorBuffer> GetNonStateInputBuffers() override;

  absl::Span<const TensorBuffer> GetNonStateOutputBuffers() const override;

 private:
  StatefulLiteRtRunnerImpl(LiteRtRunner* absl_nonnull runner,
                           std::string signature_name,
                           size_t num_non_state_inputs,
                           size_t num_non_state_outputs)
      : runner_(*runner),
        signature_name_(std::move(signature_name)),
        num_non_state_inputs_(num_non_state_inputs),
        num_non_state_outputs_(num_non_state_outputs) {}

  std::unique_ptr<LiteRtRunner> owned_runner_;
  LiteRtRunner& runner_;
  std::string signature_name_;
  size_t num_non_state_inputs_;
  size_t num_non_state_outputs_;

  int active_set_ = 0;
  std::vector<TensorBuffer> input_buffers_0_;
  std::vector<TensorBuffer> input_buffers_1_;
  std::vector<TensorBuffer> output_buffers_0_;
  std::vector<TensorBuffer> output_buffers_1_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STATEFUL_LITERT_RUNNER_H_
