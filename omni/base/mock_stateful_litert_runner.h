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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_STATEFUL_LITERT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_STATEFUL_LITERT_RUNNER_H_

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/stateful_litert_runner.h"

namespace litert::omni {

class MockStatefulLiteRtRunner : public StatefulLiteRtRunner {
 public:
  MockStatefulLiteRtRunner() = default;
  ~MockStatefulLiteRtRunner() override = default;

  MOCK_METHOD(absl::Status, Reset, (), (override));

  MOCK_METHOD(absl::StatusOr<absl::Span<const TensorBuffer>>, Step,
              (absl::Span<const TensorBuffer> non_state_inputs,
               bool auto_commit_state),
              (override));

  MOCK_METHOD(absl::Status, CommitState, (), (override));

  MOCK_METHOD(absl::Span<const TensorBuffer>, GetActiveInputStates, (),
              (const, override));

  MOCK_METHOD(absl::Span<TensorBuffer>, GetNonStateInputBuffers, (),
              (override));

  MOCK_METHOD(absl::Span<const TensorBuffer>, GetNonStateOutputBuffers, (),
              (const, override));
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_STATEFUL_LITERT_RUNNER_H_
