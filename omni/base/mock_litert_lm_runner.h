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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_LM_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_LM_RUNNER_H_

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::omni {

class MockLiteRtLmRunner : public LiteRtLmRunner {
 public:
  MockLiteRtLmRunner() = default;
  ~MockLiteRtLmRunner() override = default;

  MOCK_METHOD(absl::Status, Prefill, (const lm::ExecutorInputs& inputs),
              (override));

  MOCK_METHOD(absl::StatusOr<TensorBuffer>, Decode,
              (const lm::ExecutorInputs& inputs), (override));

  MOCK_METHOD(absl::Status, Reset, (), (override));
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_LM_RUNNER_H_
