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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_RUNNER_H_

#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"

namespace litert::omni {

class MockLiteRtRunner : public LiteRtRunner {
 public:
  MockLiteRtRunner() = default;
  ~MockLiteRtRunner() override = default;

  MOCK_METHOD(absl::StatusOr<std::vector<::litert::TensorBuffer>>,
              CreateInputBuffers, (absl::string_view signature_name),
              (override));

  MOCK_METHOD(absl::StatusOr<std::vector<::litert::TensorBuffer>>,
              CreateOutputBuffers, (absl::string_view signature_name),
              (override));

  MOCK_METHOD(absl::Status, Run,
              (absl::string_view signature_name,
               absl::Span<const ::litert::TensorBuffer> input_buffers,
               absl::Span<const ::litert::TensorBuffer> output_buffers),
              (override));
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MOCK_LITERT_RUNNER_H_
