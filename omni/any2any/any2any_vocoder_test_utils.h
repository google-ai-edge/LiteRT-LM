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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/any2any/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Fake upstream TTS stage allowing unit tests to enqueue `Any2AnyTtsOutput`
// items directly for downstream vocoder stage tests.
class FakeTtsStage : public SingleThreadedStageWithDeque<Any2AnyTtsOutput> {
 public:
  void Push(Any2AnyTtsOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

// Creates a mock `LiteRtRunner` configured with `Int32[1, input_tokens]` input
// and `Float32[1, output_samples]` output buffers.
//
// The default `Run()` behavior optionally appends each input token window to
// `recorded_windows` (if non-null) and writes deterministic PCM samples
// `code * 10 + sample_in_tok` using `samples_per_token` (or
// `output_samples / input_tokens` when `samples_per_token <= 0`).
std::unique_ptr<::testing::NiceMock<MockLiteRtRunner>> MakeMockVocoderRunner(
    int input_tokens, int output_samples,
    std::vector<std::vector<int32_t>>* recorded_windows = nullptr,
    int samples_per_token = 0);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_VOCODER_TEST_UTILS_H_
