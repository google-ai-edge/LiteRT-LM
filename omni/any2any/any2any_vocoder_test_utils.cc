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

#include "omni/any2any/any2any_vocoder_test_utils.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/mock_litert_runner.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::any2any {

using ::testing::_;
using ::testing::NiceMock;

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVocoderRunner(
    int input_tokens, int output_samples,
    std::vector<std::vector<int32_t>>* recorded_windows,
    int samples_per_token) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([input_tokens](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            support::CreateTensorBuffer<int32_t>({1, input_tokens}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([output_samples](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            support::CreateTensorBuffer<float>({1, output_samples}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [input_tokens, output_samples, recorded_windows, samples_per_token](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_ASSIGN_OR_RETURN(
                std::vector<int32_t> codes,
                support::CopyFromTensorBuffer<int32_t>(inputs[0]));
            if (recorded_windows != nullptr) {
              recorded_windows->push_back(codes);
            }
            const int spt =
                samples_per_token > 0
                    ? samples_per_token
                    : ((input_tokens > 0 && output_samples >= input_tokens)
                           ? (output_samples / input_tokens)
                           : 1);
            std::vector<float> pcm(output_samples, 0.0f);
            for (int i = 0; i < output_samples; ++i) {
              const int tok_idx = i / spt;
              const int sample_in_tok = i % spt;
              const int code =
                  tok_idx < static_cast<int>(codes.size()) ? codes[tok_idx] : 0;
              pcm[i] = static_cast<float>(code * 10 + sample_in_tok);
            }
            TensorBuffer& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });
  return runner;
}

}  // namespace litert::omni::any2any
