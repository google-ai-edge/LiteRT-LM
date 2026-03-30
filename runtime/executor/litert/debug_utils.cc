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

#include "runtime/executor/litert/debug_utils.h"

#include <cstddef>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {

void LogValues(absl::Span<const float> values, size_t num_values_to_log,
               absl::string_view debug) {
  constexpr size_t kNumExtraValuesToLog = 10;
  if (num_values_to_log * 3 + kNumExtraValuesToLog >= values.size()) {
    ABSL_LOG(INFO) << debug << "(size=" << values.size()
                   << "): " << absl::StrJoin(values, ", ");
    return;
  }

  size_t end_offset = values.size() - num_values_to_log;
  size_t mid_offset = end_offset / 2;
  ABSL_LOG(INFO) << debug << "(size=" << values.size() << "): "
                 << absl::StrJoin(values.subspan(0, num_values_to_log), ", ")
                 << " ... "
                 << absl::StrJoin(values.subspan(mid_offset, num_values_to_log),
                                  ", ")
                 << " ... " << absl::StrJoin(values.subspan(end_offset), ", ");
}

void LogTensor(TensorBuffer& tensor, size_t num_values_to_log,
               absl::string_view debug) {
  // Try to get the reference if tensor is in CPU memory.
  auto values_span = ReferTensorBufferAsSpan<float>(tensor);
  if (values_span) {
    LogValues(*values_span, num_values_to_log, debug);
    return;
  }

  // Otherwise, copy the logits from the tensor buffer to a vector.
  auto values_vector = CopyFromTensorBuffer<float>(tensor);
  if (values_vector) {
    LogValues(*values_vector, num_values_to_log, debug);
    return;
  }

  ABSL_LOG(ERROR) << debug << ": Failed to log logits.";
}

}  // namespace litert::lm
