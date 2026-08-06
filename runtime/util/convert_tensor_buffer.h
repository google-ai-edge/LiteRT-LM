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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_

#include "support/util/convert_tensor_buffer.h"

namespace litert::lm {
using ::litert::support::ConvertAndCopyToTensorBuffer;
using ::litert::support::CopyFromTensorBuffer;
using ::litert::support::CopyFromTensorBuffer2D;
using ::litert::support::CopyToTensorBuffer;
using ::litert::support::CreateTensorBuffer;
using ::litert::support::DropTokensfromTensorBuffer;
using ::litert::support::ElementTypeFor;
using ::litert::support::ReferTensorBufferAsSpan;
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_
