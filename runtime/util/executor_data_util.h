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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

// Util function for combining multiple ExecutorVisionData into a single
// ExecutorVisionData, by concatenating the vision embeddings in a single
// tensor buffer.
//
// Specifically, if the elements of input ExecutorVisionData have TensorBuffer
// with shapes,
//  [batch_size, num_token_1, feature_dim].
//  [batch_size, num_token_2, feature_dim].
//  ...
//  [batch_size, num_token_n, feature_dim].
// The output ExecutorVisionData will have TensorBuffer with shape,
// [batch_size, 1, num_token_1 + num_token_2 + ... + num_token_n,
// feature_dim].
//
// Or if the elements of input ExecutorVisionData have TensorBuffer
// with shapes,
//  [batch_size, dim1, num_token_1, feature_dim].
//  [batch_size, dim1, num_token_2, feature_dim].
//  ...
//  [batch_size, dim1, num_token_n, feature_dim].
// The output ExecutorVisionData will have TensorBuffer with shape,
// [batch_size, dim1, num_token_1 + num_token_2 + ... + num_token_n,
// feature_dim].
absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data);

// Util function for combining multiple ExecutorAudioData into a single
// ExecutorAudioData, by concatenating the audio embeddings in a single tensor
// buffer.
//
// Specifically, if the elements of input ExecutorAudioData have TensorBuffer
// with shapes,
//  [batch_size, num_token_1, feature_dim].
//  [batch_size, num_token_2, feature_dim].
//  ...
//  [batch_size, num_token_n, feature_dim].
// The output ExecutorAudioData will have TensorBuffer with shape,
// [batch_size, num_token_1 + num_token_2 + ... + num_token_n, feature_dim].
absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data);

// Util function for extracting the floating-point audio embedding soft tokens
// from an ExecutorAudioData. If the internal tensor buffer includes valid
// tokens counts smaller than its capacity, the padded garbage memory is
// identified and removed. Otherwise, the whole buffer length is returned.
absl::StatusOr<std::vector<std::vector<float>>> ExtractAudioSoftTokens(
    ExecutorAudioData& audio_data);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_
