// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_RUNTIME_PARAMS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_RUNTIME_PARAMS_H_

#include <optional>

namespace ml_drift {

struct LlmRuntimeParams {
  int token_index_offset;
  std::optional<int> active_tokens;  // for cached self-attention
  int topk;                          // for sampling during post-processing

  // indexes of values in linear int32 param buffer
  static constexpr int kTokenOffsetIndex = 0;
  static constexpr int kActiveTokensIndex = 1;
  static constexpr int kActiveTokensAlignedIndex =
      2;  // stores active tokens aligned to some value
  static constexpr int kTopKIndex = 3;
  static constexpr int kToRemoveIndicesSize =
      3;  // cannot be used together with kTopKIndex
  static constexpr int kRingOffsetIndex = 4;

  static constexpr int kTotalParamsCount = 5;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_RUNTIME_PARAMS_H_
