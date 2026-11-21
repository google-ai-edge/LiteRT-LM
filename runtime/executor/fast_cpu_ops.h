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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert

namespace litert::lm {

// Configures an op resolver with SIMD-accelerated fallback CPU kernels (such as
// SLICE) on `options` when building against the statically linked LiteRT
// runtime. No-op in dynamic or header-only runtime link builds.
absl::Status MaybeConfigureFastCpuOpResolver(litert::Options& options);

namespace internal {

inline constexpr int kMaxTensorRank = 8;

// Exposed for unit testing of dimension-coalesced N-dimensional byte slicing.
// Copies slice `begins[0..rank-1]`, `sizes[0..rank-1]` (validated >= 0 with
// `0 < rank <= kMaxTensorRank` and `begins[d] + sizes[d] <= in_dims[d]`) of
// element width `elem_bytes` from `src` into dense contiguous `dst`.
// Setting `force_portable` bypasses Highway SIMD dispatch so unit tests can
// exercise the portable strided implementation on all host architectures.
void FastSliceNdBytes(const uint8_t* src, const int64_t* in_dims,
                      const int64_t* begins, const int64_t* sizes, int rank,
                      size_t elem_bytes, uint8_t* dst,
                      bool force_portable = false);

}  // namespace internal

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_H_
