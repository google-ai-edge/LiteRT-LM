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

// Highway SIMD kernels for fast_cpu_ops.cc, compiled once per target via
// HWY_TARGET_INCLUDE + foreach_target.h (using the standard Highway toggle
// guard below).

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_INL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_INL_H_
#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_INL_H_

#if defined(                                                            \
    THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_TOGGLE
#undef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_TOGGLE
#else
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_FAST_CPU_OPS_TOGGLE
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hwy/cache_control.h"  // from @highway
#include "hwy/highway.h"  // from @highway

HWY_BEFORE_NAMESPACE();
namespace litert::lm::internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

HWY_INLINE bool BuffersOverlapSimd(const uint8_t* a, const uint8_t* b,
                                   size_t n) {
  const auto ua = reinterpret_cast<uintptr_t>(a);
  const auto ub = reinterpret_cast<uintptr_t>(b);
  return (ua < ub) ? (ua + n > ub) : (ub + n > ua);
}

HWY_INLINE void CopyBytesSimd(uint8_t* dst, const uint8_t* src, size_t n) {
  if (dst == src || n == 0) {
    return;
  }
  if (BuffersOverlapSimd(dst, src, n)) {
    std::memmove(dst, src, n);
    return;
  }

  const hn::ScalableTag<uint8_t> d;
  const size_t lanes = hn::Lanes(d);
  if (n < lanes) {
    std::memcpy(dst, src, n);
    return;
  }

  const size_t unroll_bytes = lanes * 8;
  size_t i = 0;
  for (; i + unroll_bytes <= n; i += unroll_bytes) {
    for (size_t p = 0; p < unroll_bytes; p += 64) {
      hwy::Prefetch(src + i + 512 + p);
    }
    const auto v0 = hn::LoadU(d, src + i);
    const auto v1 = hn::LoadU(d, src + i + lanes);
    const auto v2 = hn::LoadU(d, src + i + 2 * lanes);
    const auto v3 = hn::LoadU(d, src + i + 3 * lanes);
    const auto v4 = hn::LoadU(d, src + i + 4 * lanes);
    const auto v5 = hn::LoadU(d, src + i + 5 * lanes);
    const auto v6 = hn::LoadU(d, src + i + 6 * lanes);
    const auto v7 = hn::LoadU(d, src + i + 7 * lanes);
    hn::StoreU(v0, d, dst + i);
    hn::StoreU(v1, d, dst + i + lanes);
    hn::StoreU(v2, d, dst + i + 2 * lanes);
    hn::StoreU(v3, d, dst + i + 3 * lanes);
    hn::StoreU(v4, d, dst + i + 4 * lanes);
    hn::StoreU(v5, d, dst + i + 5 * lanes);
    hn::StoreU(v6, d, dst + i + 6 * lanes);
    hn::StoreU(v7, d, dst + i + 7 * lanes);
  }
  for (; i + lanes <= n; i += lanes) {
    const auto v = hn::LoadU(d, src + i);
    hn::StoreU(v, d, dst + i);
  }
  if (i < n) {
    const auto tail = hn::LoadU(d, src + n - lanes);
    hn::StoreU(tail, d, dst + n - lanes);
  }
}

void RunStridedSliceSimd(const uint8_t* src, uint8_t* dst, const size_t* counts,
                         const size_t* src_strides, const size_t* dst_strides,
                         int dim, size_t block_bytes) {
  if (dim < 0) {
    CopyBytesSimd(dst, src, block_bytes);
    return;
  }
  if (dim == 0) {
    const size_t n = counts[0];
    const size_t ss = src_strides[0];
    const size_t ds = dst_strides[0];
    for (size_t i = 0; i < n; ++i) {
      CopyBytesSimd(dst + i * ds, src + i * ss, block_bytes);
    }
    return;
  }
  const size_t n = counts[dim];
  const size_t ss = src_strides[dim];
  const size_t ds = dst_strides[dim];
  for (size_t i = 0; i < n; ++i) {
    RunStridedSliceSimd(src + i * ss, dst + i * ds, counts, src_strides,
                        dst_strides, dim - 1, block_bytes);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace litert::lm::internal
HWY_AFTER_NAMESPACE();

#endif  // toggle guard
