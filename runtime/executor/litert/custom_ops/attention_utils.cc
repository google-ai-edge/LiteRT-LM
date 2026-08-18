/* Copyright 2026 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "runtime/executor/litert/custom_ops/attention_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ruy/matrix.h"  // from @ruy
#include "ruy/mul_params.h"  // from @ruy
#include "ruy/ruy.h"  // from @ruy

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

#ifndef __aarch64__
inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return vmlaq_f32(a, b, c);
}

inline float32x4_t vrndnq_f32(float32x4_t x) {
  const uint32x4_t sign_mask = vcltq_f32(x, vdupq_n_f32(0.0f));
  const float32x4_t sign_half =
      vbslq_f32(sign_mask, vdupq_n_f32(-0.5f), vdupq_n_f32(0.5f));
  return vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(x, sign_half)));
}

inline float vaddvq_f32(float32x4_t v) {
  float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);
}

inline float vmaxvq_f32(float32x4_t v) {
  float32x2_t mx = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
  mx = vpmax_f32(mx, mx);
  return vget_lane_f32(mx, 0);
}
#endif
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace litert {
namespace lm {
namespace {

constexpr float kMaskedLogit = -1e20f;

// Per-thread persistent state. This function is called once per (head,
// query-chunk) — tens of times per layer — so anything allocated per call
// lands directly in the profile. ruy::Context in particular builds a thread
// pool and resolves CPU tuning on construction.
struct ThreadScratch {
  std::vector<float> logits;  // gt_len * s_limit
  std::vector<float> aux;     // per-row reciprocals / output staging
  ruy::Context ruy_context;
};

ThreadScratch& Scratch() {
  thread_local ThreadScratch s;
  return s;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// exp() on four floats: Cephes range reduction + degree-6 polynomial.
// Softmax always feeds this values <= 0; over that range the error is under
// 1e-7 relative, well inside fp32 noise.
inline float32x4_t ExpF32(float32x4_t x) {
  const float32x4_t kLog2e = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t kLn2hi = vdupq_n_f32(0.693359375f);
  const float32x4_t kLn2lo = vdupq_n_f32(-2.12194440e-4f);

  // exp(-88) already flushes to ~0 in fp32; clamping keeps 2^n representable.
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

  const float32x4_t n = vrndnq_f32(vmulq_f32(x, kLog2e));
  float32x4_t r = vmlsq_f32(x, n, kLn2hi);
  r = vmlsq_f32(r, n, kLn2lo);

  float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
  y = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), y, r);
  y = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), y, r);
  y = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), y, r);
  y = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), y, r);
  y = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), y, r);
  y = vmlaq_f32(r, y, vmulq_f32(r, r));
  y = vaddq_f32(y, vdupq_n_f32(1.0f));

  // Multiply by 2^n by injecting n straight into the exponent field.
  const int32x4_t pow2n =
      vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(pow2n));
}
#endif

#if defined(__AVX2__)
// AVX2 is specified without implying FMA, so degrade rather than fail to
// compile on an AVX2-without-FMA build.
inline __m256 MulAdd256(__m256 a, __m256 b, __m256 c) {  // a * b + c
#if defined(__FMA__)
  return _mm256_fmadd_ps(a, b, c);
#else
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

inline __m256 NegMulAdd256(__m256 a, __m256 b, __m256 c) {  // c - a * b
#if defined(__FMA__)
  return _mm256_fnmadd_ps(a, b, c);
#else
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

inline float HSum256(__m256 v) {
  __m128 lo = _mm_add_ps(
      _mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

inline float HMax256(__m256 v) {
  __m128 lo = _mm_max_ps(
      _mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

// Eight-lane twin of the NEON ExpF32 above: same Cephes range reduction and
// same degree-6 polynomial, so the two paths stay numerically in step.
inline __m256 ExpF256(__m256 x) {
  const __m256 kLog2e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 kLn2hi = _mm256_set1_ps(0.693359375f);
  const __m256 kLn2lo = _mm256_set1_ps(-2.12194440e-4f);

  x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

  const __m256 n = _mm256_round_ps(
      _mm256_mul_ps(x, kLog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = NegMulAdd256(n, kLn2hi, x);
  r = NegMulAdd256(n, kLn2lo, r);

  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = MulAdd256(y, r, _mm256_set1_ps(1.3981999507E-3f));
  y = MulAdd256(y, r, _mm256_set1_ps(8.3334519073E-3f));
  y = MulAdd256(y, r, _mm256_set1_ps(4.1665795894E-2f));
  y = MulAdd256(y, r, _mm256_set1_ps(1.6666665459E-1f));
  y = MulAdd256(y, r, _mm256_set1_ps(5.0000001201E-1f));
  y = MulAdd256(y, _mm256_mul_ps(r, r), r);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

  const __m256i pow2n = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
}
#endif

inline float DotProductSIMD(const float* a, const float* b, int32_t len) {
  int32_t i = 0;
  float sum_float = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  for (; i + 15 < len; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 3 < len; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  sum_float = vaddvq_f32(acc0);
#elif defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 15 < len; i += 16) {
    acc0 = MulAdd256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 =
        MulAdd256(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 7 < len; i += 8) {
    acc0 = MulAdd256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  sum_float = HSum256(_mm256_add_ps(acc0, acc1));
#endif
  for (; i < len; ++i) {
    sum_float += a[i] * b[i];
  }
  return sum_float;
}

// Dots one shared vector `b` against up to 4 separate `a` rows in a single
// pass. The point is that `b` — a slice of the KV cache — is streamed from
// memory exactly once regardless of how many query rows consume it.
inline void DotProductMultiSIMD(const float* const* a, int32_t n,
                                const float* b, int32_t len, float* out,
                                int32_t out_stride) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (n == 2) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t c0 = vdupq_n_f32(0.0f), c1 = vdupq_n_f32(0.0f);
    const float* p0 = a[0];
    const float* p1 = a[1];
    int32_t i = 0;
    for (; i + 7 < len; i += 8) {
      const float32x4_t v0 = vld1q_f32(b + i);
      const float32x4_t v1 = vld1q_f32(b + i + 4);
      a0 = vfmaq_f32(a0, vld1q_f32(p0 + i), v0);
      a1 = vfmaq_f32(a1, vld1q_f32(p0 + i + 4), v1);
      c0 = vfmaq_f32(c0, vld1q_f32(p1 + i), v0);
      c1 = vfmaq_f32(c1, vld1q_f32(p1 + i + 4), v1);
    }
    for (; i + 3 < len; i += 4) {
      const float32x4_t v0 = vld1q_f32(b + i);
      a0 = vfmaq_f32(a0, vld1q_f32(p0 + i), v0);
      c0 = vfmaq_f32(c0, vld1q_f32(p1 + i), v0);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0, a1));
    float s1 = vaddvq_f32(vaddq_f32(c0, c1));
    for (; i < len; ++i) {
      s0 += p0[i] * b[i];
      s1 += p1[i] * b[i];
    }
    out[0] = s0;
    out[out_stride] = s1;
    return;
  }
#elif defined(__AVX2__)
  if (n == 2) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    const float* p0 = a[0];
    const float* p1 = a[1];
    int32_t i = 0;
    for (; i + 15 < len; i += 16) {
      const __m256 v0 = _mm256_loadu_ps(b + i);
      const __m256 v1 = _mm256_loadu_ps(b + i + 8);
      a0 = MulAdd256(_mm256_loadu_ps(p0 + i), v0, a0);
      a1 = MulAdd256(_mm256_loadu_ps(p0 + i + 8), v1, a1);
      c0 = MulAdd256(_mm256_loadu_ps(p1 + i), v0, c0);
      c1 = MulAdd256(_mm256_loadu_ps(p1 + i + 8), v1, c1);
    }
    for (; i + 7 < len; i += 8) {
      const __m256 v0 = _mm256_loadu_ps(b + i);
      a0 = MulAdd256(_mm256_loadu_ps(p0 + i), v0, a0);
      c0 = MulAdd256(_mm256_loadu_ps(p1 + i), v0, c0);
    }
    float s0 = HSum256(_mm256_add_ps(a0, a1));
    float s1 = HSum256(_mm256_add_ps(c0, c1));
    for (; i < len; ++i) {
      s0 += p0[i] * b[i];
      s1 += p1[i] * b[i];
    }
    out[0] = s0;
    out[out_stride] = s1;
    return;
  }
#endif
  for (int32_t g = 0; g < n; ++g) {
    out[g * out_stride] = DotProductSIMD(a[g], b, len);
  }
}

inline void ValueAccumulateSIMD(float* out_ptr, const float* v_row, float p,
                                int32_t len) {
  if (p == 0.0f) return;
  int32_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t p_vec = vdupq_n_f32(p);
  for (; i + 7 < len; i += 8) {
    vst1q_f32(out_ptr + i,
              vfmaq_f32(vld1q_f32(out_ptr + i), p_vec, vld1q_f32(v_row + i)));
    vst1q_f32(out_ptr + i + 4, vfmaq_f32(vld1q_f32(out_ptr + i + 4), p_vec,
                                         vld1q_f32(v_row + i + 4)));
  }
  for (; i + 3 < len; i += 4) {
    vst1q_f32(out_ptr + i,
              vfmaq_f32(vld1q_f32(out_ptr + i), p_vec, vld1q_f32(v_row + i)));
  }
#elif defined(__AVX2__)
  const __m256 p_vec = _mm256_set1_ps(p);
  for (; i + 7 < len; i += 8) {
    _mm256_storeu_ps(out_ptr + i,
                     MulAdd256(p_vec, _mm256_loadu_ps(v_row + i),
                               _mm256_loadu_ps(out_ptr + i)));
  }
#endif
  for (; i < len; ++i) {
    out_ptr[i] += p * v_row[i];
  }
}

// Adds the mask row into `logits` in place and returns the row maximum.
// `logit_cap` is tested once here rather than once per element.
inline float ApplyMaskAndMax(float* logits, const float* mask, int32_t len,
                             std::optional<float> logit_cap) {
  if (logit_cap.has_value()) {
    const float cap = logit_cap.value();
    float mx = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < len; ++i) {
      const float v = std::tanh(logits[i] / cap) * cap + mask[i];
      logits[i] = v;
      if (v > mx) mx = v;
    }
    return mx;
  }

  int32_t i = 0;
  float mx = -std::numeric_limits<float>::infinity();
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t mx0 = vdupq_n_f32(-INFINITY);
  float32x4_t mx1 = vdupq_n_f32(-INFINITY);
  for (; i + 7 < len; i += 8) {
    const float32x4_t v0 = vaddq_f32(
        vld1q_f32(logits + i), vld1q_f32(mask + i));
    const float32x4_t v1 =
        vaddq_f32(vld1q_f32(logits + i + 4), vld1q_f32(mask + i + 4));
    vst1q_f32(logits + i, v0);
    vst1q_f32(logits + i + 4, v1);
    mx0 = vmaxq_f32(mx0, v0);
    mx1 = vmaxq_f32(mx1, v1);
  }
  for (; i + 3 < len; i += 4) {
    const float32x4_t v0 = vaddq_f32(
        vld1q_f32(logits + i), vld1q_f32(mask + i));
    vst1q_f32(logits + i, v0);
    mx0 = vmaxq_f32(mx0, v0);
  }
  mx = vmaxvq_f32(vmaxq_f32(mx0, mx1));
#elif defined(__AVX2__)
  __m256 mx0 = _mm256_set1_ps(-INFINITY);
  __m256 mx1 = _mm256_set1_ps(-INFINITY);
  for (; i + 15 < len; i += 16) {
    const __m256 v0 =
        _mm256_add_ps(_mm256_loadu_ps(logits + i), _mm256_loadu_ps(mask + i));
    const __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(logits + i + 8),
                                    _mm256_loadu_ps(mask + i + 8));
    _mm256_storeu_ps(logits + i, v0);
    _mm256_storeu_ps(logits + i + 8, v1);
    mx0 = _mm256_max_ps(mx0, v0);
    mx1 = _mm256_max_ps(mx1, v1);
  }
  for (; i + 7 < len; i += 8) {
    const __m256 v0 =
        _mm256_add_ps(_mm256_loadu_ps(logits + i), _mm256_loadu_ps(mask + i));
    _mm256_storeu_ps(logits + i, v0);
    mx0 = _mm256_max_ps(mx0, v0);
  }
  mx = HMax256(_mm256_max_ps(mx0, mx1));
#endif
  for (; i < len; ++i) {
    const float v = logits[i] + mask[i];
    logits[i] = v;
    if (v > mx) mx = v;
  }
  return mx;
}

// Replaces `logits` with exp(logits - max_logit) and returns the sum. Hard
// masked entries contribute exactly zero. The caller folds 1/sum into the
// much shorter output row rather than normalizing all `len` probabilities.
inline float ExpAndSum(float* logits, int32_t len, float max_logit) {
  int32_t i = 0;
  float sum = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t mx = vdupq_n_f32(max_logit);
  const float32x4_t masked = vdupq_n_f32(kMaskedLogit);
  const float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  for (; i + 7 < len; i += 8) {
    const float32x4_t l0 = vld1q_f32(logits + i);
    const float32x4_t l1 = vld1q_f32(logits + i + 4);
    float32x4_t e0 = ExpF32(vsubq_f32(l0, mx));
    float32x4_t e1 = ExpF32(vsubq_f32(l1, mx));
    e0 = vbslq_f32(vcgtq_f32(l0, masked), e0, zero);
    e1 = vbslq_f32(vcgtq_f32(l1, masked), e1, zero);
    vst1q_f32(logits + i, e0);
    vst1q_f32(logits + i + 4, e1);
    acc0 = vaddq_f32(acc0, e0);
    acc1 = vaddq_f32(acc1, e1);
  }
  for (; i + 3 < len; i += 4) {
    const float32x4_t l0 = vld1q_f32(logits + i);
    float32x4_t e0 = ExpF32(vsubq_f32(l0, mx));
    e0 = vbslq_f32(vcgtq_f32(l0, masked), e0, zero);
    vst1q_f32(logits + i, e0);
    acc0 = vaddq_f32(acc0, e0);
  }
  sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(__AVX2__)
  const __m256 mx = _mm256_set1_ps(max_logit);
  const __m256 masked = _mm256_set1_ps(kMaskedLogit);
  const __m256 zero = _mm256_setzero_ps();
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 15 < len; i += 16) {
    const __m256 l0 = _mm256_loadu_ps(logits + i);
    const __m256 l1 = _mm256_loadu_ps(logits + i + 8);
    __m256 e0 = ExpF256(_mm256_sub_ps(l0, mx));
    __m256 e1 = ExpF256(_mm256_sub_ps(l1, mx));
    e0 = _mm256_blendv_ps(zero, e0, _mm256_cmp_ps(l0, masked, _CMP_GT_OQ));
    e1 = _mm256_blendv_ps(zero, e1, _mm256_cmp_ps(l1, masked, _CMP_GT_OQ));
    _mm256_storeu_ps(logits + i, e0);
    _mm256_storeu_ps(logits + i + 8, e1);
    acc0 = _mm256_add_ps(acc0, e0);
    acc1 = _mm256_add_ps(acc1, e1);
  }
  for (; i + 7 < len; i += 8) {
    const __m256 l0 = _mm256_loadu_ps(logits + i);
    __m256 e0 = ExpF256(_mm256_sub_ps(l0, mx));
    e0 = _mm256_blendv_ps(zero, e0, _mm256_cmp_ps(l0, masked, _CMP_GT_OQ));
    _mm256_storeu_ps(logits + i, e0);
    acc0 = _mm256_add_ps(acc0, e0);
  }
  sum = HSum256(_mm256_add_ps(acc0, acc1));
#endif
  for (; i < len; ++i) {
    const float l = logits[i];
    const float e = (l <= kMaskedLogit) ? 0.0f : std::exp(l - max_logit);
    logits[i] = e;
    sum += e;
  }
  return sum;
}

inline void ScaleInPlace(float* p, int32_t len, float scale) {
  int32_t i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t s = vdupq_n_f32(scale);
  for (; i + 3 < len; i += 4) {
    vst1q_f32(p + i, vmulq_f32(vld1q_f32(p + i), s));
  }
#elif defined(__AVX2__)
  const __m256 s = _mm256_set1_ps(scale);
  for (; i + 7 < len; i += 8) {
    _mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), s));
  }
#endif
  for (; i < len; ++i) p[i] *= scale;
}

// Decode-shaped attention: a handful of query rows against the whole cache.
// Streams K once and V once for the entire query group rather than once per
// row, which is what matters here because the phase is bandwidth-bound on the
// KV cache, not compute-bound.
bool ComputeSmallGt(int h_idx, const TensorRef& query, const TensorRef& key,
                    const TensorRef& value, const TensorRef& mask,
                    std::optional<float> logit_cap, int k_ts_idx, int v_ts_idx,
                    const MutableTensorRef& output, int32_t s, int32_t s_limit,
                    int32_t gt, int32_t h, int32_t t) {
  constexpr int32_t kMaxGt = 4;
  if (gt > kMaxGt) return false;

  ThreadScratch& sc = Scratch();
  sc.logits.resize(static_cast<size_t>(gt) * s_limit);
  float* logits = sc.logits.data();

  const float* q_base = query.data + static_cast<size_t>(h_idx) * gt * h;
  const float* q_rows[kMaxGt];
  for (int32_t g = 0; g < gt; ++g) q_rows[g] = q_base + g * h;

  // Q @ K^T for every query row, one pass over the K cache.
  if (k_ts_idx == 2) {
    const float* k_base = key.data + static_cast<size_t>(h_idx) * s * h;
    for (int32_t s_idx = 0; s_idx < s_limit; ++s_idx) {
      DotProductMultiSIMD(q_rows, gt, k_base + static_cast<size_t>(s_idx) * h,
                          h, logits + s_idx, s_limit);
    }
  } else {
    // K stored as [h, s]: each head-dim row is contiguous over s.
    const float* k_base = key.data + static_cast<size_t>(h_idx) * h * s;
    std::fill_n(logits, static_cast<size_t>(gt) * s_limit, 0.0f);
    for (int32_t d = 0; d < h; ++d) {
      const float* k_row = k_base + static_cast<size_t>(d) * s;
      for (int32_t g = 0; g < gt; ++g) {
        ValueAccumulateSIMD(logits + g * s_limit, k_row, q_rows[g][d], s_limit);
      }
    }
  }

  float recip[kMaxGt];
  for (int32_t g = 0; g < gt; ++g) {
    float* row = logits + static_cast<size_t>(g) * s_limit;
    const float* mrow = mask.data + static_cast<size_t>(g % t) * s;
    const float mx = ApplyMaskAndMax(row, mrow, s_limit, logit_cap);
    const float sum = ExpAndSum(row, s_limit, mx);
    recip[g] = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
  }

  // probs @ V, again streaming V once for the whole group.
  float* out_base = output.data + static_cast<size_t>(h_idx) * gt * h;
  if (v_ts_idx == 3) {
    const float* v_base = value.data + static_cast<size_t>(h_idx) * h * s;
    const float* p_rows[kMaxGt];
    for (int32_t g = 0; g < gt; ++g) p_rows[g] = logits + g * s_limit;
    for (int32_t d = 0; d < h; ++d) {
      DotProductMultiSIMD(p_rows, gt, v_base + static_cast<size_t>(d) * s,
                          s_limit, out_base + d, h);
    }
  } else {
    const float* v_base = value.data + static_cast<size_t>(h_idx) * s * h;
    std::fill_n(out_base, static_cast<size_t>(gt) * h, 0.0f);
    for (int32_t s_idx = 0; s_idx < s_limit; ++s_idx) {
      const float* v_row = v_base + static_cast<size_t>(s_idx) * h;
      for (int32_t g = 0; g < gt; ++g) {
        ValueAccumulateSIMD(out_base + g * h, v_row,
                            logits[g * s_limit + s_idx], h);
      }
    }
  }

  for (int32_t g = 0; g < gt; ++g) {
    ScaleInPlace(out_base + static_cast<size_t>(g) * h, h, recip[g]);
  }
  return true;
}

}  // namespace

bool ComputeTransposedAttention(const TensorRef& query, const TensorRef& key,
                                const TensorRef& value, const TensorRef& mask,
                                std::optional<float> logit_cap, int k_ts_idx,
                                int v_ts_idx, const MutableTensorRef& output,
                                std::optional<int32_t> active_seq_len) {
  if (query.shape.size() != 4 || key.shape.size() != 4 ||
      value.shape.size() != 4 || mask.shape.size() != 4 ||
      output.shape.size() != 4) {
    return false;
  }

  const int32_t bk = query.shape[1];
  const int32_t gt = query.shape[2];
  const int32_t h = query.shape[3];

  const int32_t c = key.shape[1];
  if (bk != c || value.shape[1] != c || output.shape[1] != bk) {
    return false;
  }

  int32_t s = 0;
  if (k_ts_idx == 2) {
    s = key.shape[2];
    if (key.shape[3] != h) return false;
  } else if (k_ts_idx == 3) {
    s = key.shape[3];
    if (key.shape[2] != h) return false;
  } else {
    return false;
  }

  if (v_ts_idx == 3) {
    if (value.shape[2] != h || value.shape[3] != s) return false;
  } else if (v_ts_idx == 2) {
    if (value.shape[2] != s || value.shape[3] != h) return false;
  } else {
    return false;
  }

  const int32_t t = mask.shape[2];
  if (mask.shape[3] != s) return false;

  if (output.shape[2] != gt || output.shape[3] != h) {
    return false;
  }

  const int32_t g = gt / t;
  if (g * t != gt) {
    return false;
  }

  for (int32_t h_idx = 0; h_idx < bk; ++h_idx) {
    if (!ComputeTransposedAttentionSingleHead(h_idx, query, key, value, mask,
                                              logit_cap, k_ts_idx, v_ts_idx,
                                              output, active_seq_len)) {
      return false;
    }
  }

  return true;
}

bool ComputeTransposedAttentionSingleHead(
    int h_idx, const TensorRef& query, const TensorRef& key,
    const TensorRef& value, const TensorRef& mask,
    std::optional<float> logit_cap, int k_ts_idx, int v_ts_idx,
    const MutableTensorRef& output, std::optional<int32_t> active_seq_len,
    std::optional<std::pair<int32_t, int32_t>> query_range,
    int32_t max_ruy_threads, float* logits_scratch_ptr, float* out_scratch_ptr,
    ruy::Context* ruy_context) {
  const int32_t s = (k_ts_idx == 2) ? key.shape[2] : key.shape[3];
  const int32_t gt = query.shape[2];
  const int32_t h = query.shape[3];
  const int32_t t = mask.shape[2];
  const int32_t s_limit =
      active_seq_len.has_value() ? active_seq_len.value() : s;

  if (s_limit < 0 || s_limit > s) {
    return false;
  }
  if (s_limit == 0) {
    return true;
  }

  if (gt <= 4) {
    return ComputeSmallGt(h_idx, query, key, value, mask, logit_cap, k_ts_idx,
                          v_ts_idx, output, s, s_limit, gt, h, t);
  }

  const int32_t gt_start =
      query_range.has_value() ? query_range.value().first : 0;
  const int32_t gt_end =
      query_range.has_value() ? query_range.value().second : gt;
  const int32_t gt_len = gt_end - gt_start;
  if (gt_len <= 0) return true;

  const float* q_ptr =
      query.data + static_cast<size_t>(h_idx) * gt * h + gt_start * h;

  ThreadScratch& sc = Scratch();
  ruy::Context* gemm_context = ruy_context ? ruy_context : &sc.ruy_context;
  gemm_context->set_max_num_threads(max_ruy_threads);

  ruy::Matrix<float> lhs_q;
  ruy::MakeSimpleLayout(gt_len, h, ruy::Order::kRowMajor,
                        lhs_q.mutable_layout());
  lhs_q.set_data(q_ptr);

  ruy::Matrix<float> rhs_k;
  if (k_ts_idx == 2) {
    ruy::MakeSimpleLayout(h, s_limit, ruy::Order::kColMajor,
                          rhs_k.mutable_layout());
    rhs_k.set_data(key.data + static_cast<size_t>(h_idx) * s * h);
  } else {
    rhs_k.mutable_layout()->set_rows(h);
    rhs_k.mutable_layout()->set_cols(s_limit);
    rhs_k.mutable_layout()->set_order(ruy::Order::kRowMajor);
    rhs_k.mutable_layout()->set_stride(s);
    rhs_k.set_data(key.data + static_cast<size_t>(h_idx) * h * s);
  }

  float* logits_ptr = logits_scratch_ptr;
  if (logits_ptr == nullptr) {
    sc.logits.resize(static_cast<size_t>(gt_len) * s_limit);
    logits_ptr = sc.logits.data();
  }
  ruy::Matrix<float> dst_logits;
  ruy::MakeSimpleLayout(gt_len, s_limit, ruy::Order::kRowMajor,
                        dst_logits.mutable_layout());
  dst_logits.set_data(logits_ptr);

  ruy::MulParams<float, float> spec_q;
  ruy::Mul(lhs_q, rhs_k, spec_q, gemm_context, &dst_logits);

  sc.aux.resize(gt_len);
  float* recip = sc.aux.data();
  for (int32_t i = 0; i < gt_len; ++i) {
    float* row = logits_ptr + static_cast<size_t>(i) * s_limit;
    const float* mrow = mask.data + static_cast<size_t>((gt_start + i) % t) * s;
    const float mx = ApplyMaskAndMax(row, mrow, s_limit, logit_cap);
    const float sum = ExpAndSum(row, s_limit, mx);
    recip[i] = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
  }

  ruy::Matrix<float> rhs_v;
  if (v_ts_idx == 2) {
    ruy::MakeSimpleLayout(s_limit, h, ruy::Order::kRowMajor,
                          rhs_v.mutable_layout());
    rhs_v.set_data(value.data + static_cast<size_t>(h_idx) * s * h);
  } else {
    rhs_v.mutable_layout()->set_rows(s_limit);
    rhs_v.mutable_layout()->set_cols(h);
    rhs_v.mutable_layout()->set_order(ruy::Order::kColMajor);
    rhs_v.mutable_layout()->set_stride(s);
    rhs_v.set_data(value.data + static_cast<size_t>(h_idx) * h * s);
  }

  // Write the second GEMM straight into the output tensor — those rows are
  // contiguous, so the staging buffer and copy the original did are avoidable.
  float* out_ptr =
      out_scratch_ptr
          ? out_scratch_ptr
          : output.data + static_cast<size_t>(h_idx) * gt * h + gt_start * h;
  ruy::Matrix<float> dst_out;
  ruy::MakeSimpleLayout(gt_len, h, ruy::Order::kRowMajor,
                        dst_out.mutable_layout());
  dst_out.set_data(out_ptr);

  ruy::MulParams<float, float> spec_v;
  ruy::Mul(dst_logits, rhs_v, spec_v, gemm_context, &dst_out);

  for (int32_t i = 0; i < gt_len; ++i) {
    ScaleInPlace(out_ptr + static_cast<size_t>(i) * h, h, recip[i]);
  }

  if (out_scratch_ptr != nullptr) {
    std::copy(out_ptr, out_ptr + static_cast<size_t>(gt_len) * h,
              output.data + static_cast<size_t>(h_idx) * gt * h + gt_start * h);
  }

  return true;
}

}  // namespace lm
}  // namespace litert
