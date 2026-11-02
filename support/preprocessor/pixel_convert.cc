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

#include "support/preprocessor/pixel_convert.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"  // from @com_google_absl

// Highway dynamic dispatch: this translation unit is compiled once per SIMD
// target (see HWY_TARGETS); HWY_EXPORT / HWY_DYNAMIC_DISPATCH below pick the
// best one at runtime based on CPUID / HWCAP.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "support/preprocessor/pixel_convert.cc"
#include "hwy/foreach_target.h"  // from @highway  // IWYU pragma: keep
#include "hwy/highway.h"  // from @highway

HWY_BEFORE_NAMESPACE();
namespace litert::support {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Promotes one quarter of a u8 vector to f32 and applies the affine transform.
// `part` selects which `Lanes(df)` u8 samples of `v8` are converted.
template <class DU8, class DF>
HWY_INLINE hn::Vec<DF> AffineQuarter(DU8 du8, DF df, hn::Vec<DU8> v8,
                                     hn::Vec<DF> scale, hn::Vec<DF> offset,
                                     int part) {
  const hn::Repartition<uint16_t, DU8> du16;
  const hn::Repartition<uint32_t, DU8> du32;
  const auto v16 =
      (part < 2) ? hn::PromoteLowerTo(du16, v8) : hn::PromoteUpperTo(du16, v8);
  const auto v32 = (part % 2 == 0) ? hn::PromoteLowerTo(du32, v16)
                                   : hn::PromoteUpperTo(du32, v16);
  return hn::MulAdd(hn::ConvertTo(df, v32), scale, offset);
}

void ConvertRgbUint8ToFloat(const uint8_t* HWY_RESTRICT src,
                            float* HWY_RESTRICT dst, size_t num_pixels,
                            const float* HWY_RESTRICT scale,
                            const float* HWY_RESTRICT offset) {
  const hn::ScalableTag<uint8_t> du8;
  const hn::Repartition<float, decltype(du8)> df;
  const size_t kPixelsPerIter = hn::Lanes(du8);  // 16 on 128-bit targets.
  const size_t kFloatLanes = hn::Lanes(df);      // kPixelsPerIter / 4.

  const auto s_r = hn::Set(df, scale[0]);
  const auto s_g = hn::Set(df, scale[1]);
  const auto s_b = hn::Set(df, scale[2]);
  const auto o_r = hn::Set(df, offset[0]);
  const auto o_g = hn::Set(df, offset[1]);
  const auto o_b = hn::Set(df, offset[2]);

  size_t i = 0;
  for (; i + kPixelsPerIter <= num_pixels; i += kPixelsPerIter) {
    hn::Vec<decltype(du8)> r8, g8, b8;
    hn::LoadInterleaved3(du8, src + i * 3, r8, g8, b8);
    for (int part = 0; part < 4; ++part) {
      const auto rf = AffineQuarter(du8, df, r8, s_r, o_r, part);
      const auto gf = AffineQuarter(du8, df, g8, s_g, o_g, part);
      const auto bf = AffineQuarter(du8, df, b8, s_b, o_b, part);
      hn::StoreInterleaved3(rf, gf, bf, df, dst + (i + part * kFloatLanes) * 3);
    }
  }
  for (; i < num_pixels; ++i) {
    dst[i * 3 + 0] = static_cast<float>(src[i * 3 + 0]) * scale[0] + offset[0];
    dst[i * 3 + 1] = static_cast<float>(src[i * 3 + 1]) * scale[1] + offset[1];
    dst[i * 3 + 2] = static_cast<float>(src[i * 3 + 2]) * scale[2] + offset[2];
  }
}

void ConvertGrayUint8ToFloat(const uint8_t* HWY_RESTRICT src,
                             float* HWY_RESTRICT dst, size_t num_pixels,
                             float scale, float offset) {
  const hn::ScalableTag<uint8_t> du8;
  const hn::Repartition<float, decltype(du8)> df;
  const size_t kPixelsPerIter = hn::Lanes(du8);
  const size_t kFloatLanes = hn::Lanes(df);
  const auto s = hn::Set(df, scale);
  const auto o = hn::Set(df, offset);

  size_t i = 0;
  for (; i + kPixelsPerIter <= num_pixels; i += kPixelsPerIter) {
    const auto v8 = hn::LoadU(du8, src + i);
    for (int part = 0; part < 4; ++part) {
      hn::StoreU(AffineQuarter(du8, df, v8, s, o, part), df,
                 dst + i + part * kFloatLanes);
    }
  }
  for (; i < num_pixels; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale + offset;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace litert::support
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace litert::support {

HWY_EXPORT(ConvertRgbUint8ToFloat);
HWY_EXPORT(ConvertGrayUint8ToFloat);

void ConvertUint8ToFloat(const uint8_t* src, float* dst, size_t num_pixels,
                         int channels, absl::Span<const float> scale,
                         absl::Span<const float> offset) {
  if (channels == 3) {
    HWY_DYNAMIC_DISPATCH(ConvertRgbUint8ToFloat)
    (src, dst, num_pixels, scale.data(), offset.data());
    return;
  }
  if (channels == 1) {
    HWY_DYNAMIC_DISPATCH(ConvertGrayUint8ToFloat)
    (src, dst, num_pixels, scale[0], offset[0]);
    return;
  }
  // Generic fallback for any other channel count.
  for (size_t i = 0; i < num_pixels; ++i) {
    for (int c = 0; c < channels; ++c) {
      const size_t idx = i * channels + c;
      dst[idx] = static_cast<float>(src[idx]) * scale[c] + offset[c];
    }
  }
}

}  // namespace litert::support
#endif  // HWY_ONCE
