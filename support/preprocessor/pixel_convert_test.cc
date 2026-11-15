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
#include <vector>

#include <gtest/gtest.h>

namespace litert::support {
namespace {

std::vector<uint8_t> MakePixels(size_t count) {
  std::vector<uint8_t> src(count);
  for (size_t i = 0; i < count; ++i) {
    src[i] = static_cast<uint8_t>((i * 17 + 5) % 256);
  }
  return src;
}

void ExpectAffine(const std::vector<uint8_t>& src,
                  const std::vector<float>& dst, int channels,
                  const std::vector<float>& scale,
                  const std::vector<float>& offset) {
  ASSERT_EQ(dst.size(), src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    const int c = i % channels;
    const float expected = static_cast<float>(src[i]) * scale[c] + offset[c];
    EXPECT_NEAR(dst[i], expected, 1e-5f) << "index " << i;
  }
}

TEST(PixelConvertTest, RgbMatchesScalarReferenceAroundVectorBoundaries) {
  const std::vector<float> scale = {1.0f / 255.0f, 1.0f / 128.0f,
                                    2.0f / 255.0f};
  const std::vector<float> offset = {-0.5f, 0.1f, -1.0f};
  // Sizes around multiples of 16 / 32 / 64 pixels cover the full-vector loop
  // and the scalar tail on 128-, 256- and 512-bit targets.
  for (size_t num_pixels :
       {0, 1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 256, 528, 896}) {
    const std::vector<uint8_t> src = MakePixels(num_pixels * 3);
    std::vector<float> dst(num_pixels * 3, 0.0f);
    ConvertUint8ToFloat(src.data(), dst.data(), num_pixels, /*channels=*/3,
                        scale, offset);
    ExpectAffine(src, dst, 3, scale, offset);
  }
}

TEST(PixelConvertTest, SingleChannelMatchesScalarReference) {
  const std::vector<float> scale = {1.0f / 255.0f};
  const std::vector<float> offset = {-0.5f};
  for (size_t num_pixels : {0, 1, 7, 15, 16, 17, 33, 64, 65, 100}) {
    const std::vector<uint8_t> src = MakePixels(num_pixels);
    std::vector<float> dst(num_pixels, 0.0f);
    ConvertUint8ToFloat(src.data(), dst.data(), num_pixels, /*channels=*/1,
                        scale, offset);
    ExpectAffine(src, dst, 1, scale, offset);
  }
}

TEST(PixelConvertTest, FourChannelsUsesGenericPath) {
  const std::vector<float> scale = {1.0f, 0.5f, 0.25f, 2.0f};
  const std::vector<float> offset = {0.0f, 1.0f, -1.0f, 0.5f};
  constexpr size_t kNumPixels = 37;
  const std::vector<uint8_t> src = MakePixels(kNumPixels * 4);
  std::vector<float> dst(kNumPixels * 4, 0.0f);
  ConvertUint8ToFloat(src.data(), dst.data(), kNumPixels, /*channels=*/4, scale,
                      offset);
  ExpectAffine(src, dst, 4, scale, offset);
}

}  // namespace
}  // namespace litert::support
