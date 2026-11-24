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

#include "omni/text2image/text2image_math.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK_AND_ASSIGN

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::HasSubstr;

uint32_t FloatToBits(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

TEST(Text2ImageMathTest, SampleGaussianLatentsIsDeterministicAndSeedDependent) {
  EXPECT_TRUE(SampleGaussianLatents(/*count=*/0, /*seed=*/42).empty());

  const std::vector<float> a = SampleGaussianLatents(/*count=*/16, /*seed=*/42);
  const std::vector<float> b = SampleGaussianLatents(/*count=*/16, /*seed=*/42);
  const std::vector<float> c = SampleGaussianLatents(/*count=*/16, /*seed=*/99);
  EXPECT_EQ(a.size(), 16);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(Text2ImageMathTest, ConvertNchwToRgb888ClampsAndHandlesNonFinite) {
  // 2 pixels: [-1, 0, 1] -> (0, 128, 255) and [2, 0, -2] -> (255, 128, 0).
  const std::vector<float> nchw = {
      -1.0f, 2.0f,   // R
      0.0f,  0.0f,   // G
      1.0f,  -2.0f,  // B
  };
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> rgb,
                       ConvertNchwToRgb888(nchw, /*height=*/1, /*width=*/2));
  EXPECT_THAT(rgb, ElementsAre(0, 128, 255, 255, 128, 0));

  const std::vector<float> nonfinite = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
  };
  ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> nonfinite_rgb,
      ConvertNchwToRgb888(nonfinite, /*height=*/1, /*width=*/1));
  EXPECT_THAT(nonfinite_rgb, ElementsAre(0, 255, 0));

  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/0, /*width=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/-1, /*width=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/1, /*width=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/1, /*width=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/2, /*width=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  constexpr int kMaxInt = std::numeric_limits<int>::max();
  EXPECT_THAT(
      ConvertNchwToRgb888(/*nchw=*/{}, /*height=*/kMaxInt,
                          /*width=*/kMaxInt),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
}

TEST(Text2ImageMathTest, BuildMultiAxisRotaryPosEmbedValuesAndValidation) {
  // 2 tokens, 3 axes with dimensions [2, 2, 4] -> total dim 8 per token.
  const std::vector<float> joint_ids = {
      0.0f, 0.0f, 0.0f,  // token 0: all angles 0 -> cos=1, sin=0
      1.0f, 2.0f, 3.0f,  // token 1
  };
  const std::vector<int> axes_dim = {2, 2, 4};
  constexpr float kTheta = 256.0f;

  ASSERT_OK_AND_ASSIGN(auto rope,
                       BuildMultiAxisRotaryPosEmbed(joint_ids, /*num_tokens=*/2,
                                                    axes_dim, kTheta));
  const std::vector<float>& cos_vec = rope.first;
  const std::vector<float>& sin_vec = rope.second;
  ASSERT_EQ(cos_vec.size(), 16);
  ASSERT_EQ(sin_vec.size(), 16);

  // Token 0: cos = 1, sin = 0 everywhere.
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(cos_vec[i], 1.0f);
    EXPECT_FLOAT_EQ(sin_vec[i], 0.0f);
  }

  // Token 1: repeat_interleave by 2 along each axis.
  EXPECT_THAT(cos_vec[8], FloatNear(std::cos(1.0f), 1e-5f));
  EXPECT_FLOAT_EQ(cos_vec[8], cos_vec[9]);
  EXPECT_THAT(sin_vec[8], FloatNear(std::sin(1.0f), 1e-5f));
  EXPECT_FLOAT_EQ(sin_vec[8], sin_vec[9]);

  EXPECT_THAT(cos_vec[10], FloatNear(std::cos(2.0f), 1e-5f));
  EXPECT_FLOAT_EQ(cos_vec[10], cos_vec[11]);

  const double expected_axis2_j1 = 3.0 / std::pow(256.0, 2.0 / 4.0);
  EXPECT_THAT(
      cos_vec[14],
      FloatNear(static_cast<float>(std::cos(expected_axis2_j1)), 1e-5f));
  EXPECT_FLOAT_EQ(cos_vec[14], cos_vec[15]);
  EXPECT_THAT(
      sin_vec[14],
      FloatNear(static_cast<float>(std::sin(expected_axis2_j1)), 1e-5f));
  EXPECT_FLOAT_EQ(sin_vec[14], sin_vec[15]);

  // Validation errors.
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 0, axes_dim, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, -1, axes_dim, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, {}, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, axes_dim, 0.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, axes_dim, -10.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BuildMultiAxisRotaryPosEmbed(joint_ids, 2, axes_dim,
                                   std::numeric_limits<float>::quiet_NaN()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BuildMultiAxisRotaryPosEmbed(joint_ids, 2, axes_dim,
                                   std::numeric_limits<float>::infinity()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BuildMultiAxisRotaryPosEmbed(joint_ids, 2, axes_dim,
                                   -std::numeric_limits<float>::infinity()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed({0.0f}, 2, axes_dim, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, {2, 3, 4}, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, {2, 0, 4}, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildMultiAxisRotaryPosEmbed(joint_ids, 2, {2, -2, 4}, kTheta),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Overflow guards on `sum(axes_dim)` and `num_tokens * sum(axes_dim)`.
  constexpr int kMaxInt = std::numeric_limits<int>::max();
  constexpr int kMaxEvenDim = kMaxInt - (kMaxInt % 2);
  EXPECT_THAT(
      BuildMultiAxisRotaryPosEmbed({0.0f, 0.0f}, /*num_tokens=*/1,
                                   {kMaxEvenDim, 2}, kTheta),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
  EXPECT_THAT(
      BuildMultiAxisRotaryPosEmbed(/*joint_ids=*/{},
                                   /*num_tokens=*/kMaxInt, {kMaxEvenDim},
                                   kTheta),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
}

TEST(Text2ImageMathTest,
     BuildMultiAxisRotaryPosEmbedMatchesUnHoistedFormulaBitForBit) {
  auto verify_case = [](absl::Span<const float> joint_ids, int num_tokens,
                        absl::Span<const int> axes_dim, float theta) {
    ASSERT_OK_AND_ASSIGN(
        auto actual,
        BuildMultiAxisRotaryPosEmbed(joint_ids, num_tokens, axes_dim, theta));
    const size_t num_axes = axes_dim.size();
    int total_dim = 0;
    for (int dim : axes_dim) {
      total_dim += dim;
    }
    const size_t total_elements = static_cast<size_t>(num_tokens) * total_dim;
    ASSERT_EQ(actual.first.size(), total_elements);
    ASSERT_EQ(actual.second.size(), total_elements);

    for (int n = 0; n < num_tokens; ++n) {
      const size_t row_base = static_cast<size_t>(n) * total_dim;
      size_t col_offset = 0;
      for (size_t a = 0; a < num_axes; ++a) {
        const int dim = axes_dim[a];
        const int half_dim = dim / 2;
        const double pos = static_cast<double>(
            joint_ids[static_cast<size_t>(n) * num_axes + a]);
        for (int j = 0; j < half_dim; ++j) {
          const double inv_freq =
              1.0 / std::pow(static_cast<double>(theta),
                             static_cast<double>(2 * j) / dim);
          const double angle = pos * inv_freq;
          const float expected_c = static_cast<float>(std::cos(angle));
          const float expected_s = static_cast<float>(std::sin(angle));
          const size_t idx0 =
              row_base + col_offset + static_cast<size_t>(2) * j;
          EXPECT_EQ(FloatToBits(actual.first[idx0]), FloatToBits(expected_c));
          EXPECT_EQ(FloatToBits(actual.first[idx0 + 1]),
                    FloatToBits(expected_c));
          EXPECT_EQ(FloatToBits(actual.second[idx0]), FloatToBits(expected_s));
          EXPECT_EQ(FloatToBits(actual.second[idx0 + 1]),
                    FloatToBits(expected_s));
        }
        col_offset += static_cast<size_t>(dim);
      }
    }
  };

  // FLUX.2-style 4-axis embedding with integer, negative, and fractional IDs.
  const std::vector<float> flux_ids = {
      0.0f,  0.0f,  0.0f,  0.0f,    // token 0
      0.0f,  0.0f,  0.0f,  17.0f,   // token 1
      1.0f,  15.0f, 31.0f, 0.0f,    // token 2
      -2.5f, 3.25f, 64.0f, 255.0f,  // token 3
  };
  verify_case(flux_ids, /*num_tokens=*/4, {32, 32, 32, 32}, /*theta=*/2000.0f);

  // Non-uniform axes dimensions with a non-power-of-two theta.
  const std::vector<float> uneven_ids = {
      0.0f,  -1.0f,  2.5f,  7.0f,   // token 0
      12.0f, 0.125f, -9.0f, 42.0f,  // token 1
      -0.0f, 100.0f, 0.0f,  3.75f,  // token 2
  };
  verify_case(uneven_ids, /*num_tokens=*/3, {2, 4, 6, 8}, /*theta=*/1234.5f);
}

}  // namespace
}  // namespace litert::omni::text2image
