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

#include "runtime/components/preprocessor/pil_resize.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

// All golden buffers below were produced by Pillow itself:
//   img.resize((w, h), PIL.Image.BICUBIC).tobytes()
// The source is an 8x6 RGB image with a different linear gradient per channel,
// chosen so that every output pixel depends on several inputs and so that the
// red channel wraps around 255 (exercising clamping).
constexpr int kSrcWidth = 8;
constexpr int kSrcHeight = 6;

constexpr uint8_t kSource[] = {
    0,   0,   0,   31,  13,  97,  62,  26,  194, 93,  39,  35,  //
    124, 52,  132, 155, 65,  229, 186, 78,  70,  217, 91,  167,  //
    7,   53,  3,   38,  66,  100, 69,  79,  197, 100, 92,  38,  //
    131, 105, 135, 162, 118, 232, 193, 131, 73,  224, 144, 170,  //
    14,  106, 6,   45,  119, 103, 76,  132, 200, 107, 145, 41,  //
    138, 158, 138, 169, 171, 235, 200, 184, 76,  231, 197, 173,  //
    21,  159, 9,   52,  172, 106, 83,  185, 203, 114, 198, 44,  //
    145, 211, 141, 176, 224, 238, 207, 237, 79,  238, 250, 176,  //
    28,  212, 12,  59,  225, 109, 90,  238, 206, 121, 251, 47,  //
    152, 8,   144, 183, 21,  241, 214, 34,  82,  245, 47,  179,  //
    35,  9,   15,  66,  22,  112, 97,  35,  209, 128, 48,  50,  //
    159, 61,  147, 190, 74,  244, 221, 87,  85,  252, 100, 182,
};

// Pillow: source.resize((3, 2), BICUBIC).
constexpr uint8_t kDownscaled3x2[] = {
    36,  79,  86,  116, 112, 131, 197, 146, 148,
    55,  151, 95,  135, 134, 140, 216, 118, 157,
};

// Pillow: source.resize((10, 9), BICUBIC).
constexpr uint8_t kUpscaled10x9[] = {
    0,   0,   0,   20,  5,   63,  47,  17,  162, 71,  27,  157,  //
    96,  37,  34,  121, 48,  121, 146, 58,  219, 171, 69,  150,  //
    197, 80,  86,  218, 89,  171, 2,   21,  0,   23,  30,  64,  //
    50,  42,  163, 74,  52,  158, 99,  62,  35,  124, 73,  122,  //
    149, 83,  220, 174, 94,  151, 200, 105, 87,  221, 114, 172,  //
    7,   61,  0,   28,  70,  67,  55,  82,  166, 79,  92,  161,  //
    104, 102, 38,  129, 113, 125, 154, 123, 223, 179, 134, 154,  //
    205, 145, 90,  226, 154, 175, 12,  96,  2,   33,  105, 68,  //
    60,  117, 167, 84,  127, 162, 109, 137, 39,  134, 148, 126,  //
    159, 158, 224, 184, 169, 155, 210, 180, 91,  231, 189, 176,  //
    17,  132, 3,   38,  141, 71,  65,  153, 170, 89,  162, 165,  //
    114, 174, 42,  139, 199, 129, 164, 210, 227, 189, 221, 158,  //
    215, 232, 94,  236, 241, 179, 21,  170, 6,   42,  179, 73,  //
    69,  191, 172, 93,  202, 167, 118, 209, 44,  143, 190, 131,  //
    168, 197, 229, 193, 209, 160, 219, 220, 96,  240, 229, 181,  //
    26,  217, 7,   47,  226, 74,  74,  238, 173, 98,  255, 168,  //
    123, 243, 45,  148, 44,  132, 173, 31,  230, 198, 50,  161,  //
    224, 61,  97,  245, 70,  182, 31,  107, 10,  52,  116, 77,  //
    79,  128, 176, 103, 142, 171, 128, 139, 48,  153, 32,  135,  //
    178, 29,  233, 203, 44,  164, 229, 55,  100, 250, 64,  185,  //
    34,  0,   11,  55,  4,   78,  82,  16,  177, 106, 25,  172,  //
    131, 37,  49,  156, 62,  136, 181, 74,  234, 206, 84,  165,  //
    232, 95,  101, 253, 104, 186,
};

// Compares `actual` against a golden buffer byte by byte. gmock's
// ElementsAreArray is avoided so this test needs no gmock dependency.
void ExpectBytesEq(const std::vector<uint8_t>& actual, const uint8_t* expected,
                   size_t expected_size) {
  ASSERT_EQ(actual.size(), expected_size);
  for (size_t i = 0; i < expected_size; ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "mismatch at byte " << i;
  }
}

TEST(PilResizeTest, DownscaleMatchesPillow) {
  const std::vector<uint8_t> resized = PilResizeBicubicRgb(
      kSource, kSrcWidth, kSrcHeight, /*dst_w=*/3, /*dst_h=*/2);
  ExpectBytesEq(resized, kDownscaled3x2, std::size(kDownscaled3x2));
}

TEST(PilResizeTest, UpscaleMatchesPillow) {
  const std::vector<uint8_t> resized = PilResizeBicubicRgb(
      kSource, kSrcWidth, kSrcHeight, /*dst_w=*/10, /*dst_h=*/9);
  ExpectBytesEq(resized, kUpscaled10x9, std::size(kUpscaled10x9));
}

TEST(PilResizeTest, SameSizeIsIdentity) {
  const std::vector<uint8_t> resized =
      PilResizeBicubicRgb(kSource, kSrcWidth, kSrcHeight, kSrcWidth,
                          kSrcHeight);
  ExpectBytesEq(resized, kSource, std::size(kSource));
}

TEST(PilResizeTest, FlatImageStaysFlat) {
  constexpr int kWidth = 9;
  constexpr int kHeight = 5;
  const std::vector<uint8_t> flat(kWidth * kHeight * 3, 123);
  // A cubic kernel overshoots at edges; a constant field must not ring.
  const std::vector<uint8_t> resized =
      PilResizeBicubicRgb(flat.data(), kWidth, kHeight, /*dst_w=*/4,
                          /*dst_h=*/11);
  EXPECT_EQ(resized.size(), 4 * 11 * 3);
  for (const uint8_t value : resized) {
    EXPECT_EQ(value, 123);
  }
}

TEST(PilResizeTest, DegenerateInputsReturnEmpty) {
  EXPECT_TRUE(
      PilResizeBicubicRgb(nullptr, kSrcWidth, kSrcHeight, 3, 2).empty());
  EXPECT_TRUE(PilResizeBicubicRgb(kSource, 0, kSrcHeight, 3, 2).empty());
  EXPECT_TRUE(PilResizeBicubicRgb(kSource, kSrcWidth, -1, 3, 2).empty());
  EXPECT_TRUE(
      PilResizeBicubicRgb(kSource, kSrcWidth, kSrcHeight, 0, 2).empty());
  EXPECT_TRUE(
      PilResizeBicubicRgb(kSource, kSrcWidth, kSrcHeight, 3, -5).empty());
}

}  // namespace
}  // namespace litert::lm
