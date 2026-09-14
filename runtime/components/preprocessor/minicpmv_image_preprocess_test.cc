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

#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Appends `value` to `out` as a little-endian 32-bit word.
void AppendLe32(uint32_t value, std::string& out) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  }
}

// Appends `value` to `out` as a little-endian 16-bit word.
void AppendLe16(uint16_t value, std::string& out) {
  for (int i = 0; i < 2; ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  }
}

// Encodes an uncompressed 24-bit BMP. BMP is used rather than PNG because it
// needs no compressor, and stb_image (the decoder under test) reads it.
// The pixel at (x, y) is a deterministic function of its coordinates unless
// `solid` is true, in which case every pixel is mid-white.
std::string MakeBmp(int width, int height, bool solid) {
  const int row_stride = (width * 3 + 3) / 4 * 4;
  const uint32_t pixel_bytes = static_cast<uint32_t>(row_stride) * height;
  std::string bmp;
  bmp.append("BM");
  AppendLe32(54 + pixel_bytes, bmp);
  AppendLe16(0, bmp);
  AppendLe16(0, bmp);
  AppendLe32(54, bmp);
  AppendLe32(40, bmp);
  AppendLe32(static_cast<uint32_t>(width), bmp);
  AppendLe32(static_cast<uint32_t>(height), bmp);
  AppendLe16(1, bmp);
  AppendLe16(24, bmp);
  AppendLe32(0, bmp);
  AppendLe32(pixel_bytes, bmp);
  AppendLe32(0, bmp);
  AppendLe32(0, bmp);
  AppendLe32(0, bmp);
  AppendLe32(0, bmp);
  // BMP rows are stored bottom-up, as BGR triples.
  for (int y = height - 1; y >= 0; --y) {
    int written = 0;
    for (int x = 0; x < width; ++x) {
      const uint8_t r = solid ? 255 : static_cast<uint8_t>((x * 7) % 256);
      const uint8_t g = solid ? 255 : static_cast<uint8_t>((y * 11) % 256);
      const uint8_t b = solid ? 255 : static_cast<uint8_t>((x + y) % 256);
      bmp.push_back(static_cast<char>(b));
      bmp.push_back(static_cast<char>(g));
      bmp.push_back(static_cast<char>(r));
      written += 3;
    }
    while (written < row_stride) {
      bmp.push_back('\0');
      ++written;
    }
  }
  return bmp;
}

// Checks the internal consistency every slice must satisfy.
void ExpectSliceIsWellFormed(const MinicpmvSlice& slice) {
  EXPECT_GT(slice.tgt_h, 0);
  EXPECT_GT(slice.tgt_w, 0);
  EXPECT_EQ(slice.num_patches, slice.tgt_h * slice.tgt_w);
  // The fused vision signature cannot accept more patches than its capacity.
  EXPECT_LE(slice.num_patches, kMinicpmvMaxPatchLen);
  EXPECT_EQ(slice.strip.size(),
            static_cast<size_t>(3) * kMinicpmvPatchSize * slice.num_patches *
                kMinicpmvPatchSize);
  EXPECT_EQ(slice.position_ids.size(),
            static_cast<size_t>(slice.num_patches));
  for (const int64_t id : slice.position_ids) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, kMinicpmvNumPatchesPerSide * kMinicpmvNumPatchesPerSide);
  }
}

TEST(MinicpmvImagePreprocessTest, SmallImageProducesThumbnailOnly) {
  // 280x280 is well under scale_resolution^2, so no slicing grid is chosen.
  const auto sliced = PreprocessImageSliced(MakeBmp(280, 280, /*solid=*/false));
  ASSERT_OK(sliced);
  EXPECT_EQ(sliced->grid_x, 0);
  EXPECT_EQ(sliced->grid_y, 0);
  ASSERT_EQ(sliced->slices.size(), 1);
  ExpectSliceIsWellFormed(sliced->slices[0]);
}

TEST(MinicpmvImagePreprocessTest, LargeImageIsSlicedIntoThumbnailPlusGrid) {
  // A wide image large enough to trigger the multi-slice path.
  const auto sliced =
      PreprocessImageSliced(MakeBmp(1400, 500, /*solid=*/false));
  ASSERT_OK(sliced);
  EXPECT_GT(sliced->grid_x, 0);
  EXPECT_GT(sliced->grid_y, 0);
  // A thumbnail plus one sub-image per grid cell.
  EXPECT_EQ(sliced->slices.size(),
            1 + static_cast<size_t>(sliced->grid_x) * sliced->grid_y);
  // A wide image should be split into more columns than rows.
  EXPECT_GE(sliced->grid_x, sliced->grid_y);
  for (const MinicpmvSlice& slice : sliced->slices) {
    ExpectSliceIsWellFormed(slice);
  }
}

TEST(MinicpmvImagePreprocessTest, NormalizationMapsWhiteToOne) {
  // With mean 0.5 and std 0.5, a saturated pixel normalizes to exactly 1.0,
  // and resampling a constant field cannot introduce any other value.
  const auto sliced = PreprocessImageSliced(MakeBmp(300, 300, /*solid=*/true));
  ASSERT_OK(sliced);
  ASSERT_FALSE(sliced->slices.empty());
  for (const float value : sliced->slices[0].strip) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

TEST(MinicpmvImagePreprocessTest, PositionIdsAreRowMajorOnTheViTGrid) {
  const auto sliced = PreprocessImageSliced(MakeBmp(280, 280, /*solid=*/false));
  ASSERT_OK(sliced);
  ASSERT_EQ(sliced->slices.size(), 1);
  const MinicpmvSlice& slice = sliced->slices[0];
  // Ids are bucketized row index * grid side + bucketized column index, so
  // they are non-decreasing in row-major order and the first is always 0.
  EXPECT_EQ(slice.position_ids.front(), 0);
  for (size_t i = 1; i < slice.position_ids.size(); ++i) {
    const bool same_row =
        (i % slice.tgt_w) != 0;
    if (same_row) {
      EXPECT_GT(slice.position_ids[i], slice.position_ids[i - 1]);
    }
  }
}

TEST(MinicpmvImagePreprocessTest, RejectsUndecodableBytes) {
  EXPECT_FALSE(PreprocessImageSliced("not an image").ok());
}

}  // namespace
}  // namespace litert::lm
