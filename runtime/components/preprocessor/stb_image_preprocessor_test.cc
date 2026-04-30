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

#include "runtime/components/preprocessor/stb_image_preprocessor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/preprocessor/testdata/";

TEST(StbImagePreprocessorTest, PreprocessSuccess) {
  StbImagePreprocessor preprocessor;

  // Load the image file.
  const std::string image_path =
      (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / "apple.png")
          .string();
  std::ifstream file_stream(image_path, std::ios::binary);
  ASSERT_TRUE(file_stream.is_open())
      << "Failed to open image file: " << image_path;
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  std::string image_bytes = buffer.str();
  // Target dimensions: Batch=1, Height=224, Width=224, Channels=3 (RGB)
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 224, 224, 3});

  auto input_image = InputImage(image_bytes);
  ASSERT_OK_AND_ASSIGN(auto preprocessed_image,
                       preprocessor.Preprocess(input_image, parameter));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_tensor,
                       preprocessed_image.GetPreprocessedImageTensor());

  // Verify the output tensor properties.
  auto buffer_type = preprocessed_tensor->BufferType();
  ASSERT_TRUE(buffer_type.HasValue());
  EXPECT_EQ(buffer_type.Value(), ::litert::TensorBufferType::kHostMemory);
  auto tensor_type = preprocessed_tensor->TensorType();
  ASSERT_TRUE(tensor_type.HasValue());
  EXPECT_THAT(tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 224, 224, 3));

  // Verify pixel values are in the range [0.0, 1.0].
  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *preprocessed_tensor, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(output_tensor_lock_and_addr.HasValue());
  const float* data =
      static_cast<const float*>(output_tensor_lock_and_addr->second);
  ASSERT_NE(data, nullptr);
  size_t num_elements = 224 * 224 * 3;
  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_GE(data[i], 0.0f);
    EXPECT_LE(data[i], 1.0f);
  }

  // Sample a few fixed locations to detect issues like incorrect rotation,
  // color channel order, or scaling.
  constexpr float kTolerance = 1e-6f;
  const int height = 224;
  const int width = 224;
  const int channels = 3;

  // Helper to get the starting index for a pixel (y, x).
  auto get_pixel_index = [&](int y, int x) {
    return (y * width + x) * channels;
  };

  // --- Sample 1: Top-Left Pixel (0,0) ---
  const float expected_0_0_r = 0.26666668f;
  const float expected_0_0_g = 0.20000000f;
  const float expected_0_0_b = 0.12941177f;
  int idx_0_0 = get_pixel_index(0, 0);
  EXPECT_NEAR(data[idx_0_0 + 0], expected_0_0_r, kTolerance) << "R at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 1], expected_0_0_g, kTolerance) << "G at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 2], expected_0_0_b, kTolerance) << "B at (0,0)";

  // --- Sample 2: Top-Right Pixel (0, 223) ---
  const float expected_0_223_r = 0.96470588f;
  const float expected_0_223_g = 0.96078432f;
  const float expected_0_223_b = 0.94509804f;
  int idx_0_223 = get_pixel_index(0, width - 1);
  EXPECT_NEAR(data[idx_0_223 + 0], expected_0_223_r, kTolerance)
      << "R at (0,223)";
  EXPECT_NEAR(data[idx_0_223 + 1], expected_0_223_g, kTolerance)
      << "G at (0,223)";
  EXPECT_NEAR(data[idx_0_223 + 2], expected_0_223_b, kTolerance)
      << "B at (0,223)";

  // --- Sample 3: Center Pixel (112, 112) ---
  const float expected_112_112_r = 0.37647059f;
  const float expected_112_112_g = 0.00784314f;
  const float expected_112_112_b = 0.00392157f;
  int idx_112_112 = get_pixel_index(height / 2, width / 2);
  EXPECT_NEAR(data[idx_112_112 + 0], expected_112_112_r, kTolerance)
      << "R at (112,112)";
  EXPECT_NEAR(data[idx_112_112 + 1], expected_112_112_g, kTolerance)
      << "G at (112,112)";
  EXPECT_NEAR(data[idx_112_112 + 2], expected_112_112_b, kTolerance)
      << "B at (112,112)";

  // --- Sample 4: Bottom-Left Pixel (223, 0) ---
  const float expected_223_0_r = 0.41960785f;
  const float expected_223_0_g = 0.26274511f;
  const float expected_223_0_b = 0.16470589f;
  int idx_223_0 = get_pixel_index(height - 1, 0);
  EXPECT_NEAR(data[idx_223_0 + 0], expected_223_0_r, kTolerance)
      << "R at (223,0)";
  EXPECT_NEAR(data[idx_223_0 + 1], expected_223_0_g, kTolerance)
      << "G at (223,0)";
  EXPECT_NEAR(data[idx_223_0 + 2], expected_223_0_b, kTolerance)
      << "B at (223,0)";

  // --- Sample 5: Bottom-Right Pixel (223, 223) ---
  const float expected_223_223_r = 0.78823530f;
  const float expected_223_223_g = 0.72156864f;
  const float expected_223_223_b = 0.64705884f;
  int idx_223_223 = get_pixel_index(height - 1, width - 1);
  EXPECT_NEAR(data[idx_223_223 + 0], expected_223_223_r, kTolerance)
      << "R at (223,223)";
  EXPECT_NEAR(data[idx_223_223 + 1], expected_223_223_g, kTolerance)
      << "G at (223,223)";
  EXPECT_NEAR(data[idx_223_223 + 2], expected_223_223_b, kTolerance)
      << "B at (223,223)";
}

TEST(StbImagePreprocessorTest, PreprocessFailedWithInvalidDimensions) {
  StbImagePreprocessor preprocessor;
  std::string dummy_bytes = "dummy";
  // Invalid dimensions size (e.g., missing channels).
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 224, 224});

  EXPECT_THAT(preprocessor.Preprocess(InputImage(dummy_bytes), parameter),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(StbImagePreprocessorTest, PreprocessFailedWithInvalidImage) {
  StbImagePreprocessor preprocessor;
  std::string invalid_image_bytes = "invalid_image_bytes";
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 224, 224, 3});

  EXPECT_THAT(
      preprocessor.Preprocess(InputImage(invalid_image_bytes), parameter),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Failed to decode image. Reason: unknown image type"));
}

TEST(StbImagePreprocessorTest, PreprocessWithPatchify) {
  StbImagePreprocessor preprocessor;

  // Load the image file.
  const std::string image_path =
      (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / "apple.png")
          .string();
  std::ifstream file_stream(image_path, std::ios::binary);
  ASSERT_TRUE(file_stream.is_open())
      << "Failed to open image file: " << image_path;
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  std::string image_bytes = buffer.str();

  ImagePreprocessParameter parameter;
  constexpr int kPatchSize = 16;
  parameter.SetPatchifyConfig({.patch_width = kPatchSize,
                               .patch_height = kPatchSize,
                               .max_num_patches = 4096,
                               .pooling_kernel_size = 3});

  auto input_image = InputImage(image_bytes);
  ASSERT_OK_AND_ASSIGN(auto preprocessed_image,
                       preprocessor.Preprocess(input_image, parameter));

  ASSERT_TRUE(preprocessed_image.IsTensorBufferMap());
  ASSERT_OK_AND_ASSIGN(auto tensor_map,
                       preprocessed_image.GetPreprocessedImageTensorMap());
  ASSERT_NE(tensor_map, nullptr);
  EXPECT_TRUE(tensor_map->contains("images"));
  EXPECT_TRUE(tensor_map->contains("positions_xy"));

  const auto& images_tensor = tensor_map->at("images");
  auto images_tensor_type = images_tensor.TensorType();
  ASSERT_TRUE(images_tensor_type.HasValue());
  // The apple.png is 1024x1024.
  // 1024 / 16 = 64. 64 * 64 = 4096 patches.
  // 16 * 16 * 3 = 768 elements per patch.
  EXPECT_THAT(images_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 3969, 768));

  const auto& positions_tensor = tensor_map->at("positions_xy");
  auto positions_tensor_type = positions_tensor.TensorType();
  ASSERT_TRUE(positions_tensor_type.HasValue());
  EXPECT_THAT(positions_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 3969, 2));

  // Verify positions.
  auto positions_lock = ::litert::TensorBufferScopedLock::Create(
      positions_tensor, ::litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(positions_lock.HasValue());
  const int32_t* positions_ptr =
      reinterpret_cast<const int32_t*>(positions_lock->second);
  for (int h = 0; h < 63; ++h) {
    for (int w = 0; w < 63; ++w) {
      int idx = h * 63 + w;
      EXPECT_EQ(positions_ptr[idx * 2], w);
      EXPECT_EQ(positions_ptr[idx * 2 + 1], h);
    }
  }

  // Verify image values.
  auto images_lock = ::litert::TensorBufferScopedLock::Create(
      images_tensor, ::litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(images_lock.HasValue());
  const float* data = reinterpret_cast<const float*>(images_lock->second);

  constexpr float kTolerance = 1e-6f;
  constexpr int height = 1008;
  constexpr int width = 1008;
  constexpr int channels = 3;

  // Helper to get the starting index for a pixel (y, x).
  auto get_pixel_index = [&](int y, int x) {
    const int stride = kPatchSize * kPatchSize * channels;
    return ((y / kPatchSize) * (width / kPatchSize) + x / kPatchSize) * stride +
           ((y % kPatchSize) * kPatchSize + x % kPatchSize) * channels;
  };

  // --- Sample 1: Top-Left Pixel (0,0) ---
  const float expected_0_0_r = 0.27058824f;
  const float expected_0_0_g = 0.20392157f;
  const float expected_0_0_b = 0.12549020f;
  int idx_0_0 = get_pixel_index(0, 0);
  EXPECT_NEAR(data[idx_0_0 + 0], expected_0_0_r, kTolerance) << "R at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 1], expected_0_0_g, kTolerance) << "G at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 2], expected_0_0_b, kTolerance) << "B at (0,0)";

  // --- Sample 2: Top-Right Pixel (0, 1023) ---
  const float expected_0_1023_r = 0.97647058f;
  const float expected_0_1023_g = 0.97254902f;
  const float expected_0_1023_b = 0.95686275f;
  int idx_0_1023 = get_pixel_index(0, width - 1);
  EXPECT_NEAR(data[idx_0_1023 + 0], expected_0_1023_r, kTolerance)
      << "R at (0,1023)";
  EXPECT_NEAR(data[idx_0_1023 + 1], expected_0_1023_g, kTolerance)
      << "G at (0,1023)";
  EXPECT_NEAR(data[idx_0_1023 + 2], expected_0_1023_b, kTolerance)
      << "B at (0,1023)";

  // --- Sample 3: Center Pixel (512, 512) ---
  const float expected_512_512_r = 0.35294118f;
  const float expected_512_512_g = 0.00784314f;
  const float expected_512_512_b = 0.00392157f;
  int idx_512_512 = get_pixel_index(height / 2, width / 2);
  EXPECT_NEAR(data[idx_512_512 + 0], expected_512_512_r, kTolerance)
      << "R at (512,512)";
  EXPECT_NEAR(data[idx_512_512 + 1], expected_512_512_g, kTolerance)
      << "G at (512,512)";
  EXPECT_NEAR(data[idx_512_512 + 2], expected_512_512_b, kTolerance)
      << "B at (512,512)";

  // --- Sample 4: Bottom-Left Pixel (1023, 0) ---
  const float expected_1023_0_r = 0.42745098f;
  const float expected_1023_0_g = 0.26274511f;
  const float expected_1023_0_b = 0.16470589f;
  int idx_1023_0 = get_pixel_index(height - 1, 0);
  EXPECT_NEAR(data[idx_1023_0 + 0], expected_1023_0_r, kTolerance)
      << "R at (1023,0)";
  EXPECT_NEAR(data[idx_1023_0 + 1], expected_1023_0_g, kTolerance)
      << "G at (1023,0)";
  EXPECT_NEAR(data[idx_1023_0 + 2], expected_1023_0_b, kTolerance)
      << "B at (1023,0)";

  // --- Sample 5: Bottom-Right Pixel (1023, 1023) ---
  const float expected_1023_1023_r = 0.68235296f;
  const float expected_1023_1023_g = 0.61568630f;
  const float expected_1023_1023_b = 0.51372552f;
  int idx_1023_1023 = get_pixel_index(height - 1, width - 1);
  EXPECT_NEAR(data[idx_1023_1023 + 0], expected_1023_1023_r, kTolerance)
      << "R at (1023,1023)";
  EXPECT_NEAR(data[idx_1023_1023 + 1], expected_1023_1023_g, kTolerance)
      << "G at (1023,1023)";
  EXPECT_NEAR(data[idx_1023_1023 + 2], expected_1023_1023_b, kTolerance)
      << "B at (1023,1023)";
}

TEST(StbImagePreprocessorTest, PreprocessWithPatchifyResize) {
  StbImagePreprocessor preprocessor;

  // Load the image file.
  const std::string image_path =
      (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / "apple.png")
          .string();
  std::ifstream file_stream(image_path, std::ios::binary);
  ASSERT_TRUE(file_stream.is_open())
      << "Failed to open image file: " << image_path;
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  std::string image_bytes = buffer.str();

  ImagePreprocessParameter parameter;
  // Max 49 patches means it should resize to 112x112 (112/16 = 7, 7*7=49).
  constexpr int kPatchSize = 16;
  parameter.SetPatchifyConfig({.patch_width = kPatchSize,
                               .patch_height = kPatchSize,
                               .max_num_patches = 49,
                               .pooling_kernel_size = 3});

  auto input_image = InputImage(image_bytes);
  ASSERT_OK_AND_ASSIGN(auto preprocessed_image,
                       preprocessor.Preprocess(input_image, parameter));

  ASSERT_TRUE(preprocessed_image.IsTensorBufferMap());
  ASSERT_OK_AND_ASSIGN(auto tensor_map,
                       preprocessed_image.GetPreprocessedImageTensorMap());
  const auto& images_tensor = tensor_map->at("images");
  auto images_tensor_type = images_tensor.TensorType();
  ASSERT_TRUE(images_tensor_type.HasValue());
  EXPECT_THAT(images_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 36, 768));

  const auto& positions_tensor = tensor_map->at("positions_xy");
  auto positions_tensor_type = positions_tensor.TensorType();
  ASSERT_TRUE(positions_tensor_type.HasValue());
  EXPECT_THAT(positions_tensor_type.Value().Layout().Dimensions(),
              ElementsAre(1, 36, 2));

  // Verify image values.
  auto images_lock = ::litert::TensorBufferScopedLock::Create(
      images_tensor, ::litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(images_lock.HasValue());
  const float* data = reinterpret_cast<const float*>(images_lock->second);

  constexpr float kTolerance = 1e-6f;
  constexpr int height = 96;
  constexpr int width = 96;
  constexpr int channels = 3;

  // Helper to get the starting index for a pixel (y, x).
  auto get_pixel_index = [&](int y, int x) {
    const int stride = kPatchSize * kPatchSize * channels;
    return ((y / kPatchSize) * (width / kPatchSize) + x / kPatchSize) * stride +
           ((y % kPatchSize) * kPatchSize + x % kPatchSize) * channels;
  };

  // --- Sample 1: Top-Left Pixel (0,0) ---
  const float expected_0_0_r = 0.26666668f;
  const float expected_0_0_g = 0.20000000f;
  const float expected_0_0_b = 0.13333334f;
  int idx_0_0 = get_pixel_index(0, 0);
  EXPECT_NEAR(data[idx_0_0 + 0], expected_0_0_r, kTolerance) << "R at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 1], expected_0_0_g, kTolerance) << "G at (0,0)";
  EXPECT_NEAR(data[idx_0_0 + 2], expected_0_0_b, kTolerance) << "B at (0,0)";

  // --- Sample 2: Top-Right Pixel (0, 111) ---
  const float expected_0_111_r = 0.94901961f;
  const float expected_0_111_g = 0.94117647f;
  const float expected_0_111_b = 0.92156863f;
  int idx_0_111 = get_pixel_index(0, width - 1);
  EXPECT_NEAR(data[idx_0_111 + 0], expected_0_111_r, kTolerance)
      << "R at (0,111)";
  EXPECT_NEAR(data[idx_0_111 + 1], expected_0_111_g, kTolerance)
      << "G at (0,111)";
  EXPECT_NEAR(data[idx_0_111 + 2], expected_0_111_b, kTolerance)
      << "B at (0,111)";

  // --- Sample 3: Center Pixel (56, 56) ---
  const float expected_56_56_r = 0.41568628f;
  const float expected_56_56_g = 0.01176471f;
  const float expected_56_56_b = 0.011764706f;
  int idx_56_56 = get_pixel_index(height / 2, width / 2);
  EXPECT_NEAR(data[idx_56_56 + 0], expected_56_56_r, kTolerance)
      << "R at (56,56)";
  EXPECT_NEAR(data[idx_56_56 + 1], expected_56_56_g, kTolerance)
      << "G at (56,56)";
  EXPECT_NEAR(data[idx_56_56 + 2], expected_56_56_b, kTolerance)
      << "B at (56,56)";

  // --- Sample 4: Bottom-Left Pixel (111, 0) ---
  const float expected_111_0_r = 0.42745098f;
  const float expected_111_0_g = 0.27450982f;
  const float expected_111_0_b = 0.16862746f;
  int idx_111_0 = get_pixel_index(height - 1, 0);
  EXPECT_NEAR(data[idx_111_0 + 0], expected_111_0_r, kTolerance)
      << "R at (111,0)";
  EXPECT_NEAR(data[idx_111_0 + 1], expected_111_0_g, kTolerance)
      << "G at (111,0)";
  EXPECT_NEAR(data[idx_111_0 + 2], expected_111_0_b, kTolerance)
      << "B at (111,0)";

  // --- Sample 5: Bottom-Right Pixel (111, 111) ---
  const float expected_111_111_r = 0.83137256f;
  const float expected_111_111_g = 0.76862746f;
  const float expected_111_111_b = 0.70196080f;
  int idx_111_111 = get_pixel_index(height - 1, width - 1);
  EXPECT_NEAR(data[idx_111_111 + 0], expected_111_111_r, kTolerance)
      << "R at (111,111)";
  EXPECT_NEAR(data[idx_111_111 + 1], expected_111_111_g, kTolerance)
      << "G at (111,111)";
  EXPECT_NEAR(data[idx_111_111 + 2], expected_111_111_b, kTolerance)
      << "B at (111,111)";
}

}  // namespace
}  // namespace litert::lm
