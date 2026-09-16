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

#include "runtime/conversation/model_data_processor/minicpmv_data_processor.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

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

// Encodes an uncompressed 24-bit BMP. BMP needs no compressor and stb_image
// (used by the slicing preprocessor) reads it.
std::string MakeBmp(int width, int height) {
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
      bmp.push_back(static_cast<char>((x + y) % 256));
      bmp.push_back(static_cast<char>((y * 11) % 256));
      bmp.push_back(static_cast<char>((x * 7) % 256));
      written += 3;
    }
    while (written < row_stride) {
      bmp.push_back('\0');
      ++written;
    }
  }
  return bmp;
}

json MessagesWithImage(int width, int height) {
  const json content = json::array(
      {json::object({{"type", "image"}, {"bytes", MakeBmp(width, height)}}),
       json::object({{"type", "text"}, {"text", "what is this?"}})});
  return json::array({json::object({{"role", "user"}, {"content", content}})});
}

// Returns the text of `data`, or an empty string if it is not an InputText.
std::string TextOf(const InputData& data) {
  if (!std::holds_alternative<InputText>(data)) return "";
  const auto text = std::get<InputText>(data).GetRawTextString();
  return text.ok() ? std::string(*text) : "";
}

// Checks that an image entry carries the three tensors the fused MiniCPM-V
// vision signature expects, each padded to the signature length.
void ExpectSliceTensors(const InputData& data) {
  ASSERT_TRUE(std::holds_alternative<InputImage>(data));
  const InputImage& image = std::get<InputImage>(data);
  ASSERT_TRUE(image.IsTensorBufferMap());
  ASSERT_OK_AND_ASSIGN(const auto* tensors,
                       image.GetPreprocessedImageTensorMap());
  EXPECT_TRUE(tensors->contains("images"));
  EXPECT_TRUE(tensors->contains("positions_xy"));
  EXPECT_TRUE(tensors->contains("vit_positions"));
  const auto images_type = tensors->at("images").TensorType();
  ASSERT_TRUE(images_type.HasValue());
  const auto& dims = images_type->Layout().Dimensions();
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0], 1);
  // Every slice is padded to the fused signature length so that the executor
  // always derives exactly kMiniCpmVTokensPerSlice soft tokens.
  EXPECT_EQ(dims[1], kMiniCpmVMaxPatchLen);
  EXPECT_EQ(dims[2], 3 * kMiniCpmVPatchSize * kMiniCpmVPatchSize);
}

TEST(MiniCpmVDataProcessorTest, SmallImageEmitsThumbnailOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector("before<image_soft_token>after",
                                   MessagesWithImage(280, 280), {}));

  // prefix, <image> open, image, <image> close, suffix.
  ASSERT_EQ(input_data.size(), 5);
  EXPECT_EQ(TextOf(input_data[0]), "before");
  EXPECT_EQ(TextOf(input_data[1]), "<image_id>0</image_id><image>");
  ExpectSliceTensors(input_data[2]);
  EXPECT_EQ(TextOf(input_data[3]), "</image>");
  EXPECT_EQ(TextOf(input_data[4]), "after");
}

TEST(MiniCpmVDataProcessorTest, LargeImageEmitsThumbnailAndSlices) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector("<image_soft_token>",
                                   MessagesWithImage(1400, 500), {}));

  int num_images = 0;
  int num_slice_markers = 0;
  for (const InputData& data : input_data) {
    if (std::holds_alternative<InputImage>(data)) {
      ++num_images;
      ExpectSliceTensors(data);
    } else if (TextOf(data) == "<slice>") {
      ++num_slice_markers;
    }
  }
  // A thumbnail plus at least one sub-slice, each sub-slice fenced by <slice>.
  EXPECT_GT(num_images, 1);
  EXPECT_EQ(num_slice_markers, num_images - 1);
  EXPECT_EQ(TextOf(input_data[0]), "<image_id>0</image_id><image>");
}

TEST(MiniCpmVDataProcessorTest, TextOnlyMessageEmitsPlainText) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  const json messages = json::array(
      {json::object({{"role", "user"}, {"content", "no image here"}})});
  ASSERT_OK_AND_ASSIGN(const std::vector<InputData> input_data,
                       processor->ToInputDataVector("prompt", messages, {}));
  ASSERT_EQ(input_data.size(), 1);
  EXPECT_EQ(TextOf(input_data[0]), "prompt");
}

TEST(MiniCpmVDataProcessorTest, MarkerWithoutImageInMessagesIsRejected) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  const json messages = json::array(
      {json::object({{"role", "user"}, {"content", "no image here"}})});
  EXPECT_FALSE(
      processor->ToInputDataVector("<image_soft_token>", messages, {}).ok());
}

TEST(MiniCpmVDataProcessorTest, Base64BlobImageIsSupported) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  const json content = json::array(
      {json::object({{"type", "image"},
                     {"blob", absl::Base64Escape(MakeBmp(280, 280))}}),
       json::object({{"type", "text"}, {"text", "what is this?"}})});
  const json messages =
      json::array({json::object({{"role", "user"}, {"content", content}})});
  ASSERT_OK_AND_ASSIGN(const std::vector<InputData> input_data,
                       processor->ToInputDataVector(
                           "before<image_soft_token>after", messages, {}));
  ASSERT_EQ(input_data.size(), 5);
  ExpectSliceTensors(input_data[2]);
}

TEST(MiniCpmVDataProcessorTest, NonExistentImagePathIsRejected) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  const json content = json::array(
      {json::object(
           {{"type", "image"}, {"path", "/nonexistent/path/to/image.bmp"}}),
       json::object({{"type", "text"}, {"text", "what is this?"}})});
  const json messages =
      json::array({json::object({{"role", "user"}, {"content", content}})});
  EXPECT_FALSE(
      processor->ToInputDataVector("<image_soft_token>", messages, {}).ok());
}

TEST(MiniCpmVDataProcessorTest, ToMessageWrapsResponseText) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"a cat"}),
                           std::monostate{}));
  EXPECT_EQ(message,
            json({{"role", "assistant"},
                  {"content", {{{"type", "text"}, {"text", "a cat"}}}}}));
}

TEST(MiniCpmVDataProcessorTest, MultiImageEmitsIndexedImageIdsAndSlices) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  const json content = json::array(
      {json::object({{"type", "image"}, {"bytes", MakeBmp(280, 280)}}),
       json::object({{"type", "image"}, {"bytes", MakeBmp(280, 280)}}),
       json::object({{"type", "text"}, {"text", "compare these"}})});
  const json messages =
      json::array({json::object({{"role", "user"}, {"content", content}})});
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(
          "first:<image_soft_token> second:<image_soft_token> end", messages,
          {}));
  ASSERT_EQ(input_data.size(), 9);
  EXPECT_EQ(TextOf(input_data[0]), "first:");
  EXPECT_EQ(TextOf(input_data[1]), "<image_id>0</image_id><image>");
  ExpectSliceTensors(input_data[2]);
  EXPECT_EQ(TextOf(input_data[3]), "</image>");
  EXPECT_EQ(TextOf(input_data[4]), " second:");
  EXPECT_EQ(TextOf(input_data[5]), "<image_id>1</image_id><image>");
  ExpectSliceTensors(input_data[6]);
  EXPECT_EQ(TextOf(input_data[7]), "</image>");
  EXPECT_EQ(TextOf(input_data[8]), " end");
}

TEST(MiniCpmVDataProcessorTest, MissingSoftTokenMarkerIsRejected) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  EXPECT_FALSE(processor
                   ->ToInputDataVector("prompt without marker",
                                       MessagesWithImage(280, 280), {})
                   .ok());
}

TEST(MiniCpmVDataProcessorTest, MismatchedImageAndMarkerCountIsRejected) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpmVDataProcessor::Create(
                                           MiniCpmVDataProcessorConfig{},
                                           PromptTemplateCapabilities{}));
  EXPECT_FALSE(
      processor
          ->ToInputDataVector("<image_soft_token> and <image_soft_token>",
                              MessagesWithImage(280, 280), {})
          .ok());
}

}  // namespace
}  // namespace litert::lm
