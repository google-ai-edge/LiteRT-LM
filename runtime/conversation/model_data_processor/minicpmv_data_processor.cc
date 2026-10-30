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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

// The chat template renders each image as the literal "<image_soft_token>".
// That is only a boundary marker, not a vocabulary token.
constexpr absl::string_view kImageSoftToken = "<image_soft_token>";

// Pack one preprocessed slice into the stock map-based Encode InputImage
// contract used by VisionLiteRtCompiledModelExecutor for the fused vision
// tflite:
//   "images"        [1, L, 3*patch*patch] float  (patch-flattened)
//   "positions_xy"  [1, L, 2] int32             (contiguous (w,h); -1 pad)
//   "vit_positions" [1, L, 2] int32             (bucketized ViT (w,h); -1 pad)
// L is padded to kMiniCpmVMaxPatchLen so that
// shrink = L / kMiniCpmVTokensPerSlice always yields exactly
// kMiniCpmVTokensPerSlice soft tokens per slice. The resampler pos_embed is
// baked into the graph, so only the coordinates are runtime inputs.
absl::StatusOr<InputImage> BuildSliceImage(const MiniCpmVSlice& slice) {
  const int num_patches = slice.num_patches;
  constexpr int kPatchDim = 3 * kMiniCpmVPatchSize * kMiniCpmVPatchSize;
  const MiniCpmVPackedSlice packed = PackSlicePatches(slice);

  LITERT_ASSIGN_OR_RETURN(
      auto images_tensor,
      CopyToTensorBuffer<float>(
          absl::MakeConstSpan(packed.images),
          ::litert::Dimensions({1, num_patches, kPatchDim})));
  LITERT_ASSIGN_OR_RETURN(
      auto positions_tensor,
      CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(packed.positions_xy),
                                  ::litert::Dimensions({1, num_patches, 2})));
  LITERT_ASSIGN_OR_RETURN(
      auto vit_positions_tensor,
      CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(packed.vit_positions),
                                  ::litert::Dimensions({1, num_patches, 2})));

  absl::flat_hash_map<std::string, ::litert::TensorBuffer> tensors;
  tensors.emplace("images", std::move(images_tensor));
  tensors.emplace("positions_xy", std::move(positions_tensor));
  tensors.emplace("vit_positions", std::move(vit_positions_tensor));
  return InputImage(std::move(tensors));
}

// Returns the bytes of all image content items found in `messages` in order,
// reading from "path", base64 "blob", or taking inline "bytes".
absl::StatusOr<std::vector<std::string>> ExtractAllImageBytes(
    const ordered_json& messages) {
  std::vector<std::string> images;
  for (const auto& message : messages) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    for (const auto& item : message["content"]) {
      if (!item.is_object() || !item.contains("type") ||
          item["type"] != "image") {
        continue;
      }
      if (item.contains("bytes")) {
        images.push_back(item["bytes"].get<std::string>());
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                            LoadItemData(item));
      if (mmap_file == nullptr) {
        return absl::InvalidArgumentError(
            "Failed to load image data from item.");
      }
      images.emplace_back(static_cast<const char*>(mmap_file->data()),
                          mmap_file->length());
    }
  }
  return images;
}

// Appends `text` as an InputText, skipping empty segments.
void AppendText(absl::string_view text, std::vector<InputData>& output) {
  if (!text.empty()) {
    output.emplace_back(InputText(std::string(text)));
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<MiniCpmVDataProcessor>>
MiniCpmVDataProcessor::Create(
    MiniCpmVDataProcessorConfig config,
    std::unique_ptr<ImagePreprocessor> image_preprocessor) {
  if (image_preprocessor == nullptr) {
    image_preprocessor = ImagePreprocessor::Create();
  }
  return absl::WrapUnique(
      new MiniCpmVDataProcessor(config, std::move(image_preprocessor)));
}

absl::StatusOr<std::vector<InputData>>
MiniCpmVDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const MiniCpmVDataProcessorArguments& args) const {
  // The chat template renders each image as the literal "<image_soft_token>".
  // We replace each marker with the official multi-slice layout
  //   <image_id>i</image_id><image>{64 placeholders}</image>
  //   [<slice>{64}</slice>...]
  // and each InputImage becomes 64 vision (-1) tokens that the embedding
  // splice fills from the fused encoder.
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::string> image_bytes_list,
                        ExtractAllImageBytes(messages));

  std::vector<size_t> marker_positions;
  size_t search_pos = 0;
  while (true) {
    const size_t pos =
        rendered_template_prompt.find(kImageSoftToken, search_pos);
    if (pos == std::string::npos) break;
    marker_positions.push_back(pos);
    search_pos = pos + kImageSoftToken.size();
  }

  if (marker_positions.size() != image_bytes_list.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MiniCPM-V: number of images in messages (%d) does not match "
        "number of <image_soft_token> markers (%d) in rendered prompt.",
        image_bytes_list.size(), marker_positions.size()));
  }

  std::vector<InputData> output;
  size_t text_cursor = 0;
  const MiniCpmVSliceConfig slice_config;

  for (size_t img_idx = 0; img_idx < image_bytes_list.size(); ++img_idx) {
    const size_t marker_pos = marker_positions[img_idx];
    AppendText(absl::string_view(rendered_template_prompt)
                   .substr(text_cursor, marker_pos - text_cursor),
               output);
    text_cursor = marker_pos + kImageSoftToken.size();

    // Slice, then pad each strip to the fused signature length. The map-based
    // Encode token count is ceil(L / shrink) with
    // shrink = L / kMiniCpmVTokensPerSlice (= 17), so L = 1088 always yields
    // 64 soft tokens.
    if (image_preprocessor_ == nullptr) {
      return absl::InternalError("Image preprocessor is null.");
    }
    ABSL_ASSIGN_OR_RETURN(
        MiniCpmVSliced sliced,
        PreprocessImageSliced(image_bytes_list[img_idx], *image_preprocessor_,
                              slice_config));
    const int grid_x = sliced.grid_x;
    const int grid_y = sliced.grid_y;
    for (auto& slice : sliced.slices) {
      if (slice.num_patches > kMiniCpmVMaxPatchLen) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "MiniCPM-V slice has %d patches > fused signature capacity %d",
            slice.num_patches, kMiniCpmVMaxPatchLen));
      }
      if (slice.num_patches != kMiniCpmVMaxPatchLen) {
        PadSliceToSignatureLength(slice, kMiniCpmVMaxPatchLen);
      }
    }

    // Image id and thumbnail.
    AppendText(absl::StrFormat("<image_id>%d</image_id><image>", img_idx),
               output);
    ABSL_ASSIGN_OR_RETURN(auto thumbnail, BuildSliceImage(sliced.slices[0]));
    output.emplace_back(std::move(thumbnail));
    AppendText("</image>", output);

    // Sub-slices (index 1..N-1) laid out in grid row-major order; rows are
    // joined by a newline.
    int sub_slice = 1;
    for (int row = 0; row < grid_y; ++row) {
      for (int col = 0; col < grid_x; ++col) {
        if (sub_slice >= static_cast<int>(sliced.slices.size())) break;
        AppendText("<slice>", output);
        ABSL_ASSIGN_OR_RETURN(auto slice_image,
                              BuildSliceImage(sliced.slices[sub_slice]));
        output.emplace_back(std::move(slice_image));
        AppendText("</slice>", output);
        ++sub_slice;
      }
      if (row + 1 < grid_y) AppendText("\n", output);
    }
  }

  AppendText(absl::string_view(rendered_template_prompt).substr(text_cursor),
             output);
  return output;
}

}  // namespace litert::lm
