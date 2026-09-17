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

#include <algorithm>
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
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
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
  constexpr int kPatch = kMiniCpmVPatchSize;
  constexpr int kPatchesPerSide = kMiniCpmVNumPatchesPerSide;
  const int num_patches = slice.num_patches;
  const int patch_dim = 3 * kPatch * kPatch;

  // strip is [3, patch, num_patches*patch] (channel, py, patch*patch_w+px).
  // Rearrange to [num_patches, 3*patch*patch] for the fused linear patch
  // embedding.
  std::vector<float> images(static_cast<size_t>(num_patches) * patch_dim);
  const int strip_w = num_patches * kPatch;
  for (int p = 0; p < num_patches; ++p) {
    for (int ch = 0; ch < 3; ++ch) {
      for (int py = 0; py < kPatch; ++py) {
        for (int px = 0; px < kPatch; ++px) {
          const int col = p * kPatch + px;
          const float value =
              slice.strip[(static_cast<size_t>(ch) * kPatch + py) * strip_w +
                          col];
          images[(static_cast<size_t>(p) * patch_dim) +
                 ((ch * kPatch + py) * kPatch + px)] = value;
        }
      }
    }
  }

  // Contiguous (w,h) for the baked resampler table gather; pad rows are
  // (-1,-1). Valid rows use the slice's own target grid (row-major, so the
  // column is i % tgt_w). vit_positions hold the bucketized 70x70 coordinates
  // for the ViT absolute position embedding.
  std::vector<int32_t> positions_xy(static_cast<size_t>(num_patches) * 2);
  std::vector<int32_t> vit_positions(static_cast<size_t>(num_patches) * 2);
  const int tgt_w = slice.tgt_w > 0 ? slice.tgt_w : 1;
  for (int i = 0; i < num_patches; ++i) {
    const int64_t id = slice.position_ids[i];
    if (id < 0) {
      positions_xy[static_cast<size_t>(i) * 2 + 0] = -1;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = -1;
      vit_positions[static_cast<size_t>(i) * 2 + 0] = -1;
      vit_positions[static_cast<size_t>(i) * 2 + 1] = -1;
    } else {
      positions_xy[static_cast<size_t>(i) * 2 + 0] = i % tgt_w;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = i / tgt_w;
      vit_positions[static_cast<size_t>(i) * 2 + 0] =
          static_cast<int32_t>(id % kPatchesPerSide);
      vit_positions[static_cast<size_t>(i) * 2 + 1] =
          static_cast<int32_t>(id / kPatchesPerSide);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto images_tensor,
      CopyToTensorBuffer<float>(
          absl::MakeConstSpan(images),
          ::litert::Dimensions({1, num_patches, patch_dim})));
  LITERT_ASSIGN_OR_RETURN(
      auto positions_tensor,
      CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(positions_xy),
                                  ::litert::Dimensions({1, num_patches, 2})));
  LITERT_ASSIGN_OR_RETURN(
      auto vit_positions_tensor,
      CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(vit_positions),
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

// Widens a slice strip from its valid width to the fused signature length,
// padding with zeros. The strip is [3, patch, num_patches*patch], so widening
// must re-pack rows: a flat resize would keep the old stride and mis-read the
// later channels.
void PadSliceToSignatureLength(MiniCpmVSlice& slice, int padded_num_patches) {
  constexpr int kPatch = kMiniCpmVPatchSize;
  const int old_w = slice.num_patches * kPatch;
  const int new_w = padded_num_patches * kPatch;
  std::vector<float> wide(static_cast<size_t>(3) * kPatch * new_w, 0.0f);
  for (int row = 0; row < 3 * kPatch; ++row) {
    std::copy(slice.strip.data() + static_cast<size_t>(row) * old_w,
              slice.strip.data() + static_cast<size_t>(row) * old_w + old_w,
              wide.data() + static_cast<size_t>(row) * new_w);
  }
  slice.strip = std::move(wide);
  slice.num_patches = padded_num_patches;
  // tgt_w is kept so that positions_xy stays contiguous; pad rows are marked
  // with position_ids = -1.
  slice.position_ids.resize(padded_num_patches, -1);
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
    const PromptTemplateCapabilities& capabilities,
    std::unique_ptr<ImagePreprocessor> image_preprocessor) {
  if (image_preprocessor == nullptr) {
    image_preprocessor = ImagePreprocessor::Create();
  }
  return absl::WrapUnique(new MiniCpmVDataProcessor(
      config, capabilities, std::move(image_preprocessor)));
}

absl::StatusOr<ordered_json> MiniCpmVDataProcessor::FormatTools(
    const ordered_json& tools) const {
  return tools;
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

absl::StatusOr<Message> MiniCpmVDataProcessor::ToMessageImpl(
    const Responses& responses,
    const MiniCpmVDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json content = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return ordered_json::object({{"role", "assistant"}, {"content", content}});
}

absl::Status MiniCpmVDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<MiniCpmVDataProcessorConfig,
                                     MiniCpmVDataProcessorArguments>& other) {
  return absl::OkStatus();
}

}  // namespace litert::lm
