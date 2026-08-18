// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/model_data_processor/minicpmv_data_processor.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"            // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/str_format.h"       // from @com_google_absl
#include "absl/types/span.h"               // from @com_google_absl
#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using nlohmann::ordered_json;

// Pack one preprocessed slice into the stock map-Encode InputImage contract
// used by VisionLiteRtCompiledModelExecutor for the fused vision tflite:
//   "images"        [1, L, 3*patch*patch] float  (patch-flattened)
//   "positions_xy"  [1, L, 2] int32             (contiguous (w,h); -1 pad)
//   "vit_positions" [1, L, 2] int32             (bucketized ViT (w,h); -1 pad)
// L is padded to kMinicpmvMaxPatchLen so shrink = L / kMinicpmvTokensPerSlice
// always yields exactly kMinicpmvTokensPerSlice soft tokens per slice.
// Resampler pos_embed is baked in the graph; only coords are runtime inputs.
absl::StatusOr<InputImage> BuildSliceImage(const MinicpmvSlice& sl) {
  constexpr int kPatch = kMinicpmvPatchSize;
  constexpr int kPps = kMinicpmvNumPatchesPerSide;
  const int num_patches = sl.num_patches;
  const int patch_dim = 3 * kPatch * kPatch;
  // strip is [3, patch, num_patches*patch] (channel, py, patch*patch_w+px).
  // Rearrange to [num_patches, 3*patch*patch] for the fused linear patch embed.
  std::vector<float> images(static_cast<size_t>(num_patches) * patch_dim);
  const int strip_w = num_patches * kPatch;
  for (int p = 0; p < num_patches; ++p) {
    for (int ch = 0; ch < 3; ++ch) {
      for (int py = 0; py < kPatch; ++py) {
        for (int px = 0; px < kPatch; ++px) {
          const int col = p * kPatch + px;
          const float v =
              sl.strip[(static_cast<size_t>(ch) * kPatch + py) * strip_w + col];
          images[(static_cast<size_t>(p) * patch_dim) +
                 ((ch * kPatch + py) * kPatch + px)] = v;
        }
      }
    }
  }
  // Contiguous (w,h) for the baked resampler table gather; pad rows are (-1,-1).
  // Valid rows use the slice's own tgt grid (row-major: col = i % tgt_w).
  // vit_positions hold bucketized 70x70 coords for the ViT absolute pos embed.
  std::vector<int32_t> positions_xy(static_cast<size_t>(num_patches) * 2);
  std::vector<int32_t> vit_positions(static_cast<size_t>(num_patches) * 2);
  const int tgt_w = sl.tgt_w > 0 ? sl.tgt_w : 1;
  for (int i = 0; i < num_patches; ++i) {
    const int64_t id = sl.position_ids[i];
    if (id < 0) {
      positions_xy[static_cast<size_t>(i) * 2 + 0] = -1;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = -1;
      vit_positions[static_cast<size_t>(i) * 2 + 0] = -1;
      vit_positions[static_cast<size_t>(i) * 2 + 1] = -1;
    } else {
      positions_xy[static_cast<size_t>(i) * 2 + 0] = i % tgt_w;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = i / tgt_w;
      vit_positions[static_cast<size_t>(i) * 2 + 0] =
          static_cast<int32_t>(id % kPps);
      vit_positions[static_cast<size_t>(i) * 2 + 1] =
          static_cast<int32_t>(id / kPps);
    }
  }
  LITERT_ASSIGN_OR_RETURN(
      auto images_t,
      CopyToTensorBuffer<float>(
          absl::MakeConstSpan(images),
          ::litert::Dimensions({1, num_patches, patch_dim})));
  LITERT_ASSIGN_OR_RETURN(
      auto pos_t,
      CopyToTensorBuffer<int32_t>(
          absl::MakeConstSpan(positions_xy),
          ::litert::Dimensions({1, num_patches, 2})));
  LITERT_ASSIGN_OR_RETURN(
      auto vit_t,
      CopyToTensorBuffer<int32_t>(
          absl::MakeConstSpan(vit_positions),
          ::litert::Dimensions({1, num_patches, 2})));
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> m;
  m.emplace("images", std::move(images_t));
  m.emplace("positions_xy", std::move(pos_t));
  m.emplace("vit_positions", std::move(vit_t));
  return InputImage(std::move(m));
}

}  // namespace

absl::StatusOr<std::unique_ptr<MinicpmvDataProcessor>>
MinicpmvDataProcessor::Create(MinicpmvDataProcessorConfig config,
                              const PromptTemplateCapabilities& capabilities) {
  return absl::WrapUnique(new MinicpmvDataProcessor(config, capabilities));
}

absl::StatusOr<ordered_json> MinicpmvDataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  return message;
}

absl::StatusOr<ordered_json> MinicpmvDataProcessor::FormatTools(
    const ordered_json& tools) const {
  return tools;
}

absl::StatusOr<std::vector<InputData>>
MinicpmvDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const ordered_json& messages,
    const MinicpmvDataProcessorArguments& args) const {
  // Chat template (model_type_utils kMinicpmv) renders each image as the
  // literal "<image_soft_token>". That is only a boundary marker (not a vocab
  // token): we replace it with the official multi-slice layout
  //   <image_id>0</image_id><image>{64 placeholders}</image>
  //   [<slice>{64}</slice>...]
  // and ProcessMultimodalPrompt turns each InputImage into 64 vision (-1)
  // tokens that the embedding splice fills from the fused encoder.

  // 1. First image content item (path or bytes).
  std::string image_bytes;
  for (const auto& message : messages) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    for (const auto& item : message["content"]) {
      if (!item.is_object() || !item.contains("type") ||
          item["type"] != "image") {
        continue;
      }
      if (item.contains("path")) {
        std::ifstream f(item["path"].get<std::string>(), std::ios::binary);
        std::ostringstream ss;
        ss << f.rdbuf();
        image_bytes = ss.str();
      } else if (item.contains("bytes")) {
        image_bytes = item["bytes"].get<std::string>();
      }
      break;
    }
    if (!image_bytes.empty()) {
      break;
    }
  }
  if (image_bytes.empty()) {
    return absl::InvalidArgumentError("MiniCPM-V: no image bytes in messages.");
  }

  // 2. Slice + pad each strip to the fused signature length.
  // map-Encode token count is ceil(L / shrink) with shrink = L /
  // kMinicpmvTokensPerSlice (= 17), so L=1088 always yields 64 soft tokens.
  // strip is [3, patch, num_patches*patch]; widening must re-pack rows (a flat
  // resize keeps the old stride and would mis-read later channels).
  MinicpmvSliceConfig scfg;
  ASSIGN_OR_RETURN(MinicpmvSliced sliced,
                   PreprocessImageSliced(image_bytes, scfg));
  const int gx = sliced.grid_x, gy = sliced.grid_y;
  for (auto& sl : sliced.slices) {
    const int np = sl.num_patches;
    const int padded = kMinicpmvMaxPatchLen;
    if (np > padded) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MiniCPM-V slice has %d patches > fused signature capacity %d", np,
          padded));
    }
    if (padded != np) {
      constexpr int kPatch = kMinicpmvPatchSize;
      const int old_w = np * kPatch;
      const int new_w = padded * kPatch;
      std::vector<float> wide(static_cast<size_t>(3) * kPatch * new_w, 0.0f);
      for (int r = 0; r < 3 * kPatch; ++r) {
        std::copy(sl.strip.data() + static_cast<size_t>(r) * old_w,
                  sl.strip.data() + static_cast<size_t>(r) * old_w + old_w,
                  wide.data() + static_cast<size_t>(r) * new_w);
      }
      sl.strip = std::move(wide);
      sl.num_patches = padded;
      // Keep tgt_w for contiguous positions_xy; pad rows get position_ids=-1.
      sl.position_ids.resize(padded, -1);
    }
  }

  // 3. Split the rendered prompt at the single soft-token marker.
  std::string rendered(rendered_template_prompt);
  const std::string marker = "<image_soft_token>";
  const size_t mpos = rendered.find(marker);
  const std::string pre =
      (mpos == std::string::npos) ? rendered : rendered.substr(0, mpos);
  const std::string post = (mpos == std::string::npos)
                               ? ""
                               : rendered.substr(mpos + marker.size());

  // 4. Build interleaved InputData.
  std::vector<InputData> out;
  auto push_text = [&](const std::string& t) {
    if (!t.empty()) out.emplace_back(InputText(t));
  };
  push_text(pre);
  // image_id + thumbnail
  push_text("<image_id>0</image_id><image>");
  ASSIGN_OR_RETURN(auto thumb, BuildSliceImage(sliced.slices[0]));
  out.emplace_back(std::move(thumb));
  push_text("</image>");
  // sub-slices (index 1..N-1) laid out grid row-major; rows joined by \n.
  int sub = 1;
  for (int r = 0; r < gy; ++r) {
    for (int col = 0; col < gx; ++col) {
      if (sub >= static_cast<int>(sliced.slices.size())) break;
      push_text("<slice>");
      ASSIGN_OR_RETURN(auto sl_img, BuildSliceImage(sliced.slices[sub]));
      out.emplace_back(std::move(sl_img));
      push_text("</slice>");
      ++sub;
    }
    if (r + 1 < gy) push_text("\n");
  }
  push_text(post);
  return out;
}

absl::StatusOr<Message> MinicpmvDataProcessor::ToMessageImpl(
    const Responses& responses,
    const MinicpmvDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json content = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return ordered_json::object({{"role", "assistant"}, {"content", content}});
}

absl::Status MinicpmvDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<MinicpmvDataProcessorConfig,
                                     MinicpmvDataProcessorArguments>& other) {
  return absl::OkStatus();
}

}  // namespace litert::lm
