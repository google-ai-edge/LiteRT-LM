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

#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/preprocessor/image_preprocessor_utils.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {

using ::litert::support::DecodedImage;
using ::litert::support::ImageSize;
using TensorBufferMap =
    absl::flat_hash_map<std::string, ::litert::TensorBuffer>;

// Largest accepted `width` / `height` of a raw video frame.
constexpr int64_t kMaxVideoFrameSide = 16384;

// Input names of vision encoders that take two temporal frames per call
// (`temporal_patch_size = 2`): `images` is the first frame and `images_t1` the
// second.
constexpr absl::string_view kVisionImagesInput = "images";
constexpr absl::string_view kVisionImagesT1Input = "images_t1";

// Returns the decoded byte count of `non_ws_len` non-whitespace base64
// characters ending with `pad` '=' padding characters, or nullopt if invalid.
std::optional<size_t> Base64DecodedLengthFromCounts(size_t non_ws_len,
                                                    size_t pad) {
  if (pad > 2 || (pad > 0 && non_ws_len % 4 != 0)) {
    return std::nullopt;
  }
  const size_t data_len = non_ws_len - pad;
  const size_t rem = data_len % 4;
  if (rem == 1) {
    return std::nullopt;
  }
  return (data_len / 4) * 3 + (rem == 0 ? 0 : rem - 1);
}

// Computes the decoded byte count of a base64 string without decoding it, or
// returns nullopt if its length or padding is invalid. Uses an O(1) fast path
// when the string has no whitespace and falls back to ignoring ASCII whitespace
// only if needed (matching `absl::Base64Unescape`).
std::optional<size_t> Base64DecodedByteCount(absl::string_view encoded,
                                             size_t expected_bytes) {
  size_t pad = 0;
  while (pad < encoded.size() && pad < 3 &&
         encoded[encoded.size() - 1 - pad] == '=') {
    ++pad;
  }
  if (const std::optional<size_t> fast =
          Base64DecodedLengthFromCounts(encoded.size(), pad);
      fast.has_value() && *fast == expected_bytes) {
    return fast;
  }
  // Whitespace can only make `encoded` longer than a dense base64 string, so
  // scan for whitespace only when `encoded` is at least as long as needed.
  if (encoded.size() < expected_bytes) {
    return std::nullopt;
  }
  size_t non_ws = 0;
  size_t trailing_pad = 0;
  for (char c : encoded) {
    if (absl::ascii_isspace(static_cast<unsigned char>(c))) {
      continue;
    }
    ++non_ws;
    if (c == '=') {
      ++trailing_pad;
    } else if (trailing_pad > 0) {
      return std::nullopt;
    }
  }
  return Base64DecodedLengthFromCounts(non_ws, trailing_pad);
}

// Checks that a raw video frame's `pixels` carries `width * height * 3` bytes,
// without decoding base64.
absl::Status ValidateRawVideoFrameSize(const VideoFrame& frame) {
  const size_t expected_bytes =
      static_cast<size_t>(frame.width) * frame.height * 3;
  const auto& pixels = (*frame.item)["pixels"];
  if (pixels.is_binary()) {
    const size_t actual_bytes = pixels.get_binary().size();
    if (actual_bytes != expected_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Video frame `pixels` carries ", actual_bytes, " bytes, but ",
          frame.width, "x", frame.height, " RGB needs ", expected_bytes, "."));
    }
    return absl::OkStatus();
  }
  const absl::string_view encoded = pixels.get_ref<const std::string&>();
  const std::optional<size_t> decoded_bytes =
      Base64DecodedByteCount(encoded, expected_bytes);
  if (!decoded_bytes.has_value()) {
    return absl::InvalidArgumentError(
        "Video frame `pixels` has invalid base64 length or padding.");
  }
  if (*decoded_bytes != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Video frame `pixels` carries ", *decoded_bytes, " bytes, but ",
        frame.width, "x", frame.height, " RGB needs ", expected_bytes, "."));
  }
  return absl::OkStatus();
}

// Decodes the raw RGB888 `pixels` of a video frame (base64 string or binary
// JSON value) into `frame_image`, reusing its buffer. `base64_scratch` is a
// reused buffer for base64 decoding.
absl::Status DecodeRawVideoFrame(const VideoFrame& frame,
                                 std::string& base64_scratch,
                                 DecodedImage& frame_image) {
  frame_image.width = frame.width;
  frame_image.height = frame.height;
  const auto& pixels = (*frame.item)["pixels"];
  if (pixels.is_binary()) {
    const auto& binary = pixels.get_binary();
    frame_image.pixels.assign(binary.begin(), binary.end());
    return absl::OkStatus();
  }
  if (!absl::Base64Unescape(pixels.get_ref<const std::string&>(),
                            &base64_scratch)) {
    return absl::InvalidArgumentError(
        "Failed to decode base64 video frame `pixels`.");
  }
  frame_image.pixels.assign(base64_scratch.begin(), base64_scratch.end());
  return absl::OkStatus();
}

// Returns an error unless `size` equals `video_size`, which is set to `size`
// for the first frame.
absl::Status CheckVideoFrameSize(const ImageSize& size,
                                 std::optional<ImageSize>& video_size) {
  if (!video_size.has_value()) {
    video_size = size;
  } else if (size.width != video_size->width ||
             size.height != video_size->height) {
    return absl::InvalidArgumentError(
        "All frames of a video must have the same size.");
  }
  return absl::OkStatus();
}

// Returns the tensor map of a patchified image.
absl::StatusOr<const TensorBufferMap*> GetPatchifiedTensorMap(
    const InputImage& image) {
  if (!image.IsTensorBufferMap()) {
    return absl::InvalidArgumentError(
        "Expected a patchified image (a tensor map with `images`).");
  }
  return image.GetPreprocessedImageTensorMap();
}

}  // namespace

absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    const MultimodalPromptProcessingOptions& options) {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  std::deque<std::unique_ptr<MemoryMappedFile>> audio_files;
  // Video items are expanded by `options.video_expander`, so only keep
  // references.
  std::deque<const nlohmann::ordered_json*> video_items;

  // 1. Find all images, audio and videos contained in the messages.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        if (!item.contains("type")) {
          continue;
        }
        if (item["type"] == "video") {
          if (options.video_expander == nullptr ||
              config.video_token_regex.empty()) {
            return absl::UnimplementedError(
                "Video input is not supported by this model.");
          }
          video_items.push_back(&item);
          continue;
        }
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                              LoadItemData(item));
        if (item["type"] == "image") {
          image_files.push_back(std::move(mmap_file));
        } else if (item["type"] == "audio") {
          audio_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  // 2. Setup regex delimiters and matchers.
  RE2 re_delimiter(config.delimiter_regex);
  RE2 re_image(config.image_token_regex);
  RE2 re_audio(config.audio_token_regex);
  RE2 re_video(config.video_token_regex);

  absl::string_view prompt_view = rendered_template_prompt;
  const char* start = prompt_view.data();
  std::string part;

  // 3. Setup image preprocess parameters if preprocessor is present.
  ImagePreprocessParameter img_params;
  if (image_preprocessor != nullptr) {
    if (!options.image_params.has_value()) {
      return absl::FailedPreconditionError(
          "Image preprocessor is configured but no image parameters are "
          "provided.");
    }
    img_params = *options.image_params;
    if (options.image_params->GetPatchifyConfig().has_value()) {
      int max_num_patches =
          options.image_params->GetPatchifyConfig()->max_num_patches;
      if (options.visual_token_budget.has_value()) {
        int budget = options.visual_token_budget.value();
        if (budget <= 0) {
          return absl::InvalidArgumentError(
              "Visual token budget must be positive.");
        }
        // Overwrite the max_num_patches if the visual token budget is provided.
        max_num_patches =
            budget *
            options.image_params->GetPatchifyConfig()->pooling_kernel_size *
            options.image_params->GetPatchifyConfig()->pooling_kernel_size;
      }
      img_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
          .patch_width = options.image_params->GetPatchifyConfig()->patch_width,
          .patch_height =
              options.image_params->GetPatchifyConfig()->patch_height,
          .max_num_patches = max_num_patches,
          .pooling_kernel_size =
              options.image_params->GetPatchifyConfig()->pooling_kernel_size,
          .emit_positions =
              options.image_params->GetPatchifyConfig()->emit_positions,
          .merge_patches =
              options.image_params->GetPatchifyConfig()->merge_patches});
    }
    img_params.SetDryRun(options.dry_run);
  }

  // 4. Parse the rendered prompt and replace placeholders.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();

    if (RE2::FullMatch(part, re_image)) {
      if (image_preprocessor == nullptr) {
        return absl::FailedPreconditionError(
            "Image placeholder found but no image preprocessor is configured.");
      }

      std::string text =
          absl::StrCat(text_part, config.image_prefix, config.boi_token);
      if (!text.empty()) {
        input_data.emplace_back(InputText(std::move(text)));
      }

      if (image_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less images than expected in the prompt.");
      }
      auto image_file = std::move(image_files.front());
      image_files.pop_front();

      ABSL_ASSIGN_OR_RETURN(auto preprocessed_image,
                            image_preprocessor->Preprocess(
                                InputImage(std::string(static_cast<const char*>(
                                                           image_file->data()),
                                                       image_file->length())),
                                img_params));
      input_data.emplace_back(InputImage(std::move(preprocessed_image)));

      if (config.add_image_end) {
        input_data.emplace_back(InputImageEnd());
      }
      if (!config.image_suffix.empty()) {
        input_data.emplace_back(InputText(config.image_suffix));
      }
    } else if (RE2::FullMatch(part, re_audio)) {
      if (audio_preprocessor == nullptr) {
        return absl::FailedPreconditionError(
            "Audio placeholder found but no audio preprocessor is configured.");
      }

      std::string text =
          absl::StrCat(text_part, config.audio_prefix, config.boa_token);
      if (!text.empty()) {
        input_data.emplace_back(InputText(std::move(text)));
      }

      if (audio_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less audio than expected in the prompt.");
      }
      auto audio_file = std::move(audio_files.front());
      audio_files.pop_front();

      InputAudio raw_audio(std::string(
          static_cast<const char*>(audio_file->data()), audio_file->length()));
      ABSL_ASSIGN_OR_RETURN(
          InputAudio preprocessed_audio,
          options.dry_run ? audio_preprocessor->PreprocessDryRun(raw_audio)
                          : audio_preprocessor->Preprocess(raw_audio));
      audio_preprocessor->Reset();

      input_data.emplace_back(std::move(preprocessed_audio));

      if (config.add_audio_end) {
        input_data.emplace_back(InputAudioEnd());
      }
      if (!config.audio_suffix.empty()) {
        input_data.emplace_back(InputText(config.audio_suffix));
      }
    } else if (!video_items.empty() && RE2::FullMatch(part, re_video)) {
      if (image_preprocessor == nullptr) {
        return absl::FailedPreconditionError(
            "Video placeholder found but no image preprocessor is configured.");
      }
      if (!text_part.empty()) {
        input_data.emplace_back(InputText(std::string(text_part)));
      }
      const nlohmann::ordered_json* video_item = video_items.front();
      video_items.pop_front();
      const VideoExpandContext video_context{
          .base_image_params = *options.image_params,
          .visual_token_budget = options.visual_token_budget,
          .dry_run = options.dry_run,
      };
      ABSL_ASSIGN_OR_RETURN(
          std::vector<InputData> video_inputs,
          options.video_expander->ExpandVideo(*video_item, video_context));
      for (auto& video_input : video_inputs) {
        input_data.push_back(std::move(video_input));
      }
    } else if (!config.video_token_regex.empty() &&
               RE2::FullMatch(part, re_video)) {
      return absl::InvalidArgumentError(
          "Provided less videos than expected in the prompt.");
    }
  }

  // 5. Verification.
  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }
  if (!audio_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more audio than expected in the prompt.");
  }
  if (!video_items.empty()) {
    return absl::InvalidArgumentError(
        "Provided more videos than expected in the prompt.");
  }

  // 6. Append remaining prompt text.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }

  return input_data;
}

absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    std::optional<ImagePreprocessParameter> image_params,
    std::optional<int> visual_token_budget, bool dry_run) {
  return ProcessMultimodalPrompt(rendered_template_prompt, messages,
                                 image_preprocessor, audio_preprocessor, config,
                                 MultimodalPromptProcessingOptions{
                                     .image_params = std::move(image_params),
                                     .visual_token_budget = visual_token_budget,
                                     .dry_run = dry_run,
                                 });
}

absl::StatusOr<std::vector<VideoFrame>> LoadVideoFrames(
    const nlohmann::ordered_json& video_item, double default_fps) {
  if (!video_item.contains("frames") || !video_item["frames"].is_array() ||
      video_item["frames"].empty()) {
    return absl::InvalidArgumentError(
        "A video item must have a non-empty `frames` array.");
  }
  double fps = default_fps;
  if (video_item.contains("fps")) {
    if (!video_item["fps"].is_number() ||
        video_item["fps"].get<double>() <= 0) {
      return absl::InvalidArgumentError(
          "Video `fps` must be a positive number.");
    }
    fps = video_item["fps"].get<double>();
  }

  const auto& items = video_item["frames"];
  std::vector<VideoFrame> frames;
  frames.reserve(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (!item.is_object()) {
      return absl::InvalidArgumentError("Each video frame must be an object.");
    }
    VideoFrame frame;
    frame.item = &item;
    if (item.contains("pixels")) {
      if (!item.contains("width") || !item.contains("height") ||
          !item["width"].is_number_integer() ||
          !item["height"].is_number_integer()) {
        return absl::InvalidArgumentError(
            "A video frame with `pixels` must have integer `width` and "
            "`height`.");
      }
      // Checked as int64 before narrowing; the bound also keeps
      // `width * height` within `int` in the resize size computation.
      const int64_t width = item["width"].get<int64_t>();
      const int64_t height = item["height"].get<int64_t>();
      if (width <= 0 || height <= 0 || width > kMaxVideoFrameSide ||
          height > kMaxVideoFrameSide) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Video frame `width` and `height` must be in [1, ",
            kMaxVideoFrameSide, "], got ", width, "x", height, "."));
      }
      if (!item["pixels"].is_string() && !item["pixels"].is_binary()) {
        return absl::InvalidArgumentError(
            "Video frame `pixels` must be a base64 string or binary value.");
      }
      frame.is_raw = true;
      frame.width = static_cast<int>(width);
      frame.height = static_cast<int>(height);
    } else if (!item.contains("path") && !item.contains("blob")) {
      return absl::InvalidArgumentError(
          "A video frame must contain `pixels`, `path` or `blob`.");
    }
    frame.timestamp = static_cast<double>(i) / fps;
    if (item.contains("timestamp")) {
      if (!item["timestamp"].is_number()) {
        return absl::InvalidArgumentError(
            "Video frame `timestamp` must be a number (seconds).");
      }
      frame.timestamp = item["timestamp"].get<double>();
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

absl::StatusOr<std::vector<InputImage>> PreprocessVideoFrames(
    ImagePreprocessor& image_preprocessor,
    const std::vector<VideoFrame>& frames,
    const ImagePreprocessParameter& parameter) {
  std::optional<ImageSize> video_size;
  // Reused across frames so that only one decoded frame is resident at a time
  // and buffers are allocated once per video, not once per frame.
  std::string base64_scratch;
  DecodedImage frame_image;

  std::vector<InputImage> images;
  images.reserve(frames.size());
  for (const VideoFrame& frame : frames) {
    if (frame.item == nullptr) {
      return absl::InvalidArgumentError("Video frame has no JSON item.");
    }
    if (frame.is_raw) {
      ABSL_ASSIGN_OR_RETURN(ImageSize size,
                            ::litert::support::ComputeResizedImageSize(
                                frame.width, frame.height, parameter));
      ABSL_RETURN_IF_ERROR(CheckVideoFrameSize(size, video_size));
      ABSL_RETURN_IF_ERROR(ValidateRawVideoFrameSize(frame));
      if (parameter.IsDryRun()) {
        // Only the output shape matters (e.g. for token counting), so the
        // pixels are not decoded.
        ABSL_ASSIGN_OR_RETURN(
            InputImage image,
            ::litert::support::CreateDryRunImage(
                InputImage(absl::string_view()), size, parameter));
        images.push_back(std::move(image));
        continue;
      }
      ABSL_RETURN_IF_ERROR(
          DecodeRawVideoFrame(frame, base64_scratch, frame_image));
      ABSL_ASSIGN_OR_RETURN(InputImage image, image_preprocessor.Preprocess(
                                                  frame_image, parameter));
      images.push_back(std::move(image));
    } else {
      // Encoded frames are preprocessed straight from the loaded (memory
      // mapped or base64 decoded) bytes, without copying them.
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> bytes,
                            LoadMediaBytes(*frame.item));
      const InputImage encoded(absl::string_view(
          static_cast<const char*>(bytes->data()), bytes->length()));
      ABSL_ASSIGN_OR_RETURN(
          ImageSize size,
          image_preprocessor.GetResizedImageSize(encoded, parameter));
      ABSL_RETURN_IF_ERROR(CheckVideoFrameSize(size, video_size));
      ABSL_ASSIGN_OR_RETURN(InputImage image,
                            image_preprocessor.Preprocess(encoded, parameter));
      images.push_back(std::move(image));
    }
  }
  return images;
}

absl::StatusOr<InputImage> CreateTemporalPairImage(const InputImage& frame_0,
                                                   const InputImage& frame_1) {
  ABSL_ASSIGN_OR_RETURN(const TensorBufferMap* tensors_0,
                        GetPatchifiedTensorMap(frame_0));
  ABSL_ASSIGN_OR_RETURN(const TensorBufferMap* tensors_1,
                        GetPatchifiedTensorMap(frame_1));
  const auto images_1 = tensors_1->find(kVisionImagesInput);
  if (images_1 == tensors_1->end()) {
    return absl::InvalidArgumentError("Preprocessed frame has no `images`.");
  }
  TensorBufferMap pair;
  pair.reserve(tensors_0->size() + 1);
  for (const auto& [name, buffer] : *tensors_0) {
    LITERT_ASSIGN_OR_RETURN(auto shared, buffer.Duplicate());
    pair.insert_or_assign(name, std::move(shared));
  }
  LITERT_ASSIGN_OR_RETURN(auto shared_images_1, images_1->second.Duplicate());
  pair.insert_or_assign(kVisionImagesT1Input, std::move(shared_images_1));
  return InputImage(std::move(pair));
}

absl::Status AddRepeatedTemporalFrame(std::vector<InputData>& input_data) {
  for (auto& input : input_data) {
    auto* image = std::get_if<InputImage>(&input);
    if (image == nullptr || !image->IsTensorBufferMap()) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(const TensorBufferMap* tensors,
                          image->GetPreprocessedImageTensorMap());
    if (tensors->contains(kVisionImagesT1Input) ||
        !tensors->contains(kVisionImagesInput)) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(InputImage repeated,
                          CreateTemporalPairImage(*image, *image));
    input = std::move(repeated);
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
