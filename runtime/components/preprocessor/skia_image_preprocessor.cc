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

#include "runtime/components/preprocessor/skia_image_preprocessor.h"

#include <cstddef>
#include <memory>
#include <tuple>  // IWYU pragma: keep
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "include/codec/SkBmpDecoder.h"  // from @skia
#include "include/codec/SkCodec.h"  // from @skia
#include "include/codec/SkEncodedOrigin.h"  // from @skia
#include "include/codec/SkPixmapUtils.h"  // from @skia
#include "include/core/SkAlphaType.h"  // from @skia
#include "include/core/SkBitmap.h"  // from @skia
#include "include/core/SkColor.h"  // from @skia
#include "include/core/SkImage.h"  // from @skia
#include "include/core/SkImageInfo.h"  // from @skia
#include "include/core/SkRefCnt.h"  // from @skia
#include "include/core/SkSamplingOptions.h"  // from @skia

namespace litert::lm {

std::unique_ptr<ImagePreprocessor> ImagePreprocessor::Create() {
  return std::make_unique<SkiaImagePreprocessor>();
}

namespace {

absl::StatusOr<sk_sp<SkImage>> DecodeDataAsImage(sk_sp<SkData> data) {
  if (!data) {
    return absl::InvalidArgumentError("Null data.");
  }

  std::unique_ptr<SkCodec> codec;
  if (SkBmpDecoder::IsBmp(data->bytes(), data->size())) {
    codec = SkBmpDecoder::Decode(data, nullptr);
  }

  if (!codec) {
    return absl::InvalidArgumentError("Failed to decode image.");
  }

  // Using premul makes the image filtering look better in many cases.
  SkImageInfo info = codec->getInfo().makeAlphaType(kPremul_SkAlphaType);
  // If the image is rotated due to Exif or similar metadata, we need to
  // rotate the dimensions provided by the codec.
  if (SkEncodedOriginSwapsWidthHeight(codec->getOrigin())) {
    info = SkPixmapUtils::SwapWidthHeight(info);
  }
  auto [image, result] = codec->getImage(info);
  if (result != SkCodec::Result::kSuccess || image == nullptr) {
    return absl::InvalidArgumentError("Failed to create SkImage.");
  }
  return image;
}

}  // namespace

absl::StatusOr<InputImage> SkiaImagePreprocessor::Preprocess(
    const InputImage& input_image, const ImagePreprocessParameter& parameter) {
  if (input_image.IsTensorBuffer()) {
    ASSIGN_OR_RETURN(auto processed_image_tensor,
                     input_image.GetPreprocessedImageTensor());
    LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                            processed_image_tensor->Duplicate());
    InputImage processed_image(
        std::move(processed_image_tensor_with_reference));
    return processed_image;
  }

  if (parameter.GetPatchifyConfig().has_value()) {
    // TODO(b/515828541): Support patchify in SkiaImagePreprocessor.
    return absl::UnimplementedError(
        "Patchify is not supported in SkiaImagePreprocessor yet.");
  }

  const Dimensions& target_dimensions = parameter.GetTargetDimensions();

  if (target_dimensions.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Target dimensions must be (batch, height, width, "
                     "channels). Got dimensions size: ",
                     target_dimensions.size()));
  }

  ASSIGN_OR_RETURN(absl::string_view image_bytes,
                   input_image.GetRawImageBytes());
  sk_sp<SkData> image_data =
      SkData::MakeWithoutCopy(image_bytes.data(), image_bytes.size());
  ASSIGN_OR_RETURN(sk_sp<SkImage> image, DecodeDataAsImage(image_data));

  const int batch_size = target_dimensions.at(0);
  const int target_height = target_dimensions.at(1);
  const int target_width = target_dimensions.at(2);
  const int target_channels = target_dimensions.at(3);

  // Resize the image to the target size.
  SkImageInfo image_info =
      SkImageInfo::MakeN32(target_width, target_height, image->alphaType());

  const SkSamplingOptions options(SkCubicResampler::Mitchell());
  sk_sp<SkImage> scaled_image = image->makeScaled(image_info, options);
  if (!scaled_image) {
    return absl::InvalidArgumentError("Failed to scale image.");
  }

  SkPixmap resized_pixels;
  if (!scaled_image->peekPixels(&resized_pixels)) {
    return absl::InvalidArgumentError("Failed to peek pixels.");
  }

  const int num_elements =
      batch_size * target_height * target_width * target_channels;
  const size_t buffer_size = num_elements * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<float>(
              {batch_size, target_height, target_width, target_channels}),
          buffer_size));

  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          processed_tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* float_buffer_ptr =
      reinterpret_cast<float*>(processed_tensor_lock_and_addr.second);

  // Load pixels into the output buffer. The output buffer expects a specific
  // format, and in the case of 3 input channels, 24 bytes per pixel. Skia does
  // not support 24bpp natively, so this is somewhat manual.
  for (int h = 0; h < resized_pixels.height(); ++h) {
    for (int w = 0; w < resized_pixels.width(); ++w) {
      SkColor pixel = resized_pixels.getColor(w, h);
      float_buffer_ptr[0] = static_cast<float>(SkColorGetR(pixel)) / 255.0f;
      float_buffer_ptr[1] = static_cast<float>(SkColorGetG(pixel)) / 255.0f;
      float_buffer_ptr[2] = static_cast<float>(SkColorGetB(pixel)) / 255.0f;
      if (target_channels == 4) {
        float_buffer_ptr[3] = static_cast<float>(SkColorGetA(pixel)) / 255.0f;
      }
      float_buffer_ptr += target_channels;
    }
  }

  InputImage processed_image(std::move(processed_tensor_buffer));

  return processed_image;
}

}  // namespace litert::lm
