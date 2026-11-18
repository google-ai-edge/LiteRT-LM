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

#include "omni/any2any/any2any_image_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::omni::any2any {

Any2AnyImageStage::Any2AnyImageStage(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    std::vector<TensorBuffer> vision_input_buffers,
    std::vector<TensorBuffer> vision_output_buffers, int hidden_dim,
    int num_output_tokens, EncodeImageFn encode_image)
    : image_queue_(*image_queue),
      hidden_dim_(hidden_dim),
      num_output_tokens_(num_output_tokens),
      image_preprocessor_(std::move(image_preprocessor)),
      vision_encoder_runner_(std::move(vision_encoder_runner)),
      vision_input_buffers_(std::move(vision_input_buffers)),
      vision_output_buffers_(std::move(vision_output_buffers)),
      encode_image_(std::move(encode_image)) {}

Any2AnyImageStage::~Any2AnyImageStage() = default;

absl::StatusOr<std::unique_ptr<Any2AnyImageStage>> Any2AnyImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const Options& options, InitEncoderFn init_encoder) {
  return Create(image_queue, std::move(vision_encoder_runner), options,
                init_encoder, lm::ImagePreprocessor::Create());
}

absl::StatusOr<std::unique_ptr<Any2AnyImageStage>> Any2AnyImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const Options& options, InitEncoderFn init_encoder,
    std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor) {
  if (options.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyImageStage llm_hidden_size must be positive.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> vision_input_buffers,
                        vision_encoder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> vision_output_buffers,
                        vision_encoder_runner->CreateOutputBuffers(""));
  if (vision_input_buffers.empty() || vision_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "Any2AnyImageStage vision_encoder_runner must provide non-empty "
        "input and output buffers.");
  }

  const size_t hidden_bytes =
      static_cast<size_t>(options.llm_hidden_size) * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(size_t vision_out_bytes,
                          vision_output_buffers[0].Size());
  if (vision_out_bytes == 0 || vision_out_bytes % hidden_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyImageStage vision output buffer size (", vision_out_bytes,
        " bytes) must be a positive multiple of llm_hidden_size * "
        "sizeof(float) (",
        hidden_bytes, " bytes)."));
  }
  const int num_output_tokens =
      static_cast<int>(vision_out_bytes / hidden_bytes);

  ABSL_ASSIGN_OR_RETURN(
      EncodeImageFn encode_image,
      init_encoder(vision_input_buffers, options.llm_hidden_size,
                   num_output_tokens));
  if (!encode_image) {
    return absl::InvalidArgumentError(
        "Any2AnyImageStage encoder callback must not be null.");
  }

  return absl::WrapUnique(new Any2AnyImageStage(
      image_queue, std::move(image_preprocessor),
      std::move(vision_encoder_runner), std::move(vision_input_buffers),
      std::move(vision_output_buffers), options.llm_hidden_size,
      num_output_tokens, std::move(encode_image)));
}

bool Any2AnyImageStage::NeedScheduleInternal() const {
  return !image_queue_.Empty();
}

absl::StatusOr<Any2AnyImageStage::RgbImageView>
Any2AnyImageStage::DecodeImageToRgb(const ImageInput& image,
                                    std::vector<uint8_t>& owned_storage) const {
  if (image.image_bytes.empty()) {
    return absl::InvalidArgumentError(
        "Any2AnyImageStage ImageInput image_bytes must not be empty.");
  }
  if (image.width < 0 || image.height < 0 ||
      (image.width > 0) != (image.height > 0)) {
    return absl::InvalidArgumentError(
        "ImageInput width and height must both be positive for raw pixels or "
        "both be zero for encoded images.");
  }

  if (image.width > 0 && image.height > 0) {
    if (image.channels != 1 && image.channels != 3 && image.channels != 4) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Any2AnyImageStage raw ImageInput channels must be 1, 3, or 4, got ",
          image.channels, "."));
    }
    const uint64_t num_pixels =
        static_cast<uint64_t>(image.width) * image.height;
    const uint64_t expected_bytes = num_pixels * image.channels;
    if (static_cast<uint64_t>(image.image_bytes.size()) != expected_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Any2AnyImageStage raw ImageInput byte size (",
          image.image_bytes.size(), ") does not match width*height*channels (",
          expected_bytes, ")."));
    }

    if (image.channels == 3) {
      return RgbImageView{
          .pixels = absl::MakeConstSpan(image.image_bytes),
          .width = image.width,
          .height = image.height,
      };
    }

    if (image.channels == 4) {
      owned_storage.clear();
      owned_storage.reserve(static_cast<size_t>(num_pixels) * 3);
      for (size_t i = 0; i < image.image_bytes.size(); i += 4) {
        owned_storage.push_back(image.image_bytes[i]);
        owned_storage.push_back(image.image_bytes[i + 1]);
        owned_storage.push_back(image.image_bytes[i + 2]);
      }
      return RgbImageView{
          .pixels = absl::MakeConstSpan(owned_storage),
          .width = image.width,
          .height = image.height,
      };
    }

    if (num_pixels > std::numeric_limits<size_t>::max() / 3) {
      return absl::InvalidArgumentError(
          "Any2AnyImageStage grayscale ImageInput is too large to expand to "
          "RGB888.");
    }
    owned_storage.clear();
    owned_storage.reserve(static_cast<size_t>(num_pixels) * 3);
    for (uint8_t g : image.image_bytes) {
      owned_storage.push_back(g);
      owned_storage.push_back(g);
      owned_storage.push_back(g);
    }
    return RgbImageView{
        .pixels = absl::MakeConstSpan(owned_storage),
        .width = image.width,
        .height = image.height,
    };
  }

  if (image_preprocessor_ == nullptr) {
    return absl::FailedPreconditionError(
        "Any2AnyImageStage requires an ImagePreprocessor to decode compressed "
        "image bytes.");
  }
  const absl::string_view encoded(
      reinterpret_cast<const char*>(image.image_bytes.data()),
      image.image_bytes.size());
  ABSL_ASSIGN_OR_RETURN(lm::DecodedImage decoded,
                        image_preprocessor_->Decode(encoded));
  if (decoded.width <= 0 || decoded.height <= 0 ||
      static_cast<uint64_t>(decoded.pixels.size()) !=
          static_cast<uint64_t>(decoded.width) * decoded.height * 3) {
    return absl::InternalError(absl::StrCat(
        "Any2AnyImageStage decoded image has inconsistent dimensions (",
        decoded.width, "x", decoded.height, ") and byte size (",
        decoded.pixels.size(), ")."));
  }
  owned_storage = std::move(decoded.pixels);
  return RgbImageView{
      .pixels = absl::MakeConstSpan(owned_storage),
      .width = decoded.width,
      .height = decoded.height,
  };
}

absl::StatusOr<std::vector<float>> Any2AnyImageStage::RunVisionEncoder() {
  ABSL_RETURN_IF_ERROR(vision_encoder_runner_->Run("", vision_input_buffers_,
                                                   vision_output_buffers_));
  const size_t num_floats =
      static_cast<size_t>(num_output_tokens_) * hidden_dim_;
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      TensorBufferScopedLock::Create<const float>(
          vision_output_buffers_[0], TensorBuffer::LockMode::kRead));
  const float* src = lock_and_addr.second;
  return std::vector<float>(src, src + num_floats);
}

absl::Status Any2AnyImageStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("Image queue has no output.");
  }

  while (NeedScheduleInternal()) {
    ABSL_ASSIGN_OR_RETURN(const Sequenced<ImageInput>* sequenced_input,
                          image_queue_.PeekFront());
    const uint64_t seqnum = sequenced_input->seqnum;
    if (sequenced_input->item.image_bytes.empty()) {
      ABSL_RETURN_IF_ERROR(image_queue_.PopFront().status());
      PushOutput(Sequenced<Any2AnyImageEmbeddings>{
          .seqnum = seqnum,
          .item = Any2AnyImageEmbeddings{},
      });
      continue;
    }
    std::vector<uint8_t> owned_rgb;
    ABSL_ASSIGN_OR_RETURN(const RgbImageView rgb_view,
                          DecodeImageToRgb(sequenced_input->item, owned_rgb));
    ABSL_ASSIGN_OR_RETURN(
        Any2AnyImageEmbeddings embeddings,
        encode_image_(rgb_view, absl::MakeSpan(vision_input_buffers_),
                      [this]() { return RunVisionEncoder(); }));
    ABSL_RETURN_IF_ERROR(image_queue_.PopFront().status());
    PushOutput(Sequenced<Any2AnyImageEmbeddings>{
        .seqnum = seqnum,
        .item = std::move(embeddings),
    });
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
