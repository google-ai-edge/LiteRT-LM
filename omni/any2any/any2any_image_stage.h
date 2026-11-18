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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_IMAGE_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_IMAGE_STAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::omni::any2any {

// Concrete vision encoder stage shared across Any2Any model families:
// - Consumes `Sequenced<ImageInput>` from `Any2AnyInputFanOutStage`.
// - Emits an empty `Any2AnyImageEmbeddings{}` for empty `ImageInput` items.
// - Validates raw RGB888/RGBA8888/grayscale buffers (passing a zero-copy
//   `RgbImageView` directly over `ImageInput::image_bytes` for raw RGB888) or
//   decodes compressed image bytes (PNG/JPEG/BMP) via `lm::ImagePreprocessor`.
// - Delegates model-specific tensor preprocessing and encoder invocation to the
//   configured `EncodeImageFn`, outputting `Sequenced<Any2AnyImageEmbeddings>`.
//   Model-specific image stages wrap this stage to frame the emitted embeddings
//   into their model-specific embedding segments.
class Any2AnyImageStage
    : public SingleThreadedStageWithDeque<Sequenced<Any2AnyImageEmbeddings>> {
 public:
  // Configuration for `Any2AnyImageStage`, shared across Any2Any model families
  // (e.g., MiniCPM-o, Qwen2.5-Omni).
  struct Options {
    // Hidden dimension of the downstream LLM embedding space.
    int llm_hidden_size = 0;
  };

  // Non-owning view of interleaved RGB888 pixel bytes (`height * width * 3`).
  struct RgbImageView {
    absl::Span<const uint8_t> pixels;
    int width = 0;
    int height = 0;
  };

  // Executes the underlying vision encoder runner with the already-populated
  // input buffers and returns flattened `[NumOutputTokens(), HiddenDim()]`
  // float32 embeddings from the first output buffer.
  using RunEncoderFn = absl::FunctionRef<absl::StatusOr<std::vector<float>>()>;

  // Preprocesses `image` into `input_buffers`, invokes `run_encoder` as needed,
  // and returns the resulting `Any2AnyImageEmbeddings`.
  using EncodeImageFn =
      absl::AnyInvocable<absl::StatusOr<Any2AnyImageEmbeddings>(
          const RgbImageView& image, absl::Span<TensorBuffer> input_buffers,
          RunEncoderFn run_encoder)>;

  // Validates model-specific input tensor buffers during `Create()` and returns
  // the per-image `EncodeImageFn` callback.
  using InitEncoderFn = absl::FunctionRef<absl::StatusOr<EncodeImageFn>(
      absl::Span<const TensorBuffer> input_buffers, int hidden_dim,
      int num_output_tokens)>;

  // Creates an `Any2AnyImageStage` with the default
  // `lm::ImagePreprocessor::Create()` decoder.
  static absl::StatusOr<std::unique_ptr<Any2AnyImageStage>> Create(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      const Options& options, InitEncoderFn init_encoder);

  // Creates an `Any2AnyImageStage` with an explicit `image_preprocessor`
  // (e.g., for testing custom or unavailable compressed-image decoders).
  static absl::StatusOr<std::unique_ptr<Any2AnyImageStage>> Create(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      const Options& options, InitEncoderFn init_encoder,
      std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor);

  ~Any2AnyImageStage() override;

  int HiddenDim() const { return hidden_dim_; }
  int NumOutputTokens() const { return num_output_tokens_; }

  // Validates and resolves raw RGB/RGBA/grayscale pixel buffers or decodes
  // compressed image bytes (`width == 0 && height == 0`) into an interleaved
  // RGB888 view.
  //
  // For raw RGB888 inputs (`channels == 3`), returns a zero-copy view directly
  // referencing `image.image_bytes` without modifying `owned_storage`. For
  // RGBA, grayscale, or compressed inputs, stores the converted/decoded bytes
  // in `owned_storage` and returns a view referencing `owned_storage`.
  absl::StatusOr<RgbImageView> DecodeImageToRgb(
      const ImageInput& image, std::vector<uint8_t>& owned_storage) const;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  Any2AnyImageStage(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      std::vector<TensorBuffer> vision_input_buffers,
      std::vector<TensorBuffer> vision_output_buffers, int hidden_dim,
      int num_output_tokens, EncodeImageFn encode_image);

  absl::StatusOr<std::vector<float>> RunVisionEncoder();

  FanOutOutputQueue<OmniSession::Input, ImageInput>& image_queue_;
  int hidden_dim_;
  int num_output_tokens_;
  std::unique_ptr<lm::ImagePreprocessor> image_preprocessor_;
  std::unique_ptr<LiteRtRunner> vision_encoder_runner_;
  std::vector<TensorBuffer> vision_input_buffers_;
  std::vector<TensorBuffer> vision_output_buffers_;
  EncodeImageFn encode_image_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_IMAGE_STAGE_H_
