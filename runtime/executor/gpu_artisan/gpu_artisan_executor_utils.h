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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_GPU_ARTISAN_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_GPU_ARTISAN_EXECUTOR_UTILS_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace odml::infra {

// Creates a ml_drift::TensorFloat32 embedding tensor from a
// litert::TensorBuffer.
//
// The input litert::TensorBuffer is expected to have the shape
// [batch_size, 1, tokens_num, embedding_dim] or [batch_size, tokens_num,
// embedding_dim].
//
// The output ml_drift::TensorFloat32 will have the shape
// [batch_size, 1, tokens_num, embedding_dim].
//
// Note: currently only support batch_size of 1.
absl::StatusOr<ml_drift::TensorFloat32>
CreateMLDriftTensorEmbeddingFromLitertTensorBuffer(
    const ::litert::TensorBuffer& embedding);

// Creates a per-layer embedding vector of ml_drift::TensorFloat32 tensors
// from a litert::TensorBuffer.
//
// The input litert::TensorBuffer is expected to have the shape:
// [batch_size, layer_num, tokens_num, per_layer_embedding_dim].
//
// The output is a std::vector of ml_drift::TensorFloat32 tensors, where:
// - The vector's size is equal to layer_num.
// - Each ml_drift::TensorFloat32 in the vector has the shape:
//   [batch_size, 1, tokens_num, per_layer_embedding_dim].
//
// Note: Currently, only a batch_size of 1 is supported.
absl::StatusOr<std::vector<ml_drift::TensorFloat32>>
CreateMLDriftTensorPLEFromLitertTensorBuffer(const ::litert::TensorBuffer& ple);

struct ImageTokenInfo {
  // The number of active tokens in the image, e.g., the number of kSpecialToken
  // tokens.
  int active_tokens = 0;
  // Whether the image has an end of image token, e.g., kEnd.
  bool has_eoi = false;

  int total_tokens() const { return active_tokens + (has_eoi ? 1 : 0); }
};

// Count and parse the vision tokens for each image in a sequence of tokens.
std::vector<ImageTokenInfo> GetImageTokenInfos(absl::Span<const int> tokens,
                                               int vision_token_id,
                                               int vision_end_token_id);

// A contiguous range of input tokens that is prefilled with a single call to
// the GPU runner.
struct PrefillChunkRange {
  // The token range [token_begin, token_end) of the chunk in the input tokens.
  int token_begin = 0;
  int token_end = 0;
  // The image range [image_begin, image_end) of the chunk, indexing the images
  // returned by GetImageTokenInfos() for the same input tokens.
  int image_begin = 0;
  int image_end = 0;
  // The audio token range [audio_token_begin, audio_token_end) of the chunk,
  // counting only the audio tokens of the input tokens, i.e. indexing the rows
  // of the audio embeddings of all the audio clips in the input. Empty if the
  // chunk has no audio.
  int audio_token_begin = 0;
  int audio_token_end = 0;

  // Returns whether the chunk has an audio clip.
  bool HasAudio() const;
};

// Splits `tokens` into contiguous chunks that are prefilled one after another,
// so that the media embeddings of each chunk fit in the media embedding cache:
//
// * An image is a run of consecutive `vision_token_id` tokens and an audio clip
//   is a run of consecutive `audio_token_id` tokens, plus the
//   `audio_end_token_id` token right after it, if any. Neither is ever split
//   across chunks.
// * Each chunk has at most one audio clip. The media embeddings of an audio
//   clip (one per token) are stored in the media embedding cache after the
//   images of its chunk, which has room for `max_images_per_chunk` images
//   followed by `max_num_audio_tokens` audio embeddings.
// * A chunk without audio has at most `max_images_per_chunk` images. A chunk
//   with an audio clip of N tokens has at most
//   (max_images_per_chunk * vision_tokens_per_image + max_num_audio_tokens -
//   N) / vision_tokens_per_image images (and at most `max_images_per_chunk`),
//   so that the images and the audio fit in the media embedding cache.
//
// A new chunk is started at the first token of the image or audio clip that
// would exceed these limits, so the tokens in between stay in the previous
// chunk.
//
// args:
//   tokens: The input token ids.
//   vision_token_id: The token id that marks a vision token.
//   audio_token_id: The token id that marks an audio token.
//   audio_end_token_id: The token id that marks the end of an audio clip.
//   max_images_per_chunk: The maximum number of images in a chunk. If it is not
//     positive, the number of images is not limited.
//   vision_tokens_per_image: The number of tokens of an image slot in the media
//     embedding cache. Must be positive if `max_images_per_chunk` is positive.
//   max_num_audio_tokens: The number of audio embeddings the media embedding
//     cache has room for after the slots of `max_images_per_chunk` images.
// returns:
//   The chunks in order. They cover all of `tokens`, and there is always at
//   least one chunk, even if `tokens` is empty.
std::vector<PrefillChunkRange> SplitTokensForPrefill(
    absl::Span<const int> tokens, int vision_token_id, int audio_token_id,
    int audio_end_token_id, int max_images_per_chunk,
    int vision_tokens_per_image, int max_num_audio_tokens);

// Converts the litert::lm::proto::SamplerParameters to
// odml::infra::proto::SamplerParameters.
proto::SamplerParameters ConvertToOdmlSamplerParams(
    const ::litert::lm::proto::SamplerParameters& litert_sampler_params);

}  // namespace odml::infra

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_GPU_ARTISAN_EXECUTOR_UTILS_H_
