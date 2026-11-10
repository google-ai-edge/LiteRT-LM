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

#include "runtime/executor/gpu_artisan/gpu_artisan_executor_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace odml::infra {

absl::StatusOr<ml_drift::TensorFloat32>
CreateMLDriftTensorEmbeddingFromLitertTensorBuffer(
    const ::litert::TensorBuffer& litert_embedding) {
  ml_drift::TensorFloat32 embedding;
  LITERT_ASSIGN_OR_RETURN(
      embedding.spanned_data,
      ::litert::lm::ReferTensorBufferAsSpan<float>(litert_embedding));
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, litert_embedding.TensorType());
  auto embedding_shape = tensor_type.Layout().Dimensions();
  if (embedding_shape.size() != 3 && embedding_shape.size() != 4) {
    return absl::InvalidArgumentError(
        "The embedding tensor should have either 3 dimensions [batch_size, "
        "tokens_num, embedding_dim] or 4 dimensions [batch_size, 1, "
        "tokens_num, embedding_dim], but got " +
        std::to_string(embedding_shape.size()));
  }
  if (embedding_shape.size() == 3) {
    // If the embedding tensor has 3 dimensions, add a 1 to the shape to make
    // it a 4-D tensor with shape [batch_size, 1, tokens_num, embedding_dim].
    embedding.shape = ml_drift::BHWC(embedding_shape[0], 1, embedding_shape[1],
                                     embedding_shape[2]);
  } else {
    embedding.shape = ml_drift::BHWC(embedding_shape[0], embedding_shape[1],
                                     embedding_shape[2], embedding_shape[3]);
  }

  return std::move(embedding);
}

absl::StatusOr<std::vector<ml_drift::TensorFloat32>>
CreateMLDriftTensorPLEFromLitertTensorBuffer(
    const ::litert::TensorBuffer& litert_ple) {
  LITERT_ASSIGN_OR_RETURN(const auto tensor_type, litert_ple.TensorType());
  const absl::Span<const int32_t> dimensions =
      tensor_type.Layout().Dimensions();
  if (dimensions.size() != 4) {
    return absl::InvalidArgumentError(
        "The PLE tensor should have 4 dimensions.");
  }

  std::vector<ml_drift::TensorFloat32> ple_vec(dimensions[1]);
  const size_t ple_size = dimensions[2] * dimensions[3];
  LITERT_ASSIGN_OR_RETURN(
      auto all_ple, ::litert::lm::ReferTensorBufferAsSpan<float>(litert_ple));
  for (int i = 0; i < ple_vec.size(); ++i) {
    ple_vec[i].spanned_data =
        absl::MakeSpan(all_ple).subspan(i * ple_size, ple_size);
    // Noticed the number of layer is 1 instead of dimensions[1] because each
    // tensor if for one layer, instead of the total number of layers.
    ple_vec[i].shape =
        ml_drift::BHWC(dimensions[0], 1, dimensions[2], dimensions[3]);
  }

  return std::move(ple_vec);
}

std::vector<ImageTokenInfo> GetImageTokenInfos(absl::Span<const int> tokens,
                                               int vision_token_id,
                                               int vision_end_token_id) {
  std::vector<ImageTokenInfo> infos;
  int current_count = 0;
  bool in_image = false;
  for (const int token : tokens) {
    if (token == vision_token_id) {
      in_image = true;
      current_count++;
    } else {
      if (in_image) {
        ImageTokenInfo info;
        info.active_tokens = current_count;
        info.has_eoi = (token == vision_end_token_id);
        infos.push_back(info);
        current_count = 0;
        in_image = false;
      }
    }
  }
  if (in_image) {
    ImageTokenInfo info;
    info.active_tokens = current_count;
    info.has_eoi = false;
    infos.push_back(info);
  }
  return infos;
}

bool PrefillChunkRange::HasAudio() const {
  return audio_token_end > audio_token_begin;
}

namespace {

// Returns the maximum number of images in a chunk with an audio clip of
// `num_audio_tokens` tokens.
int MaxImagesWithAudio(int max_images_per_chunk, int vision_tokens_per_image,
                       int max_num_audio_tokens, int num_audio_tokens) {
  const int room = max_images_per_chunk * vision_tokens_per_image +
                   max_num_audio_tokens - num_audio_tokens;
  return room > 0
             ? std::min(max_images_per_chunk, room / vision_tokens_per_image)
             : 0;
}

// Returns the number of consecutive `token_id` tokens in `tokens` starting at
// `begin`.
int RunLength(absl::Span<const int> tokens, int begin, int token_id) {
  int end = begin;
  while (end < tokens.size() && tokens[end] == token_id) {
    ++end;
  }
  return end - begin;
}

}  // namespace

std::vector<PrefillChunkRange> SplitTokensForPrefill(
    absl::Span<const int> tokens, int vision_token_id, int audio_token_id,
    int audio_end_token_id, int max_images_per_chunk,
    int vision_tokens_per_image, int max_num_audio_tokens) {
  std::vector<PrefillChunkRange> chunks;
  PrefillChunkRange current;
  // The number of media embeddings of the audio clip in `current`, including
  // its end-of-audio token.
  int current_audio_media_tokens = 0;
  int num_images = 0;
  int num_audio_tokens = 0;
  int i = 0;
  while (i < tokens.size()) {
    if (tokens[i] == vision_token_id) {
      // An image starts here.
      const int images_in_chunk = current.image_end - current.image_begin;
      const int max_images =
          current.HasAudio()
              ? MaxImagesWithAudio(
                    max_images_per_chunk, vision_tokens_per_image,
                    max_num_audio_tokens, current_audio_media_tokens)
              : max_images_per_chunk;
      if (max_images_per_chunk > 0 && images_in_chunk + 1 > max_images &&
          i > current.token_begin) {
        current.token_end = i;
        chunks.push_back(current);
        current = PrefillChunkRange{.token_begin = i,
                                    .image_begin = num_images,
                                    .image_end = num_images,
                                    .audio_token_begin = num_audio_tokens,
                                    .audio_token_end = num_audio_tokens};
      }
      ++num_images;
      current.image_end = num_images;
      i += RunLength(tokens, i, vision_token_id);
    } else if (tokens[i] == audio_token_id) {
      // An audio clip starts here.
      const int clip_tokens = RunLength(tokens, i, audio_token_id);
      const bool has_end_token = i + clip_tokens < tokens.size() &&
                                 tokens[i + clip_tokens] == audio_end_token_id;
      const int clip_media_tokens = clip_tokens + (has_end_token ? 1 : 0);
      const int images_in_chunk = current.image_end - current.image_begin;
      const bool exceeds_images =
          max_images_per_chunk > 0 &&
          images_in_chunk >
              MaxImagesWithAudio(max_images_per_chunk, vision_tokens_per_image,
                                 max_num_audio_tokens, clip_media_tokens);
      if ((current.HasAudio() || exceeds_images) && i > current.token_begin) {
        current.token_end = i;
        chunks.push_back(current);
        current = PrefillChunkRange{.token_begin = i,
                                    .image_begin = num_images,
                                    .image_end = num_images,
                                    .audio_token_begin = num_audio_tokens,
                                    .audio_token_end = num_audio_tokens};
      }
      current.audio_token_begin = num_audio_tokens;
      num_audio_tokens += clip_tokens;
      current.audio_token_end = num_audio_tokens;
      current_audio_media_tokens = clip_media_tokens;
      i += clip_media_tokens;
    } else {
      ++i;
    }
  }
  current.token_end = tokens.size();
  chunks.push_back(current);
  return chunks;
}

proto::SamplerParameters ConvertToOdmlSamplerParams(
    const ::litert::lm::proto::SamplerParameters& litert_sampler_params) {
  proto::SamplerParameters odml_sampler_params;
  switch (litert_sampler_params.type()) {
    case ::litert::lm::proto::SamplerParameters::TOP_K:
      odml_sampler_params.set_type(proto::SamplerParameters::TOP_K);
      break;
    case ::litert::lm::proto::SamplerParameters::TOP_P:
      odml_sampler_params.set_type(proto::SamplerParameters::TOP_P);
      break;
    case ::litert::lm::proto::SamplerParameters::GREEDY:
      odml_sampler_params.set_type(proto::SamplerParameters::GREEDY);
      break;
    default:
      odml_sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
      break;
  }
  odml_sampler_params.set_k(litert_sampler_params.k());
  odml_sampler_params.set_p(litert_sampler_params.p());
  odml_sampler_params.set_temperature(litert_sampler_params.temperature());
  if (litert_sampler_params.has_seed()) {
    odml_sampler_params.set_seed(litert_sampler_params.seed());
  }
  return odml_sampler_params;
}

}  // namespace odml::infra
