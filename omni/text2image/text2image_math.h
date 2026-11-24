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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_MATH_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_MATH_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::text2image {

// Samples `count` standard normal N(0, 1) floats using `std::mt19937_64(seed)`.
//
// args
// - count: Number of latent elements to sample.
// - seed: Deterministic PRNG seed.
//
// returns
// - Vector of `count` sampled floats.
std::vector<float> SampleGaussianLatents(size_t count, uint64_t seed);

// Converts planar NCHW float image output of shape `[1, 3, height, width]` in
// `[-1, 1]` to interleaved RGB888 bytes of shape `[height, width, 3]` via
// `round(clamp(y * 0.5 + 0.5, 0.0, 1.0) * 255.0)`. Non-finite `NaN` values map
// safely to `0`; `+/-Inf` clamp to `255`/`0`.
//
// args
// - nchw: Planar NCHW float buffer of length `3 * height * width`.
// - height: Positive image height in pixels.
// - width: Positive image width in pixels.
//
// returns
// - Interleaved RGB888 bytes of length `3 * height * width`, or
//   `absl::InvalidArgumentError` on invalid dimensions or buffer length.
absl::StatusOr<std::vector<uint8_t>> ConvertNchwToRgb888(
    absl::Span<const float> nchw, int height, int width);

// Computes multi-axis rotary positional embeddings `(cos, sin)` of shape
// `[1, num_tokens, 1, sum(axes_dim)]` from `joint_ids` of shape
// `[num_tokens, axes_dim.size()]`, using `repeat_interleave` by 2 along each
// axis (matching Diffusers `Flux2PosEmbed` / `ZImageTransformer2DModel`).
//
// args
// - joint_ids: Flattened position IDs of length `num_tokens * axes_dim.size()`.
// - num_tokens: Positive number of tokens.
// - axes_dim: Positive even rotary dimension per axis.
// - theta: Positive finite RoPE base frequency parameter.
//
// returns
// - Pair of `(cos, sin)` flattened float vectors, each of length
//   `num_tokens * sum(axes_dim)`, or `absl::InvalidArgumentError` on invalid
//   inputs.
absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildMultiAxisRotaryPosEmbed(absl::Span<const float> joint_ids, int num_tokens,
                             absl::Span<const int> axes_dim,
                             float theta = 2000.0f);

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_MATH_H_
