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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::text2image {

// Computes the empirical FLUX.2-klein FlowMatch-Euler sigma schedule of length
// `steps + 1` (ending with `0.0f`) as a function of `steps` and `image_tokens`
// (`grid_h * grid_w`).
absl::StatusOr<std::vector<float>> ComputeFlowMatchSigmas(int steps,
                                                          int image_tokens);

// Builds 4D image token position IDs of shape `[grid_h * grid_w, 4]` where row
// `(h, w)` is `[0.0f, h, w, 0.0f]`.
std::vector<float> BuildImagePositionIds(int grid_h, int grid_w);

// Builds 4D text token position IDs of shape `[seq_len, 4]` where row `i` is
// `[0.0f, 0.0f, 0.0f, i]`.
std::vector<float> BuildTextPositionIds(int seq_len);

// Samples `count` standard normal N(0, 1) floats using `std::mt19937_64(seed)`.
std::vector<float> SampleGaussianLatents(size_t count, uint64_t seed);

// Applies per-packed-channel BatchNorm affine transform
// (`z = lat * bn_scale + bn_shift`) and 2x2 unpatchifies packed latents of
// shape `[1, grid_h * grid_w, 128]` into NCHW VAE latents of shape
// `[1, 32, 2 * grid_h, 2 * grid_w]`.
absl::StatusOr<std::vector<float>> UnpatchifyLatents(
    absl::Span<const float> packed_latents, int grid_h, int grid_w,
    absl::Span<const float> bn_scale, absl::Span<const float> bn_shift);

// Converts planar NCHW float image output of shape `[1, 3, height, width]` in
// `[-1, 1]` to interleaved RGB888 bytes of shape `[height, width, 3]` via
// `round(clamp(y * 0.5 + 0.5, 0.0, 1.0) * 255.0)`.
absl::StatusOr<std::vector<uint8_t>> ConvertNchwToRgb888(
    absl::Span<const float> nchw, int height, int width);

// TODO(b/568027544): Move generic utility functions (e.g., `Fp16ToFp32` and
// `LookupFp16TokenEmbeddings`) to a common directory such as `base/` or
// `util/`.
// Converts an IEEE-754 binary16 (FP16) value to IEEE-754 binary32 (float).
float Fp16ToFp32(uint16_t h);

// Looks up `token_ids` in a little-endian FP16 embedding table of row width
// `hidden_dim` (`[vocab_size, hidden_dim]`) and returns `float` embeddings of
// shape `[1, token_ids.size(), hidden_dim]`.
absl::StatusOr<std::vector<float>> LookupFp16TokenEmbeddings(
    absl::Span<const int32_t> token_ids, absl::string_view fp16_table_bytes,
    int hidden_dim);

// Builds a 4D additive causal + padding attention mask of shape
// `[1, num_heads, seq_len, seq_len]` where entry `(q, k)` is
// `(k > q ? neg_inf : 0.0f) + (k >= active_len ? neg_inf : 0.0f)`.
absl::StatusOr<std::vector<float>> BuildCausalPaddingAttentionMask(
    int seq_len, int active_len, int num_heads, float neg_inf = -1e9f);

// Computes 1D Qwen3 rotary positional embeddings `(cos, sin)` of shape
// `[1, seq_len, head_dim]` where `emb = concat(freqs, freqs, dim=-1)`.
absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildQwen3RotaryPosEmbed(int seq_len, int head_dim = 128,
                         float theta = 1000000.0f);

// Interleaves three text encoder tap tensors of shape
// `[1, seq_len, hidden_dim]` along the feature dimension into
// `[1, seq_len, 3 * hidden_dim]`
// (`stack([tap0, tap1, tap2], dim=1).permute(0, 2, 1, 3).reshape(1, S, 3*D)`).
absl::StatusOr<std::vector<float>> InterleaveThreeEncoderTaps(
    absl::Span<const float> tap0, absl::Span<const float> tap1,
    absl::Span<const float> tap2, int seq_len, int hidden_dim);

// Computes multi-axis FLUX.2 rotary positional embeddings `(cos, sin)` of shape
// `[1, num_tokens, 1, sum(axes_dim)]` from `joint_ids` of shape
// `[num_tokens, axes_dim.size()]`, matching `Flux2PosEmbed`
// (`repeat_interleave` by 2 along each axis).
absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildFlux2RotaryPosEmbed(absl::Span<const float> joint_ids, int num_tokens,
                         absl::Span<const int> axes_dim, float theta = 2000.0f);

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_MATH_H_
