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

#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_event.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "tensor/arithmetic.h"  // from @litert
#include "tensor/backends/tflite/arithmetic_tflite.h"  // from @litert
#include "tensor/buffer.h"  // from @litert
#include "tensor/datatypes.h"  // from @litert
#include "tensor/runners/litert/lambda_model_runner.h"  // from @litert
#include "tensor/runners/litert/litert_buffer.h"  // from @litert
#include "tensor/tensor.h"  // from @litert
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

// Mirrors the values ApplyBitmapImpl assigns on the host. The FP32 value must
// be -inf rather than a large finite floor, which would still leave a banned
// token outranking any legitimately lower-scoring one. FP16 has no usable
// -inf, so host and graphs both settle for the most negative finite half.
constexpr float kDisallowedBiasF32 = -std::numeric_limits<float>::infinity();
constexpr float kMaxFiniteF16 = 65504.0f;
constexpr float kDisallowedBiasF16 = -kMaxFiniteF16;

// Summary of a mask tree, computed before unpacking it, that says which staging
// buffers the step can need.
struct StagingNeeds {
  // Any weight that multiplies positive logits and divides negative ones, which
  // needs one of the sign-dependent graphs.
  bool sign_dependent = false;
  // Any weight whose negative-branch multiplier is not its reciprocal, which
  // the sign-dependent graphs can only express by taking weight_neg as an
  // explicit input. Over-approximates MaskFeatures::reciprocal_neg, since it
  // does not know which entries UnpackMask will skip.
  bool non_reciprocal = false;
};

// Accumulates `mask`'s contribution into `needs`.
void ScanStagingNeeds(const LogitMask* mask, ElementType element_type,
                      StagingNeeds& needs) {
  if (mask == nullptr) {
    return;
  }
  switch (mask->GetType()) {
    case MaskType::kBitmap:
      // In fp16 a ban zeroes the weight (see ban_token) and zero has no finite
      // reciprocal. In fp32 a ban is a bias only.
      needs.non_reciprocal |= (element_type == ElementType::Float16);
      return;
    case MaskType::kSparse: {
      const auto* sm = static_cast<const SparseLogitMask*>(mask);
      for (const auto& entry : sm->entries()) {
        if (entry.weight == 1.0f) {
          continue;
        }
        needs.sign_dependent |= entry.sign_dependent_weight;
        needs.non_reciprocal |=
            (!entry.sign_dependent_weight || entry.weight == 0.0f);
      }
      return;
    }
    case MaskType::kComposite: {
      const auto* cm = static_cast<const CompositeLogitMask*>(mask);
      for (const auto& m : cm->masks()) {
        ScanStagingNeeds(m.get(), element_type, needs);
      }
      return;
    }
    case MaskType::kCustom:
      // UnpackMask rejects a custom mask, so this only has to be conservative.
      needs.non_reciprocal = true;
      return;
  }
}

// Returns the multiplier to apply to non-positive logits for `entry`: dividing
// by the weight is the same as multiplying by its reciprocal. Like
// SparseLogitMask::Apply, a zero weight leaves the logit unchanged.
float NegativeWeight(const SparseLogitMask::Entry& entry) {
  if (!entry.sign_dependent_weight) {
    return entry.weight;
  }
  return entry.weight != 0.0f ? 1.0f / entry.weight : 1.0f;
}

// Views a host staging buffer as the raw bytes expected by ModelRunner inputs.
template <typename T>
absl::Span<const std::byte> AsBytes(const std::vector<T>& buffer) {
  return absl::MakeConstSpan(reinterpret_cast<const std::byte*>(buffer.data()),
                             buffer.size() * sizeof(T));
}

absl::StatusOr<tensor::TensorHandle> MakeTensorHandle(
    TensorBuffer& tb, const std::string& name, tensor::Type type,
    const std::vector<int32_t>& shape) {
  LITERT_ASSIGN_OR_RETURN(auto dup, tb.Duplicate());
  tensor::TensorInit init;
  init.name = name;
  init.type = type;
  init.shape = shape;
  init.buffer = std::make_shared<tensor::LitertBuffer>(std::move(dup));
  return tensor::TensorHandle(init);
}

// Per-element-type constants and conversions for the host staging buffers.
template <typename T>
struct StagingTraits;

template <>
struct StagingTraits<float> {
  static constexpr float kDisallowedBias = kDisallowedBiasF32;
  static float FromFloat(float v) { return v; }
  static float ToFloat(float v) { return v; }
};

template <>
struct StagingTraits<tflite::half> {
  static constexpr float kDisallowedBias = kDisallowedBiasF16;
  // Clamped so an out-of-range accumulated weight or bias saturates instead of
  // becoming an infinity that would poison the whole slice with NaNs.
  static tflite::half FromFloat(float v) {
    return tflite::half(std::clamp(v, -kMaxFiniteF16, kMaxFiniteF16));
  }
  static float ToFloat(tflite::half v) { return static_cast<float>(v); }
};

// Splits `mask` into its bitmap and sparse constituents, descending one level
// into a composite, mirroring the grouping ApplyCompositeImpl performs on the
// host.
//
// Returns an error for masks this runner cannot faithfully express (custom
// masks and nested composites); silently dropping them would let disallowed
// tokens through, so callers must fall back to host masking instead.
absl::Status CollectMasks(const LogitMask* mask,
                          std::vector<const BitmapLogitMask*>& bitmaps,
                          std::vector<const SparseLogitMask*>& sparses,
                          bool allow_composite) {
  if (mask == nullptr) {
    return absl::OkStatus();
  }
  switch (mask->GetType()) {
    case MaskType::kBitmap:
      bitmaps.push_back(static_cast<const BitmapLogitMask*>(mask));
      return absl::OkStatus();
    case MaskType::kSparse:
      sparses.push_back(static_cast<const SparseLogitMask*>(mask));
      return absl::OkStatus();
    case MaskType::kComposite: {
      if (!allow_composite) {
        return absl::UnimplementedError(
            "LiteRtLogitMaskRunner cannot express a CompositeLogitMask nested "
            "inside another CompositeLogitMask; mask the logits on the host "
            "with LogitMask::Apply instead.");
      }
      const auto* cm = static_cast<const CompositeLogitMask*>(mask);
      for (const auto& m : cm->masks()) {
        ABSL_RETURN_IF_ERROR(
            CollectMasks(m.get(), bitmaps, sparses, /*allow_composite=*/false));
      }
      return absl::OkStatus();
    }
    case MaskType::kCustom:
      break;
  }
  return absl::UnimplementedError(
      "LiteRtLogitMaskRunner cannot express a custom LogitMask; mask the "
      "logits on the host with LogitMask::Apply instead.");
}

template <typename T>
absl::Status ApplyMasksToSpan(absl::Span<T> span,
                              absl::Span<const LogitMask* const> masks,
                              int vocab_size) {
  RET_CHECK_GE(static_cast<int>(span.size()),
               static_cast<int>(masks.size()) * vocab_size);
  for (size_t i = 0; i < masks.size(); ++i) {
    if (masks[i] != nullptr) {
      ABSL_RETURN_IF_ERROR(
          masks[i]->Apply(span.subspan(i * vocab_size, vocab_size)));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<LiteRtLogitMaskRunner>>
LiteRtLogitMaskRunner::Create(Environment& env, HwAccelerators accelerator,
                              bool force_graph_on_host) {
  return std::unique_ptr<LiteRtLogitMaskRunner>(
      new LiteRtLogitMaskRunner(&env, accelerator, force_graph_on_host));
}

absl::StatusOr<std::unique_ptr<LiteRtLogitMaskRunner>>
LiteRtLogitMaskRunner::CreateForHost() {
  return std::unique_ptr<LiteRtLogitMaskRunner>(
      new LiteRtLogitMaskRunner(/*env=*/nullptr, HwAccelerators::kNone,
                                /*force_graph_on_host=*/false));
}

absl::Status LiteRtLogitMaskRunner::InitGpuOptions() {
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                          options_.GetOptions<::litert::GpuOptions>());

  // The logits buffer handed to Apply*() is produced by another GPU model (the
  // decode graph), and CompiledModel can only bind it without a copy if the
  // delegate treats that tensor as "external", i.e. exposes it in its native
  // layout (PHWC4 for ML Drift). Every other tensor stays non-external, and so
  // packed, on purpose: writing a vocab-sized host buffer into a packed buffer
  // is a straight queue write, while writing one into a native-layout buffer
  // makes the runtime repack the whole tensor on the CPU, one element at a time
  // (see LiteRtUnlockWebGpuMemory) -- ~4.8 ms per step for the weight tensor
  // alone on an A100 with a 262144-entry vocabulary. The staging tensors are
  // rewritten from the host every step, so the delegate converts them to its
  // native layout on device as part of the dispatch instead.
  LITERT_RETURN_IF_ERROR(gpu_options.EnableExternalTensorsMode(false));
  LITERT_RETURN_IF_ERROR(gpu_options.AddExternalTensorPattern("logits"));
  LITERT_RETURN_IF_ERROR(gpu_options.AddExternalTensorPattern("masked_logits"));

  // Linear buffers rather than textures, which is the only option for
  // vocab-sized tensors anyway. The graphs are elementwise, so the native
  // layout does not affect the result as long as every tensor shares it.
  LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(
      ::litert::GpuOptions::BufferStorageType::kBuffer));
  LITERT_RETURN_IF_ERROR(
      gpu_options.SetPrecision(config_.element_type == ElementType::Float32
                                   ? ::litert::GpuOptions::Precision::kFp32
                                   : ::litert::GpuOptions::Precision::kFp16));
  return absl::OkStatus();
}

LiteRtLogitMaskRunner::GraphKind LiteRtLogitMaskRunner::SelectGraphKind(
    const MaskFeatures& features) {
  if (features.has_sign_dependent) {
    if (features.reciprocal_neg) {
      return features.has_bias ? GraphKind::kSignDependentReciprocalBias
                               : GraphKind::kSignDependentReciprocal;
    }
    return features.has_bias ? GraphKind::kSignDependentBias
                             : GraphKind::kSignDependent;
  }
  if (features.has_weights) {
    return features.has_bias ? GraphKind::kWeightBias : GraphKind::kWeight;
  }
  return GraphKind::kBias;
}

bool LiteRtLogitMaskRunner::NeedsBias(GraphKind kind) {
  switch (kind) {
    case GraphKind::kBias:
    case GraphKind::kWeightBias:
    case GraphKind::kSignDependentBias:
    case GraphKind::kSignDependentReciprocalBias:
      return true;
    default:
      return false;
  }
}

bool LiteRtLogitMaskRunner::NeedsWeight(GraphKind kind) {
  return kind != GraphKind::kBias;
}

bool LiteRtLogitMaskRunner::NeedsWeightNeg(GraphKind kind) {
  return kind == GraphKind::kSignDependent ||
         kind == GraphKind::kSignDependentBias;
}

absl::Status LiteRtLogitMaskRunner::EnsureRunner(GraphKind kind) {
  auto& slot = runners_[static_cast<size_t>(kind)];
  if (slot != nullptr) {
    return absl::OkStatus();
  }
  RET_CHECK_NE(env_, nullptr) << "Environment must not be null.";

  const tensor::Type tensor_type =
      (config_.element_type == ElementType::Float32) ? tensor::Type::kFP32
                                                     : tensor::Type::kFP16;
  const std::vector<int32_t> shape = {
      config_.batch_size, config_.sequence_length, config_.vocab_size};
  auto prototype = [&](const char* name) {
    return tensor::Tensor<tensor::TfLiteMixinTag>(
        {.name = name, .type = tensor_type, .shape = shape});
  };

  tensor::TensorsMap inputs;
  inputs.emplace("logits", prototype("logits"));
  if (NeedsWeight(kind)) {
    inputs.emplace("weight", prototype("weight"));
  }
  if (NeedsWeightNeg(kind)) {
    inputs.emplace("weight_neg", prototype("weight_neg"));
  }
  if (NeedsBias(kind)) {
    inputs.emplace("bias", prototype("bias"));
  }

  slot = std::make_unique<MaskRunner>(
      *env_, options_, std::move(inputs), [kind](const auto& in) {
        const tensor::Tensor<tensor::TfLiteMixinTag>& logits = in.at("logits");
        tensor::Tensor<tensor::TfLiteMixinTag> masked_logits = logits;
        switch (kind) {
          case GraphKind::kBias:
            // The bias Add below is the whole transform.
            break;
          case GraphKind::kWeight:
          case GraphKind::kWeightBias:
            masked_logits = tensor::Mul(logits, in.at("weight"));
            break;
          case GraphKind::kSignDependent:
          case GraphKind::kSignDependentBias:
          case GraphKind::kSignDependentReciprocal:
          case GraphKind::kSignDependentReciprocalBias: {
            // Relu(z) == max(z, 0) and z - Relu(z) == min(z, 0) split the
            // logits into their positive and negative parts, so each part can
            // get its own multiplier without a data-dependent branch. Exactly
            // one of the two is non-zero for any logit.
            tensor::Tensor positive = tensor::Relu(logits);
            tensor::Tensor negative = tensor::Sub(logits, positive);
            // The reciprocal variants divide by the weight instead of taking a
            // second multiplier tensor. SelectGraphKind only picks them when
            // every staged weight_neg is 1 / weight, so this is exact.
            tensor::Tensor scaled_negative =
                NeedsWeightNeg(kind)
                    ? tensor::Mul(negative, in.at("weight_neg"))
                    : tensor::Div(negative, in.at("weight"));
            masked_logits = tensor::Add(tensor::Mul(positive, in.at("weight")),
                                        scaled_negative);
            break;
          }
          case GraphKind::kNumGraphKinds:
            break;
        }
        if (NeedsBias(kind)) {
          masked_logits = tensor::Add(masked_logits, in.at("bias"));
        }
        masked_logits.SetName("masked_logits");
        return tensor::TensorsMap{{"masked_logits", masked_logits}};
      });
  return absl::OkStatus();
}

absl::Status LiteRtLogitMaskRunner::Init() {
  LITERT_ASSIGN_OR_RETURN(options_, Options::Create());
  options_.SetHardwareAccelerators(config_.accelerator);

  if (config_.accelerator & HwAccelerators::kGpu) {
    ABSL_RETURN_IF_ERROR(InitGpuOptions());
    needs_copy_back_ = true;
  }

  tensor::Type tensor_type = (config_.element_type == ElementType::Float32)
                                 ? tensor::Type::kFP32
                                 : tensor::Type::kFP16;
  std::vector<int32_t> shape = {config_.batch_size, config_.sequence_length,
                                config_.vocab_size};

  // Copy-back runner: masked_logits = Add(logits, 0). Only built for
  // accelerators that cannot alias a graph's input and output (see
  // `needs_copy_back_`). Being a separate model, it is a separate dispatch, so
  // the two bindings are never in the same synchronization scope.
  if (needs_copy_back_) {
    RET_CHECK_NE(env_, nullptr) << "Environment must not be null.";
    copy_runner_ = std::make_unique<MaskRunner>(
        *env_, options_,
        tensor::TensorsMap{
            {"logits",
             tensor::Tensor<tensor::TfLiteMixinTag>(
                 {.name = "logits", .type = tensor_type, .shape = shape})}},
        [tensor_type](const auto& inputs) {
          // The `Add(tensor, float)` overload builds an fp32 constant, which an
          // elementwise op refuses to mix with fp16 logits, so spell the zero
          // out with the graph's own element type.
          tensor::Tensor<tensor::TfLiteMixinTag> zero(
              {.type = tensor_type,
               .shape = {},
               .buffer =
                   tensor_type == tensor::Type::kFP32
                       ? tensor::OwningCpuBuffer::Copy<tensor::Type::kFP32>(
                             {0.0f})
                       : tensor::OwningCpuBuffer::Copy<tensor::Type::kFP16>(
                             {0.0f})});
          tensor::Tensor masked_logits = tensor::Add(inputs.at("logits"), zero);
          masked_logits.SetName("masked_logits");
          return tensor::TensorsMap{{"masked_logits", masked_logits}};
        });
  }

  int total_elements =
      config_.batch_size * config_.sequence_length * config_.vocab_size;
  if (config_.element_type == ElementType::Float32) {
    host_bias_f32_.assign(total_elements, 0.0f);
    host_weight_f32_.assign(total_elements, 1.0f);
    host_weight_neg_f32_.clear();
    host_bias_f16_.clear();
    host_weight_f16_.clear();
    host_weight_neg_f16_.clear();
  } else {
    host_bias_f16_.assign(total_elements, tflite::half(0.0f));
    host_weight_f16_.assign(total_elements, tflite::half(1.0f));
    host_weight_neg_f16_.clear();
    host_bias_f32_.clear();
    host_weight_f32_.clear();
    host_weight_neg_f32_.clear();
  }

  // The staging buffers are back to the identity transform and resized, so the
  // incremental-reset bookkeeping no longer describes them.
  dirty_tokens_.assign(config_.batch_size * config_.sequence_length, {});
  disallowed_stamp_.assign(config_.vocab_size, 0);
  disallowed_generation_ = 0;

  initialized_ = true;
  return absl::OkStatus();
}

template <typename T>
absl::Status LiteRtLogitMaskRunner::UnpackMask(const LogitMask* mask,
                                               const MaskSlices<T>& slices,
                                               SliceDirty& dirty,
                                               MaskFeatures& features) {
  using Traits = StagingTraits<T>;
  const T kIdentityWeight = Traits::FromFloat(1.0f);
  const T kZero = Traits::FromFloat(0.0f);
  const T kDisallowed = Traits::FromFloat(Traits::kDisallowedBias);

  // The staging buffers persist across steps, so undo exactly what the previous
  // call wrote before repopulating the slice. Past a fraction of the slice a
  // straight fill is cheaper than walking the list; the threshold only has to
  // be in the right ballpark, since either cost is small on both sides of it.
  constexpr int kFullResetDivisor = 8;
  const int full_reset_threshold = config_.vocab_size / kFullResetDivisor;
  if (static_cast<int>(dirty.bias.size()) >= full_reset_threshold) {
    std::fill(slices.bias.begin(), slices.bias.end(), kZero);
  } else {
    for (const int token_id : dirty.bias) {
      slices.bias[token_id] = kZero;
    }
  }
  if (static_cast<int>(dirty.weight.size()) >= full_reset_threshold) {
    std::fill(slices.weight.begin(), slices.weight.end(), kIdentityWeight);
    std::fill(slices.weight_neg.begin(), slices.weight_neg.end(),
              kIdentityWeight);
  } else {
    const bool has_weight_neg = !slices.weight_neg.empty();
    for (const int token_id : dirty.weight) {
      slices.weight[token_id] = kIdentityWeight;
      if (has_weight_neg) {
        slices.weight_neg[token_id] = kIdentityWeight;
      }
    }
  }
  dirty.bias.clear();
  dirty.weight.clear();

  // Bumping the generation invalidates every stamp in one step. Wrapping back
  // onto a stale stamp would make an untouched token look banned, so reset the
  // whole array on the (astronomically rare) wrap.
  if (static_cast<int>(disallowed_stamp_.size()) != config_.vocab_size) {
    disallowed_stamp_.assign(config_.vocab_size, 0);
    disallowed_generation_ = 0;
  }
  if (++disallowed_generation_ == 0) {
    std::fill(disallowed_stamp_.begin(), disallowed_stamp_.end(), 0);
    disallowed_generation_ = 1;
  }

  if (mask == nullptr) {
    return absl::OkStatus();
  }

  std::vector<const BitmapLogitMask*> bitmaps;
  std::vector<const SparseLogitMask*> sparses;
  ABSL_RETURN_IF_ERROR(
      CollectMasks(mask, bitmaps, sparses, /*allow_composite=*/true));

  // Only the sparse pass below reads the stamps, and the generation bumped
  // above already retires the ones an earlier call left behind, so with no
  // sparse mask in play stamping is pure cost -- one store per banned token,
  // which a hard constraint makes nearly vocabulary sized.
  const bool needs_stamp = !sparses.empty();

  // Records `count` consecutive token ids starting at `base_token` in `list`,
  // stopping at the threshold past which the next call resets the whole slice
  // and ignores the list. A grammar constraint bans almost all of a 262k-token
  // vocabulary on every step, so the appends beyond it are pure waste.
  const auto record_dirty = [&](std::vector<int>& list, int base_token,
                                int count) {
    const int room = full_reset_threshold - static_cast<int>(list.size());
    for (int i = 0; i < std::min(count, room); ++i) {
      list.push_back(base_token + i);
    }
  };

  // Bans `token_id`, reproducing the host's `logits[token_id] = kDisallowed`
  // assignment. In FP32 kDisallowed is -inf, so adding it is already an
  // assignment for any finite logit. The FP16 sentinel is finite, and adding it
  // would overflow moderately negative logits to -inf and cancel large positive
  // ones to 0.0, so the weight is zeroed too and the graph evaluates
  // `logit * 0.0 + (-65504.0)` exactly.
  const auto ban_token = [&](int token_id) {
    features.has_bias = true;
    slices.bias[token_id] = kDisallowed;
    record_dirty(dirty.bias, token_id, 1);
    if constexpr (std::is_same_v<T, tflite::half>) {
      features.has_weights = true;
      // A zero weight has no finite reciprocal, so the graph cannot derive
      // weight_neg by dividing and needs it as an explicit input.
      features.reciprocal_neg = false;
      slices.weight[token_id] = kZero;
      if (!slices.weight_neg.empty()) {
        slices.weight_neg[token_id] = kZero;
      }
      record_dirty(dirty.weight, token_id, 1);
    }
    if (needs_stamp) {
      disallowed_stamp_[token_id] = disallowed_generation_;
    }
  };

  // Bans the 64 tokens starting at `base_token`, for a bitmap word that
  // disallows all of them. Whole disallowed words are what a hard constraint
  // is mostly made of, so they are filled in one sweep rather than one
  // predicated store per bit, as the `word == 0` case of ApplyBitmapImpl does.
  const auto ban_block = [&](int base_token) {
    features.has_bias = true;
    std::fill_n(slices.bias.begin() + base_token, 64, kDisallowed);
    record_dirty(dirty.bias, base_token, 64);
    if constexpr (std::is_same_v<T, tflite::half>) {
      features.has_weights = true;
      features.reciprocal_neg = false;
      std::fill_n(slices.weight.begin() + base_token, 64, kZero);
      if (!slices.weight_neg.empty()) {
        std::fill_n(slices.weight_neg.begin() + base_token, 64, kZero);
      }
      record_dirty(dirty.weight, base_token, 64);
    }
    if (needs_stamp) {
      std::fill_n(disallowed_stamp_.begin() + base_token, 64,
                  disallowed_generation_);
    }
  };

  // 1. Hard constraints. Several bitmaps are fused by intersecting their
  // allowed sets, which is what ApplyCompositeImpl does via `fused_word &=`.
  if (!bitmaps.empty()) {
    int min_vocab_size = bitmaps[0]->vocab_size();
    for (const auto* bm : bitmaps) {
      min_vocab_size = std::min(min_vocab_size, bm->vocab_size());
    }
    // A mask can be built with any vocabulary size, and a negative one would
    // otherwise run the padding loop below from a negative index straight off
    // the front of the slice.
    const int limit = std::clamp(min_vocab_size, 0, config_.vocab_size);
    const int num_words = (min_vocab_size > 0) ? (min_vocab_size + 63) / 64 : 0;

    for (int word_idx = 0; word_idx < num_words; ++word_idx) {
      const int base_token = word_idx * 64;
      if (base_token >= limit) break;

      uint64_t fused_word = ~uint64_t{0};
      for (const auto* bm : bitmaps) {
        fused_word &= bm->words()[word_idx];
        if (fused_word == 0) break;
      }
      if (fused_word == ~uint64_t{0} && base_token + 64 <= limit) {
        // Every token in this word is allowed by every bitmap.
        continue;
      }

      const int count = std::min(64, limit - base_token);
      if (fused_word == 0 && count == 64) {
        // No token in this word survives the intersection.
        ban_block(base_token);
        continue;
      }

      const uint64_t disallowed_bits = ~fused_word;
      for (int bit = 0; bit < count; ++bit) {
        if ((disallowed_bits >> bit) & 1) {
          ban_token(base_token + bit);
        }
      }
    }

    // Tokens past the narrowest bitmap's vocabulary are padding and are always
    // disallowed. ApplyBitmapImpl and ApplyCompositeImpl both do this; omitting
    // it here would let padded tokens be sampled.
    for (int i = limit; i < config_.vocab_size; ++i) {
      ban_token(i);
    }
  }

  // 2. Soft constraints, composed in order. Applying (weight w, bias b) on top
  // of the already staged transform z * W + B yields
  //   (z * W + B) * w + b = z * (W * w) + (B * w + b),
  // which is how the host path composes chained SparseLogitMask::Apply calls.
  for (const auto* sm : sparses) {
    for (const auto& entry : sm->entries()) {
      const int token_id = entry.token_id;
      if (token_id < 0 || token_id >= config_.vocab_size) {
        continue;
      }
      // A token already banned by a hard constraint stays banned; the host path
      // gets this for free by skipping logits that are already -inf. The
      // bitmap pass already recorded those tokens as dirty.
      if (disallowed_stamp_[token_id] == disallowed_generation_) {
        continue;
      }
      record_dirty(dirty.bias, token_id, 1);

      const float weight = entry.weight;
      const float weight_neg = NegativeWeight(entry);
      const float old_bias = Traits::ToFloat(slices.bias[token_id]);
      const float old_weight = Traits::ToFloat(slices.weight[token_id]);

      // The host applies a sign-dependent entry to the running value `z*W + B`,
      // scaling the bias it already carries by the branch multiplier it takes,
      // while the graph splits on the sign of the original logit `z` and adds a
      // single bias tensor at the end. The two agree only if the staged
      // transform leaves the sign and zero point of the logit alone:
      //   * a non-zero `B` would have to be scaled by `weight` on the positive
      //     branch and by `weight_neg` on the negative one, which a single bias
      //     tensor cannot represent;
      //   * a negative `W` makes the host branch on the opposite sign from the
      //     graph, so the two pick different multipliers outright.
      // A weight of 1.0 is exempt: both branches multiply by 1.0 and the entry
      // degenerates into a plain bias, which composes linearly.
      if (entry.sign_dependent_weight && weight != 1.0f &&
          (old_bias != 0.0f || old_weight < 0.0f)) {
        return absl::UnimplementedError(
            "LiteRtLogitMaskRunner cannot compose a sign-dependent weight on "
            "top of an existing non-zero bias or sign-flipping weight; mask "
            "the logits on the host with LogitMask::Apply instead.");
      }

      if (weight != 1.0f) {
        features.has_weights = true;
        features.has_sign_dependent |= entry.sign_dependent_weight;
        // The reciprocal variants derive weight_neg as 1 / weight, which this
        // entry only matches if it is sign dependent with a non-zero weight.
        if (!entry.sign_dependent_weight || weight == 0.0f) {
          features.reciprocal_neg = false;
        }
        slices.weight[token_id] = Traits::FromFloat(old_weight * weight);
        if (!slices.weight_neg.empty()) {
          const float old_weight_neg =
              Traits::ToFloat(slices.weight_neg[token_id]);
          slices.weight_neg[token_id] =
              Traits::FromFloat(old_weight_neg * weight_neg);
        }
        record_dirty(dirty.weight, token_id, 1);
      }

      // Scaling the staged bias by this entry's weight lets biases from earlier
      // masks pass through the later multiply, as the sequential host
      // evaluation does. Sign-dependent entries would need two such scalings,
      // which is what the guard above rejects.
      const float new_bias = old_bias * weight + entry.bias;
      if (new_bias != 0.0f) {
        features.has_bias = true;
      }
      slices.bias[token_id] = Traits::FromFloat(new_bias);
    }
  }

  return absl::OkStatus();
}

LiteRtLogitMaskRunner::MaskSlices<float>
LiteRtLogitMaskRunner::StagingSlicesF32(int offset) {
  return MaskSlices<float>{
      .bias =
          absl::MakeSpan(host_bias_f32_).subspan(offset, config_.vocab_size),
      .weight =
          absl::MakeSpan(host_weight_f32_).subspan(offset, config_.vocab_size),
      .weight_neg = host_weight_neg_f32_.empty()
                        ? absl::Span<float>()
                        : absl::MakeSpan(host_weight_neg_f32_)
                              .subspan(offset, config_.vocab_size)};
}

LiteRtLogitMaskRunner::MaskSlices<tflite::half>
LiteRtLogitMaskRunner::StagingSlicesF16(int offset) {
  return MaskSlices<tflite::half>{
      .bias =
          absl::MakeSpan(host_bias_f16_).subspan(offset, config_.vocab_size),
      .weight =
          absl::MakeSpan(host_weight_f16_).subspan(offset, config_.vocab_size),
      .weight_neg = host_weight_neg_f16_.empty()
                        ? absl::Span<tflite::half>()
                        : absl::MakeSpan(host_weight_neg_f16_)
                              .subspan(offset, config_.vocab_size)};
}

void LiteRtLogitMaskRunner::EnsureSignDependentStaging() {
  const int total_elements =
      config_.batch_size * config_.sequence_length * config_.vocab_size;
  if (config_.element_type == ElementType::Float32) {
    host_weight_neg_f32_.resize(total_elements, 1.0f);
  } else {
    host_weight_neg_f16_.resize(total_elements, tflite::half(1.0f));
  }
}

bool LiteRtLogitMaskRunner::HasSignDependentStaging() const {
  return config_.element_type == ElementType::Float32
             ? !host_weight_neg_f32_.empty()
             : !host_weight_neg_f16_.empty();
}

absl::Status LiteRtLogitMaskRunner::UnpackSlices(
    absl::Span<const LogitMask* const> masks, MaskFeatures& features) {
  features = MaskFeatures();
  for (size_t i = 0; i < masks.size(); ++i) {
    const int offset = static_cast<int>(i) * config_.vocab_size;
    ABSL_RETURN_IF_ERROR(config_.element_type == ElementType::Float32
                             ? UnpackMask(masks[i], StagingSlicesF32(offset),
                                          dirty_tokens_[i], features)
                             : UnpackMask(masks[i], StagingSlicesF16(offset),
                                          dirty_tokens_[i], features));
  }
  return absl::OkStatus();
}

absl::Status LiteRtLogitMaskRunner::RunMaskGraph(TensorBuffer& logits,
                                                 GraphKind kind) {
  const bool is_f32 = config_.element_type == ElementType::Float32;
  tensor::Type tensor_type = is_f32 ? tensor::Type::kFP32 : tensor::Type::kFP16;
  std::vector<int32_t> shape = {config_.batch_size, config_.sequence_length,
                                config_.vocab_size};

  // The graphs were compiled for exactly `shape`, and the handle below binds
  // the whole buffer. A buffer that is padded or otherwise sized differently
  // would be silently reinterpreted, so reject it instead.
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits.TensorType());
  const auto& dims = logits_type.Layout().Dimensions();
  RET_CHECK_EQ(dims.size(), 3)
      << "Expected logits with dimensions [batch_size, sequence_length, "
         "vocab_size].";
  RET_CHECK_EQ(static_cast<int>(dims[0]), config_.batch_size)
      << "Logits batch size does not match the compiled graph.";
  RET_CHECK_EQ(static_cast<int>(dims[1]), config_.sequence_length)
      << "Logits sequence length does not match the compiled graph.";
  RET_CHECK_EQ(static_cast<int>(dims[2]), config_.vocab_size)
      << "Logits vocab size does not match the compiled graph.";
  RET_CHECK(logits_type.ElementType() == config_.element_type)
      << "Logits element type does not match the compiled graph.";

  // CompiledModel rejects output buffers that carry an event, so an
  // asynchronously produced logits buffer has to be waited on before it is
  // bound as this graph's output; dropping the event instead would let the
  // graph read logits that are not written yet. Duplicate() shares the buffer,
  // so this must precede the handles below.
  if (logits.HasEvent()) {
    LITERT_ASSIGN_OR_RETURN(auto event, logits.GetEvent());
    LITERT_RETURN_IF_ERROR(event.Wait(/*timeout_in_ms=*/-1));
    LITERT_RETURN_IF_ERROR(logits.ClearEvent());
  }

  LITERT_ASSIGN_OR_RETURN(
      auto logits_handle,
      MakeTensorHandle(logits, "logits", tensor_type, shape));

  MaskRunner* runner = runners_[static_cast<size_t>(kind)].get();
  RET_CHECK(runner != nullptr) << "Mask graph was not compiled.";

  // Only the staging tensors this variant declares are bound; see GraphKind.
  ABSL_RETURN_IF_ERROR(runner->SetInput("logits", logits_handle));
  if (NeedsBias(kind)) {
    ABSL_RETURN_IF_ERROR(runner->SetInput(
        "bias", is_f32 ? AsBytes(host_bias_f32_) : AsBytes(host_bias_f16_)));
  }
  if (NeedsWeight(kind)) {
    ABSL_RETURN_IF_ERROR(
        runner->SetInput("weight", is_f32 ? AsBytes(host_weight_f32_)
                                          : AsBytes(host_weight_f16_)));
  }
  if (NeedsWeightNeg(kind)) {
    ABSL_RETURN_IF_ERROR(
        runner->SetInput("weight_neg", is_f32 ? AsBytes(host_weight_neg_f32_)
                                              : AsBytes(host_weight_neg_f16_)));
  }
  if (copy_runner_ == nullptr) {
    // In place: the caller's buffer is both the input and the output.
    ABSL_RETURN_IF_ERROR(runner->SetOutput("masked_logits", logits_handle));
    return runner->Run();
  }

  // Out of place: leave the mask graph bound to its own output buffer and copy
  // that buffer back over the caller's logits in a second dispatch.
  ABSL_RETURN_IF_ERROR(runner->Run());
  LITERT_ASSIGN_OR_RETURN(auto masked_handle,
                          runner->GetOutput("masked_logits"));
  ABSL_RETURN_IF_ERROR(copy_runner_->SetInput("logits", masked_handle));
  ABSL_RETURN_IF_ERROR(copy_runner_->SetOutput("masked_logits", logits_handle));
  return copy_runner_->Run();
}

absl::Status LiteRtLogitMaskRunner::ApplySlices(
    TensorBuffer& logits, absl::Span<const LogitMask* const> masks,
    int expected_batch_size, int expected_seq_len) {
  bool all_null = true;
  for (const LogitMask* mask : masks) {
    if (mask != nullptr) {
      all_null = false;
      break;
    }
  }
  if (all_null) {
    return absl::OkStatus();
  }

  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits.TensorType());
  const auto& dims = logits_type.Layout().Dimensions();
  RET_CHECK_EQ(dims.size(), 3)
      << "Expected logits with dimensions [batch_size, sequence_length, "
         "vocab_size].";
  const int actual_batch = static_cast<int>(dims[0]);
  const int actual_seq = static_cast<int>(dims[1]);
  const int vocab_size = static_cast<int>(dims[2]);
  const ElementType element_type = logits_type.ElementType();
  RET_CHECK(element_type == ElementType::Float32 ||
            element_type == ElementType::Float16)
      << "Unsupported logits element type.";
  RET_CHECK_EQ(actual_batch, expected_batch_size)
      << "Logits batch size does not match expected batch size.";
  RET_CHECK_EQ(actual_seq, expected_seq_len)
      << "Logits sequence length does not match expected sequence length.";
  RET_CHECK_EQ(static_cast<int>(masks.size()), actual_batch * actual_seq);

  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits.BufferType());

  // 1. Host-memory fast path: mask in place via spans, without compiling or
  // invoking a graph, unless a test asked for the graph path on host buffers.
  if (buffer_type == TensorBufferType::kHostMemory && !force_graph_on_host_) {
    if (element_type == ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(auto span,
                              ReferTensorBufferAsSpan<float>(logits));
      return ApplyMasksToSpan(span, masks, vocab_size);
    } else {
      LITERT_ASSIGN_OR_RETURN(auto span,
                              ReferTensorBufferAsSpan<tflite::half>(logits));
      return ApplyMasksToSpan(span, masks, vocab_size);
    }
  }

  // 2. Device / LiteRT graph execution path:
  if (config_.accelerator != HwAccelerators::kNone) {
    // Lazily (re)initialise if dimensions or dtype changed. The graphs
    // themselves are compiled on first use by EnsureRunner().
    if (!initialized_ || config_.batch_size != actual_batch ||
        config_.sequence_length != actual_seq ||
        config_.vocab_size != vocab_size ||
        config_.element_type != element_type) {
      config_.batch_size = actual_batch;
      config_.sequence_length = actual_seq;
      config_.vocab_size = vocab_size;
      config_.element_type = element_type;
      for (auto& runner : runners_) {
        runner.reset();
      }
      ABSL_RETURN_IF_ERROR(Init());
    }

    StagingNeeds needs;
    for (const LogitMask* mask : masks) {
      ScanStagingNeeds(mask, config_.element_type, needs);
    }
    // The reciprocal graph variants derive the negative multipliers with a Div
    // and never read weight_neg, which covers the common case (an fp32
    // repetition penalty, on its own or under a bitmap), so the vocab-sized
    // buffer is only paid for when some weight has no reciprocal.
    if (needs.sign_dependent && needs.non_reciprocal) {
      EnsureSignDependentStaging();
    }

    // One dirty-token list per slice, carried across calls. Init() sizes this
    // to the slice count whenever the configuration changes.
    if (dirty_tokens_.size() != masks.size()) {
      dirty_tokens_.assign(masks.size(), {});
    }

    MaskFeatures features;
    absl::Status unpack_status = UnpackSlices(masks, features);
    if (unpack_status.ok()) {
      GraphKind kind = SelectGraphKind(features);
      if (NeedsWeightNeg(kind) && !HasSignDependentStaging()) {
        // The pre-scan reads the masks while the staged values pick the graph,
        // so if the two ever disagree, stage weight_neg and unpack again rather
        // than bind an empty input. Unpacking is idempotent: the dirty-token
        // lists undo whatever the first pass wrote.
        EnsureSignDependentStaging();
        unpack_status = UnpackSlices(masks, features);
        kind = SelectGraphKind(features);
      }
      if (unpack_status.ok()) {
        if (!features.has_bias && !features.has_weights) {
          return absl::OkStatus();
        }
        ABSL_RETURN_IF_ERROR(EnsureRunner(kind));
        return RunMaskGraph(logits, kind);
      }
    }
    if (!absl::IsUnimplemented(unpack_status)) {
      return unpack_status;
    }
  }

  // 3. CPU fallback: no accelerator, or a mask the graphs cannot express.
  if (buffer_type == TensorBufferType::kHostMemory) {
    if (element_type == ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(auto span,
                              ReferTensorBufferAsSpan<float>(logits));
      return ApplyMasksToSpan(span, masks, vocab_size);
    } else {
      LITERT_ASSIGN_OR_RETURN(auto span,
                              ReferTensorBufferAsSpan<tflite::half>(logits));
      return ApplyMasksToSpan(span, masks, vocab_size);
    }
  }
  if (element_type == ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto vec, CopyFromTensorBuffer<float>(logits));
    ABSL_RETURN_IF_ERROR(
        ApplyMasksToSpan(absl::MakeSpan(vec), masks, vocab_size));
    LITERT_RETURN_IF_ERROR(logits.Write(absl::MakeConstSpan(vec)));
    return absl::OkStatus();
  } else {
    LITERT_ASSIGN_OR_RETURN(auto vec,
                            CopyFromTensorBuffer<tflite::half>(logits));
    ABSL_RETURN_IF_ERROR(
        ApplyMasksToSpan(absl::MakeSpan(vec), masks, vocab_size));
    LITERT_RETURN_IF_ERROR(logits.Write(absl::MakeConstSpan(vec)));
    return absl::OkStatus();
  }
}

absl::Status LiteRtLogitMaskRunner::Apply(TensorBuffer& logits,
                                          const LogitMask* mask) {
  if (mask == nullptr) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits.TensorType());
  const auto& dims = logits_type.Layout().Dimensions();
  RET_CHECK_EQ(dims.size(), 3)
      << "Expected logits with dimensions [batch_size, sequence_length, "
         "vocab_size].";
  const int batch_size = static_cast<int>(dims[0]);
  std::vector<const LogitMask*> batch_masks(batch_size, mask);
  return ApplySlices(logits, absl::MakeConstSpan(batch_masks),
                     /*expected_batch_size=*/batch_size,
                     /*expected_seq_len=*/1);
}

absl::Status LiteRtLogitMaskRunner::ApplySequence(
    TensorBuffer& logits, absl::Span<const std::unique_ptr<LogitMask>> masks) {
  std::vector<const LogitMask*> raw_masks;
  raw_masks.reserve(masks.size());
  for (const auto& m : masks) {
    raw_masks.push_back(m.get());
  }
  return ApplySlices(logits, absl::MakeConstSpan(raw_masks),
                     /*expected_batch_size=*/1,
                     /*expected_seq_len=*/static_cast<int>(masks.size()));
}

absl::Status LiteRtLogitMaskRunner::ApplyBatch(
    TensorBuffer& logits, absl::Span<const LogitMask* const> masks) {
  return ApplySlices(logits, masks,
                     /*expected_batch_size=*/static_cast<int>(masks.size()),
                     /*expected_seq_len=*/1);
}

int LiteRtLogitMaskRunner::NumCompiledGraphsForTesting() const {
  int count = 0;
  for (const auto& runner : runners_) {
    if (runner != nullptr) {
      ++count;
    }
  }
  return count;
}

}  // namespace litert::lm
