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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LITERT_LOGIT_MASK_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LITERT_LOGIT_MASK_RUNNER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

// Applies LogitMasks to a logits buffer, either on device through lazily
// compiled LiteRT graphs or directly on the host CPU.
//
// Supports BitmapLogitMask, SparseLogitMask (including sign-dependent weights,
// which need a dedicated graph) and one level of CompositeLogitMask. Custom
// masks and nested composites cannot be expressed as a graph and fall back to
// LogitMask::Apply on the host.
//
// The execution path is picked per call:
// 1. Host memory: masked in place on the CPU, for every accelerator including
//    `HwAccelerators::kCpu`. A host-visible buffer is cheaper to touch directly
//    than to push through a vocab-wide elementwise graph.
// 2. `accelerator != HwAccelerators::kNone`: masked in place on device by a
//    lazily compiled graph, avoiding a device-to-host readback stall.
// 3. Otherwise, or when a mask cannot be expressed as a graph: masked on the
//    host, copying the buffer back and forth if it lives on device.
class LiteRtLogitMaskRunner {
 public:
  // Creates a LiteRtLogitMaskRunner for the specified hardware accelerator.
  //
  // @param env LiteRT environment used to compile the mask graphs. Must outlive
  // the runner.
  // @param accelerator The accelerator the mask graphs are compiled for. Pass
  // `HwAccelerators::kNone` to always mask on the host.
  // @param force_graph_on_host Test-only override that sends host-memory
  // buffers down the graph path (2) instead of the in-place fast path (1), so
  // that the graphs can be exercised on CPU-only test machines. Has no effect
  // when `accelerator` is `HwAccelerators::kNone`, which has no graph to run.
  static absl::StatusOr<std::unique_ptr<LiteRtLogitMaskRunner>> Create(
      Environment& env, HwAccelerators accelerator = HwAccelerators::kNone,
      bool force_graph_on_host = false);

  // Creates a host-only LiteRtLogitMaskRunner (`HwAccelerators::kNone`)
  // that always masks logits on the host and does not require a LiteRT
  // environment.
  static absl::StatusOr<std::unique_ptr<LiteRtLogitMaskRunner>> CreateForHost();

  ~LiteRtLogitMaskRunner();

  // Applies a single mask to a logits buffer of shape [1, 1, vocab_size] or
  // [batch_size, 1, vocab_size] (applied to all sequences in the batch).
  absl::Status Apply(TensorBuffer& logits, const LogitMask* mask);

  // Applies a sequence of masks to a logits buffer of shape [1, seq_len,
  // vocab_size]. masks.size() must match sequence_length.
  absl::Status ApplySequence(
      TensorBuffer& logits, absl::Span<const std::unique_ptr<LogitMask>> masks);

  // Applies masks to a batch of sequences of shape [batch_size, 1, vocab_size].
  // masks.size() must match batch_size.
  absl::Status ApplyBatch(TensorBuffer& logits,
                          absl::Span<const LogitMask* const> masks);

  // Test-only: the number of mask graphs compiled so far, which stays at zero
  // for as long as masking runs on the host.
  int NumCompiledGraphsForTesting() const;

 private:
  // Identifies the pre-compiled subgraph that expresses a staged transform.
  // Each variant declares only the staging tensors its formula needs: an
  // all-identity tensor left out of the signature is never uploaded, and one
  // vocab-sized upload dominates the cost of a masking step.
  enum class GraphKind {
    // masked_logits = logits + bias
    kBias,
    // masked_logits = logits * weight
    kWeight,
    // masked_logits = logits * weight + bias
    kWeightBias,
    // masked_logits = relu(logits) * weight + min(logits, 0) * weight_neg
    kSignDependent,
    // ... + bias
    kSignDependentBias,
    // masked_logits = relu(logits) * weight + min(logits, 0) / weight.
    // Equivalent to kSignDependent when every staged weight_neg is the
    // reciprocal of its weight, and saves the weight_neg upload.
    kSignDependentReciprocal,
    // ... + bias
    kSignDependentReciprocalBias,
    kNumGraphKinds,
  };

  // The lazily compiled mask graphs, defined in the implementation file.
  //
  // Holding them behind an incomplete type keeps the LiteRT graph builder out
  // of this header: the runner type is a class template whose destructor alone
  // instantiates a CompiledModelRunner and its hash maps, which every
  // translation unit that destroys a LiteRtLogitMaskRunner would otherwise
  // have to emit.
  struct MaskGraphs;

  struct Config {
    int batch_size = 0;
    int sequence_length = 0;
    int vocab_size = 0;
    ElementType element_type = ElementType::Float32;
    HwAccelerators accelerator = HwAccelerators::kNone;
  };

  LiteRtLogitMaskRunner(Environment* env, HwAccelerators accelerator,
                        bool force_graph_on_host);

  absl::Status Init();

  // Unified implementation for Apply, ApplySequence, and ApplyBatch.
  absl::Status ApplySlices(TensorBuffer& logits,
                           absl::Span<const LogitMask* const> masks,
                           int expected_batch_size, int expected_seq_len);

  // Configures the GPU delegate options so that the graphs' tensors live in
  // the same kind of GPU memory as the logits buffers the caller passes in.
  // Only called when the runner targets a GPU accelerator.
  absl::Status InitGpuOptions();

  // Host staging buffers describing the transform to apply to a single
  // [vocab_size] slice of logits:
  //   z' = z * weight + bias      for z > 0
  //   z' = z * weight_neg + bias  for z <= 0
  template <typename T>
  struct MaskSlices {
    absl::Span<T> bias;
    absl::Span<T> weight;
    // Multiplier for non-positive logits. Equal to `weight` unless an entry is
    // sign dependent, and empty when no mask being unpacked is sign dependent.
    absl::Span<T> weight_neg;
  };

  // Summary of the unpacked masks, used to pick the cheapest graph that can
  // express them.
  struct MaskFeatures {
    bool has_bias = false;            // Any non-zero bias.
    bool has_weights = false;         // Any weight != 1.
    bool has_sign_dependent = false;  // Any weight that depends on logit sign.
    // True while every staged weight_neg is the reciprocal of the weight staged
    // for the same token, which is what lets the reciprocal graph variants
    // derive weight_neg on device. Cleared by a zero weight (no finite
    // reciprocal) or a non-sign-dependent weight (weight_neg is the weight).
    bool reciprocal_neg = true;
  };

  // Token ids UnpackMask wrote into one [vocab_size] staging slice, so that the
  // next call can undo just those entries. Biases and weights are tracked
  // separately: a hard constraint bans most of the vocabulary through the bias
  // alone, and a shared list would then make the weights look wide enough to
  // refill for nothing. `weight` covers weight_neg, which is only ever written
  // alongside weight. Each list stops growing once it is long enough to make
  // the next call reset the whole slice instead of walking it.
  struct SliceDirty {
    std::vector<int> bias;
    std::vector<int> weight;
  };

  // Returns the [vocab_size] staging slices starting at `offset`. The
  // weight_neg slice is empty until EnsureSignDependentStaging() has grown its
  // buffer, which only happens once a mask needs explicit negative weights.
  MaskSlices<float> StagingSlicesF32(int offset);
  MaskSlices<tflite::half> StagingSlicesF16(int offset);

  // Unpacks one mask per slice into the staging buffers, resetting `features`
  // to the combined summary of all of them.
  absl::Status UnpackSlices(absl::Span<const LogitMask* const> masks,
                            MaskFeatures& features);

  // Unpacks a single mask into the host bias and weight staging slices,
  // reproducing the semantics of LogitMask::Apply for the bitmap, sparse and
  // composite mask types. Defined for float and tflite::half.
  //
  // The staging buffers persist across steps, so `dirty` carries the token ids
  // the previous call wrote into this slice; those are reset to the identity
  // transform and `dirty` is refilled with the tokens this call writes.
  //
  // Returns an Unimplemented error for masks the graphs cannot express (custom
  // masks and nested composites) rather than silently ignoring them, which
  // would let disallowed tokens through.
  template <typename T>
  absl::Status UnpackMask(const LogitMask* mask, const MaskSlices<T>& slices,
                          SliceDirty& dirty, MaskFeatures& features);

  // Grows the weight_neg staging buffer to the current configuration, on first
  // use so that callers that never need explicit negative weights do not pay
  // for it.
  void EnsureSignDependentStaging();

  // Whether EnsureSignDependentStaging() has run for the current `config_`.
  bool HasSignDependentStaging() const;

  // Returns the cheapest graph variant that can express `features`.
  static GraphKind SelectGraphKind(const MaskFeatures& features);

  // Whether `kind` declares the named staging tensor as a graph input.
  static bool NeedsBias(GraphKind kind);
  static bool NeedsWeight(GraphKind kind);
  static bool NeedsWeightNeg(GraphKind kind);

  // Compiles `kind` if it has not been compiled yet. Variants are built on
  // demand: each one costs a model build plus a delegate compilation and a
  // caller typically only ever exercises one.
  absl::Status EnsureRunner(GraphKind kind);

  // Feeds the staged buffers into `kind` and updates `logits` with the result.
  // The graph runs in place on `logits` unless `needs_copy_back_` is set, in
  // which case it writes to its own output buffer and the copy-back graph
  // copies that back into `logits`.
  absl::Status RunMaskGraph(TensorBuffer& logits, GraphKind kind);

  Environment* env_ = nullptr;
  Config config_;
  Options options_;

  // Test-only: send host-memory buffers through the graph path instead of
  // masking them in place. See Create().
  const bool force_graph_on_host_ = false;

  // Whether Init() has run for the current `config_`. The graphs themselves are
  // compiled lazily by EnsureRunner().
  bool initialized_ = false;

  // The lazily compiled mask graphs, and the copy-back graph that
  // `needs_copy_back_` selects. Always non-null; see MaskGraphs for why these
  // live behind a pointer rather than directly in this class.
  std::unique_ptr<MaskGraphs> graphs_;

  // Whether the target accelerator refuses to alias a graph's input and output.
  // A mask graph otherwise runs in place, binding the caller's buffer as both
  // "logits" and "masked_logits", which is safe because the graphs are strictly
  // elementwise. WebGPU and Vulkan reject a dispatch that binds one buffer as
  // both read-only and read-write storage, so there the graph writes to its own
  // output and the copy-back graph copies it back in a second dispatch.
  bool needs_copy_back_ = false;

  // Persistent host staging buffers to avoid per-step dynamic allocations. The
  // weight_neg buffers stay empty until a graph needs them as an explicit
  // input, which the reciprocal variants never do.
  std::vector<float> host_bias_f32_;
  std::vector<float> host_weight_f32_;
  std::vector<float> host_weight_neg_f32_;
  std::vector<tflite::half> host_bias_f16_;
  std::vector<tflite::half> host_weight_f16_;
  std::vector<tflite::half> host_weight_neg_f16_;

  // Marks the tokens of the slice currently being unpacked that a hard (bitmap)
  // constraint has already disallowed, so that soft sparse penalties skip them
  // the way SparseLogitMask::Apply leaves an already -inf logit untouched. A
  // token is disallowed iff its entry equals `disallowed_generation_`, which
  // UnpackMask bumps on entry; stamping rather than clearing keeps the per-step
  // cost proportional to the number of banned tokens.
  std::vector<uint32_t> disallowed_stamp_;
  uint32_t disallowed_generation_ = 0;

  // One SliceDirty per staging slice (batch_size * sequence_length entries).
  // May contain duplicates; resetting an entry twice is harmless.
  std::vector<SliceDirty> dirty_tokens_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LITERT_LOGIT_MASK_RUNNER_H_
