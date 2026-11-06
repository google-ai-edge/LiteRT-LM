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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_MTP_DRAFTER_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_MTP_DRAFTER_SUPPORT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_opaque_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_requirements.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

// What the shared MTP drafter needs from NVIDIA's TensorRT-RTX backend, which
// LiteRT reaches through its generic NPU route (Backend::NPU with the generic
// compiler plugin). Everything NVIDIA-specific about the drafter lives here:
// the drafter itself only calls AddNvidiaSharedValueCacheOptions when it
// compiles and CreateMtpGreedySampler when it samples.
//
// The drafter reads the base model's KV caches and writes none of its own.
// The two models are compiled separately but share those buffers, so the
// physical layout the base model's compiler chose for a value cache has to be
// told to the drafter's compiler; and the sampler can take the token ids
// straight from CUDA logits instead of downloading them.
namespace litert::lm {

// The NVIDIA dispatch library the runtime has loaded already, or nullopt
// (another instance would not share the tensor buffers' CUDA allocator and
// runtime context).
absl::StatusOr<std::optional<SharedLibrary>> LoadedDispatchLibrary(
    const Environment& env);

// Whether the base model stores a logical [1, H, D, S] value cache in NVIDIA's
// physical [1, H, S, D] layout, from the producer's buffer requirements: true
// for CUDA buffers with the transposed strides, false for other buffer types
// and for dense CUDA caches, an error for CUDA strides of any other shape.
absl::StatusOr<bool> UsesNvidiaTransposedValueCache(
    const Layout& logical_layout,
    const TensorBufferRequirements& producer_requirements);

// The compilation options that tell NVIDIA's compiler which of the drafter's
// cache inputs are read-only views of transposed value caches. The names are
// sorted and owned by the options, and the options hash into the compiled
// drafter's cache key, so two drafters sharing the same caches share an engine.
Expected<OpaqueOptions> CreateNvidiaReadOnlyValueCacheOptions(
    std::vector<std::string> transposed_input_names);

// The compilation options that tell NVIDIA's compiler which of the drafter's
// inputs are KV caches the base model keeps in place, so the drafter's engine
// reads them with the storage the base model's engine gave them (FP16 for the
// float32 caches of Gemma 4 26B). Sorted, owned and hashed like the above.
Expected<OpaqueOptions> CreateNvidiaSharedCacheInputs(
    std::vector<std::string> input_names);

// Adds those options to the drafter's compilation options when `backend` is
// the generic NPU route: every `kv_cache_` input of the drafter as a shared
// cache input, and the ones the base model's compiled decode signature stores
// transposed as read-only value caches. Other backends leave the options as
// they were; a transposed key cache is an error.
absl::Status AddNvidiaSharedValueCacheOptions(Backend backend,
                                              const Model& drafter_model,
                                              CompiledModel& base_model,
                                              Options& compilation_options);

// Metadata-only eligibility checks. Malformed shapes on otherwise matching
// NVIDIA float buffers are errors; other backends, types and configurations
// are unsupported. The greedy sampler takes the fixed greedy configuration
// (TOP_P, k=1, p=0, temperature 1); the candidate selection any TOP_P
// configuration with 1 <= k <= 1024 (LiteRtDispatchNvidiaTopKCandidatesMaxK).
absl::StatusOr<bool> SupportsNvidiaGreedySampler(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type);

absl::StatusOr<bool> SupportsNvidiaTopKCandidates(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type);

// Optional MTP argmax extension. Returns nullptr to keep the existing CPU/GPU
// sampler when disabled, unsupported, or absent from the already-loaded NVIDIA
// dispatch library. Once matched, operational errors are returned, not hidden.
// On unless LITERT_NVIDIA_MTP_GPU_SAMPLING=0; never loads a new dispatch
// instance.
absl::StatusOr<std::unique_ptr<Sampler>> TryCreateNvidiaGreedySampler(
    const Environment& env, Backend backend,
    const proto::SamplerParameters& sampler_params,
    const TensorBuffer& logits_tensor);

// LiteRT-LM's sampling policy on the candidates NVIDIA's dispatch selected:
// TopKTopPSampling's (runtime/components/sampling_cpu_util.cc: its Softmax,
// top-p prefix, pick and score) on the rows' k candidate logits in the
// candidates' order (logit descending, then index ascending: the order the
// CPU sampler's own selection gives), picking each row's token with its value
// of `uniforms` (one per row, in [0, 1): what DrawSamplingUniforms draws for
// the CPU sampler), the compact ids mapped back through `candidate_ids`.
// Equal probabilities keep that order, as the device's sampling of the steps
// ahead does. Whenever the k-th largest logit of a row is unique and the
// uniform does not land on tied probabilities, the candidates, the
// probabilities, the pick and the score are those of the CPU sampler on the
// full logits with the same uniforms, bit for bit; where the CPU's heap keeps
// an unspecified subset of the tokens tied at the k-th value, the device
// keeps the lowest indices, and where its unstable sort orders tied
// probabilities, these keep theirs: the sampled token differs only when the
// uniform lands on such a token. ids_tensor gets the rows' tokens;
// scores_tensor, when given, the log of each token's probability among its
// candidates, as TopPSampler writes it without exact log probabilities.
absl::Status SampleNvidiaCandidates(absl::Span<const int32_t> candidate_ids,
                                    absl::Span<const float> candidate_values,
                                    int rows, int k, float p, float temperature,
                                    absl::Span<const float> uniforms,
                                    TensorBuffer& ids_tensor,
                                    TensorBuffer* scores_tensor);

// The NPU route's sampler, with the signature of
// runtime/components/sampler_factory.h's CreateSampler so the drafter calls
// it in that one's place: the drafter's greedy sampler, and the verifier's
// sampler before its session configuration. It decides on the first logits
// it is given: on CUDA buffers of a dense float type NVIDIA's argmax for the
// greedy configuration, or NVIDIA's candidate selection sampled with
// SampleNvidiaCandidates for any other TOP_P configuration up to k = 1024
// (LITERT_NVIDIA_MTP_GPU_SAMPLING=0 opts out of both); otherwise the generic
// CPU sampler it keeps (the generic factory has no NPU sampler). Both paths
// pick the tokens with the caller's uniforms, so a session's stream is one
// stream whichever path samples a round. What the candidate
// path cannot do, the host sampler does before any randomness is consumed:
// scores with exact log probabilities, a NaN among the logits (the dispatch
// flags it), logits the dispatch declines at selection, k above the device's
// reach. The argmax path is upstream's: a NaN is the first maximum's
// business (a NaN at index zero wins, other NaNs are ignored), and a
// dispatch refusal is an error; a scores tensor with the fixed greedy
// configuration takes log(1) = 0 per row, as the host sampler writes for k
// 1, unless exact log probabilities are asked for (the host's). Host logits
// and a batch above 1 ([N, rows, vocab]) take the host sampler; a padded
// CUDA float buffer is an error at the sampler's decision, as the greedy
// sampler has it. The other backends get the factory's sampler, as before.
//
// `tells_step_feed`: the sampler is the one that samples the session's
// tokens (the executor's own, when the session has no sampler), and tells
// the decode step that feeds itself how it samples (LiteRT's
// step_sampler_c_api.h); a drafter's or a verifier's sampler does not, its
// configuration not being the session's.
absl::StatusOr<std::unique_ptr<Sampler>> CreateMtpGreedySampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env = std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt,
    bool tells_step_feed = false);

// The executor's sampler, with the signature of the factory's CreateSampler
// plus the executor's backend, so the executor calls it in that one's place:
// for an executor on NVIDIA's TensorRT-RTX backend (Backend::NPU, with
// NVIDIA's dispatch library loaded) whose sampler backend is the CPU, the
// route's default, the NPU route's sampler of CreateMtpGreedySampler, which
// also tells the decode step that feeds itself how it samples
// (LITERT_NVIDIA_MTP_GPU_SAMPLING=0 keeps it on the host); otherwise the
// factory's sampler for `sampler_backend`.
absl::StatusOr<std::unique_ptr<Sampler>> CreateExecutorSampler(
    Backend executor_backend, Backend sampler_backend, int batch_size,
    proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env = std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

// The MTP verifier's session sampler, with the signature of the factory's
// UpdateOrRecreateSampler so the drafter calls it in that one's place. The
// sampler of CreateMtpGreedySampler (the NPU route's, known by what it is,
// whatever backend the drafter names for its session sampler: the CPU on
// that route) takes the configuration, recreating its host sampler as the
// factory would when k outgrows it, with the sequence size, vocabulary and
// activation type given here; a
// null sampler on Backend::NPU is created as that when NVIDIA's dispatch
// library is loaded; any other sampler, on any backend, goes through
// UpdateOrRecreateSampler. `max_top_k` is the
// capacity the sampler has seen, raised after an update that took.
absl::Status UpdateOrRecreateMtpVerifierSampler(
    std::unique_ptr<Sampler>& sampler, int& max_top_k, Backend backend,
    int batch_size, const proto::SamplerParameters& sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env = std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_MTP_DRAFTER_SUPPORT_H_
