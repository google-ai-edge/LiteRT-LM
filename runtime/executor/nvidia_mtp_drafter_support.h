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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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

// Metadata-only eligibility check. Malformed shapes on otherwise matching
// NVIDIA float buffers are errors; other backends/types/configs are
// unsupported.
absl::StatusOr<bool> SupportsNvidiaGreedySampler(
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

// The drafter's and the verifier's greedy sampler, with the signature of
// runtime/components/sampler_factory.h's CreateSampler so the drafter calls
// it in that one's place. On the NPU route the sampler decides on the first
// logits it is given: NVIDIA's CUDA argmax when they are CUDA buffers of a
// dense float type (LITERT_NVIDIA_MTP_GPU_SAMPLING=0 opts out), otherwise the
// generic CPU sampler it was created with (the generic factory has no NPU
// sampler). The other backends get the factory's sampler, as before.
absl::StatusOr<std::unique_ptr<Sampler>> CreateMtpGreedySampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env = std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_MTP_DRAFTER_SUPPORT_H_
