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

#include "runtime/executor/nvidia_mtp_drafter_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"  // from @litert
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_opaque_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_requirements.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "litert/vendors/nvidia/dispatch/greedy_sampler_c_api.h"  // from @litert
#include "litert/vendors/nvidia/dispatch/step_sampler_c_api.h"  // from @litert
#include "litert/vendors/nvidia/dispatch/topk_candidates_c_api.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/components/sampling_uniforms.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// An option payload (a C struct of `num_inputs` and `input_names`) with the
// storage of its names: the struct points into `names`, which stay alive as
// long as the options do.
template <typename Payload>
struct OwnedCacheInputNames : Payload {
  explicit OwnedCacheInputNames(std::vector<std::string> cache_input_names)
      : names(std::move(cache_input_names)) {
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
      name_pointers.push_back(name.c_str());
    }
    this->num_inputs = static_cast<uint32_t>(name_pointers.size());
    this->input_names = name_pointers.data();
  }

  std::vector<std::string> names;
  std::vector<const char*> name_pointers;
};

template <typename Payload>
Expected<OpaqueOptions> CreateCacheInputNameOptions(
    const char* identifier, uint64_t (*hash)(const void*),
    std::vector<std::string> names) {
  auto payload =
      std::make_unique<OwnedCacheInputNames<Payload>>(std::move(names));
  LITERT_ASSIGN_OR_RETURN(
      auto opaque_options,
      OpaqueOptions::Create(
          identifier, static_cast<Payload*>(payload.get()), [](void* data) {
            delete static_cast<OwnedCacheInputNames<Payload>*>(
                static_cast<Payload*>(data));
          }));
  payload.release();
  LITERT_RETURN_IF_ERROR(opaque_options.SetHash(hash));
  return opaque_options;
}

}  // namespace

absl::StatusOr<bool> UsesNvidiaTransposedValueCache(
    const Layout& logical_layout,
    const TensorBufferRequirements& producer_requirements) {
  LITERT_ASSIGN_OR_RETURN(auto buffer_types,
                          producer_requirements.SupportedTypes());
  const auto cuda_type =
      static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType);
  if (absl::c_find(buffer_types, cuda_type) == buffer_types.end()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto strides, producer_requirements.Strides());
  if (strides.empty()) {
    return false;
  }

  const auto dimensions = logical_layout.Dimensions();
  bool contiguous = strides.size() == dimensions.size();
  uint64_t dense_stride = 1;
  for (size_t i = dimensions.size(); contiguous && i > 0; --i) {
    contiguous = dimensions[i - 1] > 0 && strides[i - 1] == dense_stride;
    if (contiguous) {
      dense_stride *= dimensions[i - 1];
    }
  }
  if (contiguous) {
    return false;
  }

  uint32_t expected_strides[4];
  if (strides.size() == 4 &&
      nvidia::GetTransposedValueCacheStrides(
          static_cast<const LiteRtLayout&>(logical_layout), expected_strides) &&
      std::equal(strides.begin(), strides.end(), expected_strides)) {
    return true;
  }
  return absl::InvalidArgumentError(
      "Unsupported NVIDIA shared value-cache strides");
}

Expected<OpaqueOptions> CreateNvidiaReadOnlyValueCacheOptions(
    std::vector<std::string> transposed_input_names) {
  return CreateCacheInputNameOptions<LiteRtNvidiaReadOnlyValueCacheOptions>(
      LITERT_NVIDIA_READ_ONLY_VALUE_CACHE_OPTIONS_ID,
      nvidia::HashReadOnlyValueCacheOptions, std::move(transposed_input_names));
}

Expected<OpaqueOptions> CreateNvidiaSharedCacheInputs(
    std::vector<std::string> input_names) {
  return CreateCacheInputNameOptions<LiteRtNvidiaSharedCacheInputs>(
      LITERT_NVIDIA_SHARED_CACHE_INPUTS_ID, nvidia::HashSharedCacheInputs,
      std::move(input_names));
}

absl::Status AddNvidiaSharedValueCacheOptions(Backend backend,
                                              const Model& drafter_model,
                                              CompiledModel& base_model,
                                              Options& compilation_options) {
  if (backend != Backend::NPU) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto signature, drafter_model.GetSignature(0));
  std::vector<std::string> shared_inputs;
  std::vector<std::string> transposed_inputs;
  for (absl::string_view input_name : signature.InputNames()) {
    if (!absl::StartsWith(input_name, "kv_cache_")) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto tensor, signature.InputTensor(input_name));
    shared_inputs.emplace_back(tensor.Name());
    LITERT_ASSIGN_OR_RETURN(
        auto requirements,
        base_model.GetOutputBufferRequirements("decode", input_name));
    // Use the compiled producer shape: the drafter source may still contain
    // context-length magic numbers that compilation will replace.
    LITERT_ASSIGN_OR_RETURN(auto producer_type, base_model.GetOutputTensorType(
                                                    "decode", input_name));
    ABSL_ASSIGN_OR_RETURN(
        bool transposed,
        UsesNvidiaTransposedValueCache(producer_type.Layout(), requirements));
    if (!transposed) {
      continue;
    }
    if (!absl::StrContains(input_name, "_v_")) {
      return absl::InvalidArgumentError(absl::StrCat(
          "NVIDIA transposed layout requires a value cache: ", input_name));
    }
    transposed_inputs.emplace_back(tensor.Name());
  }
  if (!shared_inputs.empty()) {
    LITERT_ASSIGN_OR_RETURN(auto shared_options,
                            CreateNvidiaSharedCacheInputs(shared_inputs));
    LITERT_RETURN_IF_ERROR(
        compilation_options.AddOpaqueOptions(std::move(shared_options)));
  }
  if (transposed_inputs.empty()) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(
      auto opaque_options,
      CreateNvidiaReadOnlyValueCacheOptions(std::move(transposed_inputs)));
  LITERT_RETURN_IF_ERROR(
      compilation_options.AddOpaqueOptions(std::move(opaque_options)));
  return absl::OkStatus();
}

// The device samplers of NVIDIA's dispatch library, for the drafter's and the
// verifier's logits: the CUDA argmax for the greedy configuration, and for
// any other top-k / top-p configuration the k candidates selected on the
// device and sampled here with LiteRT-LM's own policy.
namespace {

constexpr int kNvidiaTopKCandidatesMaxK = 1024;

bool IsFixedGreedy(const proto::SamplerParameters& params) {
  return params.type() == proto::SamplerParameters::TOP_P && params.k() == 1 &&
         params.p() == 0.0f && params.temperature() == 1.0f;
}

// A top-k / top-p configuration the candidate path takes: what the CPU
// sampler accepts (TopPSampler::Create), with k within the device's reach.
bool IsTopP(const proto::SamplerParameters& params) {
  return params.type() == proto::SamplerParameters::TOP_P && params.k() >= 1 &&
         params.k() <= kNvidiaTopKCandidatesMaxK && params.p() >= 0.0f &&
         params.p() <= 1.0f && std::isfinite(params.temperature()) &&
         params.temperature() >= 0.0f;
}

// On by default for NVIDIA logits; LITERT_NVIDIA_MTP_GPU_SAMPLING=0 keeps
// the host samplers (the logits then travel to the host every step): the MTP
// drafter's and verifier's, and the executor's own
// (CreateExecutorSampler).
bool GpuSamplingOptedOut() {
  const char* enabled = std::getenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
  return enabled != nullptr && std::strcmp(enabled, "0") == 0;
}

bool IsDense(const Layout& layout) {
  if (!layout.HasStrides()) {
    return true;
  }
  const auto dims = layout.Dimensions();
  const auto strides = layout.Strides();
  uint64_t stride = 1;
  for (size_t i = dims.size(); i > 0; --i) {
    if (dims[i - 1] <= 0 || strides[i - 1] != stride) {
      return false;
    }
    stride *= dims[i - 1];
  }
  return true;
}

absl::StatusOr<bool> SupportsLogits(TensorBufferType buffer_type,
                                    const RankedTensorType& type) {
  if (buffer_type !=
          static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType) ||
      (type.ElementType() != ElementType::Float16 &&
       type.ElementType() != ElementType::Float32)) {
    return false;
  }
  const auto dims = type.Layout().Dimensions();
  if (dims.size() == 3 && dims[0] > 1 && dims[1] > 0 && dims[2] > 0 &&
      IsDense(type.Layout())) {
    // A batch above one ([N, rows, vocab]: a session with N candidates on a
    // batched decode) samples on the host, as before the device paths.
    return false;
  }
  if (dims.size() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 ||
      !IsDense(type.Layout())) {
    return absl::InvalidArgumentError(
        "NVIDIA device samplers require dense [1, rows, vocab] logits");
  }
  return true;
}

absl::Status CheckIdsTensor(const TensorBuffer& ids_tensor, int rows) {
  LITERT_ASSIGN_OR_RETURN(auto ids_type, ids_tensor.TensorType());
  const auto ids_dims = ids_type.Layout().Dimensions();
  if (ids_type.ElementType() != ElementType::Int32 || ids_dims.size() != 2 ||
      ids_dims[0] != 1 || ids_dims[1] != rows || !IsDense(ids_type.Layout())) {
    return absl::InvalidArgumentError(
        "NVIDIA device samplers require dense int32 [1, rows] output IDs");
  }
  return absl::OkStatus();
}

}  // namespace

// The dispatch library the runtime has loaded already, or nothing (NoLoad:
// another instance would not share the tensor buffers' CUDA allocator and
// runtime context).
absl::StatusOr<std::optional<SharedLibrary>> LoadedDispatchLibrary(
    const Environment& env) {
#if (defined(__linux__) || defined(_WIN32)) && defined(RTLD_NOLOAD)
  LITERT_ASSIGN_OR_RETURN(auto options, env.GetOptions());
  auto directory =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  if (!directory.HasValue()) {
    return std::nullopt;
  }
  const auto* directory_string = std::get_if<const char*>(&directory.Value());
  if (directory_string == nullptr || *directory_string == nullptr) {
    return absl::InvalidArgumentError(
        "NVIDIA dispatch library directory is not a string");
  }
  const std::string path =
      absl::StrCat(*directory_string, "/libLiteRtDispatch_Nvidia.so");
  auto library = SharedLibrary::Load(path, RtldFlags::Now().NoLoad().Local());
  if (!library.HasValue()) {
    return std::nullopt;
  }
  return std::optional<SharedLibrary>(std::move(library.Value()));
#else
  return std::nullopt;
#endif
}

namespace {

class NvidiaGreedySampler final : public Sampler {
 public:
  using CreateFn = decltype(&LiteRtDispatchNvidiaGreedySamplerCreate);
  using DestroyFn = decltype(&LiteRtDispatchNvidiaGreedySamplerDestroy);
  using SampleFn = decltype(&LiteRtDispatchNvidiaGreedySamplerSampleBatched);

  NvidiaGreedySampler(SharedLibrary library,
                      LiteRtDispatchNvidiaGreedySampler sampler,
                      DestroyFn destroy, SampleFn sample, int rows, int vocab,
                      std::vector<int32_t> ids)
      : library_(std::move(library)),
        sampler_(sampler),
        destroy_(destroy),
        sample_(sample),
        rows_(rows),
        vocab_(vocab),
        ids_(std::move(ids)) {}

  ~NvidiaGreedySampler() override { destroy_(sampler_); }

  // Greedy sampling is deterministic, so the uniforms are only validated, as
  // the host sampler does.
  absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor, absl::Span<const float> uniforms) override {
    ABSL_RETURN_IF_ERROR(ValidateUniforms(uniforms, rows_));
    if (scores_tensor != nullptr) {
      return absl::UnimplementedError(
          "NVIDIA greedy sampler returns token IDs, not probability scores");
    }
    LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_tensor.BufferType());
    LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
    ABSL_ASSIGN_OR_RETURN(bool supported,
                          SupportsLogits(buffer_type, logits_type));
    if (!supported) {
      return absl::InvalidArgumentError(
          "NVIDIA greedy sampler logits buffer type changed after creation");
    }
    const auto logits_dims = logits_type.Layout().Dimensions();
    if (logits_dims[1] != rows_ || logits_dims[2] != vocab_) {
      return absl::InvalidArgumentError(
          "NVIDIA greedy sampler logits shape changed after creation");
    }
    ABSL_RETURN_IF_ERROR(CheckIdsTensor(ids_tensor, rows_));
    // Dispatch consumes the CUDA allocation directly and synchronizes the small
    // ID download. Do not lock/read the logits tensor on the host here.
    LITERT_RETURN_IF_ERROR(
        sample_(sampler_, logits_tensor.Get(), rows_, vocab_, ids_.data()));
    LITERT_RETURN_IF_ERROR(ids_tensor.Write<int32_t>(ids_));
    return absl::OkStatus();
  }

  absl::Status UpdateConfig(const proto::SamplerParameters& sampler_params,
                            int batch_size) override {
    if (batch_size != 1 || !IsFixedGreedy(sampler_params)) {
      return absl::UnimplementedError(
          "NVIDIA MTP sampler supports only batch-1 TOP_P with k=1 and p=0");
    }
    return absl::OkStatus();
  }

 private:
  // Keep the initialized dispatch DSO alive until sampler destruction
  // completes.
  SharedLibrary library_;
  LiteRtDispatchNvidiaGreedySampler sampler_;
  DestroyFn destroy_;
  SampleFn sample_;
  int rows_;
  int vocab_;
  std::vector<int32_t> ids_;
};

// The dispatch library's candidate selection on CUDA logits, and LiteRT-LM's
// sampling policy on what it selects.
class NvidiaCandidateSampler {
 public:
  using CreateFn = decltype(&LiteRtDispatchNvidiaTopKCandidatesCreate);
  using DestroyFn = decltype(&LiteRtDispatchNvidiaTopKCandidatesDestroy);
  using SelectFn = decltype(&LiteRtDispatchNvidiaTopKCandidatesSelect);

  NvidiaCandidateSampler(SharedLibrary library,
                         LiteRtDispatchNvidiaTopKCandidates selector,
                         DestroyFn destroy, SelectFn select, int rows,
                         int vocab, int max_k)
      : library_(std::move(library)),
        selector_(selector),
        destroy_(destroy),
        select_(select),
        rows_(rows),
        vocab_(vocab),
        max_k_(max_k) {}

  ~NvidiaCandidateSampler() { destroy_(selector_); }

  // Samples every row from its k candidates with `params`, picking with
  // `uniforms` (one per row): true when it did; false when a row's
  // candidates are unspecified (a NaN among its logits) and the host sampler
  // must run.
  absl::StatusOr<bool> TrySample(const TensorBuffer& logits_tensor,
                                 const proto::SamplerParameters& params,
                                 absl::Span<const float> uniforms,
                                 TensorBuffer& ids_tensor,
                                 TensorBuffer* scores_tensor) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_tensor.BufferType());
    LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
    ABSL_ASSIGN_OR_RETURN(bool supported,
                          SupportsLogits(buffer_type, logits_type));
    if (!supported) {
      return absl::InvalidArgumentError(
          "NVIDIA candidate sampler logits buffer type changed after creation");
    }
    const auto logits_dims = logits_type.Layout().Dimensions();
    if (logits_dims[1] != rows_ || logits_dims[2] != vocab_) {
      return absl::InvalidArgumentError(
          "NVIDIA candidate sampler logits shape changed after creation");
    }
    ABSL_RETURN_IF_ERROR(CheckIdsTensor(ids_tensor, rows_));
    ABSL_RETURN_IF_ERROR(ValidateUniforms(uniforms, rows_));
    // TopKTopPSampling takes no more candidates than the vocabulary has.
    const int k = std::min(params.k(), vocab_);
    if (k < 1 || k > max_k_) {
      return absl::InvalidArgumentError(
          absl::StrCat("NVIDIA candidate sampler takes k in [1, ", max_k_,
                       "], got ", params.k()));
    }
    const size_t slots = static_cast<size_t>(rows_) * k;
    candidate_ids_.resize(slots);
    candidate_values_.resize(slots);
    row_flags_.resize(rows_);
    const LiteRtStatus selected = select_(
        selector_, logits_tensor.Get(), rows_, vocab_, k, candidate_ids_.data(),
        candidate_values_.data(), row_flags_.data());
    if (selected == kLiteRtStatusErrorUnsupported) {
      // The dispatch declines these logits (another device, a layout it does
      // not take): the host samples them, with nothing drawn yet.
      return false;
    }
    LITERT_RETURN_IF_ERROR(selected);
    for (int32_t flags : row_flags_) {
      if ((flags & LITERT_DISPATCH_NVIDIA_TOPK_CANDIDATES_NAN) != 0) {
        return false;
      }
    }
    ABSL_RETURN_IF_ERROR(SampleNvidiaCandidates(
        candidate_ids_, candidate_values_, rows_, k, params.p(),
        params.temperature(), uniforms, ids_tensor, scores_tensor));
    return true;
  }

 private:
  SharedLibrary library_;
  LiteRtDispatchNvidiaTopKCandidates selector_;
  DestroyFn destroy_;
  SelectFn select_;
  int rows_;
  int vocab_;
  int max_k_;
  std::vector<int32_t> candidate_ids_;
  std::vector<float> candidate_values_;
  std::vector<int32_t> row_flags_;
};

absl::StatusOr<std::unique_ptr<NvidiaCandidateSampler>>
TryCreateNvidiaCandidateSampler(const Environment& env,
                                const proto::SamplerParameters& sampler_params,
                                const TensorBuffer& logits_tensor) {
  if (GpuSamplingOptedOut() || !IsTopP(sampler_params)) {
    ABSL_LOG(INFO) << "NVIDIA candidate selection not taken: "
                   << (GpuSamplingOptedOut() ? "opted out"
                                             : "not a top-p configuration");
    return std::unique_ptr<NvidiaCandidateSampler>();
  }
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_tensor.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
  ABSL_ASSIGN_OR_RETURN(bool supported,
                        SupportsLogits(buffer_type, logits_type));
  if (!supported) {
    ABSL_LOG(INFO) << "NVIDIA candidate selection not taken: logits buffer "
                      "type "
                   << static_cast<int>(buffer_type)
                   << " is not a CUDA float "
                      "buffer; the host samples";
    return std::unique_ptr<NvidiaCandidateSampler>();
  }
  ABSL_ASSIGN_OR_RETURN(auto library, LoadedDispatchLibrary(env));
  if (!library.has_value()) {
    ABSL_LOG(INFO) << "NVIDIA dispatch library not loaded by the runtime; "
                      "the MTP verifier samples on the host";
    return std::unique_ptr<NvidiaCandidateSampler>();
  }
  using MaxKFn = decltype(&LiteRtDispatchNvidiaTopKCandidatesMaxK);
  auto max_k =
      library->LookupSymbol<MaxKFn>("LiteRtDispatchNvidiaTopKCandidatesMaxK");
  auto create = library->LookupSymbol<NvidiaCandidateSampler::CreateFn>(
      "LiteRtDispatchNvidiaTopKCandidatesCreate");
  auto destroy = library->LookupSymbol<NvidiaCandidateSampler::DestroyFn>(
      "LiteRtDispatchNvidiaTopKCandidatesDestroy");
  auto select = library->LookupSymbol<NvidiaCandidateSampler::SelectFn>(
      "LiteRtDispatchNvidiaTopKCandidatesSelect");
  if (!max_k.HasValue() || !create.HasValue() || !destroy.HasValue() ||
      !select.HasValue()) {
    ABSL_LOG(INFO) << "NVIDIA dispatch library exports no top-k candidate "
                      "selection; the MTP verifier samples on the host";
    return std::unique_ptr<NvidiaCandidateSampler>();
  }
  const auto dims = logits_type.Layout().Dimensions();
  const int device_max_k =
      std::min(static_cast<int>(max_k.Value()()), kNvidiaTopKCandidatesMaxK);
  if (sampler_params.k() > device_max_k && dims[2] > device_max_k) {
    return std::unique_ptr<NvidiaCandidateSampler>();
  }
  LiteRtDispatchNvidiaTopKCandidates selector = nullptr;
  LITERT_RETURN_IF_ERROR(create.Value()(&selector));
  if (selector == nullptr) {
    return absl::InternalError(
        "NVIDIA top-k candidate selection returned a null handle");
  }
  ABSL_LOG(INFO) << "NVIDIA MTP GPU candidate selection enabled for " << dims[1]
                 << " rows, vocabulary " << dims[2] << " (k "
                 << sampler_params.k() << ", p " << sampler_params.p()
                 << ", temperature " << sampler_params.temperature() << ")";
  return std::make_unique<NvidiaCandidateSampler>(
      *std::move(library), selector, destroy.Value(), select.Value(), dims[1],
      dims[2], device_max_k);
}

}  // namespace

namespace {

// LiteRT-LM's top-k / top-p policy (sampling_cpu_util.cc's TopKTopPSampling) on
// rows of k candidates sorted by logit, descending, then by index: the order
// TopKTokenIds gives them and the order the device's step sampling (LiteRT's
// step_sample_kernel) walks them. The softmax is TopKTopPSampling's own, and
// so are the top-p prefix and the pick with each row's uniform; equal
// probabilities keep the candidates' order, which TopKTopPSampling's
// unstable sort leaves unspecified, so this host and the device pick the
// same candidate for the same draw. Each row's pick (a compact index) and the
// probability it was sampled with; the walk adds the top-p prefix up again in
// the same order, so the draw (at most the prefix's sum) always lands on a
// candidate, and the initial pick (the first candidate, probability 0) is
// never what a row keeps.
absl::Status SampleSortedCandidates(absl::Span<const float> values, int rows,
                                    int k, float p, float temperature,
                                    absl::Span<const float> uniforms,
                                    std::vector<int>& picks,
                                    std::vector<float>& probabilities) {
  std::vector<int> order(static_cast<size_t>(rows) * k);
  for (int row = 0; row < rows; ++row) {
    std::iota(order.begin() + static_cast<size_t>(row) * k,
              order.begin() + static_cast<size_t>(row + 1) * k, 0);
  }
  std::vector<std::vector<float>> max_logits;
  ABSL_ASSIGN_OR_RETURN(auto softmax,
                        Softmax(values, order, temperature, /*batch_size=*/1,
                                /*sequence_size=*/rows, max_logits));
  picks.assign(rows, 0);
  probabilities.assign(rows, 0.0f);
  if (k == 1) {
    std::fill(probabilities.begin(), probabilities.end(), 1.0f);
    return absl::OkStatus();
  }
  const float current_temp =
      std::max(temperature, std::numeric_limits<float>::epsilon());
  for (int row = 0; row < rows; ++row) {
    const float* row_probabilities =
        softmax[0].data() + static_cast<size_t>(row) * k;
    double cumulative_prob = 0.0;
    int final_sample_size = 0;
    for (int i = 0; i < k; ++i) {
      cumulative_prob += row_probabilities[i];
      final_sample_size = i + 1;
      if (cumulative_prob >= p) break;
    }
    if (cumulative_prob <= std::numeric_limits<double>::epsilon()) {
      probabilities[row] =
          std::exp((values[static_cast<size_t>(row) * k] - max_logits[0][row]) /
                   current_temp);
      continue;
    }
    const double random_sample =
        static_cast<double>(uniforms[row]) * cumulative_prob;
    double current_cumulative = 0.0;
    for (int i = 0; i < final_sample_size; ++i) {
      current_cumulative += row_probabilities[i];
      if (random_sample <= current_cumulative) {
        picks[row] = i;
        probabilities[row] = row_probabilities[i];
        break;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SampleNvidiaCandidates(absl::Span<const int32_t> candidate_ids,
                                    absl::Span<const float> candidate_values,
                                    int rows, int k, float p, float temperature,
                                    absl::Span<const float> uniforms,
                                    TensorBuffer& ids_tensor,
                                    TensorBuffer* scores_tensor) {
  if (rows <= 0 || k <= 0 ||
      candidate_ids.size() != static_cast<size_t>(rows) * k ||
      candidate_values.size() != candidate_ids.size()) {
    return absl::InvalidArgumentError(
        "NVIDIA candidates: rows * k ids and values");
  }
  ABSL_RETURN_IF_ERROR(ValidateUniforms(uniforms, rows));
  // The CPU sampler's policy on the candidates, in their order: with k
  // candidates for a vocabulary of k, TopKTopPSampling would select them all
  // in the order they come in; the softmax, the top-p prefix, the pick by
  // each row's uniform and the scores are its own, and equal probabilities
  // keep that order.
  std::vector<int> picks;
  std::vector<float> probabilities;
  ABSL_RETURN_IF_ERROR(SampleSortedCandidates(candidate_values, rows, k, p,
                                              temperature, uniforms, picks,
                                              probabilities));
  std::vector<int32_t> ids(rows);
  std::vector<float> scores(rows);
  for (int row = 0; row < rows; ++row) {
    ids[row] = candidate_ids[static_cast<size_t>(row) * k + picks[row]];
    // The scores are the log of the probability of the sampled token, as
    // TopPSampler writes them without exact log probabilities.
    scores[row] = std::log(probabilities[row]);
  }
  LITERT_RETURN_IF_ERROR(ids_tensor.Write<int32_t>(ids));
  if (scores_tensor != nullptr) {
    LITERT_RETURN_IF_ERROR(scores_tensor->Write<float>(scores));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> SupportsNvidiaGreedySampler(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type) {
  if (backend != Backend::NPU || !IsFixedGreedy(sampler_params)) {
    return false;
  }
  return SupportsLogits(buffer_type, logits_type);
}

absl::StatusOr<bool> SupportsNvidiaTopKCandidates(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type) {
  if (backend != Backend::NPU || !IsTopP(sampler_params)) {
    return false;
  }
  return SupportsLogits(buffer_type, logits_type);
}

absl::StatusOr<std::unique_ptr<Sampler>> TryCreateNvidiaGreedySampler(
    const Environment& env, Backend backend,
    const proto::SamplerParameters& sampler_params,
    const TensorBuffer& logits_tensor) {
  if (GpuSamplingOptedOut() || backend != Backend::NPU ||
      !IsFixedGreedy(sampler_params)) {
    return std::unique_ptr<Sampler>();
  }
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_tensor.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
  ABSL_ASSIGN_OR_RETURN(bool supported,
                        SupportsNvidiaGreedySampler(backend, sampler_params,
                                                    buffer_type, logits_type));
  if (!supported) {
    return std::unique_ptr<Sampler>();
  }
  ABSL_ASSIGN_OR_RETURN(auto library, LoadedDispatchLibrary(env));
  if (!library.has_value()) {
    ABSL_LOG(INFO) << "NVIDIA dispatch library not loaded by the runtime; "
                      "the MTP drafter samples on the host";
    return std::unique_ptr<Sampler>();
  }
  auto create = library->LookupSymbol<NvidiaGreedySampler::CreateFn>(
      "LiteRtDispatchNvidiaGreedySamplerCreate");
  auto destroy = library->LookupSymbol<NvidiaGreedySampler::DestroyFn>(
      "LiteRtDispatchNvidiaGreedySamplerDestroy");
  auto sample = library->LookupSymbol<NvidiaGreedySampler::SampleFn>(
      "LiteRtDispatchNvidiaGreedySamplerSampleBatched");
  if (!create.HasValue() || !destroy.HasValue() || !sample.HasValue()) {
    ABSL_LOG(INFO) << "NVIDIA dispatch library exports no greedy sampler; "
                      "the MTP drafter samples on the host";
    return std::unique_ptr<Sampler>();
  }
  const auto dims = logits_type.Layout().Dimensions();
  std::vector<int32_t> ids(dims[1]);
  LiteRtDispatchNvidiaGreedySampler sampler = nullptr;
  LITERT_RETURN_IF_ERROR(create.Value()(&sampler));
  if (sampler == nullptr) {
    return absl::InternalError("NVIDIA greedy sampler returned a null handle");
  }
  std::unique_ptr<void, NvidiaGreedySampler::DestroyFn> owned_sampler(
      sampler, destroy.Value());
  ABSL_LOG(INFO) << "NVIDIA MTP GPU greedy sampling enabled for " << dims[1]
                 << " rows, vocabulary " << dims[2];
  auto result = std::unique_ptr<Sampler>(new NvidiaGreedySampler(
      *std::move(library), sampler, destroy.Value(), sample.Value(), dims[1],
      dims[2], std::move(ids)));
  owned_sampler.release();
  return result;
}

namespace {

// The NPU route's sampler. The first logits it sees decide between NVIDIA's
// device paths (CUDA logits the dispatch accepts: the argmax for the greedy
// configuration, the candidate selection for any other top-k / top-p one)
// and the host sampler it keeps; one sampler sees one kind of logits buffer,
// so the choice is made once, and a configuration update makes it again.
// The host sampler is the generic factory's CPU sampler, kept and recreated
// as UpdateOrRecreateSampler would (a larger k than it was made for). Both
// paths pick the tokens with the caller's uniforms, so a session's stream is
// one stream whichever path samples a round.
//
// On the candidate path, one row at a time, the sampler also tells the
// decode step that feeds itself (LiteRT's step_sampler_c_api.h) how it
// samples: a step run ahead then samples its logits on the device as this
// sampler will, and the host's token confirms it.
class LazyNvidiaSampler : public Sampler {
 public:
  // `tells_step_feed`: the session's own sampler tells the decode step that
  // feeds itself how it samples; a drafter's or a verifier's sampler, made
  // the same way, does not (its configuration is not the session's).
  LazyNvidiaSampler(const Environment* env,
                    proto::SamplerParameters sampler_params,
                    std::unique_ptr<Sampler> host, int batch_size,
                    std::optional<int> sequence_size,
                    std::optional<int> vocab_size,
                    std::optional<ActivationDataType> activation_data_type,
                    bool tells_step_feed)
      : env_(env),
        sampler_params_(std::move(sampler_params)),
        host_(std::move(host)),
        host_max_top_k_(sampler_params_.k()),
        batch_size_(batch_size),
        sequence_size_(sequence_size),
        vocab_size_(vocab_size),
        activation_data_type_(activation_data_type),
        tells_step_feed_(tells_step_feed) {}

  absl::Status Init() {
    return host_->UpdateConfig(sampler_params_, batch_size_);
  }

  // Samples with the caller's uniforms (one per row, in [0, 1)).
  absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor, absl::Span<const float> uniforms) override {
    return Sample(logits_tensor, ids_tensor, scores_tensor, uniforms);
  }

  // What a recreated host sampler is made with, as UpdateOrRecreateSampler
  // takes them at every update.
  void SetRecreation(std::optional<int> sequence_size,
                     std::optional<int> vocab_size,
                     std::optional<ActivationDataType> activation_data_type) {
    sequence_size_ = sequence_size;
    vocab_size_ = vocab_size;
    activation_data_type_ = activation_data_type;
  }

  absl::Status UpdateConfig(const proto::SamplerParameters& sampler_params,
                            int batch_size) override {
    if (host_ == nullptr || sampler_params.k() > host_max_top_k_) {
      // As UpdateOrRecreateSampler: a new CPU sampler.
      std::optional<std::reference_wrapper<const Environment>> env;
      if (env_ != nullptr) {
        env = std::cref(*env_);
      }
      ABSL_ASSIGN_OR_RETURN(
          std::unique_ptr<Sampler> new_host,
          CreateSampler(Backend::CPU, batch_size, sampler_params, env,
                        sequence_size_, vocab_size_, activation_data_type_));
      RET_CHECK(new_host != nullptr)
          << "No sampler for sampler type " << sampler_params.type();
      host_ = std::move(new_host);
      host_max_top_k_ = sampler_params.k();
    }
    sampler_params_ = sampler_params;
    batch_size_ = batch_size;
    decided_ = false;
    device_greedy_.reset();
    device_candidates_.reset();
    return host_->UpdateConfig(sampler_params, batch_size);
  }

 private:
  absl::Status Sample(const TensorBuffer& logits_tensor,
                      TensorBuffer& ids_tensor, TensorBuffer* scores_tensor,
                      absl::Span<const float> uniforms) {
    if (!decided_) {
      decided_ = true;
      if (env_ != nullptr && !GpuSamplingOptedOut()) {
        if (IsFixedGreedy(sampler_params_)) {
          ABSL_ASSIGN_OR_RETURN(
              device_greedy_,
              TryCreateNvidiaGreedySampler(*env_, Backend::NPU, sampler_params_,
                                           logits_tensor));
          if (device_greedy_ != nullptr) {
            ABSL_RETURN_IF_ERROR(TellStepSampler());
          }
        } else {
          ABSL_ASSIGN_OR_RETURN(device_candidates_,
                                TryCreateNvidiaCandidateSampler(
                                    *env_, sampler_params_, logits_tensor));
          if (device_candidates_ != nullptr) {
            ABSL_RETURN_IF_ERROR(TellStepSampler());
          }
        }
      }
    }
    // The argmax returns ids only. The scores of the greedy configuration
    // are log(1) = 0 (TopPSampler's for k 1), which need no logits; exact
    // log probabilities are the host's.
    if (device_greedy_ != nullptr &&
        (scores_tensor == nullptr ||
         !sampler_params_.compute_exact_log_probs())) {
      ABSL_RETURN_IF_ERROR(device_greedy_->SampleToIdAndScoreBuffer(
          logits_tensor, ids_tensor, nullptr, uniforms));
      if (scores_tensor != nullptr) {
        LITERT_ASSIGN_OR_RETURN(const size_t bytes,
                                scores_tensor->PackedSize());
        const std::vector<float> zeros(bytes / sizeof(float), 0.0f);
        LITERT_RETURN_IF_ERROR(
            scores_tensor->Write(absl::MakeConstSpan(zeros)));
      }
      return absl::OkStatus();
    }
    // Exact log probabilities need the whole vocabulary: the host's work.
    if (device_candidates_ != nullptr &&
        !(scores_tensor != nullptr &&
          sampler_params_.compute_exact_log_probs())) {
      ABSL_ASSIGN_OR_RETURN(
          bool sampled,
          device_candidates_->TrySample(logits_tensor, sampler_params_,
                                        uniforms, ids_tensor, scores_tensor));
      if (sampled) return absl::OkStatus();
    }
    return host_->SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                           scores_tensor, uniforms);
  }

  using StepSetFn = decltype(&LiteRtDispatchNvidiaStepSamplerSet);

  // Tells the step feed how this sampler samples (a sampler that is not the
  // session's, a dispatch without the entry point, or more rows than one,
  // leave the steps ahead as they are). Only the steps ahead depend on it: a
  // call the dispatch refuses is logged and leaves them as they are, never
  // failing a sample (the host takes a step ahead only when its inputs match
  // the host's). The draws of the samples to come are the caller's
  // (`uniforms`), not this sampler's: it pushes none, so a step ahead with a
  // k above 1 samples the argmax (step_sampler_c_api.h).
  absl::Status TellStepSampler() {
    if (!tells_step_feed_ || batch_size_ != 1) {
      return absl::OkStatus();
    }
    if (step_set_ == nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto library, LoadedDispatchLibrary(*env_));
      if (!library.has_value()) return absl::OkStatus();
      auto set = library->LookupSymbol<StepSetFn>(
          "LiteRtDispatchNvidiaStepSamplerSet");
      if (!set.HasValue()) return absl::OkStatus();
      step_library_ = std::move(*library);
      step_set_ = set.Value();
    }
    if (const LiteRtStatus status =
            step_set_(sampler_params_.k(), sampler_params_.p(),
                      sampler_params_.temperature());
        status != kLiteRtStatusOk) {
      ABSL_LOG(WARNING) << "NVIDIA step feed: the dispatch does not take the "
                           "sampler's configuration (status "
                        << status << "); the steps ahead are left as they are";
    }
    return absl::OkStatus();
  }

  const Environment* env_;
  proto::SamplerParameters sampler_params_;
  std::unique_ptr<Sampler> host_;
  int host_max_top_k_;
  int batch_size_;
  std::optional<int> sequence_size_;
  std::optional<int> vocab_size_;
  std::optional<ActivationDataType> activation_data_type_;
  std::unique_ptr<Sampler> device_greedy_;
  std::unique_ptr<NvidiaCandidateSampler> device_candidates_;
  bool decided_ = false;
  std::optional<SharedLibrary> step_library_;
  StepSetFn step_set_ = nullptr;
  const bool tells_step_feed_;
};

// The NPU route's sampler over the CPU sampler of the same configuration.
absl::StatusOr<std::unique_ptr<Sampler>> CreateNpuRouteSampler(
    int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type,
    bool tells_step_feed) {
  ABSL_ASSIGN_OR_RETURN(
      auto host,
      CreateSampler(Backend::CPU, batch_size, sampler_params, env,
                    sequence_size, vocab_size, activation_data_type));
  RET_CHECK(host != nullptr)
      << "No sampler for sampler type " << sampler_params.type();
  auto sampler = std::make_unique<LazyNvidiaSampler>(
      env.has_value() ? &env->get() : nullptr, std::move(sampler_params),
      std::move(host), batch_size, sequence_size, vocab_size,
      activation_data_type, tells_step_feed);
  ABSL_RETURN_IF_ERROR(sampler->Init());
  return std::unique_ptr<Sampler>(std::move(sampler));
}

// Whether the runtime has loaded NVIDIA's dispatch library: only then do the
// executor's and the verifier's session samplers take the NPU route, so any
// other NPU keeps the factory's samplers.
bool NvidiaDispatchLoaded(
    std::optional<std::reference_wrapper<const Environment>> env) {
  if (!env.has_value()) return false;
  auto library = LoadedDispatchLibrary(env->get());
  return library.ok() && library->has_value();
}

}  // namespace

void TellNvidiaDecodeBudget(const Environment& env, Backend backend,
                            std::optional<int> max_num_tokens_to_emit) {
  if (backend != Backend::NPU || !max_num_tokens_to_emit.has_value()) return;
  using SetBudgetFn = decltype(&LiteRtDispatchNvidiaStepFeedSetBudget);
  // Looked up once per process, on the first decode step, after the runtime has
  // loaded the dispatch library; the handle then keeps that library loaded.
  static absl::once_flag once;
  static SetBudgetFn set_budget = nullptr;
  absl::call_once(once, [&env] {
    auto library = LoadedDispatchLibrary(env);
    if (!library.ok() || !library->has_value()) return;
    auto symbol = (*library)->LookupSymbol<SetBudgetFn>(
        "LiteRtDispatchNvidiaStepFeedSetBudget");
    if (!symbol.HasValue()) return;
    set_budget = symbol.Value();
    new SharedLibrary(std::move(**library));  // kept for the process
  });
  if (set_budget == nullptr) return;
  const int remaining = std::max(*max_num_tokens_to_emit - 1, 0);
  if (const LiteRtStatus status = set_budget(static_cast<uint32_t>(remaining));
      status != kLiteRtStatusOk) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "NVIDIA step feed: the dispatch does not take the decode budget "
           "(status "
        << status << ")";
  }
}

absl::StatusOr<std::unique_ptr<Sampler>> CreateMtpGreedySampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type,
    bool tells_step_feed) {
  if (backend != Backend::NPU) {
    return CreateSampler(backend, batch_size, std::move(sampler_params), env,
                         sequence_size, vocab_size, activation_data_type);
  }
  return CreateNpuRouteSampler(batch_size, std::move(sampler_params), env,
                               sequence_size, vocab_size, activation_data_type,
                               tells_step_feed);
}

absl::StatusOr<std::unique_ptr<Sampler>> CreateExecutorSampler(
    Backend executor_backend, Backend sampler_backend, int batch_size,
    proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  if (executor_backend == Backend::NPU && sampler_backend == Backend::CPU &&
      NvidiaDispatchLoaded(env)) {
    return CreateMtpGreedySampler(Backend::NPU, batch_size,
                                  std::move(sampler_params), env, sequence_size,
                                  vocab_size, activation_data_type,
                                  /*tells_step_feed=*/true);
  }
  return CreateSampler(sampler_backend, batch_size, std::move(sampler_params),
                       env, sequence_size, vocab_size, activation_data_type);
}

absl::Status UpdateOrRecreateMtpVerifierSampler(
    std::unique_ptr<Sampler>& sampler, int& max_top_k, Backend backend,
    int batch_size, const proto::SamplerParameters& sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  // The drafter names the backend of its session sampler, which on the NPU
  // route is the CPU (the generic factory has no NPU sampler): the NPU
  // route's sampler is known by what it is, and keeps the configuration
  // whatever backend is named for it.
  auto* npu_sampler = dynamic_cast<LazyNvidiaSampler*>(sampler.get());
  if (npu_sampler == nullptr &&
      (backend != Backend::NPU || sampler != nullptr ||
       !NvidiaDispatchLoaded(env))) {
    return UpdateOrRecreateSampler(sampler, max_top_k, backend, batch_size,
                                   sampler_params, env, sequence_size,
                                   vocab_size, activation_data_type);
  }
  if (sampler == nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        sampler,
        CreateNpuRouteSampler(batch_size, sampler_params, env, sequence_size,
                              vocab_size, activation_data_type,
                              /*tells_step_feed=*/false));
    npu_sampler = dynamic_cast<LazyNvidiaSampler*>(sampler.get());
  }
  if (npu_sampler != nullptr) {
    npu_sampler->SetRecreation(sequence_size, vocab_size, activation_data_type);
  }
  ABSL_RETURN_IF_ERROR(sampler->UpdateConfig(sampler_params, batch_size));
  // The NPU route's sampler keeps its host sampler at the capacity it needs:
  // the caller's capacity follows an update that took.
  max_top_k = std::max(max_top_k, sampler_params.k());
  return absl::OkStatus();
}

}  // namespace litert::lm
