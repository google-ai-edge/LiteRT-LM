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
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
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

// The CUDA argmax sampler of NVIDIA's dispatch library, for the drafter's
// and the verifier's logits.
namespace {

bool IsFixedGreedy(const proto::SamplerParameters& params) {
  return params.type() == proto::SamplerParameters::TOP_P && params.k() == 1 &&
         params.p() == 0.0f && std::isfinite(params.temperature()) &&
         params.temperature() >= 0.0f;
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
  if (dims.size() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 ||
      !IsDense(type.Layout())) {
    return absl::InvalidArgumentError(
        "NVIDIA greedy sampler requires dense [1, rows, vocab] logits");
  }
  return true;
}

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

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
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
    LITERT_ASSIGN_OR_RETURN(auto ids_type, ids_tensor.TensorType());
    const auto ids_dims = ids_type.Layout().Dimensions();
    if (ids_type.ElementType() != ElementType::Int32 || ids_dims.size() != 2 ||
        ids_dims[0] != 1 || ids_dims[1] != rows_ ||
        !IsDense(ids_type.Layout())) {
      return absl::InvalidArgumentError(
          "NVIDIA greedy sampler requires dense int32 [1, rows] output IDs");
    }
    // Dispatch consumes the CUDA allocation directly and synchronizes the small
    // ID download. Do not lock/read the logits tensor on the host here.
    LITERT_RETURN_IF_ERROR(
        sample_(sampler_, logits_tensor.Get(), rows_, vocab_, ids_.data()));
    LITERT_RETURN_IF_ERROR(ids_tensor.Write<int32_t>(ids_));
    return absl::OkStatus();
  }

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> /*rand_gen*/) override {
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

}  // namespace

absl::StatusOr<bool> SupportsNvidiaGreedySampler(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type) {
  if (backend != Backend::NPU || !IsFixedGreedy(sampler_params)) {
    return false;
  }
  return SupportsLogits(buffer_type, logits_type);
}

absl::StatusOr<std::unique_ptr<Sampler>> TryCreateNvidiaGreedySampler(
    const Environment& env, Backend backend,
    const proto::SamplerParameters& sampler_params,
    const TensorBuffer& logits_tensor) {
  // On by default for NVIDIA logits; LITERT_NVIDIA_MTP_GPU_SAMPLING=0 keeps
  // the CPU sampler (the logits then travel to the host every step).
  const char* enabled = std::getenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
  if ((enabled != nullptr && std::strcmp(enabled, "0") == 0) ||
      backend != Backend::NPU || !IsFixedGreedy(sampler_params)) {
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

#if defined(__linux__) && defined(RTLD_NOLOAD)
  LITERT_ASSIGN_OR_RETURN(auto options, env.GetOptions());
  auto directory =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  if (!directory.HasValue()) {
    return std::unique_ptr<Sampler>();
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
    return std::unique_ptr<Sampler>();
  }
  auto create = library->LookupSymbol<NvidiaGreedySampler::CreateFn>(
      "LiteRtDispatchNvidiaGreedySamplerCreate");
  auto destroy = library->LookupSymbol<NvidiaGreedySampler::DestroyFn>(
      "LiteRtDispatchNvidiaGreedySamplerDestroy");
  auto sample = library->LookupSymbol<NvidiaGreedySampler::SampleFn>(
      "LiteRtDispatchNvidiaGreedySamplerSampleBatched");
  if (!create.HasValue() || !destroy.HasValue() || !sample.HasValue()) {
    ABSL_VLOG(1)
        << "NVIDIA batched greedy sampler extension unavailable; using "
           "the existing MTP sampler";
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
      std::move(library.Value()), sampler, destroy.Value(), sample.Value(),
      dims[1], dims[2], std::move(ids)));
  owned_sampler.release();
  return result;
#else
  // NoLoad is essential: loading another dispatch instance would not share the
  // tensor buffer's CUDA allocator and runtime context.
  return std::unique_ptr<Sampler>();
#endif
}

namespace {

// The NPU route's sampler: the first logits it sees decide between NVIDIA's
// argmax (CUDA logits the dispatch accepts) and the host sampler it was
// created with. One sampler sees one kind of logits buffer, so the choice is
// made once; a configuration update makes it again.
class LazyNvidiaGreedySampler : public Sampler {
 public:
  LazyNvidiaGreedySampler(const Environment* env,
                          proto::SamplerParameters sampler_params,
                          std::unique_ptr<Sampler> host)
      : env_(env),
        sampler_params_(std::move(sampler_params)),
        host_(std::move(host)) {}

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    if (!decided_) {
      decided_ = true;
      if (env_ != nullptr) {
        ABSL_ASSIGN_OR_RETURN(
            device_, TryCreateNvidiaGreedySampler(
                         *env_, Backend::NPU, sampler_params_, logits_tensor));
      }
    }
    Sampler& sampler = device_ != nullptr ? *device_ : *host_;
    return sampler.SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                            scores_tensor);
  }

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    sampler_params_ = sampler_params;
    decided_ = false;
    device_.reset();
    return host_->UpdateConfig(sampler_params, batch_size, std::move(rand_gen));
  }

 private:
  const Environment* env_;
  proto::SamplerParameters sampler_params_;
  std::unique_ptr<Sampler> host_;
  std::unique_ptr<Sampler> device_;
  bool decided_ = false;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Sampler>> CreateMtpGreedySampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  if (backend != Backend::NPU) {
    return CreateSampler(backend, batch_size, std::move(sampler_params), env,
                         sequence_size, vocab_size, activation_data_type);
  }
  ABSL_ASSIGN_OR_RETURN(
      auto host,
      CreateSampler(Backend::CPU, batch_size, sampler_params, env,
                    sequence_size, vocab_size, activation_data_type));
  return std::unique_ptr<Sampler>(
      new LazyNvidiaGreedySampler(env.has_value() ? &env->get() : nullptr,
                                  std::move(sampler_params), std::move(host)));
}

}  // namespace litert::lm
