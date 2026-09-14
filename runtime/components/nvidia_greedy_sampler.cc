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

#include "runtime/components/nvidia_greedy_sampler.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "litert/vendors/nvidia/dispatch/greedy_sampler_c_api.h"  // from @litert
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
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

  absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
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
        ids_dims[0] != 1 || ids_dims[1] != rows_ || !IsDense(ids_type.Layout())) {
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
  // Keep the initialized dispatch DSO alive until sampler destruction completes.
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
  const char* enabled = std::getenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0 ||
      backend != Backend::NPU || !IsFixedGreedy(sampler_params)) {
    return std::unique_ptr<Sampler>();
  }
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_tensor.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
  ABSL_ASSIGN_OR_RETURN(
      bool supported, SupportsNvidiaGreedySampler(
                          backend, sampler_params, buffer_type, logits_type));
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
    ABSL_VLOG(1) << "NVIDIA batched greedy sampler extension unavailable; using "
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

}  // namespace litert::lm
