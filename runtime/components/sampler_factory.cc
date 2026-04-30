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

#include "runtime/components/sampler_factory.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <utility>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Common type definitions for sampler C APIs.
using LiteRtTopKSampler_Sampler = void;
using LiteRtTopKSampler_ActivationDataType = void;
using LiteRtTopKSampler_SamplerParameters = void;

// OpenCL Sampler C API function pointers.
extern "C" int (*LiteRtTopKOpenClSampler_Create_Static)(
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKOpenClSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

// WebGPU Sampler C API function pointers.
extern "C" int (*LiteRtTopKWebGpuSampler_Create_Static)(
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKWebGpuSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (
    *LiteRtTopKWebGpuSampler_SetInputTensorsAndInferenceFunc_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    int (*run_inference_func)(void* arg), void* arg,
    char** error_msg) = nullptr;

// Metal Sampler C API function pointers.
extern "C" int (*LiteRtTopKMetalSampler_Create_Static)(
    LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKMetalSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_SetInputTensorsAndInferenceFunc_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    int (*run_inference_func)(void* arg), void* arg,
    char** error_msg) = nullptr;

absl::Status CreateStatus(int error_code, const char* error_msg) {
  absl::StatusCode code = static_cast<absl::StatusCode>(error_code);
  return absl::Status(code, error_msg);
}

absl::Status CreateStatusAndFreeErrorMsg(int error_code, char* error_msg) {
  absl::Cleanup cleanup = [error_msg] { free(error_msg); };
  return error_code == 0 ? absl::OkStatus()
                         : CreateStatus(error_code, error_msg);
}

// A base wrapper of TopK Sampler C API functions.
class TopKCApiSampler : public Sampler {
 public:
  using LiteRtTopKSampler_Create = int (*)(
      LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
      const LiteRtTopKSampler_ActivationDataType* absl_nullable
          activation_data_type,
      const LiteRtTopKSampler_SamplerParameters* absl_nullable sampler_params,
      LiteRtTopKSampler_Sampler** sampler_out, char** absl_nullable error_msg);
  using LiteRtTopKSampler_Destroy =
      void (*)(LiteRtTopKSampler_Sampler* sampler);
  using LiteRtTopKSampler_SampleToIdAndScoreBuffer =
      int (*)(LiteRtTopKSampler_Sampler* sampler,
              LiteRtTensorBuffer logits_tensor, LiteRtTensorBuffer ids_tensor,
              const LiteRtTensorBuffer* absl_nullable scores_tensor,
              char** absl_nullable error_msg);
  using LiteRtTopKSampler_UpdateConfig = int (*)(
      LiteRtTopKSampler_Sampler* sampler,
      const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
      void* absl_nullable rand_gen_shared_ptr, char** absl_nullable error_msg);
  using LiteRtTopKSampler_CanHandleInput =
      int (*)(LiteRtTopKSampler_Sampler* sampler);
  using LiteRtTopKSampler_HandlesInput =
      int (*)(LiteRtTopKSampler_Sampler* sampler);
  using LiteRtTopKSampler_SetInputTensorsAndInferenceFunc =
      int (*)(LiteRtTopKSampler_Sampler* sampler,
              LiteRtTensorBuffer absl_nullable ids_tensor,
              LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
              LiteRtTensorBuffer absl_nullable input_positions_tensor,
              LiteRtTensorBuffer absl_nullable prev_mask_tensor,
              LiteRtTensorBuffer absl_nullable mask_tensor,
              int (*run_inference_func)(void* arg), void* arg,
              char** absl_nullable error_msg);

  struct TopKSamplerCApi {
    std::optional<SharedLibrary> lib;
    LiteRtTopKSampler_Create create_func;
    LiteRtTopKSampler_Destroy destroy_func;
    LiteRtTopKSampler_SampleToIdAndScoreBuffer sample_func;
    LiteRtTopKSampler_UpdateConfig update_config_func;
    LiteRtTopKSampler_CanHandleInput can_handle_input_func;
    LiteRtTopKSampler_HandlesInput handles_input_func;
    LiteRtTopKSampler_SetInputTensorsAndInferenceFunc set_input_tensors_func;

    TopKSamplerCApi(
        std::optional<SharedLibrary> lib, LiteRtTopKSampler_Create create_func,
        LiteRtTopKSampler_Destroy destroy_func,
        LiteRtTopKSampler_SampleToIdAndScoreBuffer sample_func,
        LiteRtTopKSampler_UpdateConfig update_config_func,
        LiteRtTopKSampler_CanHandleInput can_handle_input_func = nullptr,
        LiteRtTopKSampler_HandlesInput handles_input_func = nullptr,
        LiteRtTopKSampler_SetInputTensorsAndInferenceFunc
            set_input_tensors_func = nullptr)
        : lib(std::move(lib)),
          create_func(create_func),
          destroy_func(destroy_func),
          sample_func(sample_func),
          update_config_func(update_config_func),
          can_handle_input_func(can_handle_input_func),
          handles_input_func(handles_input_func),
          set_input_tensors_func(set_input_tensors_func) {}
  };

  ~TopKCApiSampler() override { capi_->destroy_func(sampler_); }

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    char* error_msg = nullptr;
    LiteRtTensorBuffer scores_tensor_capi = nullptr;
    if (scores_tensor != nullptr) {
      scores_tensor_capi = scores_tensor->Get();
    }
    int error_code = capi_->sample_func(
        sampler_, logits_tensor.Get(), ids_tensor.Get(),
        scores_tensor_capi ? &scores_tensor_capi : nullptr, &error_msg);
    return CreateStatusAndFreeErrorMsg(error_code, error_msg);
  }

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    char* error_msg = nullptr;
    int error_code = capi_->update_config_func(
        sampler_, &sampler_params, batch_size, &rand_gen, &error_msg);
    return CreateStatusAndFreeErrorMsg(error_code, error_msg);
  }

  bool CanHandleInput() const override {
    return capi_->can_handle_input_func
               ? static_cast<bool>(capi_->can_handle_input_func(sampler_))
               : false;
  }

  bool HandlesInput() const override {
    return capi_->handles_input_func
               ? static_cast<bool>(capi_->handles_input_func(sampler_))
               : false;
  }

  absl::Status SetInputTensorsAndInferenceFunc(
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      int (*run_inference_func)(void* arg), void* arg) override {
    if (!capi_->set_input_tensors_func) {
      return absl::UnimplementedError("SetInputTensors is not implemented.");
    }

    char* error_msg = nullptr;
    int error_code = capi_->set_input_tensors_func(
        sampler_, ids_tensor ? ids_tensor->Get() : nullptr,
        prev_input_positions_tensor ? prev_input_positions_tensor->Get()
                                    : nullptr,
        input_positions_tensor ? input_positions_tensor->Get() : nullptr,
        prev_mask_tensor ? prev_mask_tensor->Get() : nullptr,
        mask_tensor ? mask_tensor->Get() : nullptr, run_inference_func, arg,
        &error_msg);
    return CreateStatusAndFreeErrorMsg(error_code, error_msg);
  }

 protected:
  TopKCApiSampler(std::unique_ptr<TopKSamplerCApi> capi,
                  LiteRtTopKSampler_Sampler* sampler)
      : capi_(std::move(capi)), sampler_(sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKSamplerCApi>> GetSamplerCApi(
      const char* lib_name, const char* create_func_name,
      const char* destroy_func_name, const char* sample_func_name,
      const char* update_config_func_name,
      const char* can_handle_input_func_name = nullptr,
      const char* handles_input_func_name = nullptr,
      const char* set_input_tensors_func_name = nullptr) {
    // Load Sampler C API library and get the symbols.
    LITERT_ASSIGN_OR_RETURN(
        auto lib, SharedLibrary::Load(lib_name, RtldFlags::Lazy().Local()));
    LITERT_ASSIGN_OR_RETURN(
        auto sampler_create_func,
        lib.LookupSymbol<LiteRtTopKSampler_Create>(create_func_name));
    RET_CHECK_NE(sampler_create_func, nullptr)
        << "Failed to load " << create_func_name;
    LITERT_ASSIGN_OR_RETURN(
        auto sampler_destroy_func,
        lib.LookupSymbol<LiteRtTopKSampler_Destroy>(destroy_func_name));
    RET_CHECK_NE(sampler_destroy_func, nullptr)
        << "Failed to load " << destroy_func_name;
    LITERT_ASSIGN_OR_RETURN(
        auto sampler_sample_func,
        lib.LookupSymbol<LiteRtTopKSampler_SampleToIdAndScoreBuffer>(
            sample_func_name));
    LITERT_ASSIGN_OR_RETURN(auto sampler_update_config_func,
                            lib.LookupSymbol<LiteRtTopKSampler_UpdateConfig>(
                                update_config_func_name));
    RET_CHECK_NE(sampler_sample_func, nullptr)
        << "Failed to load " << sample_func_name;

    LiteRtTopKSampler_CanHandleInput sampler_can_handle_input_func = nullptr;
    if (can_handle_input_func_name != nullptr) {
      LITERT_ASSIGN_OR_RETURN(
          sampler_can_handle_input_func,
          lib.LookupSymbol<LiteRtTopKSampler_CanHandleInput>(
              can_handle_input_func_name));
      RET_CHECK_NE(sampler_can_handle_input_func, nullptr)
          << "Failed to load " << can_handle_input_func_name;
    }
    LiteRtTopKSampler_HandlesInput sampler_handles_input_func = nullptr;
    if (handles_input_func_name != nullptr) {
      LITERT_ASSIGN_OR_RETURN(sampler_handles_input_func,
                              lib.LookupSymbol<LiteRtTopKSampler_HandlesInput>(
                                  handles_input_func_name));
      RET_CHECK_NE(sampler_handles_input_func, nullptr)
          << "Failed to load " << handles_input_func_name;
    }
    LiteRtTopKSampler_SetInputTensorsAndInferenceFunc
        sampler_set_input_tensors_func = nullptr;
    if (set_input_tensors_func_name != nullptr) {
      LITERT_ASSIGN_OR_RETURN(
          sampler_set_input_tensors_func,
          lib.LookupSymbol<LiteRtTopKSampler_SetInputTensorsAndInferenceFunc>(
              set_input_tensors_func_name));
      RET_CHECK_NE(sampler_set_input_tensors_func, nullptr)
          << "Failed to load " << set_input_tensors_func_name;
    }

    return std::make_unique<TopKSamplerCApi>(
        std::move(lib), sampler_create_func, sampler_destroy_func,
        sampler_sample_func, sampler_update_config_func,
        sampler_can_handle_input_func, sampler_handles_input_func,
        sampler_set_input_tensors_func);
  }

  std::unique_ptr<TopKSamplerCApi> capi_;
  LiteRtTopKSampler_Sampler* const sampler_;
};

// A wrapper of TopKOpenClSampler C API functions.
class TopKOpenClCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKOpenClCApiSampler>> Create(
      LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
    auto capi_or = GetSamplerCApi(
        "libLiteRtTopKOpenClSampler.so", "LiteRtTopKOpenClSampler_Create",
        "LiteRtTopKOpenClSampler_Destroy",
        "LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer",
        "LiteRtTopKOpenClSampler_UpdateConfig");
    if (capi_or.ok()) {
      capi = std::move(capi_or.value());
      ABSL_LOG(INFO) << "Dynamically loaded LiteRtTopKOpenClSampler C API.";
    } else {
      if (capi_or.status().code() != absl::StatusCode::kUnavailable) {
        return capi_or.status();
      }
      ABSL_LOG(WARNING) << "OpenCL sampler not available, falling back to "
                           "statically linked C API: "
                        << capi_or.status();
      auto static_capi_or = GetStaticTopKOpenClSamplerCApi();
      if (!static_capi_or.ok()) {
        return capi_or.status();
      }
      capi = std::move(static_capi_or.value());
      ABSL_LOG(INFO) << "Statically linked LiteRtTopKOpenClSampler C API.";
    }

    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    int error_code = capi->create_func(
        env, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
    RET_CHECK(sampler) << "Failed to create sampler";
    return absl::WrapUnique(
        new TopKOpenClCApiSampler(std::move(capi), sampler));
  }

 private:
  TopKOpenClCApiSampler(std::unique_ptr<TopKSamplerCApi> capi,
                        LiteRtTopKSampler_Sampler* sampler)
      : TopKCApiSampler(std::move(capi), sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKSamplerCApi>>
  GetStaticTopKOpenClSamplerCApi() {
    if (LiteRtTopKOpenClSampler_Create_Static == nullptr ||
        LiteRtTopKOpenClSampler_Destroy_Static == nullptr ||
        LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static == nullptr ||
        LiteRtTopKOpenClSampler_UpdateConfig_Static == nullptr) {
      return absl::UnavailableError(
          "Static LiteRtTopKOpenClSampler C API not available.");
    }
    return std::make_unique<TopKSamplerCApi>(
        /*lib=*/std::nullopt, LiteRtTopKOpenClSampler_Create_Static,
        LiteRtTopKOpenClSampler_Destroy_Static,
        LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static,
        LiteRtTopKOpenClSampler_UpdateConfig_Static);
  }
};

// A wrapper of TopKWebGpuSampler C API functions.
class TopKWebGpuCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKWebGpuCApiSampler>> Create(
      LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
#if defined(_WIN32)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif
    auto capi_or = GetSamplerCApi(
        "libLiteRtTopKWebGpuSampler" SO_EXT, "LiteRtTopKWebGpuSampler_Create",
        "LiteRtTopKWebGpuSampler_Destroy",
        "LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer",
        "LiteRtTopKWebGpuSampler_UpdateConfig",
        "LiteRtTopKWebGpuSampler_CanHandleInput",
        "LiteRtTopKWebGpuSampler_HandlesInput",
        "LiteRtTopKWebGpuSampler_SetInputTensorsAndInferenceFunc");
    if (capi_or.ok()) {
      capi = std::move(capi_or.value());
      ABSL_LOG(INFO) << "Dynamically loaded LiteRtTopKWebGpuSampler C API.";
    } else {
      if (capi_or.status().code() != absl::StatusCode::kUnavailable) {
        return capi_or.status();
      }
      ABSL_LOG(WARNING) << "WebGPU sampler not available, falling back to "
                           "statically linked C API: "
                        << capi_or.status();
      auto static_capi_or = GetStaticTopKWebGpuSamplerCApi();
      if (!static_capi_or.ok()) {
        return capi_or.status();
      }
      capi = std::move(static_capi_or.value());
      ABSL_LOG(INFO) << "Statically linked LiteRtTopKWebGpuSampler C API.";
    }

    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    int error_code = capi->create_func(
        env, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
    RET_CHECK(sampler) << "Failed to create sampler";
    return absl::WrapUnique(
        new TopKWebGpuCApiSampler(std::move(capi), sampler));
  }

 private:
  TopKWebGpuCApiSampler(std::unique_ptr<TopKSamplerCApi> capi,
                        LiteRtTopKSampler_Sampler* sampler)
      : TopKCApiSampler(std::move(capi), sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKSamplerCApi>>
  GetStaticTopKWebGpuSamplerCApi() {
    if (LiteRtTopKWebGpuSampler_Create_Static == nullptr ||
        LiteRtTopKWebGpuSampler_Destroy_Static == nullptr ||
        LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static == nullptr ||
        LiteRtTopKWebGpuSampler_UpdateConfig_Static == nullptr ||
        LiteRtTopKWebGpuSampler_CanHandleInput_Static == nullptr ||
        LiteRtTopKWebGpuSampler_HandlesInput_Static == nullptr ||
        LiteRtTopKWebGpuSampler_SetInputTensorsAndInferenceFunc_Static ==
            nullptr) {
      return absl::UnavailableError(
          "Static LiteRtTopKWebGpuSampler C API not available.");
    }
    return std::make_unique<TopKSamplerCApi>(
        /*lib=*/std::nullopt, LiteRtTopKWebGpuSampler_Create_Static,
        LiteRtTopKWebGpuSampler_Destroy_Static,
        LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static,
        LiteRtTopKWebGpuSampler_UpdateConfig_Static,
        LiteRtTopKWebGpuSampler_CanHandleInput_Static,
        LiteRtTopKWebGpuSampler_HandlesInput_Static,
        LiteRtTopKWebGpuSampler_SetInputTensorsAndInferenceFunc_Static);
  }
};

// A wrapper of TopKMetalSampler C API functions.
class TopKMetalCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKMetalCApiSampler>> Create(
      LiteRtEnvironment env, int batch_size, int sequence_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
    // Metal is only supported on Apple platforms (macOS/iOS/tvOS/watchOS).
    // The shared library validation will handle platform checks implicitly,
    // but typically we expect .dylib on Apple.
    auto capi_or = GetSamplerCApi(
        "libLiteRtTopKMetalSampler.dylib", "LiteRtTopKMetalSampler_Create",
        "LiteRtTopKMetalSampler_Destroy",
        "LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer",
        "LiteRtTopKMetalSampler_UpdateConfig",
        "LiteRtTopKMetalSampler_CanHandleInput",
        "LiteRtTopKMetalSampler_HandlesInput",
        "LiteRtTopKMetalSampler_SetInputTensorsAndInferenceFunc");
    if (capi_or.ok()) {
      capi = std::move(capi_or.value());
      ABSL_LOG(INFO) << "Dynamically loaded LiteRtTopKMetalSampler C API.";
    } else {
      if (capi_or.status().code() != absl::StatusCode::kUnavailable) {
        return capi_or.status();
      }
      ABSL_LOG(WARNING) << "Metal sampler not available, falling back to "
                           "statically linked C API: "
                        << capi_or.status();
      auto static_capi_or = GetStaticTopKMetalSamplerCApi();
      if (!static_capi_or.ok()) {
        return capi_or.status();
      }
      capi = std::move(static_capi_or.value());
      ABSL_LOG(INFO) << "Statically linked LiteRtTopKMetalSampler C API.";
    }

    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    int error_code = capi->create_func(
        env, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
    RET_CHECK(sampler);
    return absl::WrapUnique(new TopKMetalCApiSampler(std::move(capi), sampler));
  }

 private:
  TopKMetalCApiSampler(std::unique_ptr<TopKSamplerCApi> capi,
                       LiteRtTopKSampler_Sampler* sampler)
      : TopKCApiSampler(std::move(capi), sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKSamplerCApi>>
  GetStaticTopKMetalSamplerCApi() {
    if (LiteRtTopKMetalSampler_Create_Static == nullptr ||
        LiteRtTopKMetalSampler_Destroy_Static == nullptr ||
        LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static == nullptr ||
        LiteRtTopKMetalSampler_UpdateConfig_Static == nullptr ||
        LiteRtTopKMetalSampler_CanHandleInput_Static == nullptr ||
        LiteRtTopKMetalSampler_HandlesInput_Static == nullptr ||
        LiteRtTopKMetalSampler_SetInputTensorsAndInferenceFunc_Static ==
            nullptr) {
      return absl::UnavailableError(
          "Static LiteRtTopKMetalSampler C API not available.");
    }
    return std::make_unique<TopKSamplerCApi>(
        /*lib=*/std::nullopt, LiteRtTopKMetalSampler_Create_Static,
        LiteRtTopKMetalSampler_Destroy_Static,
        LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static,
        LiteRtTopKMetalSampler_UpdateConfig_Static,
        LiteRtTopKMetalSampler_CanHandleInput_Static,
        LiteRtTopKMetalSampler_HandlesInput_Static,
        LiteRtTopKMetalSampler_SetInputTensorsAndInferenceFunc_Static);
  }
};

absl::StatusOr<std::unique_ptr<Sampler>> CreateCpuSampler(
    int batch_size, int sequence_size,
    proto::SamplerParameters sampler_params) {
  switch (sampler_params.type()) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED:
      ABSL_LOG(INFO) << "Sampler type is unspecified. Assume the LLM Executor "
                        "handles the sampling logic.";
      return nullptr;
    case proto::SamplerParameters::TOP_P:
      return TopPSampler::Create(sampler_params.k(), sampler_params.p(),
                                 sampler_params.temperature(), batch_size,
                                 sequence_size, sampler_params.seed());
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Sampler type: ", sampler_params.type(), " not implemented yet."));
  }
}

absl::StatusOr<std::unique_ptr<Sampler>> CreateGpuSampler(
    int batch_size, proto::SamplerParameters sampler_params,
    LiteRtEnvironment env, int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  // Check environment options to determine the preferred backend.
  auto cpp_env = litert::Environment::WrapCObject(env, litert::OwnHandle::kNo);
  auto options_or = cpp_env.GetOptions();
  bool use_metal = false;
  bool use_webgpu = false;
  if (options_or.HasValue()) {
    for (const auto& option : options_or->GetOptions()) {
      if (option.tag == litert::EnvironmentOptions::Tag::kMetalDevice) {
        use_metal = true;
      } else if (option.tag == litert::EnvironmentOptions::Tag::kWebGpuDevice) {
        use_webgpu = true;
      }
    }
  }

#ifdef __ANDROID__
  if (use_webgpu) {
#if LITERT_HAS_WEBGPU_SUPPORT  // NOLINT(misc-include-cleaner)
    auto webgpu_sampler = TopKWebGpuCApiSampler::Create(
        env, batch_size, sequence_size, vocab_size, activation_data_type,
        sampler_params);
    if (webgpu_sampler.ok() ||
        webgpu_sampler.status().code() != absl::StatusCode::kUnavailable) {
      return webgpu_sampler;
    }
    ABSL_LOG(INFO) << "WebGPU sampler explicitly requested but "
                      "failed/unavailable, falling back.";
#endif  // LITERT_HAS_WEBGPU_SUPPORT
  }

#if LITERT_HAS_OPENCL_SUPPORT  // NOLINT(misc-include-cleaner)
  auto opencl_sampler =
      TopKOpenClCApiSampler::Create(env, batch_size, sequence_size, vocab_size,
                                    activation_data_type, sampler_params);
  if (opencl_sampler.ok() ||
      opencl_sampler.status().code() != absl::StatusCode::kUnavailable) {
    return opencl_sampler;
  }
  ABSL_LOG(INFO)
      << "OpenCL sampler not available, falling back to other sampler options.";
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT  // NOLINT(misc-include-cleaner)
  if (!use_webgpu) {
    auto webgpu_sampler = TopKWebGpuCApiSampler::Create(
        env, batch_size, sequence_size, vocab_size, activation_data_type,
        sampler_params);
    if (webgpu_sampler.ok() ||
        webgpu_sampler.status().code() != absl::StatusCode::kUnavailable) {
      return webgpu_sampler;
    }
    ABSL_LOG(INFO) << "WebGPU sampler not available, falling back to other "
                      "sampler options.";
  }
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#else  // !__ANDROID__
#if defined(__APPLE__)
  if (use_metal || !use_webgpu) {
    auto metal_sampler =
        TopKMetalCApiSampler::Create(env, batch_size, sequence_size, vocab_size,
                                     activation_data_type, sampler_params);
    if (metal_sampler.ok() ||
        metal_sampler.status().code() != absl::StatusCode::kUnavailable) {
      return metal_sampler;
    }
    if (use_metal) {
      ABSL_LOG(WARNING)
          << "Metal sampler explicitly requested but failed/unavailable.";
    } else {
      ABSL_LOG(INFO) << "Metal sampler not available, falling back to other "
                        "sampler options.";
    }
  }
#endif  // __APPLE__

#if LITERT_HAS_WEBGPU_SUPPORT  // NOLINT(misc-include-cleaner)
  auto webgpu_sampler =
      TopKWebGpuCApiSampler::Create(env, batch_size, sequence_size, vocab_size,
                                    activation_data_type, sampler_params);
  if (webgpu_sampler.ok() ||
      webgpu_sampler.status().code() != absl::StatusCode::kUnavailable) {
    return webgpu_sampler;
  }
  ABSL_LOG(INFO)
      << "WebGPU sampler not available, falling back to other sampler options.";
#endif                         // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT  // NOLINT(misc-include-cleaner)
  auto opencl_sampler =
      TopKOpenClCApiSampler::Create(env, batch_size, sequence_size, vocab_size,
                                    activation_data_type, sampler_params);
  if (opencl_sampler.ok() ||
      opencl_sampler.status().code() != absl::StatusCode::kUnavailable) {
    return opencl_sampler;
  }
  ABSL_LOG(INFO)
      << "OpenCL sampler not available, falling back to other sampler options.";
#endif                         // LITERT_HAS_OPENCL_SUPPORT
#endif                         // !__ANDROID__

  return absl::UnavailableError("GPU sampler not available.");
}

}  // namespace

absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    LiteRtEnvironment env, std::optional<int> sequence_size,
    std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  int sequence_size_value = sequence_size.value_or(1);
  switch (backend) {
    case Backend::GPU: {
      RET_CHECK(env != nullptr)
          << "LiteRT environment is needed for GPU sampling.";
      RET_CHECK(vocab_size.has_value())
          << "Vocabulary size is needed for GPU sampling.";
      auto sampler_or =
          CreateGpuSampler(batch_size, sampler_params, env, sequence_size_value,
                           vocab_size.value(), activation_data_type);
      if (sampler_or.ok() ||
          sampler_or.status().code() != absl::StatusCode::kUnavailable) {
        // For a normal failure or success, return the result.
        return sampler_or;
      }
      // For a failure due to GPU sampler unavailable, fall back to CPU.
      ABSL_LOG(WARNING)
          << "GPU sampler unavailable. Falling back to CPU sampling. To use "
             "GPU sampling, please make sure libLiteRtTopKWebGpuSampler.so or "
             "libLiteRtTopKOpenClSampler.so is available at LD_LIBRARY_PATH "
             "on device. You can find the shared library under prebuilt/";
      ABSL_FALLTHROUGH_INTENDED;
    }
    case Backend::CPU:
      return CreateCpuSampler(batch_size, sequence_size_value, sampler_params);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  }
}

}  // namespace litert::lm
