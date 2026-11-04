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
#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string_view>
#include <system_error>  // NOLINT
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_any.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

// Common type definitions for sampler C APIs.
using LiteRtTopKSampler_Sampler = void;
using LiteRtTopKSampler_ActivationDataType = void;
using LiteRtTopKSampler_SamplerParameters = void;

// OpenCL Sampler C API function pointers.
extern "C" int (*LiteRtTopKOpenClSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKOpenClSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (
    *LiteRtTopKOpenClSampler_SampleToIdAndScoreBufferWithUniforms_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    const float* uniforms, int num_uniforms, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKOpenClSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (
    *LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
    void* arg, LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg) = nullptr;

// WebGPU Sampler C API function pointers.
extern "C" int (*LiteRtTopKWebGpuSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKWebGpuSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (
    *LiteRtTopKWebGpuSampler_SampleToIdAndScoreBufferWithUniforms_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    const float* uniforms, int num_uniforms, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKWebGpuSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (
    *LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
    void* arg, LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg) = nullptr;

// Metal Sampler C API function pointers.
extern "C" int (*LiteRtTopKMetalSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg) = nullptr;

extern "C" void (*LiteRtTopKMetalSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) = nullptr;

extern "C" int (
    *LiteRtTopKMetalSampler_SampleToIdAndScoreBufferWithUniforms_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    const float* uniforms, int num_uniforms, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler) = nullptr;

extern "C" int (*LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
    void* arg, LiteRtTensorBuffer absl_nullable ids_tensor,
    LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
    LiteRtTensorBuffer absl_nullable input_positions_tensor,
    LiteRtTensorBuffer absl_nullable prev_mask_tensor,
    LiteRtTensorBuffer absl_nullable mask_tensor,
    LiteRtTensorBuffer absl_nullable prev_param_tensor,
    LiteRtTensorBuffer absl_nullable param_tensor, char** error_msg) = nullptr;

namespace {

#if defined(_WIN32)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif

absl::Status CreateStatus(int error_code, const char* error_msg) {
  absl::StatusCode code = static_cast<absl::StatusCode>(error_code);
  return absl::Status(code, error_msg != nullptr ? error_msg : "");
}

absl::Status CreateStatusAndFreeErrorMsg(int error_code, char* error_msg) {
  absl::Cleanup cleanup = [error_msg] { free(error_msg); };
  return error_code == 0 ? absl::OkStatus()
                         : CreateStatus(error_code, error_msg);
}

litert::Expected<SharedLibrary> LoadSamplerSharedLibrary(
    const litert::Environment& env, const char* lib_name) {
  litert::Expected<litert::EnvironmentOptions> env_options = env.GetOptions();
  if (env_options.HasValue()) {
    litert::Expected<const litert::LiteRtVariant> runtime_lib_dir_option =
        env_options->GetOption(
            litert::EnvironmentOptions::Tag::kRuntimeLibraryDir);
    if (runtime_lib_dir_option.HasValue()) {
      std::string_view dir_view;
      std::visit(
          [&dir_view](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, const char*>) {
              if (val != nullptr) {
                dir_view = val;
              }
            } else if constexpr (std::is_convertible_v<T, std::string_view>) {
              dir_view = val;
            }
          },
          *runtime_lib_dir_option);
      if (!dir_view.empty()) {
        std::filesystem::path full_path =
            std::filesystem::path(dir_view) / lib_name;
        std::error_code ec;
        if (std::filesystem::exists(full_path, ec) && !ec) {
          litert::Expected<SharedLibrary> loaded_library = SharedLibrary::Load(
              full_path.string(), RtldFlags::Lazy().Local());
          if (loaded_library.HasValue()) {
            return loaded_library;
          }
          ABSL_LOG(WARNING)
              << "Found " << full_path.string()
              << " but failed to load: " << loaded_library.Error().Message();
        }
      }
    }
  }
  return SharedLibrary::Load(lib_name, RtldFlags::Lazy().Local());
}

// A base wrapper of TopK Sampler C API functions.
class TopKCApiSampler : public Sampler {
 public:
  using LiteRtTopKSampler_Create = int (*)(
      const struct LiteRtRuntimeCApiStruct* runtime_c_api,
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
  using LiteRtTopKSampler_SampleToIdAndScoreBufferWithUniforms =
      int (*)(LiteRtTopKSampler_Sampler* sampler,
              LiteRtTensorBuffer logits_tensor, LiteRtTensorBuffer ids_tensor,
              const LiteRtTensorBuffer* absl_nullable scores_tensor,
              const float* absl_nullable uniforms, int num_uniforms,
              char** absl_nullable error_msg);
  using LiteRtTopKSampler_UpdateConfig = int (*)(
      LiteRtTopKSampler_Sampler* sampler,
      const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
      void* absl_nullable rand_gen_shared_ptr, char** absl_nullable error_msg);
  using LiteRtTopKSampler_CanHandleInput =
      int (*)(LiteRtTopKSampler_Sampler* sampler);
  using LiteRtTopKSampler_HandlesInput =
      int (*)(LiteRtTopKSampler_Sampler* sampler);
  using LiteRtTopKSampler_SetInferenceFuncAndInputTensors = int (*)(
      LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
      void* arg, LiteRtTensorBuffer absl_nullable ids_tensor,
      LiteRtTensorBuffer absl_nullable prev_input_positions_tensor,
      LiteRtTensorBuffer absl_nullable input_positions_tensor,
      LiteRtTensorBuffer absl_nullable prev_mask_tensor,
      LiteRtTensorBuffer absl_nullable mask_tensor,
      LiteRtTensorBuffer absl_nullable prev_param_tensor,
      LiteRtTensorBuffer absl_nullable param_tensor,
      char** absl_nullable error_msg);

  struct TopKSamplerCApi {
    std::optional<SharedLibrary> lib;
    LiteRtTopKSampler_Create create_func;
    LiteRtTopKSampler_Destroy destroy_func;
    LiteRtTopKSampler_SampleToIdAndScoreBuffer sample_func;
    LiteRtTopKSampler_UpdateConfig update_config_func;
    LiteRtTopKSampler_CanHandleInput can_handle_input_func;
    LiteRtTopKSampler_HandlesInput handles_input_func;
    LiteRtTopKSampler_SetInferenceFuncAndInputTensors
        set_inference_func_and_input_tensors_func;
    // Optional; null if the library doesn't export it.
    LiteRtTopKSampler_SampleToIdAndScoreBufferWithUniforms
        sample_with_uniforms_func;

    TopKSamplerCApi(
        std::optional<SharedLibrary> lib, LiteRtTopKSampler_Create create_func,
        LiteRtTopKSampler_Destroy destroy_func,
        LiteRtTopKSampler_SampleToIdAndScoreBuffer sample_func,
        LiteRtTopKSampler_UpdateConfig update_config_func,
        LiteRtTopKSampler_CanHandleInput can_handle_input_func = nullptr,
        LiteRtTopKSampler_HandlesInput handles_input_func = nullptr,
        LiteRtTopKSampler_SetInferenceFuncAndInputTensors
            set_inference_func_and_input_tensors_func = nullptr,
        LiteRtTopKSampler_SampleToIdAndScoreBufferWithUniforms
            sample_with_uniforms_func = nullptr)
        : lib(std::move(lib)),
          create_func(create_func),
          destroy_func(destroy_func),
          sample_func(sample_func),
          update_config_func(update_config_func),
          can_handle_input_func(can_handle_input_func),
          handles_input_func(handles_input_func),
          set_inference_func_and_input_tensors_func(
              set_inference_func_and_input_tensors_func),
          sample_with_uniforms_func(sample_with_uniforms_func) {}
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

  absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor, absl::Span<const float> uniforms) override {
    if (capi_->sample_with_uniforms_func == nullptr) {
      // Libraries built before the symbol was added sample with their own
      // random number generator, from the same distribution.
      ABSL_LOG_FIRST_N(WARNING, 1)
          << "The loaded sampler library doesn't support caller-supplied "
             "uniforms; sampling with its own random number generator.";
      return SampleToIdAndScoreBuffer(logits_tensor, ids_tensor, scores_tensor);
    }
    char* error_msg = nullptr;
    LiteRtTensorBuffer scores_tensor_capi = nullptr;
    if (scores_tensor != nullptr) {
      scores_tensor_capi = scores_tensor->Get();
    }
    int error_code = capi_->sample_with_uniforms_func(
        sampler_, logits_tensor.Get(), ids_tensor.Get(),
        scores_tensor_capi ? &scores_tensor_capi : nullptr, uniforms.data(),
        static_cast<int>(uniforms.size()), &error_msg);
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

  absl::Status SetInferenceFuncAndInputTensors(
      int (*run_inference_func)(void* arg), void* arg,
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      const TensorBuffer* prev_param_tensor,
      const TensorBuffer* param_tensor) override {
    if (!capi_->set_inference_func_and_input_tensors_func) {
      return absl::UnimplementedError(
          "SetInferenceFuncAndInputTensors is not implemented.");
    }

    char* error_msg = nullptr;
    int error_code = capi_->set_inference_func_and_input_tensors_func(
        sampler_, run_inference_func, arg,
        ids_tensor ? ids_tensor->Get() : nullptr,
        prev_input_positions_tensor ? prev_input_positions_tensor->Get()
                                    : nullptr,
        input_positions_tensor ? input_positions_tensor->Get() : nullptr,
        prev_mask_tensor ? prev_mask_tensor->Get() : nullptr,
        mask_tensor ? mask_tensor->Get() : nullptr,
        prev_param_tensor ? prev_param_tensor->Get() : nullptr,
        param_tensor ? param_tensor->Get() : nullptr, &error_msg);
    return CreateStatusAndFreeErrorMsg(error_code, error_msg);
  }

 protected:
  TopKCApiSampler(std::unique_ptr<TopKSamplerCApi> capi,
                  LiteRtTopKSampler_Sampler* sampler)
      : capi_(std::move(capi)), sampler_(sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKSamplerCApi>> GetSamplerCApi(
      const litert::Environment& env, const char* lib_name,
      const char* create_func_name, const char* destroy_func_name,
      const char* sample_func_name, const char* update_config_func_name,
      const char* can_handle_input_func_name = nullptr,
      const char* handles_input_func_name = nullptr,
      const char* set_inference_func_and_input_tensors_func_name = nullptr,
      const char* sample_with_uniforms_func_name = nullptr) {
    LITERT_ASSIGN_OR_RETURN(SharedLibrary lib,
                            LoadSamplerSharedLibrary(env, lib_name));

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
    LiteRtTopKSampler_SetInferenceFuncAndInputTensors
        sampler_set_inference_func_and_input_tensors_func = nullptr;
    if (set_inference_func_and_input_tensors_func_name != nullptr) {
      LITERT_ASSIGN_OR_RETURN(
          sampler_set_inference_func_and_input_tensors_func,
          lib.LookupSymbol<LiteRtTopKSampler_SetInferenceFuncAndInputTensors>(
              set_inference_func_and_input_tensors_func_name));
      RET_CHECK_NE(sampler_set_inference_func_and_input_tensors_func, nullptr)
          << "Failed to load "
          << set_inference_func_and_input_tensors_func_name;
    }

    // Optional symbol: libraries built before it was added don't export it.
    LiteRtTopKSampler_SampleToIdAndScoreBufferWithUniforms
        sampler_sample_with_uniforms_func = nullptr;
    if (sample_with_uniforms_func_name != nullptr) {
      auto func = lib.LookupSymbol<
          LiteRtTopKSampler_SampleToIdAndScoreBufferWithUniforms>(
          sample_with_uniforms_func_name);
      if (func) {
        sampler_sample_with_uniforms_func = *func;
      } else {
        ABSL_LOG(INFO) << "Optional symbol " << sample_with_uniforms_func_name
                       << " not found; the sampler ignores caller-supplied "
                          "uniforms.";
      }
    }

    return std::make_unique<TopKSamplerCApi>(
        std::move(lib), sampler_create_func, sampler_destroy_func,
        sampler_sample_func, sampler_update_config_func,
        sampler_can_handle_input_func, sampler_handles_input_func,
        sampler_set_inference_func_and_input_tensors_func,
        sampler_sample_with_uniforms_func);
  }

  std::unique_ptr<TopKSamplerCApi> capi_;
  LiteRtTopKSampler_Sampler* const sampler_;
};

// A wrapper of TopKOpenClSampler C API functions.
class TopKOpenClCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKOpenClCApiSampler>> Create(
      const litert::Environment& env, int batch_size, int sequence_size,
      int vocab_size, std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
    if (absl::StatusOr<std::unique_ptr<TopKSamplerCApi>> static_capi =
            GetStaticTopKOpenClSamplerCApi();
        static_capi.ok()) {
      capi = *std::move(static_capi);
      ABSL_VLOG(1) << "Statically linked LiteRtTopKOpenClSampler C API.";
    } else {
      ABSL_VLOG(1) << "Static OpenCL sampler not available ("
                   << static_capi.status()
                   << "), falling back to dynamically loaded C API.";
      ABSL_ASSIGN_OR_RETURN(
          capi,
          GetSamplerCApi(
              env, "libLiteRtTopKOpenClSampler" SO_EXT,
              "LiteRtTopKOpenClSampler_Create",
              "LiteRtTopKOpenClSampler_Destroy",
              "LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer",
              "LiteRtTopKOpenClSampler_UpdateConfig",
              "LiteRtTopKOpenClSampler_CanHandleInput",
              "LiteRtTopKOpenClSampler_HandlesInput",
              "LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors",
              "LiteRtTopKOpenClSampler_SampleToIdAndScoreBufferWithUniforms"));
      ABSL_VLOG(1) << "Dynamically loaded LiteRtTopKOpenClSampler C API.";
    }

    auto [runtime_c_api, env_handle] = env.GetHolderForCApi();
    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    int error_code = capi->create_func(
        runtime_c_api, env_handle, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    ABSL_RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
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
        LiteRtTopKOpenClSampler_UpdateConfig_Static == nullptr ||
        LiteRtTopKOpenClSampler_CanHandleInput_Static == nullptr ||
        LiteRtTopKOpenClSampler_HandlesInput_Static == nullptr ||
        LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static ==
            nullptr) {
      return absl::UnavailableError(
          "Static LiteRtTopKOpenClSampler C API not available.");
    }
    return std::make_unique<TopKSamplerCApi>(
        /*lib=*/std::nullopt, LiteRtTopKOpenClSampler_Create_Static,
        LiteRtTopKOpenClSampler_Destroy_Static,
        LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static,
        LiteRtTopKOpenClSampler_UpdateConfig_Static,
        LiteRtTopKOpenClSampler_CanHandleInput_Static,
        LiteRtTopKOpenClSampler_HandlesInput_Static,
        LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static,
        LiteRtTopKOpenClSampler_SampleToIdAndScoreBufferWithUniforms_Static);
  }
};

// A wrapper of TopKWebGpuSampler C API functions.
class TopKWebGpuCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKWebGpuCApiSampler>> Create(
      const litert::Environment& env, int batch_size, int sequence_size,
      int vocab_size, std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
    if (absl::StatusOr<std::unique_ptr<TopKSamplerCApi>> static_capi =
            GetStaticTopKWebGpuSamplerCApi();
        static_capi.ok()) {
      capi = *std::move(static_capi);
      ABSL_VLOG(1) << "Statically linked LiteRtTopKWebGpuSampler C API.";
    } else {
      ABSL_VLOG(1) << "Static WebGPU sampler not available ("
                   << static_capi.status()
                   << "), falling back to dynamically loaded C API.";
      ABSL_ASSIGN_OR_RETURN(
          capi,
          GetSamplerCApi(
              env, "libLiteRtTopKWebGpuSampler" SO_EXT,
              "LiteRtTopKWebGpuSampler_Create",
              "LiteRtTopKWebGpuSampler_Destroy",
              "LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer",
              "LiteRtTopKWebGpuSampler_UpdateConfig",
              "LiteRtTopKWebGpuSampler_CanHandleInput",
              "LiteRtTopKWebGpuSampler_HandlesInput",
              "LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors",
              "LiteRtTopKWebGpuSampler_SampleToIdAndScoreBufferWithUniforms"));
      ABSL_VLOG(1) << "Dynamically loaded LiteRtTopKWebGpuSampler C API.";
    }

    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    auto [runtime_c_api, env_handle] = env.GetHolderForCApi();
    int error_code = capi->create_func(
        runtime_c_api, env_handle, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    ABSL_RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
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
        LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static ==
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
        LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static,
        LiteRtTopKWebGpuSampler_SampleToIdAndScoreBufferWithUniforms_Static);
  }
};

// A wrapper of TopKMetalSampler C API functions.
class TopKMetalCApiSampler : public TopKCApiSampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKMetalCApiSampler>> Create(
      const litert::Environment& env, int batch_size, int sequence_size,
      int vocab_size, std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    std::unique_ptr<TopKSamplerCApi> capi;
    if (absl::StatusOr<std::unique_ptr<TopKSamplerCApi>> static_capi =
            GetStaticTopKMetalSamplerCApi();
        static_capi.ok()) {
      capi = *std::move(static_capi);
      ABSL_VLOG(1) << "Statically linked LiteRtTopKMetalSampler C API.";
    } else {
      ABSL_VLOG(1) << "Static Metal sampler not available ("
                   << static_capi.status()
                   << "), falling back to dynamically loaded C API.";
      // Metal is only supported on Apple platforms (macOS/iOS/tvOS/watchOS).
      // The shared library validation will handle platform checks implicitly,
      // but typically we expect .dylib on Apple.
      ABSL_ASSIGN_OR_RETURN(
          capi,
          GetSamplerCApi(
              env, "libLiteRtTopKMetalSampler.dylib",
              "LiteRtTopKMetalSampler_Create", "LiteRtTopKMetalSampler_Destroy",
              "LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer",
              "LiteRtTopKMetalSampler_UpdateConfig",
              "LiteRtTopKMetalSampler_CanHandleInput",
              "LiteRtTopKMetalSampler_HandlesInput",
              "LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors",
              "LiteRtTopKMetalSampler_SampleToIdAndScoreBufferWithUniforms"));
      ABSL_VLOG(1) << "Dynamically loaded LiteRtTopKMetalSampler C API.";
    }

    LiteRtTopKSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    auto [runtime_c_api, env_handle] = env.GetHolderForCApi();
    int error_code = capi->create_func(
        runtime_c_api, env_handle, batch_size, sequence_size, vocab_size,
        activation_data_type.has_value() ? &activation_data_type.value()
                                         : nullptr,
        &sampler_params, &sampler, &error_msg);
    ABSL_RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
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
        LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors_Static ==
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
        LiteRtTopKMetalSampler_SetInferenceFuncAndInputTensors_Static,
        LiteRtTopKMetalSampler_SampleToIdAndScoreBufferWithUniforms_Static);
  }
};

absl::StatusOr<std::unique_ptr<Sampler>> CreateCpuSampler(
    int batch_size, int sequence_size,
    proto::SamplerParameters sampler_params) {
  switch (sampler_params.type()) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED:
      ABSL_VLOG(1) << "Sampler type is unspecified. Assume the LLM Executor "
                      "handles the sampling logic.";
      return nullptr;
    case proto::SamplerParameters::TOP_P:
      return TopPSampler::Create(sampler_params.k(), sampler_params.p(),
                                 sampler_params.temperature(), batch_size,
                                 sequence_size, sampler_params.seed(),
                                 sampler_params.compute_exact_log_probs());
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Sampler type: ", sampler_params.type(), " not implemented yet."));
  }
}

absl::StatusOr<std::unique_ptr<Sampler>> CreateGpuSampler(
    int batch_size, proto::SamplerParameters sampler_params,
    const litert::Environment& env, int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  litert::Expected<litert::EnvironmentOptions> env_options = env.GetOptions();
  bool use_metal = false;
  bool use_webgpu = false;
  bool use_opencl = false;
  if (env_options.HasValue()) {
    for (const litert::EnvironmentOptions::Option& option :
         env_options->GetOptions()) {
      if (option.tag == litert::EnvironmentOptions::Tag::kMetalDevice) {
        use_metal = true;
      } else if (option.tag == litert::EnvironmentOptions::Tag::kWebGpuDevice) {
        use_webgpu = true;
      } else if (option.tag ==
                 litert::EnvironmentOptions::Tag::kOpenClDeviceId) {
        use_opencl = true;
      }
    }
  }
  const bool has_explicit_backend = use_metal || use_webgpu || use_opencl;

#if defined(__APPLE__)
  // LiteRT's GpuEnvironment creates a default Metal device on Apple even when
  // the WebGPU delegate is active (`use_webgpu == true`), so only select the
  // Metal sampler when WebGPU is not active.
  if (use_metal && !use_webgpu) {
    absl::StatusOr<std::unique_ptr<Sampler>> metal_sampler =
        TopKMetalCApiSampler::Create(env, batch_size, sequence_size, vocab_size,
                                     activation_data_type, sampler_params);
    if (metal_sampler.ok() || !absl::IsUnavailable(metal_sampler.status())) {
      return metal_sampler;
    }
    ABSL_LOG(WARNING)
        << "Metal sampler explicitly requested but failed/unavailable.";
  }
#endif  // __APPLE__

  auto try_webgpu = [&]() -> absl::StatusOr<std::unique_ptr<Sampler>> {
#if LITERT_HAS_WEBGPU_SUPPORT  // NOLINT(misc-include-cleaner)
    if (use_webgpu || !has_explicit_backend) {
      absl::StatusOr<std::unique_ptr<Sampler>> webgpu_sampler =
          TopKWebGpuCApiSampler::Create(env, batch_size, sequence_size,
                                        vocab_size, activation_data_type,
                                        sampler_params);
      if (webgpu_sampler.ok()) {
        return webgpu_sampler;
      }
      if (absl::IsUnavailable(webgpu_sampler.status()) ||
          (!use_webgpu && absl::IsNotFound(webgpu_sampler.status()))) {
        ABSL_VLOG(1) << "WebGPU sampler not available, falling back to other "
                        "sampler options.";
      } else {
        return webgpu_sampler;
      }
    }
#endif  // LITERT_HAS_WEBGPU_SUPPORT
    return absl::UnavailableError("WebGPU sampler not available.");
  };

  auto try_opencl = [&]() -> absl::StatusOr<std::unique_ptr<Sampler>> {
#if LITERT_HAS_OPENCL_SUPPORT  // NOLINT(misc-include-cleaner)
    // LiteRT's GpuEnvironment may co-populate a default kOpenClDeviceId even
    // when the WebGPU delegate is active (`use_webgpu == true`), whereas the
    // OpenCL delegate never sets kWebGpuDevice. Skip OpenCL when `use_webgpu`
    // is set so the sampler backend matches the active WebGPU delegate.
    if ((use_opencl && !use_webgpu) || !has_explicit_backend) {
      absl::StatusOr<std::unique_ptr<Sampler>> opencl_sampler =
          TopKOpenClCApiSampler::Create(env, batch_size, sequence_size,
                                        vocab_size, activation_data_type,
                                        sampler_params);
      if (opencl_sampler.ok()) {
        return opencl_sampler;
      }
      if (absl::IsUnavailable(opencl_sampler.status()) ||
          (!use_opencl && absl::IsNotFound(opencl_sampler.status()))) {
        ABSL_VLOG(1) << "OpenCL sampler not available, falling back to other "
                        "sampler options.";
      } else {
        return opencl_sampler;
      }
    }
#endif  // LITERT_HAS_OPENCL_SUPPORT
    return absl::UnavailableError("OpenCL sampler not available.");
  };

#if defined(__ANDROID__)
  if (absl::StatusOr<std::unique_ptr<Sampler>> sampler = try_opencl();
      sampler.ok() || !absl::IsUnavailable(sampler.status())) {
    return sampler;
  }
  if (absl::StatusOr<std::unique_ptr<Sampler>> sampler = try_webgpu();
      sampler.ok() || !absl::IsUnavailable(sampler.status())) {
    return sampler;
  }
#else
  if (absl::StatusOr<std::unique_ptr<Sampler>> sampler = try_webgpu();
      sampler.ok() || !absl::IsUnavailable(sampler.status())) {
    return sampler;
  }
  if (absl::StatusOr<std::unique_ptr<Sampler>> sampler = try_opencl();
      sampler.ok() || !absl::IsUnavailable(sampler.status())) {
    return sampler;
  }
#endif

  return absl::UnavailableError("GPU sampler not available.");
}

#undef SO_EXT

}  // namespace

absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const litert::Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  int sequence_size_value = sequence_size.value_or(1);
  switch (backend) {
    case Backend::GPU: {
      RET_CHECK(env.has_value())
          << "LiteRT environment is needed for GPU sampling.";
      RET_CHECK(vocab_size.has_value())
          << "Vocabulary size is needed for GPU sampling.";
      absl::StatusOr<std::unique_ptr<Sampler>> gpu_sampler = CreateGpuSampler(
          batch_size, sampler_params, env.value().get(), sequence_size_value,
          vocab_size.value(), activation_data_type);
      if (gpu_sampler.ok() || !absl::IsUnavailable(gpu_sampler.status())) {
        // For a normal failure or success, return the result.
        return gpu_sampler;
      }
      // For a failure due to GPU sampler unavailable, fall back to CPU.
      ABSL_LOG(WARNING)
          << "GPU sampler unavailable. Falling back to CPU sampling. To use "
             "GPU sampling, please make sure the sampler library matching the "
             "inference backend (libLiteRtTopKWebGpuSampler.so, "
             "libLiteRtTopKOpenClSampler.so, or "
             "libLiteRtTopKMetalSampler.dylib on Apple platforms) is linked in "
             "or available at LD_LIBRARY_PATH on device. You can find the "
             "shared library under prebuilt/";
      ABSL_FALLTHROUGH_INTENDED;
    }
    case Backend::CPU:
      return CreateCpuSampler(batch_size, sequence_size_value, sampler_params);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  }
}

absl::Status UpdateOrRecreateSampler(
    std::unique_ptr<Sampler>& sampler, int& max_top_k, Backend backend,
    int batch_size, const proto::SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen,
    std::optional<std::reference_wrapper<const litert::Environment>> env,
    std::optional<int> sequence_size, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  if (sampler == nullptr || sampler_params.k() > max_top_k) {
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<Sampler> new_sampler,
        CreateSampler(backend, batch_size, sampler_params, env, sequence_size,
                      vocab_size, activation_data_type));
    RET_CHECK(new_sampler != nullptr)
        << "No sampler for sampler type " << sampler_params.type();
    sampler = std::move(new_sampler);
    max_top_k = sampler_params.k();
  }
  return sampler->UpdateConfig(sampler_params, batch_size, std::move(rand_gen));
}

proto::SamplerParameters GreedySamplerParams() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return sampler_params;
}

bool SamplerParamsEquivalent(const std::optional<proto::SamplerParameters>& a,
                             const std::optional<proto::SamplerParameters>& b,
                             bool ignore_seed) {
  const auto is_default = [](const std::optional<proto::SamplerParameters>& p) {
    return !p.has_value() ||
           p->type() == proto::SamplerParameters::TYPE_UNSPECIFIED;
  };
  const bool a_is_default = is_default(a);
  const bool b_is_default = is_default(b);
  if (a_is_default || b_is_default) {
    return a_is_default == b_is_default;
  }
  return a->type() == b->type() && a->k() == b->k() && a->p() == b->p() &&
         a->temperature() == b->temperature() &&
         (ignore_seed || a->seed() == b->seed()) &&
         a->backend() == b->backend() &&
         a->compute_exact_log_probs() == b->compute_exact_log_probs();
}

}  // namespace litert::lm
