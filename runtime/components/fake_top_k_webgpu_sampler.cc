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

#include <cstdlib>
#include <cstring>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_c_api.h"  // from @litert
#include "litert/c/litert_common.h"  // from @litert

#if defined(_WIN32)
#define SAMPLER_EXPORT __declspec(dllexport) extern "C"
#else
#define SAMPLER_EXPORT __attribute__((visibility("default"))) extern "C"
#endif

namespace {
struct FakeWebGpuSampler {
  int token_id;
  const LiteRtRuntimeCApiStruct* runtime_c_api;
};
}  // namespace

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_Create(
    const LiteRtRuntimeCApiStruct* runtime_c_api, void* env, int batch_size,
    int sequence_size, int vocab_size, const void* activation_data_type,
    const void* sampler_params, void** sampler_out, char** error_msg) {
  if (sampler_out != nullptr) {
    *sampler_out = nullptr;
  }
  if (vocab_size == 400) {
    if (error_msg != nullptr) {
      constexpr absl::string_view kErrorMessage =
          "Invalid argument in fake WebGPU sampler (400)";
      char* buf = static_cast<char*>(std::malloc(kErrorMessage.size() + 1));
      if (buf != nullptr) {
        std::memcpy(buf, kErrorMessage.data(), kErrorMessage.size());
        buf[kErrorMessage.size()] = '\0';
      }
      *error_msg = buf;
    }
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }
  if (vocab_size == 404) {
    if (error_msg != nullptr) {
      constexpr absl::string_view kErrorMessage =
          "WebGPU device not found for fake sampler (404)";
      char* buf = static_cast<char*>(std::malloc(kErrorMessage.size() + 1));
      if (buf != nullptr) {
        std::memcpy(buf, kErrorMessage.data(), kErrorMessage.size());
        buf[kErrorMessage.size()] = '\0';
      }
      *error_msg = buf;
    }
    return static_cast<int>(absl::StatusCode::kNotFound);
  }
  if (sampler_out != nullptr) {
    *sampler_out = new FakeWebGpuSampler{42, runtime_c_api};
  }
  return 0;
}

SAMPLER_EXPORT void LiteRtTopKWebGpuSampler_Destroy(void* sampler) {
  delete static_cast<FakeWebGpuSampler*>(sampler);
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
    void* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg) {
  auto* fake_sampler = static_cast<FakeWebGpuSampler*>(sampler);
  if (fake_sampler == nullptr) {
    return 0;
  }
  void* host_mem = nullptr;
  if (fake_sampler->runtime_c_api != nullptr &&
      fake_sampler->runtime_c_api->litert_lock_tensor_buffer != nullptr &&
      fake_sampler->runtime_c_api->litert_lock_tensor_buffer(
          ids_tensor, &host_mem, kLiteRtTensorBufferLockModeWrite) ==
          kLiteRtStatusOk) {
    if (host_mem != nullptr) {
      *reinterpret_cast<int*>(host_mem) = fake_sampler->token_id;
    }
    if (fake_sampler->runtime_c_api->litert_unlock_tensor_buffer != nullptr) {
      fake_sampler->runtime_c_api->litert_unlock_tensor_buffer(ids_tensor);
    }
  }
  return 0;
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_SampleToIdAndScoreBufferWithUniforms(
    void* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    const float* uniforms, int num_uniforms, char** error_msg) {
  return LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
      sampler, logits_tensor, ids_tensor, scores_tensor, error_msg);
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_UpdateConfig(
    void* sampler, const void* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg) {
  return 0;
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_CanHandleInput(void* sampler) {
  return 1;
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_HandlesInput(void* sampler) {
  return 1;
}

SAMPLER_EXPORT int LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors(
    void* sampler, int (*run_inference_func)(void* arg), void* arg,
    LiteRtTensorBuffer ids_tensor,
    LiteRtTensorBuffer prev_input_positions_tensor,
    LiteRtTensorBuffer input_positions_tensor,
    LiteRtTensorBuffer prev_mask_tensor, LiteRtTensorBuffer mask_tensor,
    LiteRtTensorBuffer prev_param_tensor, LiteRtTensorBuffer param_tensor,
    char** error_msg) {
  return 0;
}
