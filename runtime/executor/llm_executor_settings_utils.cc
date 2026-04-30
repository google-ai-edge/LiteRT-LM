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

#include "runtime/executor/llm_executor_settings_utils.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/file_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::lm {

namespace {

// Default number of threads for WebGPU weight upload and kernel compilation.
constexpr int kDefaultNumThreadsToUpload = 2;
constexpr int kDefaultNumThreadsToCompile = 1;

}  // namespace

absl::StatusOr<Backend> GetSamplerBackend(
    const LlmExecutorSettings& executor_settings) {
  Backend backend = executor_settings.GetBackend();
  Backend sampler_backend = executor_settings.GetSamplerBackend();

  if (sampler_backend == Backend::UNSPECIFIED) {
    sampler_backend = backend;
  }

  if (sampler_backend != Backend::CPU && sampler_backend != Backend::GPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend,
                     " for backend: ", backend));
  }

  return sampler_backend;
}

absl::StatusOr<litert::Options> CreateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    const ActivationDataType& activation_data_type,
    std::optional<ModelSignatures*> signatures,
    std::optional<std::string> cache_suffix) {
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  std::string cache_path = executor_settings.GetCacheDir();

  switch (executor_settings.GetBackend()) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                              compilation_options.GetGpuOptions());
      gpu_compilation_options.EnableInfiniteFloatCapping(true);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
      } else {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp16);
      }
#if defined(__APPLE__)
      gpu_compilation_options.SetPreferTextureWeights(false);
      gpu_compilation_options.SetUseMetalArgumentBuffers(true);
#else   // !__APPLE__
      gpu_compilation_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__

      bool has_valid_model_fd =
          executor_settings.GetModelAssets().GetScopedFile().ok() &&
          executor_settings.GetModelAssets().GetScopedFile().value()->IsValid();

      auto program_cache_file = executor_settings.GetProgramCacheFile(
          ExecutorSettingsBase::kMlDriftCacheSuffix, /*check_and_clean=*/true);
      bool has_valid_program_cache_fd =
          program_cache_file.ok() &&
          !std::holds_alternative<std::string>(*program_cache_file);

      auto model_path_or_status = executor_settings.GetModelAssets().GetPath();
      if (model_path_or_status.ok()) {
        // If the model path is available, use the model name as the cache key.
        absl::string_view model_path = *model_path_or_status;
        absl::string_view model_name = Basename(model_path);
        LITERT_ASSIGN_OR_RETURN(std::string metadata_id,
                                GetFileCacheIdentifier(model_path));
        std::string cache_key = absl::StrCat(model_name, "_", metadata_id);
        gpu_compilation_options.SetModelCacheKey(cache_key.c_str());
      } else if (has_valid_model_fd && has_valid_program_cache_fd) {
        // If the model is loaded from an fd, there is no way to automatically
        // generate a cache key. But if we are loading a model from an fd, it is
        // likely that our program cache is also loaded from an fd which does
        // not require a cache key to prevent collisions. The GPU delegate will
        // still expect a cache key, so we set it to a constant value.
        gpu_compilation_options.SetModelCacheKey("fd_token");
      }

      AdvancedSettings advanced_settings;
      if (executor_settings.GetAdvancedSettings()) {
        advanced_settings = *executor_settings.GetAdvancedSettings();
      }

      bool serialization_dir_set = false;
      if (cache_path != ":nocache") {
        if (cache_path.empty()) {
          ASSIGN_OR_RETURN(auto model_path,
                           executor_settings.GetModelAssets().GetPath());
          cache_path = std::filesystem::path(std::string(model_path))
                           .parent_path()
                           .string();
          if (cache_path.empty()) {
            cache_path = std::filesystem::current_path().string();
          }
        }
        ABSL_LOG(INFO) << "Setting serialization dir: " << cache_path;
        gpu_compilation_options.SetSerializationDir(cache_path.c_str());
        serialization_dir_set = true;
        gpu_compilation_options.SetSerializeExternalTensors(true);
        gpu_compilation_options.CacheCompiledProgramsOnly(
            advanced_settings.cache_compiled_shaders_only);
      } else {
        gpu_compilation_options.SetSerializeExternalTensors(false);
      }

      if (program_cache_file.ok()) {
        if (std::holds_alternative<std::string>(*program_cache_file)) {
          if (!serialization_dir_set) {
            cache_path = std::filesystem::path(
                             std::get<std::string>(*program_cache_file))
                             .parent_path()
                             .string();
            ABSL_LOG(INFO) << "Setting program cache dir: " << cache_path;
            gpu_compilation_options.SetSerializationDir(cache_path.c_str());
          }
        } else {
          auto scoped_cache_file =
              std::get<std::shared_ptr<lm::ScopedFile>>(*program_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          gpu_compilation_options.SetProgramCacheFd(fd);
        }
        gpu_compilation_options.SetSerializeProgramCache(true);
      } else {
        gpu_compilation_options.SetSerializeProgramCache(false);
      }

      // Use NoExternalTensorsMode to get better performance.
      ASSIGN_OR_RETURN(const GpuConfig gpu_config,
                       executor_settings.GetBackendConfig<GpuConfig>());
      bool external_tensor_mode = gpu_config.external_tensor_mode;
      gpu_compilation_options.EnableExternalTensorsMode(external_tensor_mode);
      if (!external_tensor_mode) {
        // This option prevents KVCache handling from being affected by
        // BHWC conversion in NoExternalTensorsMode.
        gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
        if (signatures.has_value() &&
            signatures.value()->input_int32_param.has_value()) {
          gpu_compilation_options.AddBufferStorageTensorPattern("kv_cache_");
          gpu_compilation_options.AddExternalTensorPattern("param_tensor");
          gpu_compilation_options.AddBufferStorageTensorPattern("param_tensor");
        }
        ASSIGN_OR_RETURN(auto sampler_backend,
                         GetSamplerBackend(executor_settings));
        if (sampler_backend == Backend::GPU) {
          // GPU Sampler requires logits to be external tensors (PHWC4 format).
          gpu_compilation_options.AddExternalTensorPattern("logits");
        }
      }
      // Prefill and decode are always fully delegated to single delegate.
      gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);

      gpu_compilation_options.SetMadviseOriginalSharedTensors(
          advanced_settings.gpu_madvise_original_shared_tensors);
      gpu_compilation_options.SetConvertWeightsOnGpu(
          advanced_settings.convert_weights_on_gpu);
      gpu_compilation_options.EnableConstantTensorSharing(
          advanced_settings.share_constant_tensors);
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(
          !advanced_settings.allow_src_quantized_fc_conv_ops.has_value() ||
          advanced_settings.allow_src_quantized_fc_conv_ops.value());
      gpu_compilation_options.HintWaitingForCompletion(
          advanced_settings.hint_waiting_for_completion.has_value() &&
          advanced_settings.hint_waiting_for_completion.value());
      if (advanced_settings.hint_kernel_batch_size.has_value()) {
        gpu_compilation_options.SetKernelBatchSize(
            advanced_settings.hint_kernel_batch_size.value());
      }
      if (advanced_settings.is_benchmark) {
        gpu_compilation_options.SetSyncExecutionModeWaitType(
            GpuOptions::SyncExecutionModeWaitType::kActive);
        gpu_compilation_options.WaitForWeightsConversionComplete(
            advanced_settings
                .wait_for_weights_conversion_complete_in_benchmark);
      }
      if (advanced_settings.gpu_context_low_priority.has_value() &&
          advanced_settings.gpu_context_low_priority.value()) {
        gpu_compilation_options.SetPriority(GpuOptions::Priority::kLow);
      }
      if (!advanced_settings.preferred_device_substr.empty()) {
        gpu_compilation_options.SetPreferredDeviceSubstr(
            advanced_settings.preferred_device_substr.c_str());
      }
      gpu_compilation_options.DisableShaderOptimization(
          !advanced_settings.optimize_shader_compilation);
      // TODO b/441627719 - Select backend by runtime options.
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
      gpu_compilation_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
      // Prepare WebGPU or Vulkan command buffers ahead to reduce the overhead
      // of command buffer preparation. 2 steps ahead because KV cache is
      // swapped and the GPU resource bindings are the same as the previous
      // previous step.
      gpu_compilation_options.SetNumStepsOfCommandBufferPreparations(2);
      gpu_compilation_options.SetNumThreadsToUpload(
          advanced_settings.num_threads_to_upload >= 0
              ? advanced_settings.num_threads_to_upload
              : kDefaultNumThreadsToUpload);
      gpu_compilation_options.SetNumThreadsToCompile(
          advanced_settings.num_threads_to_compile >= 0
              ? advanced_settings.num_threads_to_compile
              : kDefaultNumThreadsToCompile);
      compilation_options.SetHardwareAccelerators(HwAccelerators::kGpu);
      break;
    }
    case Backend::CPU: {
      LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                              compilation_options.GetCpuOptions());
      ASSIGN_OR_RETURN(const CpuConfig cpu_config,
                       executor_settings.GetBackendConfig<CpuConfig>());
      const uint32_t num_threads = cpu_config.number_of_threads;
      cpu_compilation_options.SetNumThreads(num_threads);
      auto weight_cache_file = executor_settings.GetWeightCacheFile(
          cache_suffix.value_or("") +
              std::string(ExecutorSettingsBase::kXnnpackCacheSuffix),
          /*check_and_clean=*/true);
      if (weight_cache_file.ok()) {
        if (std::holds_alternative<std::string>(*weight_cache_file)) {
          cache_path = std::get<std::string>(*weight_cache_file);
          cpu_compilation_options.SetXNNPackWeightCachePath(cache_path.c_str());
        } else {
          auto scoped_cache_file =
              std::get<std::shared_ptr<ScopedFile>>(*weight_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          cpu_compilation_options.SetXNNPackWeightCacheFileDescriptor(fd);
        }
      } else {
        ABSL_LOG(WARNING) << "Can't use cache: " << weight_cache_file.status();
      }
      auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
      cpu_compilation_options.SetXNNPackFlags(
          default_xnn_options.flags |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
      LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                              compilation_options.GetRuntimeOptions());
      runtime_options.SetCompressQuantizationZeroPoints(true);
      AdvancedSettings advanced_settings;
      if (executor_settings.GetAdvancedSettings()) {
        advanced_settings = *executor_settings.GetAdvancedSettings();
      }
      runtime_options.SetDisableDelegateClustering(
          advanced_settings.disable_delegate_clustering);
      compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  return compilation_options;
}

}  // namespace litert::lm
