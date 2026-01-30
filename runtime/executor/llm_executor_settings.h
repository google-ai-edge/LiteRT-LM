// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

struct GpuArtisanConfig {
  // Number of output candidates.
  uint32_t num_output_candidates = 1;

  // Whether to wait for weight uploads before prefilling.
  bool wait_for_weight_uploads = false;

  // Number of decode steps per sync. Used by GPU only.
  uint32_t num_decode_steps_per_sync = 1;

  // Sequence batch size for encoding. Used by GPU only. Number of input
  // tokens to process at a time for batch processing. Setting this value to 1
  // means both the encoding and decoding share the same graph of sequence
  // length of 1. Setting this value to 0 means the batch size will be
  // optimized programmatically.
  uint32_t sequence_batch_size = 0;

  // The supported lora ranks for the base model. Used by GPU only. By default
  // it will be empty, meaning not supporting any lora ranks.
  std::vector<uint32_t> supported_lora_ranks = {};

  // Maximum top k, which is the max Top-K value supported for all
  // sessions created with the engine, used by GPU only. If a session with
  // Top-K value larger than this is being asked to be created, it will be
  // rejected(throw error). The max top k will be 1, which means only greedy
  // decoding is supported for any sessions created with this engine.
  uint32_t max_top_k = 1;

  // Enables decode logits.
  // AiCore uses decode logits, so this is enabled for AiCore.
  // LLM Engine defaults to disabling decode logits.
  bool enable_decode_logits = false;

  // Enables external embeddings.
  // AiCore uses external embeddings, so this is enabled for AiCore.
  // LLM Engine defaults to disabling external embeddings.
  bool enable_external_embeddings = false;
};

std::ostream& operator<<(std::ostream& os, const GpuArtisanConfig& config);

struct GpuConfig {
  // Maximum top k, which is the max Top-K value supported for all
  // sessions created with the engine, used by GPU only. If a session with
  // Top-K value larger than this is being asked to be created, it will be
  // rejected(throw error). The default max top k will be 1, which
  // means only greedy decoding is supported for any sessions created with
  // this engine.
  uint32_t max_top_k = 1;

  // Whether to use external tensor mode.
  bool external_tensor_mode = false;
};
std::ostream& operator<<(std::ostream& os, const GpuConfig& config);

struct CpuConfig {
  // The increment size of the kv-cache. This is used by dynamically exported
  // models. Each time during decode, the kv-cache size is increased by this
  // size.
  uint32_t kv_increment_size = 16;

  // The maximum number of tokens to process in a single prefill chunk. This
  // setting is only applicable to dynamically exported models. Using smaller
  // chunk sizes can reduce peak memory usage and allow for more timely
  // cancellation of long input sequences. A value of -1 indicates that no
  // chunking is applied, and the entire prefill is processed at once.
  int prefill_chunk_size = -1;

  // Number of threads. The default value is 4.
  uint32_t number_of_threads = 4;
};
std::ostream& operator<<(std::ostream& os, const CpuConfig& config);

// Optional advanced settings for the LLM executor.
struct AdvancedSettings {
  // Ordered set of the maximum number of prefill tokens processed at once when
  // the graph has dynamic prefill lengths.
  std::set<int> prefill_batch_sizes;

  // The number of output candidates, or the decode batch size.
  int num_output_candidates = 1;

  // Whether to configure magic numbers when the model contains magic numbers.
  // Magic number for the context length will be replaced with max_num_tokens_
  // in LlmExecutorSettings.
  // Magic numbers of the prefill lengths will be replaced with the
  // prefill_batch_sizes above with best match which means, a subgraph of
  // prefill will be chosen to have the smallest magic number greater than or
  // equal to the given prefill batch size.
  // The numbers that replaced magic numbers must be less than magic numbers.
  // Otherwise, default values less than magic numbers will be used that are
  // chosen by some heuristics.
  bool configure_magic_numbers = true;

  // Whether to verify magic numbers when the model contains magic numbers and
  // test signatures.
  // If true, the subgraphs replacing magic numbers with real dimensions must be
  // the same as or supersets of the subgraphs in test signatures of the same
  // dimensions.
  bool verify_magic_numbers = false;

  // Whether to clear kv cache before the first prefill step which may help to
  // disclose any issues related to kv cache.
  // When mask is in floating point and KV cache is not cleared, some
  // uninitialized values in KV cache, .e.g. NaN, may disrupt calculations
  // improperly.
  // Disable it if it's safe to keep the KV cache uninitialized, e.g. quantized,
  // so, they can't be NaN.
  bool clear_kv_cache_before_prefill = true;

  // For debugging purpose, the number of values at the beginning of logits, in
  // the middle of logits, and at the end of logits to print after each decode
  // step. If 0, disables printing logits.
  uint32_t num_logits_to_print_after_decode = 0;

  // If true, the GPU backend will madvise the original shared tensors after
  // use.
  bool gpu_madvise_original_shared_tensors = true;

  // If true, the executor is running a benchmark.
  bool is_benchmark = false;

  // Preferred WebGPU device name substring, case-insensitive.
  // If not empty, the adapter which the device name contains the substring will
  // be chosen.
  // If empty, the device will be determined by other factors.
  std::string preferred_device_substr;

  // Number of threads for WebGPU weight upload. -1 means it's determined by
  // the runtime.
  int num_threads_to_upload = -1;
  // Number of threads for WebGPU kernel shader compilation. -1 means it's
  // determined by the runtime.
  int num_threads_to_compile = -1;

  // If true, the executor will convert weights on GPU.
  // It is not supported by the all backends so this flag is ignored when using
  // non-OpenCL and non-WebGPU backends.
  bool convert_weights_on_gpu = true;

  // If true (by default), the executor enables Vulkan kernel shader
  // optimization.
  // Some GPU backends like Vulkan don't get much performance benefit from the
  // shader optimization but just increase initialization time with longer
  // shader compilation time.
  bool optimize_shader_compilation = true;

  // If true (by default), the executor enables constant tensor sharing.
  // Some GPU backends like Vulkan may degrade the performance when constant
  // tensor sharing is enabled.
  bool share_constant_tensors = true;

  // If true and the sampler supports, the sampler manipulates decode input
  // tensors including tokens, positions, and mask.
  bool sampler_handles_input = true;

  bool operator==(const AdvancedSettings& other) const {
    return prefill_batch_sizes == other.prefill_batch_sizes &&
           num_output_candidates == other.num_output_candidates &&
           configure_magic_numbers == other.configure_magic_numbers &&
           verify_magic_numbers == other.verify_magic_numbers &&
           clear_kv_cache_before_prefill ==
               other.clear_kv_cache_before_prefill &&
           num_logits_to_print_after_decode ==
               other.num_logits_to_print_after_decode &&
           gpu_madvise_original_shared_tensors ==
               other.gpu_madvise_original_shared_tensors &&
           is_benchmark == other.is_benchmark &&
           preferred_device_substr == other.preferred_device_substr &&
           num_threads_to_upload == other.num_threads_to_upload &&
           num_threads_to_compile == other.num_threads_to_compile &&
           convert_weights_on_gpu == other.convert_weights_on_gpu &&
           optimize_shader_compilation == other.optimize_shader_compilation &&
           share_constant_tensors == other.share_constant_tensors &&
           sampler_handles_input == other.sampler_handles_input;
  }
};
std::ostream& operator<<(std::ostream& os, const AdvancedSettings& settings);

// Settings for the LLM executor.
//
// This class holds the settings for the LLM executor, including the
// model assets, cache directory, maximum number of tokens, backend,
// activation data type, and backend-specific settings.
//
// The user should construct the class using ModelAssets and then set the
// remaining settings using the setter APIs.
class LlmExecutorSettings : public ExecutorSettingsBase {
 public:
  // Creates a LlmExecutorSettings with default values using the provided
  // ModelAssets.
  static absl::StatusOr<LlmExecutorSettings> CreateDefault(
      ModelAssets model_assets, Backend backend = Backend::CPU);

  uint32_t GetMaxNumTokens() const { return max_num_tokens_; }
  void SetMaxNumTokens(uint32_t max_num_tokens) {
    max_num_tokens_ = max_num_tokens;
  }

  uint32_t GetMaxNumImages() const { return max_num_images_; }
  void SetMaxNumImages(uint32_t max_num_images) {
    max_num_images_ = max_num_images;
  }

  uint32_t GetLoraRank() const { return lora_rank_; }
  void SetLoraRank(uint32_t lora_rank) { lora_rank_ = lora_rank; }

  template <typename T>
  absl::StatusOr<const T> GetBackendConfig() const {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    }
    return absl::InvalidArgumentError("Backend config is not valid.");
  }

  template <typename T>
  absl::StatusOr<T> MutableBackendConfig() {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    }
    return absl::InvalidArgumentError("Backend config is not valid.");
  }

  void SetBackendConfig(
      const std::variant<GpuArtisanConfig, GpuConfig, CpuConfig>& config) {
    backend_config_ = config;
  }

  Backend GetSamplerBackend() const { return sampler_backend_; }
  void SetSamplerBackend(Backend sampler_backend) {
    sampler_backend_ = sampler_backend;
  }

  const std::optional<AdvancedSettings>& GetAdvancedSettings() const {
    return advanced_settings_;
  }
  void SetAdvancedSettings(const AdvancedSettings& advanced_settings) {
    advanced_settings_ = advanced_settings;
  }

  absl::Status SetSupportedLoraRanks(const std::vector<uint32_t>& lora_ranks) {
    if (std::holds_alternative<GpuArtisanConfig>(backend_config_)) {
      std::get<GpuArtisanConfig>(backend_config_).supported_lora_ranks =
          lora_ranks;
      return absl::OkStatus();
    } else if (!lora_ranks.empty()) {
      // If lora_ranks is not empty, but the backend is not GpuArtisanConfig,
      // we log a warning and ignore the lora ranks.
      LOG(ERROR) << "supported_lora_ranks is only supported for "
                    "GpuArtisanConfig. The provided lora ranks will be "
                    "ignored.";
    }
    return absl::OkStatus();
  }

 private:
  explicit LlmExecutorSettings(ModelAssets model_assets)
      : ExecutorSettingsBase(std::move(model_assets)) {}

  // Maximum number of the sum of input and output tokens. It is equivalent to
  // the size of the kv-cache.
  uint32_t max_num_tokens_;

  // Maximum number of images the model can handle.
  uint32_t max_num_images_;

  // LoRA rank. 0 means LoRA is disabled.
  uint32_t lora_rank_ = 0;

  // Backend specific config.
  std::variant<GpuArtisanConfig, GpuConfig, CpuConfig> backend_config_;

  // Backend to use for sampling.
  Backend sampler_backend_ = Backend::UNSPECIFIED;

  // Optional advanced settings.
  std::optional<AdvancedSettings> advanced_settings_;

  // Declare the output stream operator as a friend such that it can be used
  // to print the LlmExecutorSettings private member.
  friend std::ostream& operator<<(std::ostream& os,
                                  const LlmExecutorSettings& config);
};
std::ostream& operator<<(std::ostream& os, const LlmExecutorSettings& config);

// Struct to host the runtime settings for the executor.
// Settings will not be changed by the executor while executing task.
// TODO: b/404279705 - Set default values in LLM Executor RuntimeConfig
struct RuntimeConfig {

  // The number of output heads.
  // Multiple output heads might be supported in the future. For now, it is
  // always 1.
  std::optional<int> output_heads;

  // The number of tokens per decode function call.
  std::optional<int> tokens_per_decode;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
