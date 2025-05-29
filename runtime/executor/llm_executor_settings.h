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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/scoped_file.h"

namespace litert::lm {

enum class Backend {
  // CPU hand-written path backend.
  CPU_ARTISAN,

  // GPU hand-written path backend.
  GPU_ARTISAN,

  // CPU LiteRT backend.
  CPU,

  // GPU LiteRT backend.
  GPU,

  // Google Tensor Emission Graph backend.
  GOOGLE_TENSOR_ARTISAN,

  // Qualcomm QNN backend.
  QNN,

  // Google Tensor Single Graph backend.
  GOOGLE_TENSOR,
};
std::ostream& operator<<(std::ostream& os, const Backend& backend);
// Returns the backend from the string.
absl::StatusOr<Backend> GetBackendFromString(absl::string_view backend_str);

enum class ActivationDataType {
  // Use float32 as the activation data type.
  FLOAT32,

  // Use float16 as the activation data type.
  FLOAT16,

  // Use int16 as the activation data type.
  INT16,

  // Use int8 as the activation data type.
  INT8,
};
std::ostream& operator<<(std::ostream& os,
                         const ActivationDataType& activation);

// Fake weights mode.
enum class FakeWeightsMode {
  // Don't use fake weights, read real weights from disk.
  FAKE_WEIGHTS_NONE,

  // Replace all weights with INT8 fakes.
  FAKE_WEIGHTS_8BITS_ALL_LAYERS,

  // Replace feedforward and embedding weights with INT4 fakes and replace
  // attention weights with INT8 fakes.
  FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4,
};
std::ostream& operator<<(std::ostream& os,
                         const FakeWeightsMode& fake_weights_mode);

enum class FileFormat {
  // .tflite file format.
  TFLITE,

  // .task file format.
  TASK,

  // .litert_lm file format.
  LITERT_LM,
};
std::ostream& operator<<(std::ostream& os, const FileFormat& file_format);

// Class to host the model assets, including base models and lora models.
class ModelAssets {
 public:
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<ScopedFile> model_file);
  static absl::StatusOr<ModelAssets> Create(absl::string_view model_path);

  // Convenience factory function to create a ModelAssets with both a model
  // path and file. Will use the scoped file if both are provided.
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<ScopedFile> model_file, absl::string_view model_path);

  bool HasScopedFile() const {
    return std::holds_alternative<std::shared_ptr<ScopedFile>>(
        path_or_scoped_file_);
  }

  // Returns the model file if it was created with the respective variant,
  // otherwise returns an error.
  absl::StatusOr<absl::string_view> GetPath() const;
  absl::StatusOr<std::shared_ptr<ScopedFile>> GetScopedFile() const;

  // Convenience method to get a read-only scoped file to the model file
  // regardless of whether this instance was created from a path or scoped file.
  absl::StatusOr<std::shared_ptr<ScopedFile>> GetOrCreateScopedFile() const;

  FakeWeightsMode fake_weights_mode() const { return fake_weights_mode_; }

  void SetFakeWeightsMode(FakeWeightsMode fake_weights_mode) {
    fake_weights_mode_ = fake_weights_mode;
  }

 private:
  explicit ModelAssets(std::shared_ptr<ScopedFile> model_file);
  explicit ModelAssets(std::string model_path);

  // TODO: b/417814685 - Consider supporting multiple model files if the need
  // case arises.
  std::variant<std::string, std::shared_ptr<ScopedFile>>
      path_or_scoped_file_;

  FakeWeightsMode fake_weights_mode_ = FakeWeightsMode::FAKE_WEIGHTS_NONE;
};
std::ostream& operator<<(std::ostream& os, const ModelAssets& model_assets);

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
};
std::ostream& operator<<(std::ostream& os, const GpuConfig& config);

struct CpuConfig {
  // Number of threads. The default value is 4.
  uint32_t number_of_threads = 4;
};
std::ostream& operator<<(std::ostream& os, const CpuConfig& config);

// Settings for the LLM executor.
//
// This class holds the settings for the LLM executor, including the
// model assets, cache directory, maximum number of tokens, backend,
// activation data type, and backend-specific settings.
//
// The user should construct the class using ModelAssets and then set the
// remaining settings using the setter APIs.
class LlmExecutorSettings {
 public:
  // Creates a LlmExecutorSettings with default values using the provided
  // ModelAssets.
  static absl::StatusOr<LlmExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend backend = Backend::CPU);

  // Getter APIs.
  const ModelAssets& GetModelAssets() const { return model_assets_; }
  uint32_t GetMaxNumTokens() const { return max_num_tokens_; }
  uint32_t GetMaxNumImages() const { return max_num_images_; }
  const Backend& GetBackend() const { return backend_; }
  const std::optional<ActivationDataType>& GetActivationDataType() const {
    return activation_data_type_;
  }

  // Should be used by consumers who want to write to a single weight cache
  // file. Returns, in order of preference:
  //   1. an open file descriptor to the weight cache file,
  //   2. the file path of the weight cache file, based on the given cache
  //      directory and/or model path. Will append `suffix`.
  //   3. an error if a weight cache file could not be determined.
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix = ".cache") const;
  // Prefer to use `GetWeightCacheFile()` if possible.
  const std::string& GetCacheDir() const { return cache_dir_; }
  // Prefer to use `GetWeightCacheFile()` if possible.
  const std::shared_ptr<litert::lm::ScopedFile>& GetScopedCacheFile() const {
    return scoped_cache_file_;
  }

  template <typename T>
  absl::StatusOr<const T> GetBackendConfig() const {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    } else {
      return absl::InvalidArgumentError("Backend config is not valid.");
    }
  }

  template <typename T>
  absl::StatusOr<T> MutableBackendConfig() {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    } else {
      return absl::InvalidArgumentError("Backend config is not valid.");
    }
  }

  // Setter APIs.
  void SetCacheDir(const std::string& cache_dir) { cache_dir_ = cache_dir; }
  void SetScopedCacheFile(std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_cache_file_ = std::move(cache_file);
  }
  void SetMaxNumTokens(uint64_t max_num_tokens) {
    max_num_tokens_ = max_num_tokens;
  }
  void SetMaxNumImages(uint32_t max_num_images) {
    max_num_images_ = max_num_images;
  }
  void SetBackend(const Backend& backend) { backend_ = backend; }
  void SetActivationDataType(const ActivationDataType& activation_data_type) {
    activation_data_type_ = activation_data_type;
  }
  void SetBackendConfig(const std::variant<GpuArtisanConfig, GpuConfig,
                                           CpuConfig>& backend_config) {
    backend_config_ = backend_config;
  }

 private:
  explicit LlmExecutorSettings(const ModelAssets& model_assets)
      : model_assets_(model_assets) {}

  // Path to the LiteRT model file.
  const ModelAssets model_assets_;

  // Directory for saving the weight cache file. If this is set and the
  // backend supports it, the re-arranged weights will be stored in the
  // directory after the 1st initialization, making the future initialization
  // to be much faster.
  //
  // Consumers should prefer to use the `cache_file_` if set.
  std::string cache_dir_;

  // Open file for writing the weight cache to and later loading cache from.
  // If set, this should be preferred over the `cache_dir_`.
  std::shared_ptr<litert::lm::ScopedFile> scoped_cache_file_;

  // Maximum number of the sum of input and output tokens. It is equivalent to
  // the size of the kv-cache.
  uint32_t max_num_tokens_;

  // Maximum number of images the model can handle.
  uint32_t max_num_images_;

  // Optional setting to use LLM executor backend.
  Backend backend_ = Backend::CPU;

  // Backend specific config.
  std::variant<GpuArtisanConfig, GpuConfig, CpuConfig> backend_config_;

  // Optional setting for specific activation data type. If not set, the
  // default activation data type for each OS & backend will be used. Setting
  // this field will override the default activation data type, for example,
  // OpenCL backend only support fp32 on Linux.
  std::optional<ActivationDataType> activation_data_type_;

  // Declare the output stream operator as a friend such that it can be used
  // to print the LlmExecutorSettings private member.
  friend std::ostream& operator<<(std::ostream& os,
                                  const LlmExecutorSettings& config);
};
std::ostream& operator<<(std::ostream& os, const LlmExecutorSettings& config);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
