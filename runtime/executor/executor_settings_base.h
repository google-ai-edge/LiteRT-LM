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

#ifndef THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_EXECUTOR_SETTINGS_BASE_H_
#define THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_EXECUTOR_SETTINGS_BASE_H_

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/data_stream.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

enum class Backend {
  // Unspecified backend.
  UNSPECIFIED,

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

  // NPU backend.
  NPU,
};
std::ostream& operator<<(std::ostream& os, const Backend& backend);
// Returns the backend enum from the string. Case-insensitive.
absl::StatusOr<Backend> GetBackendFromString(absl::string_view backend_str);
// Returns the string representation of the backend enum.
std::string GetBackendString(Backend backend);

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

absl::StatusOr<ActivationDataType> GetActivationDataTypeFromString(
    const std::string& activation_data_type);

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
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<MemoryMappedFile> model_file);
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<MemoryMappedFile> model_file,
      absl::string_view model_path);
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<DataStream> data_stream);

  // Convenience factory function to create a ModelAssets with both a model
  // path and file. Will use the scoped file if both are provided.
  static absl::StatusOr<ModelAssets> Create(
      std::shared_ptr<ScopedFile> model_file, absl::string_view model_path);

  bool HasScopedFile() const { return scoped_file_ != nullptr; }
  bool HasMemoryMappedFile() const { return memory_mapped_file_ != nullptr; }
  bool HasDataStream() const { return data_stream_ != nullptr; }

  // Returns the model file if it was created with the respective variant,
  // otherwise returns an error.
  absl::StatusOr<absl::string_view> GetPath() const;
  absl::StatusOr<std::shared_ptr<ScopedFile>> GetScopedFile() const;
  absl::StatusOr<std::shared_ptr<MemoryMappedFile>> GetMemoryMappedFile() const;
  absl::StatusOr<std::shared_ptr<DataStream>> GetDataStream() const;

  // Convenience method to get a read-only scoped file to the model file
  // regardless of whether this instance was created from a path or scoped file.
  absl::StatusOr<std::shared_ptr<ScopedFile>> GetOrCreateScopedFile() const;

  FakeWeightsMode fake_weights_mode() const { return fake_weights_mode_; }

  void SetFakeWeightsMode(FakeWeightsMode fake_weights_mode) {
    fake_weights_mode_ = fake_weights_mode;
  }

 private:
  explicit ModelAssets(std::shared_ptr<ScopedFile> model_file,
                       absl::string_view model_path);
  explicit ModelAssets(absl::string_view model_path);
  explicit ModelAssets(std::shared_ptr<MemoryMappedFile> model_file);
  explicit ModelAssets(std::shared_ptr<MemoryMappedFile> model_file,
                       absl::string_view model_path);
  explicit ModelAssets(std::shared_ptr<DataStream> data_stream);

  // TODO: b/417814685 - Consider supporting multiple model files if the need
  // case arises.
  std::string path_;
  std::shared_ptr<ScopedFile> scoped_file_;
  std::shared_ptr<MemoryMappedFile> memory_mapped_file_;
  std::shared_ptr<DataStream> data_stream_;

  FakeWeightsMode fake_weights_mode_ = FakeWeightsMode::FAKE_WEIGHTS_NONE;
};
std::ostream& operator<<(std::ostream& os, const ModelAssets& model_assets);

// Base Settings for the executor modules.
class ExecutorSettingsBase {
 public:
  // Holds centralized cache suffixes and keys for different backend topologies.
  struct CacheSuffix {
    std::string weight_suffix;            // For CPU (XNNPACK)
    std::string program_suffix;           // For GPU (MlDrift) program cache
    std::string gpu_weight_cache_suffix;  // For GPU (MlDrift) weight cache key
  };

  // Cache suffixes for different backend names. By default, the generate cache
  // file names will have the following format:
  //  XNNPACK: <model_path>.xnnpack_cache_<unique_id>
  //  ML Drift Program Cache: <model_path>_<unique_id>_mldrift_program_cache.bin
  //  ML Drift Weight Cache: <model_path>_<unique_id>
  static constexpr absl::string_view kXnnpackCacheSuffix = ".xnnpack_cache";
  static constexpr absl::string_view kMlDriftCacheSuffix =
      "_mldrift_program_cache.bin";

  // Dynamically generates cache suffix and keys systematically.
  // Args:
  //   - backend: The target backend topology. Valid values are Backend::CPU and
  //   Backend::GPU.
  //   - model_path: The absolute path to the modality main model file.
  //   - module_name: The optional sub-module identifier. Valid values are empty
  //   string (""),
  //     "vision_encoder", "vision_adapter", "streaming_audio_encoder",
  //     "audio_adapter", "static_audio_encoder".
  // Returns:
  //   - CacheSuffix: Centralized naming extensions.
  //   - absl::Status: InvalidArgumentError if validation fails on backend or
  //   module identifiers.
  static absl::StatusOr<CacheSuffix> GetCacheSuffix(
      Backend backend, absl::string_view model_path,
      absl::string_view module_name = "");

  virtual ~ExecutorSettingsBase() = default;

  // Getter APIs.
  const ModelAssets& GetModelAssets() const { return model_assets_; }

  // Backend APIs.
  const Backend& GetBackend() const { return backend_; }
  virtual absl::Status SetBackend(const Backend& backend) {
    backend_ = backend;
    return absl::OkStatus();
  }

  // Activation data type APIs.
  const std::optional<ActivationDataType>& GetActivationDataType() const {
    return activation_data_type_;
  }
  void SetActivationDataType(const ActivationDataType& activation_data_type) {
    activation_data_type_ = activation_data_type;
  }

  // Should be used by consumers who want to write to a single weight cache
  // file. Returns, in order of preference:
  //   1. an open file descriptor to the weight cache file,
  //   2. the file path of the weight cache file, based on the given cache
  //      directory and/or model path. Will append `suffix`.
  //   3. an error if a weight cache file could not be determined.
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix = ".cache",
                     bool check_and_clean = false) const;
  // Prefer to use `GetWeightCacheFile()` if possible.
  const std::string& GetCacheDir() const { return cache_dir_; }
  // Prefer to use `GetWeightCacheFile()` if possible.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedCacheFile() const {
    return scoped_cache_file_;
  }
  const std::string& GetLitertDispatchLibDir() const {
    return litert_dispatch_lib_dir_;
  }

  // Should be used by consumers who want to write to a single program cache
  // file. Returns, in order of preference:
  //   1. an open file descriptor to the program cache file,
  //   2. the file path of the program cache file, based on the given cache
  //      directory and/or model path. Will append `suffix`.
  //   3. an error if a program cache file could not be determined.
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile(absl::string_view suffix = ".program_cache",
                      bool check_and_clean = false) const;
  // Prefer to use `GetProgramCacheFile()` if possible.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedProgramCacheFile() const {
    return scoped_program_cache_file_;
  }

  // Setter APIs.
  void SetCacheDir(const std::string& cache_dir) { cache_dir_ = cache_dir; }
  void SetScopedCacheFile(std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_cache_file_ = std::move(cache_file);
  }
  void SetLitertDispatchLibDir(const std::string& litert_dispatch_lib_dir) {
    litert_dispatch_lib_dir_ = litert_dispatch_lib_dir;
  }

  void SetScopedProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_program_cache_file_ = std::move(cache_file);
  }

 protected:
  explicit ExecutorSettingsBase(ModelAssets model_assets)
      : model_assets_(std::move(model_assets)) {}
  // Optional setting to use LLM executor backend.
  Backend backend_ = Backend::CPU;

 private:
  // Path to the LiteRT model file.
  ModelAssets model_assets_;

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

  // Open file for writing the program cache to and later loading cache from.
  // If set, this should be preferred over the `cache_dir_`.
  std::shared_ptr<litert::lm::ScopedFile> scoped_program_cache_file_;

  // Optional setting for specific activation data type. If not set, the
  // default activation data type for each OS & backend will be used. Setting
  // this field will override the default activation data type, for example,
  // OpenCL backend only support fp32 on Linux.
  std::optional<ActivationDataType> activation_data_type_;

  // Optional LoRA model assets.
  std::optional<ModelAssets> lora_model_assets_;

  // LiteRT dispatch library directory. If not set, the runtime will look for
  // the library in the path defined as the environment variables.
  std::string litert_dispatch_lib_dir_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
