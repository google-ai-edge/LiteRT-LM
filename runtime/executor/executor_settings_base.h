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

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "runtime/util/data_stream.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// NOTE: `Backend` currently conflates the target hardware accelerator (`CPU`,
// `GPU`, `NPU`) with the executor implementation (`CPU_ARTISAN`, `GPU_ARTISAN`,
// `TENSOR_LLM`, and dynamically registered custom executors such as
// `gemma_cpp`). To support running a custom executor across multiple hardware
// backends (e.g., `gemma_cpp` on `CPU` vs. `GPU`), we should decouple the
// executor implementation selector (e.g., an `executor_type` / custom executor
// name in `ExecutorSettingsBase`) from the hardware `Backend` enum.
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
// Registers a custom/dynamic backend name (case-insensitive) and returns its
// assigned Backend identifier. If the name is already registered or matches a
// built-in backend, returns the existing Backend identifier.
Backend RegisterCustomBackend(absl::string_view backend_name);
// Returns true if `backend` was dynamically registered via
// `RegisterCustomBackend`.
bool IsCustomBackend(Backend backend);
// Returns the backend enum from the string. Case-insensitive.
absl::StatusOr<Backend> GetBackendFromString(absl::string_view backend_str);
// Returns the string representation of the backend enum.
std::string GetBackendString(Backend backend);
// Returns true if `backend` runs on the CPU (CPU or CPU_ARTISAN), i.e. uses no
// accelerator. Custom backends are not assumed to be CPU-only.
bool IsCpuBackend(Backend backend);

// Returns the LiteRT hardware accelerator that matches `backend`, for use when
// compiling auxiliary subgraphs (e.g. logit masking) that must run on the same
// device as the model. Returns HwAccelerators::kNone for backends that have no
// LiteRT accelerator equivalent.
::litert::HwAccelerators GetHwAcceleratorForBackend(Backend backend);

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

// Parses a tri-state GPU IR model setting. Case-insensitive.
// "enabled" -> true, "disabled" -> false, "default" -> std::nullopt.
absl::StatusOr<std::optional<bool>> GetUseIrModelFromString(
    absl::string_view use_ir_model);

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

// Vendor-agnostic NPU performance (power) mode. Each value is translated to the
// closest vendor-specific performance mode (e.g. Qualcomm HTP performance mode,
// Google Tensor performance mode) when the NPU options are built.
enum class NpuPerformanceMode {
  // Do not set any performance mode, i.e. use the vendor's default.
  kVendorDefault,

  // Lowest power consumption, lowest performance.
  kExtremePowerSaver,

  // Low power consumption.
  kPowerSaver,

  // Balance between power consumption and performance.
  kBalanced,

  // High performance.
  kHighPerformance,

  // High performance that can be sustained over a long period of time.
  kSustainedHighPerformance,

  // Highest performance, highest power consumption.
  kBurst,
};
std::ostream& operator<<(std::ostream& os, const NpuPerformanceMode& mode);

// Controls when the NPU performance mode is applied.
enum class NpuPerformanceControl {
  // The performance mode is applied when the model is loaded and held until it
  // is unloaded (Qualcomm: HtpPerfCtrlMode::kManual).
  kWhileLoaded,

  // The performance mode is applied before each inference and released shortly
  // after it, so that the NPU can return to a low power state while idle
  // (Qualcomm: HtpPerfCtrlMode::kAuto).
  kPerInference,
};
std::ostream& operator<<(std::ostream& os,
                         const NpuPerformanceControl& control);

// MediaTek NeuroPilot SDK version used when loading the MediaTek NPU dispatch
// runtime. Mirrors `litert::mediatek::MediatekOptions::NeronSDKVersion`.
enum class MediatekSdkVersion {
  // Do not set an explicit SDK version, i.e. use the vendor's default.
  kDefault,
  kVersion7,
  kVersion8,
  kVersion9,
};
std::ostream& operator<<(std::ostream& os, const MediatekSdkVersion& version);

// MediaTek-specific NPU options. Only honored by MediaTek NPUs.
struct MediatekNpuOptions {
  // MediaTek NeuroPilot SDK version. Defaults to kDefault to preserve the
  // historical behavior.
  MediatekSdkVersion sdk_version = MediatekSdkVersion::kDefault;
};
std::ostream& operator<<(std::ostream& os,
                         const MediatekNpuOptions& mediatek_options);

// NPU options applied when compiling a model for the NPU backend, including
// common cross-vendor options and vendor-specific sub-options.
struct NpuOptions {
  // The NPU performance (power) mode. Defaults to kBurst to preserve the
  // historical behavior.
  NpuPerformanceMode performance_mode = NpuPerformanceMode::kBurst;

  // When `performance_mode` is applied. Defaults to kWhileLoaded to preserve
  // the historical behavior. Currently only honored by Qualcomm NPUs.
  NpuPerformanceControl performance_control =
      NpuPerformanceControl::kWhileLoaded;

  // MediaTek-specific NPU options.
  MediatekNpuOptions mediatek_options;
};
std::ostream& operator<<(std::ostream& os, const NpuOptions& npu_options);

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

  // Returns true if the assets refer to an existing filesystem directory path
  // (rather than a regular file, `ScopedFile`, `MemoryMappedFile`, or
  // `DataStream`).
  bool IsDirectory() const;

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

  // The kind of cache file. A cache saves work done on the first model load so
  // that later loads can skip it.
  enum class CacheFileKind {
    // Weights converted to the backend's layout: XNNPACK packed weights on CPU,
    // ML Drift weights on GPU.
    kWeight,
    // Compiled GPU programs (ML Drift shaders). Only the GPU backend uses it.
    kProgram,
  };

  // The model a cache file belongs to: the main model or one of its
  // sub-models. Each one needs its own cache file. The audio executor asks for
  // kStaticAudioEncoder or kStreamingAudioEncoder, never kAudioEncoder; their
  // scoped files default to kAudioEncoder's.
  enum class CacheModel {
    kMain,
    kMtpDrafter,
    kVisionEncoder,
    kVisionAdapter,
    kAudioEncoder,
    kStaticAudioEncoder,
    kStreamingAudioEncoder,
    kAudioAdapter,
    kTextEncoder,
  };

  // Returns the name `model` puts in front of the backend suffix in its cache
  // suffixes, e.g. ".mtp_drafter" in ".mtp_drafter_mldrift_weight_cache.bin".
  // The main model's name is empty. Cache file names are built from these, so
  // they must not change.
  static constexpr absl::string_view GetCacheModelName(CacheModel model) {
    switch (model) {
      case CacheModel::kMain:
        return "";
      case CacheModel::kMtpDrafter:
        return ".mtp_drafter";
      case CacheModel::kVisionEncoder:
        return ".vision_encoder";
      case CacheModel::kVisionAdapter:
        return ".vision_adapter";
      case CacheModel::kAudioEncoder:
        return ".audio_encoder";
      case CacheModel::kStaticAudioEncoder:
        return ".static_audio_encoder";
      case CacheModel::kStreamingAudioEncoder:
        return ".streaming_audio_encoder";
      case CacheModel::kAudioAdapter:
        return ".audio_adapter";
      case CacheModel::kTextEncoder:
        return ".text_encoder";
    }
    return "";
  }

  // A cache file the engine needs. `suffix` is valid only during the
  // `CacheFileProvider` call.
  struct CacheFileRequest {
    CacheFileKind kind;
    CacheModel model;
    // The full cache suffix, e.g. ".mtp_drafter_mldrift_weight_cache.bin".
    // Hosts can use it to name the file.
    absl::string_view suffix;
  };

  // Returns an open, writable cache file for a request, or NotFound to build
  // that model without a cache. It runs on engine threads, possibly more than
  // once per request (e.g. when the MTP drafter loads lazily), so it must be
  // thread-safe and return the same file for the same request.
  using CacheFileProvider =
      std::function<absl::StatusOr<std::shared_ptr<litert::lm::ScopedFile>>(
          const CacheFileRequest&)>;

  // Cache suffixes for different backend names. By default, the generate cache
  // file names will have the following format:
  //  XNNPACK: <model_path>.xnnpack_cache_<unique_id>
  //  ML Drift Program Cache: <model_path>_<unique_id>_mldrift_program_cache.bin
  //  ML Drift Weight Cache: <model_path>_<unique_id>
  static constexpr absl::string_view kXnnpackCacheSuffix = ".xnnpack_cache";
  static constexpr absl::string_view kMlDriftCacheSuffix =
      "_mldrift_program_cache.bin";
  static constexpr absl::string_view kMlDriftWeightCacheSuffix =
      "_mldrift_weight_cache.bin";
  // Suffixes used by `GetWeightCacheFile()` / `GetProgramCacheFile()` when no
  // suffix is given.
  static constexpr absl::string_view kDefaultWeightCacheSuffix = ".cache";
  static constexpr absl::string_view kDefaultProgramCacheSuffix =
      ".program_cache";

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
  void SetModelAssets(ModelAssets model_assets) {
    model_assets_ = std::move(model_assets);
  }

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

  // Mixed precision APIs.
  bool IsMixedPrecisionEnabled() const { return enable_mixed_precision_; }
  void SetEnableMixedPrecision(bool enable) {
    enable_mixed_precision_ = enable;
  }

  // Should be used by consumers who want to write to a single weight cache
  // file. `suffix` is the model's name (see `GetCacheModelName()`) followed by
  // the backend suffix. Returns, in order of preference:
  //   1. the scoped cache file set for the model,
  //   2. the file from the `CacheFileProvider`, if one is set,
  //   3. the file path of the weight cache file, based on the given cache
  //      directory and/or model path. Will append `suffix`.
  //   4. an error if a weight cache file could not be determined. A sub-model
  //      with no file of its own gets NotFound if the main model has a scoped
  //      file or a provider is set, rather than sharing the main model's file.
  virtual absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix, bool check_and_clean) const;

  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile() const {
    return GetWeightCacheFile(kDefaultWeightCacheSuffix, false);
  }

  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix) const {
    return GetWeightCacheFile(suffix, false);
  }

  // Prefer to use `GetWeightCacheFile()` if possible.
  const std::string& GetCacheDir() const { return cache_dir_; }
  // Prefer to use `GetWeightCacheFile()` if possible.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedCacheFile() const {
    return GetScopedCacheFile(CacheModel::kMain, CacheFileKind::kWeight);
  }
  const std::string& GetLitertDispatchLibDir() const {
    return litert_dispatch_lib_dir_;
  }

  // Should be used by consumers who want to write to a single program cache
  // file. Same order of preference as `GetWeightCacheFile()`.
  virtual absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile(absl::string_view suffix, bool check_and_clean) const;

  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile() const {
    return GetProgramCacheFile(kDefaultProgramCacheSuffix, false);
  }

  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile(absl::string_view suffix) const {
    return GetProgramCacheFile(suffix, false);
  }
  // Prefer to use `GetProgramCacheFile()` if possible.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedProgramCacheFile() const {
    return GetScopedCacheFile(CacheModel::kMain, CacheFileKind::kProgram);
  }

  // Returns the scoped cache file set for `model`, or nullptr.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedCacheFile(
      CacheModel model, CacheFileKind kind) const;

  // Setter APIs.
  void SetCacheDir(const std::string& cache_dir) { cache_dir_ = cache_dir; }
  void SetScopedCacheFile(std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    SetScopedCacheFile(CacheModel::kMain, CacheFileKind::kWeight,
                       std::move(cache_file));
  }
  void SetLitertDispatchLibDir(const std::string& litert_dispatch_lib_dir) {
    litert_dispatch_lib_dir_ = litert_dispatch_lib_dir;
  }

  void SetScopedProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    SetScopedCacheFile(CacheModel::kMain, CacheFileKind::kProgram,
                       std::move(cache_file));
  }

  // Sets an open, writable cache file for `model`, for hosts that can't let
  // the engine create files (e.g. a sandboxed process). Each model needs its
  // own file. nullptr clears it.
  void SetScopedCacheFile(CacheModel model, CacheFileKind kind,
                          std::shared_ptr<litert::lm::ScopedFile> cache_file);

  // Sets where cache files come from for models with no scoped cache file.
  // Lets the host open files on demand, for any model, including sub-models
  // such as the MTP drafter. Takes precedence over the cache directory.
  void SetCacheFileProvider(CacheFileProvider provider) {
    cache_file_provider_ = std::move(provider);
  }
  bool HasCacheFileProvider() const { return cache_file_provider_ != nullptr; }

  void SetDisableWeightCache(bool disable) { disable_weight_cache_ = disable; }
  void SetDisableProgramCache(bool disable) {
    disable_program_cache_ = disable;
  }
  bool IsWeightCacheDisabled() const { return disable_weight_cache_; }
  bool IsProgramCacheDisabled() const { return disable_program_cache_; }

  // Sets whether the GPU backend uses the IR model path. std::nullopt means
  // "default", which is currently resolved to disabled.
  void SetUseIrModel(std::optional<bool> use_ir_model) {
    use_ir_model_ = use_ir_model;
  }
  const std::optional<bool>& UseIrModel() const { return use_ir_model_; }

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

  // Open files for writing caches to and later loading them from, per model.
  // If set, these are preferred over `cache_file_provider_` and `cache_dir_`.
  absl::flat_hash_map<CacheModel, std::shared_ptr<litert::lm::ScopedFile>>
      scoped_weight_cache_files_;
  absl::flat_hash_map<CacheModel, std::shared_ptr<litert::lm::ScopedFile>>
      scoped_program_cache_files_;

  // Supplies cache files for models with no scoped cache file. If set, this
  // is preferred over `cache_dir_`.
  CacheFileProvider cache_file_provider_;

  // Optional setting for specific activation data type. If not set, the
  // default activation data type for each OS & backend will be used. Setting
  // this field will override the default activation data type, for example,
  // OpenCL backend only support fp32 on Linux.
  std::optional<ActivationDataType> activation_data_type_;

  // Optional setting to enable mixed precision. If true, it will override
  // activation data type to FP32 which underlying for mix precision.
  bool enable_mixed_precision_ = false;

  // Optional LoRA model assets.
  std::optional<ModelAssets> lora_model_assets_;

  // LiteRT dispatch library directory. If not set, the runtime will look for
  // the library in the path defined as the environment variables.
  std::string litert_dispatch_lib_dir_;

  bool disable_weight_cache_ = false;
  bool disable_program_cache_ = false;
  // Optional setting to use the IR model path on the GPU backend. If not set,
  // the default (disabled) is used.
  std::optional<bool> use_ir_model_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
