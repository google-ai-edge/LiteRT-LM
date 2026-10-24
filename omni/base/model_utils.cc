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

#include "omni/base/model_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_custom_op.h"  // from @litert
#include "omni/base/litert_lm_engine_runner.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_cache.h"
#include "omni/omni_engine.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/huggingface_tokenizer.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni {

namespace {

class TfLiteModelResources : public lm::ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteModelResources>> Create(
      absl::string_view path) {
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(std::string(path)));
    return absl::WrapUnique(new TfLiteModelResources(std::move(model)));
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == lm::proto::LlmMetadata::TF_LITE_PREFILL_DECODE ||
        model_type == lm::proto::LlmMetadata::TF_LITE_MTP_DRAFTER) {
      return &model_;
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("GetTFLiteModelBuffer not implemented.");
  }

  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("GetTokenizer not implemented.");
  }

  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("GetLlmMetadata not implemented.");
  }

  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

 private:
  explicit TfLiteModelResources(Model model) : model_(std::move(model)) {}

  Model model_;
};

absl::StatusOr<std::unique_ptr<lm::ModelResources>> CreateModelResources(
    absl::string_view path, std::shared_ptr<lm::ScopedFile> scoped_file) {
  auto format_or = lm::GetFileFormat(path, scoped_file);
  if (format_or.ok() && *format_or == lm::FileFormat::TFLITE) {
    return TfLiteModelResources::Create(path);
  }
  if (format_or.ok() && *format_or == lm::FileFormat::TASK) {
    ABSL_ASSIGN_OR_RETURN(auto resources,
                          lm::ModelAssetBundleResources::Create(
                              /*tag=*/"", std::move(scoped_file)));
    return lm::ModelResourcesTask::Create(std::move(resources));
  }
  ABSL_ASSIGN_OR_RETURN(auto loader,
                        lm::LitertLmLoader::Create(std::move(scoped_file)));
  return lm::ModelResourcesLitertLm::Create(std::move(loader));
}

absl::StatusOr<std::vector<std::string>> GetStateTensorPatterns(
    const Model& model, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs) {
  std::vector<std::string> patterns;
  LITERT_ASSIGN_OR_RETURN(const auto& in_names,
                          model.GetSignatureInputNames(signature_name));
  for (size_t i = num_non_state_inputs; i < in_names.size(); ++i) {
    patterns.push_back(std::string(in_names[i]));
  }

  LITERT_ASSIGN_OR_RETURN(const auto& out_names,
                          model.GetSignatureOutputNames(signature_name));
  for (size_t i = num_non_state_outputs; i < out_names.size(); ++i) {
    patterns.push_back(std::string(out_names[i]));
  }
  return patterns;
}

std::string JoinPath(absl::string_view dir, absl::string_view filename) {
  return (std::filesystem::path(std::string_view(dir)) /
          std::string_view(filename))
      .string();
}

}  // namespace

absl::StatusOr<OmniEngine::Settings::Backend> ParseOmniBackend(
    absl::string_view backend_str) {
  const std::string lower = absl::AsciiStrToLower(backend_str);
  if (lower == "cpu") {
    return OmniEngine::Settings::Backend::kCpu;
  }
  if (lower == "gpu") {
    return OmniEngine::Settings::Backend::kGpu;
  }
  if (lower == "npu") {
    return OmniEngine::Settings::Backend::kNpu;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported backend: ", backend_str));
}

lm::Backend ToLmBackend(OmniEngine::Settings::Backend backend) {
  switch (backend) {
    case OmniEngine::Settings::Backend::kGpu:
      return lm::Backend::GPU;
    case OmniEngine::Settings::Backend::kNpu:
      return lm::Backend::NPU;
    case OmniEngine::Settings::Backend::kCpu:
    default:
      return lm::Backend::CPU;
  }
}

ModelOptions ToLmModelOptions(const OmniEngine::Settings& settings) {
  ModelOptions lm_options;
  lm_options.backend = ToLmBackend(settings.backend);
  lm_options.num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  lm_options.cache_dir = settings.cache_dir.empty() ? std::string(kNoCacheDir)
                                                    : settings.cache_dir;
  lm_options.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  return lm_options;
}

absl::StatusOr<std::unique_ptr<::litert::support::Tokenizer>> CreateTokenizer(
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view tokenizer_path) {
  if (lm_resources != nullptr) {
    return lm_resources->GetTokenizer();
  }
  if (tokenizer_path.empty()) {
    return absl::InvalidArgumentError(
        "Tokenizer path must be provided when not using a .litertlm model.");
  }
  ABSL_RETURN_IF_ERROR(CheckFileReadable(tokenizer_path));
  return ::litert::support::HuggingFaceTokenizer::CreateFromFile(
      tokenizer_path);
}

absl::Status CheckFileReadable(absl::string_view path) {
  std::string path_str(path);
  std::ifstream in(path_str);
  if (!in.good()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> LoadFile(absl::string_view path) {
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  std::string path_str(path);
  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("File not found or unreadable: ", path));
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string content(size, '\0');
  if (size > 0 && !file.read(content.data(), size)) {
    return absl::InternalError(absl::StrCat("Failed to read file: ", path));
  }
  return content;
}

absl::StatusOr<std::string> LoadFile(absl::string_view model_dir,
                                     absl::string_view filename) {
  return LoadFile(JoinPath(model_dir, filename));
}

absl::Status DownloadFileWithCurl(absl::string_view url,
                                  absl::string_view target_path) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
  (void)url;
  (void)target_path;
  return absl::UnimplementedError(
      "DownloadFileWithCurl is not supported on iOS.");
#else
  ABSL_LOG(INFO) << "Downloading " << url << " to " << target_path;
  std::string cmd =
      absl::StrCat("mkdir -p $(dirname \"", target_path,
                   "\") && curl -L -s -o \"", target_path, "\" \"", url, "\"");
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    return absl::InternalError(
        absl::StrCat("Failed to download from ", url, " to ", target_path));
  }
  return absl::OkStatus();
#endif
}

std::string ResolveLitertLmPath(absl::string_view model_folder) {
  constexpr absl::string_view kLitertLmExtension = ".litertlm";
  std::filesystem::path path{std::string(model_folder)};
  std::error_code ec;
  if (std::filesystem::is_regular_file(path, ec) &&
      path.extension().string() == kLitertLmExtension) {
    return path.string();
  }
  if (std::filesystem::is_directory(path, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
      if (entry.is_regular_file(ec) &&
          entry.path().extension().string() == kLitertLmExtension) {
        return entry.path().string();
      }
    }
  }
  return "";
}

absl::StatusOr<std::shared_ptr<lm::ModelResources>> CreateLmModelResources(
    absl::string_view model_path) {
  ABSL_RETURN_IF_ERROR(CheckFileReadable(model_path));
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(model_path));
  ABSL_ASSIGN_OR_RETURN(auto loader,
                        lm::LitertLmLoader::Create(std::move(scoped_file)));
  ABSL_ASSIGN_OR_RETURN(auto resources,
                        lm::ModelResourcesLitertLm::Create(std::move(loader)));
  return std::shared_ptr<lm::ModelResources>(std::move(resources));
}

namespace {

// `model_path` is the file the model weights are read from; it only feeds the
// on-disk cache key so caches are invalidated when that file changes.
absl::StatusOr<Options> CreateCompilationOptions(const ModelOptions& options,
                                                 absl::string_view model_name,
                                                 absl::string_view model_path) {
  LITERT_ASSIGN_OR_RETURN(auto comp_options, Options::Create());
  bool target_gpu = options.backend == lm::Backend::GPU;
  ABSL_ASSIGN_OR_RETURN(
      const ModelCacheConfig cache,
      ResolveModelCache(model_name, model_path, options.cache_dir));

  if (target_gpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            comp_options.GetOptions<::litert::GpuOptions>());
    gpu_compilation_options.EnableInfiniteFloatCapping(true);
    gpu_compilation_options.SetPrecision(options.force_fp32_on_gpu
                                             ? GpuOptions::Precision::kFp32
                                             : GpuOptions::Precision::kFp16);
#if defined(__APPLE__)
    // Metal argument buffers setting.
    gpu_compilation_options.SetPreferTextureWeights(false);
    gpu_compilation_options.SetUseMetalArgumentBuffers(true);
#elif defined(__ANDROID__)
    gpu_compilation_options.SetPreferTextureWeights(true);
#endif
    gpu_compilation_options.EnableConstantTensorSharing(true);
    gpu_compilation_options.SetMadviseOriginalSharedTensors(true);
    gpu_compilation_options.SetConvertWeightsOnGpu(true);
    gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
    LrtSetGpuOptionsUseIrModel(gpu_compilation_options.Get(), false);
    for (const auto& pattern : options.external_tensor_patterns) {
      gpu_compilation_options.AddExternalTensorPattern(pattern.c_str());
      gpu_compilation_options.AddBufferStorageTensorPattern(pattern.c_str());
    }
    ABSL_RETURN_IF_ERROR(
        ConfigureGpuModelCache(model_name, cache, gpu_compilation_options));
    comp_options.SetHardwareAccelerators(HwAccelerators::kGpu);
  } else {
    comp_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            comp_options.GetOptions<::litert::CpuOptions>());
    ABSL_RETURN_IF_ERROR(lm::SetCpuOptions(cpu_options, options.num_threads));
    ABSL_RETURN_IF_ERROR(
        ConfigureCpuModelCache(model_name, cache, cpu_options));
  }

  // Register custom bidirectional LSTM op.
  // Must happen before CompiledModel::Create: the op resolver is built there,
  // so a kernel added afterwards is never seen.
  LITERT_RETURN_IF_ERROR(
      litert::custom_ops::RegisterBidirectionalLstmCustomOp(comp_options));
  return comp_options;
}

}  // namespace

absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  std::string path = JoinPath(options.model_dir, model_filename);
  ABSL_ASSIGN_OR_RETURN(
      auto comp_options,
      CreateCompilationOptions(options, model_filename, /*model_path=*/path));
  bool target_gpu = options.backend == lm::Backend::GPU;
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, path, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                               target_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

absl::StatusOr<std::unique_ptr<CompiledModel>> CreateCompiledModel(
    const OmniEngine::Settings& settings, Environment& env, Options& options,
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view model_type) {
  if (lm_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view model_buffer,
                          lm_resources->GetTFLiteModelBuffer(model_type));
    LITERT_ASSIGN_OR_RETURN(
        auto comp_model,
        CompiledModel::Create(
            env,
            BufferRef<uint8_t>(
                reinterpret_cast<const uint8_t*>(model_buffer.data()),
                model_buffer.size()),
            options));
    return std::make_unique<CompiledModel>(std::move(comp_model));
  }

  ABSL_RETURN_IF_ERROR(CheckFileReadable(settings.model_path));
  LITERT_ASSIGN_OR_RETURN(
      auto comp_model,
      CompiledModel::Create(env, settings.model_path, options));
  return std::make_unique<CompiledModel>(std::move(comp_model));
}

absl::StatusOr<CompiledModel> CreateCompiledModelFromBuffer(
    Environment& env, const ModelOptions& options,
    absl::string_view model_buffer, absl::string_view model_name) {
  // The buffer is a section of the `.litertlm` container at `model_dir`.
  ABSL_ASSIGN_OR_RETURN(
      auto comp_options,
      CreateCompilationOptions(options, model_name,
                               ResolveLitertLmPath(options.model_dir)));
  bool target_gpu = options.backend == lm::Backend::GPU;
  BufferRef<uint8_t> buffer_ref(
      reinterpret_cast<const uint8_t*>(model_buffer.data()),
      model_buffer.size());
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, buffer_ref, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created from buffer with ",
                               target_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

absl::StatusOr<CompiledModel> CreateCompiledModelForStatefulRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs) {
  if (options.backend != lm::Backend::GPU) {
    return CreateCompiledModel(env, options, model_filename);
  }

  std::string path = JoinPath(options.model_dir, model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));

  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromFile(path));
  LITERT_ASSIGN_OR_RETURN(
      auto state_patterns,
      GetStateTensorPatterns(model, signature_name, num_non_state_inputs,
                             num_non_state_outputs));

  ModelOptions stateful_options = options;
  stateful_options.external_tensor_patterns.insert(
      stateful_options.external_tensor_patterns.end(), state_patterns.begin(),
      state_patterns.end());

  return CreateCompiledModel(env, stateful_options, model_filename);
}

absl::StatusOr<std::unique_ptr<LiteRtLmRunner>> CreateLmRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  std::string path = JoinPath(options.model_dir, model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(path));
  auto shared_scoped_file =
      std::make_shared<lm::ScopedFile>(std::move(scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto model_assets,
                        lm::ModelAssets::Create(shared_scoped_file, path));
  // `LiteRtLmRunner` exposes `DecodeLogits()` for CPU-side decoding, so keep
  // `sampler_backend` on CPU to preserve the model's FP32 output logits buffer
  // even when `options.backend == lm::Backend::GPU` uses FP16 precision.
  ABSL_ASSIGN_OR_RETURN(auto executor_settings,
                        lm::LlmExecutorSettings::CreateDefault(
                            std::move(model_assets), options.backend,
                            /*sampler_backend=*/lm::Backend::CPU));
  if (options.backend == lm::Backend::CPU && options.num_threads > 0) {
    lm::CpuConfig cpu_config;
    cpu_config.number_of_threads = options.num_threads;
    executor_settings.SetBackendConfig(cpu_config);
  }
  if (options.backend == lm::Backend::GPU) {
    executor_settings.SetActivationDataType(
        options.force_fp32_on_gpu ? lm::ActivationDataType::FLOAT32
                                  : lm::ActivationDataType::FLOAT16);
  }
  if (!options.cache_dir.empty()) {
    executor_settings.SetCacheDir(options.cache_dir);
  }
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        CreateModelResources(path, shared_scoped_file));
  LITERT_ASSIGN_OR_RETURN(auto executor,
                          lm::CreateLlmLiteRtCompiledModelExecutor(
                              executor_settings, env, *model_resources));
  return std::make_unique<LiteRtLmRunnerImpl>(std::move(executor),
                                              std::move(model_resources));
}

absl::StatusOr<std::unique_ptr<LiteRtLmEngineRunner>> CreateLmEngineRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  std::string path = JoinPath(options.model_dir, model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(path));
  auto shared_scoped_file =
      std::make_shared<lm::ScopedFile>(std::move(scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto model_assets,
                        lm::ModelAssets::Create(shared_scoped_file, path));
  ABSL_ASSIGN_OR_RETURN(auto engine_settings,
                        lm::EngineSettings::CreateDefault(
                            std::move(model_assets), options.backend,
                            /*vision_backend=*/std::nullopt,
                            /*audio_backend=*/options.backend));
  if (options.backend == lm::Backend::CPU && options.num_threads > 0) {
    lm::CpuConfig cpu_config;
    cpu_config.number_of_threads = options.num_threads;
    engine_settings.GetMutableMainExecutorSettings().SetBackendConfig(
        cpu_config);
  }
  if (options.backend == lm::Backend::GPU) {
    const auto activation_type = options.force_fp32_on_gpu
                                     ? lm::ActivationDataType::FLOAT32
                                     : lm::ActivationDataType::FLOAT16;
    engine_settings.GetMutableMainExecutorSettings().SetActivationDataType(
        activation_type);
    if (engine_settings.GetMutableAudioExecutorSettings().has_value()) {
      engine_settings.GetMutableAudioExecutorSettings()->SetActivationDataType(
          activation_type);
    }
  }
  if (!options.cache_dir.empty()) {
    engine_settings.GetMutableMainExecutorSettings().SetCacheDir(
        options.cache_dir);
    if (engine_settings.GetMutableAudioExecutorSettings().has_value()) {
      engine_settings.GetMutableAudioExecutorSettings()->SetCacheDir(
          options.cache_dir);
    }
  }
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        CreateModelResources(path, shared_scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto engine, lm::EngineFactory::CreateDefault(
                                         std::move(engine_settings)));

  auto tok_status = model_resources->GetTokenizer();
  support::Tokenizer* tokenizer =
      (tok_status.ok() && *tok_status != nullptr)
          ? tok_status->get()
          : &const_cast<support::Tokenizer&>(engine->GetTokenizer());

  int stop_token_id = -1;
  auto metadata_status = model_resources->GetLlmMetadata();
  if (metadata_status.ok() && *metadata_status != nullptr) {
    for (const auto& stop_token : (*metadata_status)->stop_tokens()) {
      if (stop_token.has_token_ids() && !stop_token.token_ids().ids().empty()) {
        stop_token_id = stop_token.token_ids().ids(0);
        break;
      } else if (stop_token.has_token_str()) {
        auto id = tokenizer->TokenToId(stop_token.token_str());
        if (id.ok()) {
          stop_token_id = *id;
          break;
        }
      }
    }
  }
  if (stop_token_id < 0) {
    auto id = tokenizer->TokenToId("<eos>");
    if (id.ok()) {
      stop_token_id = *id;
    }
  }

  lm::SessionConfig session_config = lm::SessionConfig::CreateDefault();
  session_config.SetApplyPromptTemplateInSession(false);
  session_config.SetAudioModalityEnabled(true);
  if (stop_token_id >= 0) {
    session_config.GetMutableStopTokenIds().push_back({stop_token_id});
  }
  if (metadata_status.ok() && *metadata_status != nullptr) {
    for (const auto& stop_token : (*metadata_status)->stop_tokens()) {
      if (stop_token.has_token_ids() && !stop_token.token_ids().ids().empty()) {
        std::vector<int> ids(stop_token.token_ids().ids().begin(),
                             stop_token.token_ids().ids().end());
        if (std::find(session_config.GetMutableStopTokenIds().begin(),
                      session_config.GetMutableStopTokenIds().end(),
                      ids) == session_config.GetMutableStopTokenIds().end()) {
          session_config.GetMutableStopTokenIds().push_back(std::move(ids));
        }
      } else if (stop_token.has_token_str()) {
        auto id = tokenizer->TokenToId(stop_token.token_str());
        if (id.ok()) {
          std::vector<int> ids{*id};
          if (std::find(session_config.GetMutableStopTokenIds().begin(),
                        session_config.GetMutableStopTokenIds().end(),
                        ids) == session_config.GetMutableStopTokenIds().end()) {
            session_config.GetMutableStopTokenIds().push_back(std::move(ids));
          }
        }
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(auto session, engine->CreateSession(session_config));

  return std::make_unique<LiteRtLmEngineRunnerImpl>(
      std::move(engine), std::move(session), session_config,
      std::move(model_resources));
}

absl::StatusOr<size_t> ResolveInputIndex(const CompiledModel& model,
                                         absl::string_view tensor_name) {
  auto names_res = model.GetSignatureInputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == tensor_name) {
        return i;
      }
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "Input tensor '", tensor_name, "' not found in model signatures"));
}

absl::StatusOr<size_t> ResolveOutputIndex(const CompiledModel& model,
                                          absl::string_view tensor_name) {
  auto names_res = model.GetSignatureOutputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == tensor_name) {
        return i;
      }
    }
  }
  return absl::NotFoundError(absl::StrCat(
      "Output tensor '", tensor_name, "' not found in model signatures"));
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithText(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  if (type.ElementType() != ElementType::Int32) {
    return absl::InvalidArgumentError(
        "Expected Int32 TensorBuffer for text token IDs");
  }
  lm::ExecutorTextData text_data(std::move(dup));
  return lm::ExecutorInputs(std::move(text_data), std::nullopt, std::nullopt);
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithAudio(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& dims = type.Layout().Dimensions();
  int num_tokens = dims.size() >= 2 ? dims[dims.size() - 2] : 1;
  std::vector<int32_t> tokens(num_tokens, lm::ExecutorAudioData::kSpecialToken);
  LITERT_ASSIGN_OR_RETURN(
      auto token_buf, support::CreateTensorBuffer<int32_t>({1, num_tokens}));
  LITERT_RETURN_IF_ERROR(token_buf.Write<int32_t>(absl::MakeConstSpan(tokens)));

  lm::ExecutorAudioData audio_data;
  audio_data.SetProjectedAudioEmbeddings(std::move(dup));
  lm::ExecutorTextData text_data(std::move(token_buf));
  return lm::ExecutorInputs(std::move(text_data), std::nullopt,
                            std::move(audio_data));
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithVision(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& dims = type.Layout().Dimensions();
  int num_tokens = dims.size() >= 2 ? dims[dims.size() - 2] : 1;
  std::vector<int32_t> tokens(num_tokens,
                              lm::ExecutorVisionData::kSpecialToken);
  LITERT_ASSIGN_OR_RETURN(
      auto token_buf, support::CreateTensorBuffer<int32_t>({1, num_tokens}));
  LITERT_RETURN_IF_ERROR(token_buf.Write<int32_t>(absl::MakeConstSpan(tokens)));

  lm::ExecutorVisionData vision_data;
  vision_data.SetEmbeddings(std::move(dup));
  lm::ExecutorTextData text_data(std::move(token_buf));
  return lm::ExecutorInputs(std::move(text_data), std::move(vision_data),
                            std::nullopt);
}

}  // namespace litert::omni
