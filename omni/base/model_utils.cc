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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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
#include "omni/base/litert_lm_runner.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
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
      lm::ModelType model_type) override {
    if (model_type == lm::ModelType::kTfLitePrefillDecode ||
        model_type == lm::ModelType::kTfLiteMtpDrafter) {
      return &model_;
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::ModelType model_type) override {
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
      lm::ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::ModelType model_type) override {
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

}  // namespace

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
  return LoadFile(absl::StrCat(model_dir, "/", filename));
}

absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  LITERT_ASSIGN_OR_RETURN(auto comp_options, Options::Create());
  bool target_gpu = options.backend == lm::Backend::GPU;

  if (target_gpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            comp_options.GetOptions<::litert::GpuOptions>());
    gpu_compilation_options.EnableInfiniteFloatCapping(true);
    gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
#if defined(__APPLE__)
    // Metal argument buffers setting.
    gpu_compilation_options.SetUseMetalArgumentBuffers(true);
#endif  // !__APPLE__
    gpu_compilation_options.EnableConstantTensorSharing(true);
    gpu_compilation_options.SetMadviseOriginalSharedTensors(true);
    gpu_compilation_options.SetConvertWeightsOnGpu(true);
    gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
    comp_options.SetHardwareAccelerators(HwAccelerators::kGpu);
  } else {
    comp_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            comp_options.GetOptions<::litert::CpuOptions>());
    ABSL_RETURN_IF_ERROR(lm::SetCpuOptions(cpu_options, options.num_threads));

    if (!options.cache_dir.empty()) {
      std::string cache_path = absl::StrCat(options.cache_dir, "/",
                                            model_filename, ".xnnpack_cache");
      absl::StatusOr<std::variant<std::string, std::shared_ptr<lm::ScopedFile>>>
          cache_variant(cache_path);
      ABSL_RETURN_IF_ERROR(
          lm::SetCpuCacheOptions(cache_variant, model_filename, cpu_options));
    }
  }

  std::string path = absl::StrCat(options.model_dir, "/", model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, path, comp_options));
  ABSL_VLOG(2) << absl::StrCat("Compiled model created successfully with ",
                               target_gpu ? "GPU" : "CPU", " backend");
  return std::move(compiled_model);
}

absl::StatusOr<std::unique_ptr<LiteRtLmRunner>> CreateLmRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename) {
  std::string path = absl::StrCat(options.model_dir, "/", model_filename);
  ABSL_RETURN_IF_ERROR(CheckFileReadable(path));
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, lm::ScopedFile::Open(path));
  auto shared_scoped_file =
      std::make_shared<lm::ScopedFile>(std::move(scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto model_assets,
                        lm::ModelAssets::Create(shared_scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto executor_settings,
                        lm::LlmExecutorSettings::CreateDefault(
                            std::move(model_assets), options.backend));
  if (options.backend == lm::Backend::CPU && options.num_threads > 0) {
    lm::CpuConfig cpu_config;
    cpu_config.number_of_threads = options.num_threads;
    executor_settings.SetBackendConfig(cpu_config);
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
