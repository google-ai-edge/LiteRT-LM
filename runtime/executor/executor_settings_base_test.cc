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

#include "runtime/executor/executor_settings_base.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/file_data_stream.h"
#include "runtime/util/file_util.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::VariantWith;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

std::string GetTestModelPath() {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  return model_path.string();
}

TEST(LlmExecutorConfigTest, Backend) {
  Backend backend;
  std::stringstream oss;
  backend = Backend::GPU_ARTISAN;
  oss << backend;
  EXPECT_EQ(oss.str(), "GPU_ARTISAN");

  backend = Backend::GPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "GPU");

  backend = Backend::CPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "CPU");

  backend = Backend::GOOGLE_TENSOR_ARTISAN;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "GOOGLE_TENSOR_ARTISAN");

  backend = Backend::NPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "NPU");
}

TEST(LlmExecutorConfigTest, StringToBackend) {
  auto backend = GetBackendFromString("gpu_artisan");
  EXPECT_EQ(*backend, Backend::GPU_ARTISAN);
  backend = GetBackendFromString("gpu");
  EXPECT_EQ(*backend, Backend::GPU);
  backend = GetBackendFromString("cpu");
  EXPECT_EQ(*backend, Backend::CPU);
  backend = GetBackendFromString("google_tensor_artisan");
  EXPECT_EQ(*backend, Backend::GOOGLE_TENSOR_ARTISAN);
  backend = GetBackendFromString("npu");
  EXPECT_EQ(*backend, Backend::NPU);
  EXPECT_THAT(GetBackendFromString("cpu_artisan"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmExecutorConfigTest, IsCpuBackend) {
  EXPECT_TRUE(IsCpuBackend(Backend::CPU));
  EXPECT_FALSE(IsCpuBackend(Backend::GPU));
  EXPECT_FALSE(IsCpuBackend(Backend::GPU_ARTISAN));
  EXPECT_FALSE(IsCpuBackend(Backend::NPU));
  EXPECT_FALSE(IsCpuBackend(Backend::GOOGLE_TENSOR_ARTISAN));
  EXPECT_FALSE(IsCpuBackend(Backend::UNSPECIFIED));
  EXPECT_FALSE(IsCpuBackend(RegisterCustomBackend("is_cpu_backend_custom")));
}

TEST(LlmExecutorConfigTest, StringToActivationDataType) {
  auto activation_data_type = GetActivationDataTypeFromString("float32");
  EXPECT_EQ(*activation_data_type, ActivationDataType::FLOAT32);
  activation_data_type = GetActivationDataTypeFromString("float16");
  EXPECT_EQ(*activation_data_type, ActivationDataType::FLOAT16);
  activation_data_type = GetActivationDataTypeFromString("int16");
  EXPECT_EQ(*activation_data_type, ActivationDataType::INT16);
  activation_data_type = GetActivationDataTypeFromString("int8");
  EXPECT_EQ(*activation_data_type, ActivationDataType::INT8);
  activation_data_type = GetActivationDataTypeFromString("invalid");
  EXPECT_EQ(activation_data_type.status(),
            absl::InvalidArgumentError(
                "Unsupported activation data type: invalid. Supported "
                "activation data types are: [FLOAT32, "
                "FLOAT16, INT16, INT8]"));
}

TEST(LlmExecutorConfigTest, StringToUseIrModel) {
  ASSERT_OK_AND_ASSIGN(auto use_ir_model, GetUseIrModelFromString("enabled"));
  EXPECT_THAT(use_ir_model, Optional(true));
  ASSERT_OK_AND_ASSIGN(use_ir_model, GetUseIrModelFromString("ENABLED"));
  EXPECT_THAT(use_ir_model, Optional(true));
  ASSERT_OK_AND_ASSIGN(use_ir_model, GetUseIrModelFromString("disabled"));
  EXPECT_THAT(use_ir_model, Optional(false));
  ASSERT_OK_AND_ASSIGN(use_ir_model, GetUseIrModelFromString("default"));
  EXPECT_EQ(use_ir_model, std::nullopt);
  EXPECT_THAT(GetUseIrModelFromString("true"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetUseIrModelFromString(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmExecutorConfigTest, ActivatonDataType) {
  ActivationDataType act;
  std::stringstream oss;
  act = ActivationDataType::FLOAT32;
  oss << act;
  EXPECT_EQ(oss.str(), "FLOAT32");

  act = ActivationDataType::FLOAT16;
  oss.str("");
  oss << act;
  EXPECT_EQ(oss.str(), "FLOAT16");
}

TEST(LlmExecutorConfigTest, FakeWeightsMode) {
  FakeWeightsMode fake_weights_mode;
  std::stringstream oss;
  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_NONE;
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_NONE");

  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_8BITS_ALL_LAYERS;
  oss.str("");
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_8BITS_ALL_LAYERS");

  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4;
  oss.str("");
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4");
}

TEST(LlmExecutorConfigTest, FileFormat) {
  std::stringstream oss;

  oss.str("");
  oss << FileFormat::TFLITE;
  EXPECT_EQ(oss.str(), "TFLITE");

  oss.str("");
  oss << FileFormat::TASK;
  EXPECT_EQ(oss.str(), "TASK");

  oss.str("");
  oss << FileFormat::LITERT_LM;
  EXPECT_EQ(oss.str(), "LITERT_LM");
}

TEST(LlmExecutorConfigTest, NpuPerformanceMode) {
  std::stringstream oss;

  oss << NpuPerformanceMode::kVendorDefault;
  EXPECT_EQ(oss.str(), "VENDOR_DEFAULT");

  oss.str("");
  oss << NpuPerformanceMode::kExtremePowerSaver;
  EXPECT_EQ(oss.str(), "EXTREME_POWER_SAVER");

  oss.str("");
  oss << NpuPerformanceMode::kPowerSaver;
  EXPECT_EQ(oss.str(), "POWER_SAVER");

  oss.str("");
  oss << NpuPerformanceMode::kBalanced;
  EXPECT_EQ(oss.str(), "BALANCED");

  oss.str("");
  oss << NpuPerformanceMode::kHighPerformance;
  EXPECT_EQ(oss.str(), "HIGH_PERFORMANCE");

  oss.str("");
  oss << NpuPerformanceMode::kSustainedHighPerformance;
  EXPECT_EQ(oss.str(), "SUSTAINED_HIGH_PERFORMANCE");

  oss.str("");
  oss << NpuPerformanceMode::kBurst;
  EXPECT_EQ(oss.str(), "BURST");
}

TEST(LlmExecutorConfigTest, NpuPerformanceControl) {
  std::stringstream oss;
  oss << NpuPerformanceControl::kWhileLoaded;
  EXPECT_EQ(oss.str(), "WHILE_LOADED");

  oss.str("");
  oss << NpuPerformanceControl::kPerInference;
  EXPECT_EQ(oss.str(), "PER_INFERENCE");
}

TEST(LlmExecutorConfigTest, MediatekSdkVersion) {
  std::stringstream oss;
  oss << MediatekSdkVersion::kDefault;
  EXPECT_EQ(oss.str(), "DEFAULT");

  oss.str("");
  oss << MediatekSdkVersion::kVersion7;
  EXPECT_EQ(oss.str(), "VERSION_7");

  oss.str("");
  oss << MediatekSdkVersion::kVersion8;
  EXPECT_EQ(oss.str(), "VERSION_8");

  oss.str("");
  oss << MediatekSdkVersion::kVersion9;
  EXPECT_EQ(oss.str(), "VERSION_9");
}

TEST(LlmExecutorConfigTest, MediatekNpuOptions) {
  MediatekNpuOptions mediatek_options;
  EXPECT_EQ(mediatek_options.sdk_version, MediatekSdkVersion::kDefault);

  std::stringstream oss;
  mediatek_options.sdk_version = MediatekSdkVersion::kVersion9;
  oss << mediatek_options;
  EXPECT_EQ(oss.str(), "MediatekNpuOptions{sdk_version: VERSION_9}");
}

TEST(LlmExecutorConfigTest, NpuOptions) {
  NpuOptions npu_options;
  EXPECT_EQ(npu_options.performance_mode, NpuPerformanceMode::kBurst);
  EXPECT_EQ(npu_options.performance_control,
            NpuPerformanceControl::kWhileLoaded);
  EXPECT_EQ(npu_options.mediatek_options.sdk_version,
            MediatekSdkVersion::kDefault);

  std::stringstream oss;
  npu_options.performance_mode = NpuPerformanceMode::kPowerSaver;
  npu_options.performance_control = NpuPerformanceControl::kPerInference;
  npu_options.mediatek_options.sdk_version = MediatekSdkVersion::kVersion9;
  oss << npu_options;
  EXPECT_EQ(oss.str(),
            "NpuOptions{performance_mode: POWER_SAVER, performance_control: "
            "PER_INFERENCE, mediatek_options: MediatekNpuOptions{sdk_version: "
            "VERSION_9}}");
}

TEST(LlmExecutorConfigTest, ModelAssets) {
  auto model_assets = ModelAssets::Create("/path/to/model1");
  ASSERT_OK(model_assets);
  std::stringstream oss;
  oss << *model_assets;
  const std::string expected_output = R"(model_path: /path/to/model1
fake_weights_mode: FAKE_WEIGHTS_NONE
)";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, ModelAssetsMemoryMapped) {
  const std::string model_content = "some fake model content";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       InMemoryFile::Create(model_content));
  MemoryMappedFile* raw_ptr = memory_mapped_file.get();
  auto model_assets = ModelAssets::Create(std::move(memory_mapped_file));
  ASSERT_OK(model_assets);
  EXPECT_TRUE(model_assets->HasMemoryMappedFile());
  ASSERT_OK_AND_ASSIGN(auto retrieved_mmf, model_assets->GetMemoryMappedFile());
  EXPECT_EQ(retrieved_mmf.get(), raw_ptr);
  EXPECT_EQ(retrieved_mmf->length(), model_content.length());
  EXPECT_EQ(
      absl::string_view(reinterpret_cast<const char*>(retrieved_mmf->data()),
                        retrieved_mmf->length()),
      model_content);

  std::stringstream oss;
  oss << *model_assets;
  EXPECT_THAT(oss.str(), testing::HasSubstr("model_file memory mapped file"));
  EXPECT_THAT(oss.str(), testing::HasSubstr("FAKE_WEIGHTS_NONE"));

  // Without path, GetOrCreateScopedFile on MemoryMappedFile should fail.
  EXPECT_FALSE(model_assets->GetOrCreateScopedFile().ok());
}

TEST(LlmExecutorConfigTest, ModelAssetsMemoryMappedWithPath) {
  const auto model_path = GetTestModelPath();
  ASSERT_OK_AND_ASSIGN(auto memory_mapped_file,
                       MemoryMappedFile::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create(std::move(memory_mapped_file), model_path));
  EXPECT_TRUE(model_assets.HasMemoryMappedFile());
  EXPECT_OK(model_assets.GetPath());
  // With path, GetOrCreateScopedFile should succeed by opening the path.
  ASSERT_OK_AND_ASSIGN(auto scoped_file, model_assets.GetOrCreateScopedFile());
  ASSERT_NE(scoped_file, nullptr);
  EXPECT_TRUE(scoped_file->IsValid());
}

TEST(LlmExecutorConfigTest, ModelAssetsDataStream) {
  ASSERT_OK_AND_ASSIGN(auto data_stream,
                       FileDataStream::Create(GetTestModelPath()));

  auto model_assets = ModelAssets::Create(data_stream);
  ASSERT_OK(model_assets);
  EXPECT_TRUE(model_assets->HasDataStream());
  ASSERT_OK_AND_ASSIGN(auto retrieved_stream, model_assets->GetDataStream());
  EXPECT_EQ(retrieved_stream, data_stream);

  std::stringstream oss;
  oss << *model_assets;
  EXPECT_THAT(oss.str(),
              testing::HasSubstr("model_file is loading from a data stream"));
  EXPECT_THAT(oss.str(), testing::HasSubstr("FAKE_WEIGHTS_NONE"));
}

TEST(LlmExecutorConfigTest, GetCacheSuffixValidCpu) {
  auto result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::CPU, "/path/to/model.litertlm", "");
  ASSERT_OK(result);
  EXPECT_EQ(result->weight_suffix, ".xnnpack_cache");
  EXPECT_TRUE(result->program_suffix.empty());
  EXPECT_TRUE(result->gpu_weight_cache_suffix.empty());

  result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::CPU, "/path/to/model.litertlm", "vision_encoder");
  ASSERT_OK(result);
  EXPECT_EQ(result->weight_suffix, ".vision_encoder.xnnpack_cache");
  EXPECT_TRUE(result->program_suffix.empty());
  EXPECT_TRUE(result->gpu_weight_cache_suffix.empty());
}

TEST(LlmExecutorConfigTest, GetCacheSuffixValidGpu) {
  auto result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::GPU, "/path/to/model.litertlm", "");
  ASSERT_OK(result);
  EXPECT_TRUE(result->weight_suffix.empty());
  EXPECT_EQ(result->program_suffix, "_mldrift_program_cache.bin");
  EXPECT_EQ(result->gpu_weight_cache_suffix, "model.litertlm");

  result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::GPU, "/path/to/model.litertlm", "vision_encoder");
  ASSERT_OK(result);
  EXPECT_TRUE(result->weight_suffix.empty());
  EXPECT_EQ(result->program_suffix,
            ".mldrift_program_cache.vision_encoder.bin");
  EXPECT_EQ(result->gpu_weight_cache_suffix, "model.litertlm.vision_encoder");
}

TEST(LlmExecutorConfigTest, GetCacheSuffixInvalidBackend) {
  auto result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::NPU, "/path/to/model.litertlm", "");
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("Unsupported backend"));
}

TEST(LlmExecutorConfigTest, GetCacheSuffixInvalidModule) {
  auto result = ExecutorSettingsBase::GetCacheSuffix(
      Backend::CPU, "/path/to/model.litertlm", "invalid_module");
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("Invalid module name"));
}

class TestExecutorSettings : public ExecutorSettingsBase {
 public:
  explicit TestExecutorSettings(ModelAssets model_assets)
      : ExecutorSettingsBase(std::move(model_assets)) {}
};

constexpr absl::string_view kMtpDrafterName =
    ExecutorSettingsBase::GetCacheModelName(
        ExecutorSettingsBase::CacheModel::kMtpDrafter);

TEST(LlmExecutorConfigTest, GetWeightCacheFileWithNoCache) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir(":nocache");

  auto result = settings.GetWeightCacheFile();
  EXPECT_FALSE(result.ok());
}

TEST(LlmExecutorConfigTest, GetWeightCacheFileWithDisableWeightCache) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");
  settings.SetDisableWeightCache(true);

  auto result = settings.GetWeightCacheFile();
  EXPECT_FALSE(result.ok());
}

TEST(LlmExecutorConfigTest, GetWeightCacheFileWithScopedFileDoesNotError) {
  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(GetTestModelPath()));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(scoped_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));
  TestExecutorSettings settings(model_assets);
  settings.SetCacheDir("/cache/dir");
  settings.SetScopedCacheFile(model_file_ptr);

  auto result = settings.GetWeightCacheFile();
  EXPECT_TRUE(result.ok());
}

TEST(LlmExecutorConfigTest, GetProgramCacheFileWithNoCache) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir(":nocache");

  auto result = settings.GetProgramCacheFile();
  EXPECT_FALSE(result.ok());
}

TEST(LlmExecutorConfigTest, GetProgramCacheFileWithDisableProgramCache) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");
  settings.SetDisableProgramCache(true);

  auto result = settings.GetProgramCacheFile();
  EXPECT_FALSE(result.ok());
}

TEST(LlmExecutorConfigTest, GetProgramCacheFileWithScopedFileDoesNotError) {
  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(GetTestModelPath()));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(scoped_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_file_ptr));
  TestExecutorSettings settings(model_assets);
  settings.SetCacheDir("/cache/dir");
  settings.SetScopedProgramCacheFile(model_file_ptr);

  auto result = settings.GetProgramCacheFile();
  EXPECT_TRUE(result.ok());
}

TEST(LlmExecutorConfigTest, ScopedCacheFileIsOnlyUsedForMainModelSuffixes) {
  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(GetTestModelPath()));
  auto cache_file = std::make_shared<ScopedFile>(std::move(scoped_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(cache_file));
  TestExecutorSettings settings(model_assets);
  settings.SetScopedCacheFile(cache_file);

  for (absl::string_view suffix :
       {ExecutorSettingsBase::kDefaultWeightCacheSuffix,
        ExecutorSettingsBase::kXnnpackCacheSuffix,
        ExecutorSettingsBase::kMlDriftWeightCacheSuffix}) {
    SCOPED_TRACE(suffix);
    EXPECT_THAT(
        settings.GetWeightCacheFile(suffix, /*check_and_clean=*/true),
        IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(cache_file)));
  }
  // The MTP drafter must not write into the main model's cache file.
  for (absl::string_view backend_suffix :
       {ExecutorSettingsBase::kXnnpackCacheSuffix,
        ExecutorSettingsBase::kMlDriftWeightCacheSuffix}) {
    std::string suffix = absl::StrCat(kMtpDrafterName, backend_suffix);
    SCOPED_TRACE(suffix);
    EXPECT_THAT(settings.GetWeightCacheFile(suffix, /*check_and_clean=*/true),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST(LlmExecutorConfigTest,
     ScopedProgramCacheFileIsOnlyUsedForMainModelSuffixes) {
  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(GetTestModelPath()));
  auto cache_file = std::make_shared<ScopedFile>(std::move(scoped_file));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(cache_file));
  TestExecutorSettings settings(model_assets);
  settings.SetScopedProgramCacheFile(cache_file);

  for (absl::string_view suffix :
       {ExecutorSettingsBase::kDefaultProgramCacheSuffix,
        ExecutorSettingsBase::kMlDriftCacheSuffix}) {
    SCOPED_TRACE(suffix);
    EXPECT_THAT(
        settings.GetProgramCacheFile(suffix, /*check_and_clean=*/true),
        IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(cache_file)));
  }
  // The MTP drafter must not write into the main model's cache file.
  std::string drafter_suffix =
      absl::StrCat(kMtpDrafterName, ExecutorSettingsBase::kMlDriftCacheSuffix);
  EXPECT_THAT(
      settings.GetProgramCacheFile(drafter_suffix, /*check_and_clean=*/true),
      StatusIs(absl::StatusCode::kNotFound));
}

using CacheFileKind = ExecutorSettingsBase::CacheFileKind;
using CacheModel = ExecutorSettingsBase::CacheModel;

// Hands out one cache file per model and records each request it gets.
class FakeCacheFileProvider {
 public:
  struct Request {
    CacheFileKind kind;
    CacheModel model;
    std::string suffix;
    bool operator==(const Request&) const = default;
  };

  // Gives `model` the file `file`. Other models get NotFound.
  void AddFile(CacheModel model, std::shared_ptr<ScopedFile> file) {
    files_[model] = std::move(file);
  }

  ExecutorSettingsBase::CacheFileProvider AsProvider() {
    return [this](const ExecutorSettingsBase::CacheFileRequest& request)
               -> absl::StatusOr<std::shared_ptr<ScopedFile>> {
      requests_.push_back(
          {request.kind, request.model, std::string(request.suffix)});
      auto it = files_.find(request.model);
      if (it == files_.end()) {
        return absl::NotFoundError("No file for this model.");
      }
      return it->second;
    };
  }

  const std::vector<Request>& requests() const { return requests_; }

 private:
  absl::flat_hash_map<CacheModel, std::shared_ptr<ScopedFile>> files_;
  std::vector<Request> requests_;
};

std::shared_ptr<ScopedFile> OpenTestFile() {
  absl::StatusOr<ScopedFile> file = ScopedFile::Open(GetTestModelPath());
  if (!file.ok()) {
    return nullptr;
  }
  return std::make_shared<ScopedFile>(*std::move(file));
}

TEST(LlmExecutorConfigTest, CacheModelNamesMatchCacheFileNames) {
  // Cache file names are built from these, so changing one would orphan
  // existing caches.
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kMain), "");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kMtpDrafter),
            ".mtp_drafter");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kVisionEncoder),
            ".vision_encoder");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kVisionAdapter),
            ".vision_adapter");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kAudioEncoder),
            ".audio_encoder");
  EXPECT_EQ(
      ExecutorSettingsBase::GetCacheModelName(CacheModel::kStaticAudioEncoder),
      ".static_audio_encoder");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(
                CacheModel::kStreamingAudioEncoder),
            ".streaming_audio_encoder");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kAudioAdapter),
            ".audio_adapter");
  EXPECT_EQ(ExecutorSettingsBase::GetCacheModelName(CacheModel::kTextEncoder),
            ".text_encoder");
}

TEST(LlmExecutorConfigTest, ScopedCacheFilesArePerModel) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto main_file = OpenTestFile();
  auto drafter_file = OpenTestFile();
  ASSERT_TRUE(main_file && drafter_file);
  settings.SetScopedCacheFile(main_file);
  settings.SetScopedCacheFile(CacheModel::kMtpDrafter, CacheFileKind::kWeight,
                              drafter_file);

  EXPECT_EQ(settings.GetScopedCacheFile(), main_file);
  EXPECT_EQ(
      settings.GetScopedCacheFile(CacheModel::kMain, CacheFileKind::kWeight),
      main_file);
  EXPECT_EQ(settings.GetScopedCacheFile(CacheModel::kMtpDrafter,
                                        CacheFileKind::kWeight),
            drafter_file);
  EXPECT_EQ(settings.GetScopedCacheFile(CacheModel::kMtpDrafter,
                                        CacheFileKind::kProgram),
            nullptr);
  EXPECT_THAT(
      settings.GetWeightCacheFile(ExecutorSettingsBase::kXnnpackCacheSuffix,
                                  /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(main_file)));
  EXPECT_THAT(
      settings.GetWeightCacheFile(
          absl::StrCat(kMtpDrafterName,
                       ExecutorSettingsBase::kXnnpackCacheSuffix),
          /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(drafter_file)));

  settings.SetScopedCacheFile(CacheModel::kMtpDrafter, CacheFileKind::kWeight,
                              nullptr);
  EXPECT_EQ(settings.GetScopedCacheFile(CacheModel::kMtpDrafter,
                                        CacheFileKind::kWeight),
            nullptr);
}

TEST(LlmExecutorConfigTest, SubModelScopedCacheFileIgnoresDisabledCache) {
  // Sub-model scoped files have always been used even when the cache is
  // disabled, unlike the main model's.
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto main_file = OpenTestFile();
  auto encoder_file = OpenTestFile();
  ASSERT_TRUE(main_file && encoder_file);
  settings.SetScopedCacheFile(main_file);
  settings.SetScopedCacheFile(CacheModel::kVisionEncoder,
                              CacheFileKind::kWeight, encoder_file);
  settings.SetDisableWeightCache(true);

  EXPECT_THAT(settings.GetWeightCacheFile(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      settings.GetWeightCacheFile(".vision_encoder.xnnpack_cache"),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(encoder_file)));
}

TEST(LlmExecutorConfigTest, CacheFileProviderGivesEachModelItsOwnFile) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto main_weights = OpenTestFile();
  auto drafter_weights = OpenTestFile();
  auto main_programs = OpenTestFile();
  auto drafter_programs = OpenTestFile();
  ASSERT_TRUE(main_weights && drafter_weights && main_programs &&
              drafter_programs);
  FakeCacheFileProvider weight_files;
  weight_files.AddFile(CacheModel::kMain, main_weights);
  weight_files.AddFile(CacheModel::kMtpDrafter, drafter_weights);
  FakeCacheFileProvider program_files;
  program_files.AddFile(CacheModel::kMain, main_programs);
  program_files.AddFile(CacheModel::kMtpDrafter, drafter_programs);
  settings.SetCacheFileProvider(
      [weights = weight_files.AsProvider(),
       programs = program_files.AsProvider()](
          const ExecutorSettingsBase::CacheFileRequest& request) {
        return request.kind == CacheFileKind::kWeight ? weights(request)
                                                      : programs(request);
      });

  const std::string drafter_weight_suffix = absl::StrCat(
      kMtpDrafterName, ExecutorSettingsBase::kMlDriftWeightCacheSuffix);
  const std::string drafter_program_suffix =
      absl::StrCat(kMtpDrafterName, ExecutorSettingsBase::kMlDriftCacheSuffix);
  EXPECT_THAT(
      settings.GetWeightCacheFile(
          ExecutorSettingsBase::kMlDriftWeightCacheSuffix,
          /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(main_weights)));
  EXPECT_THAT(
      settings.GetWeightCacheFile(drafter_weight_suffix,
                                  /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(drafter_weights)));
  EXPECT_THAT(
      settings.GetProgramCacheFile(ExecutorSettingsBase::kMlDriftCacheSuffix,
                                   /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(main_programs)));
  EXPECT_THAT(
      settings.GetProgramCacheFile(drafter_program_suffix,
                                   /*check_and_clean=*/true),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(drafter_programs)));

  using Request = FakeCacheFileProvider::Request;
  EXPECT_THAT(
      weight_files.requests(),
      ElementsAre(
          Request{CacheFileKind::kWeight, CacheModel::kMain,
                  std::string(ExecutorSettingsBase::kMlDriftWeightCacheSuffix)},
          Request{CacheFileKind::kWeight, CacheModel::kMtpDrafter,
                  drafter_weight_suffix}));
  EXPECT_THAT(
      program_files.requests(),
      ElementsAre(
          Request{CacheFileKind::kProgram, CacheModel::kMain,
                  std::string(ExecutorSettingsBase::kMlDriftCacheSuffix)},
          Request{CacheFileKind::kProgram, CacheModel::kMtpDrafter,
                  drafter_program_suffix}));
}

TEST(LlmExecutorConfigTest, CacheFileProviderGetsModelFromSuffix) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  FakeCacheFileProvider provider;
  settings.SetCacheFileProvider(provider.AsProvider());

  for (absl::string_view suffix :
       {".cache", ".xnnpack_cache", "_mldrift_weight_cache.bin",
        ".mtp_drafter.xnnpack_cache",
        ".vision_encoder_mldrift_weight_cache.bin",
        ".vision_adapter.xnnpack_cache", ".audio_encoder.xnnpack_cache",
        ".static_audio_encoder.xnnpack_cache",
        ".streaming_audio_encoder.xnnpack_cache", ".text_encoder"}) {
    settings.GetWeightCacheFile(suffix).IgnoreError();
  }
  for (absl::string_view suffix :
       {".program_cache", "_mldrift_program_cache.bin",
        ".audio_adapter_mldrift_program_cache.bin"}) {
    settings.GetProgramCacheFile(suffix).IgnoreError();
  }

  std::vector<CacheModel> models;
  for (const auto& request : provider.requests()) {
    models.push_back(request.model);
  }
  EXPECT_THAT(
      models,
      ElementsAre(CacheModel::kMain, CacheModel::kMain, CacheModel::kMain,
                  CacheModel::kMtpDrafter, CacheModel::kVisionEncoder,
                  CacheModel::kVisionAdapter, CacheModel::kAudioEncoder,
                  CacheModel::kStaticAudioEncoder,
                  CacheModel::kStreamingAudioEncoder, CacheModel::kTextEncoder,
                  CacheModel::kMain, CacheModel::kMain,
                  CacheModel::kAudioAdapter));
}

TEST(LlmExecutorConfigTest, AudioEncoderScopedFileServesStaticAndStreaming) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto encoder_file = OpenTestFile();
  auto static_encoder_file = OpenTestFile();
  ASSERT_TRUE(encoder_file && static_encoder_file);
  settings.SetScopedCacheFile(CacheModel::kAudioEncoder, CacheFileKind::kWeight,
                              encoder_file);

  EXPECT_THAT(
      settings.GetWeightCacheFile(".static_audio_encoder.xnnpack_cache"),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(encoder_file)));
  EXPECT_THAT(
      settings.GetWeightCacheFile(".streaming_audio_encoder.xnnpack_cache"),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(encoder_file)));

  // A file set for the specific encoder takes precedence.
  settings.SetScopedCacheFile(CacheModel::kStaticAudioEncoder,
                              CacheFileKind::kWeight, static_encoder_file);
  EXPECT_THAT(
      settings.GetWeightCacheFile(".static_audio_encoder.xnnpack_cache"),
      IsOkAndHolds(
          VariantWith<std::shared_ptr<ScopedFile>>(static_encoder_file)));
}

TEST(LlmExecutorConfigTest, UnknownSuffixGetsNoHostFile) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  FakeCacheFileProvider provider;
  settings.SetCacheFileProvider(provider.AsProvider());

  EXPECT_THAT(settings.GetWeightCacheFile(".custom.xnnpack_cache"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(provider.requests(), IsEmpty());
}

TEST(LlmExecutorConfigTest, CacheFileProviderNotFoundDoesNotFallBackToPath) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  settings.SetCacheDir("/cache/dir");
  FakeCacheFileProvider provider;
  settings.SetCacheFileProvider(provider.AsProvider());

  EXPECT_THAT(settings.GetWeightCacheFile(".xnnpack_cache"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(settings.GetProgramCacheFile("_mldrift_program_cache.bin"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmExecutorConfigTest, CacheFileProviderNullFileIsAnError) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  settings.SetCacheFileProvider(
      [](const ExecutorSettingsBase::CacheFileRequest&)
          -> absl::StatusOr<std::shared_ptr<ScopedFile>> { return nullptr; });

  EXPECT_THAT(settings.GetWeightCacheFile(),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(LlmExecutorConfigTest, ScopedFileIsPreferredOverCacheFileProvider) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto scoped_file = OpenTestFile();
  auto provided_main_file = OpenTestFile();
  auto provided_drafter_file = OpenTestFile();
  ASSERT_TRUE(scoped_file && provided_main_file && provided_drafter_file);
  settings.SetScopedCacheFile(scoped_file);
  FakeCacheFileProvider provider;
  provider.AddFile(CacheModel::kMain, provided_main_file);
  provider.AddFile(CacheModel::kMtpDrafter, provided_drafter_file);
  settings.SetCacheFileProvider(provider.AsProvider());

  EXPECT_THAT(
      settings.GetWeightCacheFile(),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(scoped_file)));
  // The provider covers the models with no scoped file.
  EXPECT_THAT(settings.GetWeightCacheFile(".mtp_drafter.xnnpack_cache"),
              IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(
                  provided_drafter_file)));
}

TEST(LlmExecutorConfigTest, CacheFileProviderIsNotAskedWhenCacheIsDisabled) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  FakeCacheFileProvider provider;
  settings.SetCacheFileProvider(provider.AsProvider());
  settings.SetDisableWeightCache(true);
  settings.SetDisableProgramCache(true);

  EXPECT_THAT(settings.GetWeightCacheFile(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(settings.GetProgramCacheFile(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(provider.requests(), IsEmpty());
}

TEST(LlmExecutorConfigTest, CacheFilesAreCopiedWithSettings) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/path/to/model.tflite"));
  TestExecutorSettings settings(model_assets);
  auto main_file = OpenTestFile();
  auto drafter_file = OpenTestFile();
  ASSERT_TRUE(main_file && drafter_file);
  settings.SetScopedCacheFile(main_file);
  FakeCacheFileProvider provider;
  provider.AddFile(CacheModel::kMtpDrafter, drafter_file);
  settings.SetCacheFileProvider(provider.AsProvider());

  TestExecutorSettings copy = settings;
  EXPECT_TRUE(copy.HasCacheFileProvider());
  EXPECT_EQ(copy.GetScopedCacheFile(), main_file);
  EXPECT_THAT(
      copy.GetWeightCacheFile(".mtp_drafter.xnnpack_cache"),
      IsOkAndHolds(VariantWith<std::shared_ptr<ScopedFile>>(drafter_file)));
}

TEST(LlmExecutorConfigTest, GetProgramCacheFile) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetProgramCacheFile();
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  EXPECT_THAT(std::get<std::string>(*result),
              testing::HasSubstr("model.tflite.program_cache"));
}

TEST(LlmExecutorConfigTest, GetProgramCacheFileWithSuffix) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetProgramCacheFile(".mysuffix");
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  EXPECT_THAT(std::get<std::string>(*result),
              testing::HasSubstr("model.tflite.mysuffix"));
}

TEST(LlmExecutorConfigTest, GetProgramCacheFileWithIdentifier) {
  ASSERT_OK_AND_ASSIGN(std::string temp_file,
                       JoinPath(testing::TempDir(), "test_model.tflite"));
  std::ofstream ofs(temp_file);
  ofs << "test data";
  ofs.close();

  auto model_assets = ModelAssets::Create(temp_file);
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetProgramCacheFile();
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  std::string path = std::get<std::string>(*result);

  EXPECT_THAT(path, testing::HasSubstr("test_model.tflite_"));
  EXPECT_THAT(path, testing::EndsWith("_9.program_cache"));
}

TEST(LlmExecutorConfigTest,
     GetProgramCacheFileCheckAndClean_DeletesStaleCaches) {
  ASSERT_OK_AND_ASSIGN(
      std::string model_file,
      JoinPath(testing::TempDir(), "stale_test_model_prog.tflite"));
  std::ofstream ofs(model_file);
  ofs << "test data";
  ofs.close();

  ASSERT_OK_AND_ASSIGN(std::string cache_dir,
                       JoinPath(testing::TempDir(), "cache_dir_prog"));
  std::error_code ec;
  std::filesystem::create_directory(cache_dir, ec);

  ASSERT_OK_AND_ASSIGN(
      std::string stale_cache,
      JoinPath(cache_dir, "stale_test_model_prog.tflite_oldid.program_cache"));
  std::ofstream ofs_stale(stale_cache);
  ofs_stale << "stale test data";
  ofs_stale.close();

  EXPECT_TRUE(std::filesystem::exists(stale_cache));

  auto model_assets = ModelAssets::Create(model_file);
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir(cache_dir);

  // check_and_clean = true
  auto result = settings.GetProgramCacheFile(".program_cache", true);
  ASSERT_OK(result);

  // Stale cache should be deleted.
  EXPECT_FALSE(std::filesystem::exists(stale_cache));
}

TEST(LlmExecutorConfigTest, GetWeightCacheFile) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetWeightCacheFile();
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  EXPECT_THAT(std::get<std::string>(*result),
              testing::HasSubstr("model.tflite.cache"));
}

TEST(LlmExecutorConfigTest, GetWeightCacheFileWithSuffix) {
  auto model_assets = ModelAssets::Create("/path/to/model.tflite");
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetWeightCacheFile(".mysuffix");
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  EXPECT_THAT(std::get<std::string>(*result),
              testing::HasSubstr("model.tflite.mysuffix"));
}

TEST(LlmExecutorConfigTest, GetWeightCacheFileWithIdentifier) {
  ASSERT_OK_AND_ASSIGN(
      std::string temp_file,
      JoinPath(testing::TempDir(), "test_model_weight.tflite"));
  std::ofstream ofs(temp_file);
  ofs << "test data";
  ofs.close();

  auto model_assets = ModelAssets::Create(temp_file);
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir("/cache/dir");

  auto result = settings.GetWeightCacheFile();
  ASSERT_OK(result);
  EXPECT_TRUE(std::holds_alternative<std::string>(*result));
  std::string path = std::get<std::string>(*result);

  EXPECT_THAT(path, testing::HasSubstr("test_model_weight.tflite_"));
  EXPECT_THAT(path, testing::EndsWith("_9.cache"));
}

TEST(LlmExecutorConfigTest,
     GetWeightCacheFileCheckAndClean_DeletesStaleCaches) {
  ASSERT_OK_AND_ASSIGN(
      std::string model_file,
      JoinPath(testing::TempDir(), "stale_test_model_weight.tflite"));
  std::ofstream ofs(model_file);
  ofs << "test data";
  ofs.close();

  ASSERT_OK_AND_ASSIGN(std::string cache_dir,
                       JoinPath(testing::TempDir(), "cache_dir_weight"));
  std::error_code ec;
  std::filesystem::create_directory(cache_dir, ec);

  ASSERT_OK_AND_ASSIGN(
      std::string stale_cache,
      JoinPath(cache_dir, "stale_test_model_weight.tflite_oldid.cache"));
  std::ofstream ofs_stale(stale_cache);
  ofs_stale << "stale test data";
  ofs_stale.close();

  EXPECT_TRUE(std::filesystem::exists(stale_cache));

  auto model_assets = ModelAssets::Create(model_file);
  ASSERT_OK(model_assets);
  TestExecutorSettings settings(*model_assets);
  settings.SetCacheDir(cache_dir);

  // check_and_clean = true
  auto result = settings.GetWeightCacheFile(".cache", true);
  ASSERT_OK(result);

  // Stale cache should be deleted.
  EXPECT_FALSE(std::filesystem::exists(stale_cache));
}

}  // namespace
}  // namespace litert::lm
