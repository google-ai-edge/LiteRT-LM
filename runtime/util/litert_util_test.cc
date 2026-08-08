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

#include "runtime/util/litert_util.h"

#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

class FakeModelResources : public ModelResources {
 public:
  FakeModelResources() = default;
  ~FakeModelResources() override = default;

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    return nullptr;
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }
};

// Loads a real TFLite model from testdata. `magic_test_context_length.tflite`
// embeds a magic number for the context length, which triggers magic-number
// configuration in CreateEnvironment.
absl::StatusOr<Model> LoadModelFromFile(absl::string_view model_path) {
  auto model_path_in_srcdir =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/" /
      std::string(model_path);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(model_path_in_srcdir.string()));
  return model;
}

// Returns the path to the shared test .task model used to build EngineSettings.
std::string TestTaskPath() {
  return (std::filesystem::path(::testing::SrcDir()) /
          "google3/runtime/testdata/"
          "test_lm_new_metadata.task")
      .string();
}

// Fake ModelResources that returns a caller-provided model for every model type
// and lets the test control the TF_LITE_AUX buffer lookup. Whether that buffer
// is missing/empty is exactly what CreateEnvironment uses to decide whether an
// NPU model runs through the generic LiteRT compiler-plugin path.
class FakeNpuModelResources : public ModelResources {
 public:
  // If `aux_buffer` is nullopt, the aux model is reported as missing, which
  // selects the generic compiler-plugin path. Otherwise the provided bytes are
  // returned, which selects the specialized TF_LITE_AUX NPU executor path.
  FakeNpuModelResources(const Model& model,
                        std::optional<std::string> aux_buffer)
      : model_(model), aux_buffer_(std::move(aux_buffer)) {}
  ~FakeNpuModelResources() override = default;

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    return &model_;
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLiteAux) {
      if (!aux_buffer_.has_value()) {
        return absl::NotFoundError("No aux model buffer.");
      }
      return absl::string_view(*aux_buffer_);
    }
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  const Model& model_;
  std::optional<std::string> aux_buffer_;
};

TEST(LiteRtUtilTest, CreatetEnvironment_CPUGPUFirst_ExcludesNPUOptions) {
  std::string task_path =
      (std::filesystem::path(::testing::SrcDir()) /
       "google3/runtime/testdata/"
       "test_lm_new_metadata.task")
          .string();
  std::string expected_path =
      std::filesystem::path(task_path).parent_path().string();

  auto model_assets = ModelAssets::Create(task_path);
  ASSERT_OK(model_assets);

  auto cpu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(cpu_settings);
  cpu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  FakeModelResources fake_resources;
  auto env_status = CreateEnvironment(*cpu_settings, &fake_resources);
  ASSERT_OK(env_status);
  auto& env = env_status->env;

  auto options_status = env.GetOptions();
  ASSERT_TRUE(options_status.HasValue());
  const auto& options = *options_status;

  auto dispatch_lib_status =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  ASSERT_FALSE(dispatch_lib_status.HasValue())
      << "kDispatchLibraryDir was initialized on the Environment singleton "
         "when CPU was used first.";

  auto compiler_plugin_lib_status =
      options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
  ASSERT_FALSE(compiler_plugin_lib_status.HasValue())
      << "kCompilerPluginLibraryDir was initialized on the Environment "
         "singleton when CPU was used first.";

  // Initialize NPU second in the same process session.
  auto npu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  auto npu_env_status = CreateEnvironment(*npu_settings, nullptr);
  ASSERT_OK(npu_env_status);
  auto& npu_env = npu_env_status->env;

  auto npu_options_status = npu_env.GetOptions();
  ASSERT_TRUE(npu_options_status.HasValue());
  const auto& npu_options = *npu_options_status;

  auto npu_dispatch_lib_status =
      npu_options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  auto npu_compiler_plugin_lib_status =
      npu_options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
#if !defined(LITERT_DISABLE_NPU)
  ASSERT_TRUE(npu_dispatch_lib_status.HasValue())
      << "kDispatchLibraryDir was not initialized when NPU was used second.";
  ASSERT_TRUE(npu_compiler_plugin_lib_status.HasValue())
      << "kCompilerPluginLibraryDir was not initialized when NPU was used "
         "second.";

  auto npu_dispatch_lib_dir = std::get<const char*>(*npu_dispatch_lib_status);
  EXPECT_EQ(std::string(npu_dispatch_lib_dir), expected_path);
  auto npu_compiler_plugin_lib_dir =
      std::get<const char*>(*npu_compiler_plugin_lib_status);
  EXPECT_EQ(std::string(npu_compiler_plugin_lib_dir), expected_path);
#else
  ASSERT_FALSE(npu_dispatch_lib_status.HasValue());
  ASSERT_FALSE(npu_compiler_plugin_lib_status.HasValue());
#endif
}

TEST(LiteRtUtilTest, GetEnvironment_NPUFirst_IncludesNPUOptions) {
  std::string task_path =
      (std::filesystem::path(::testing::SrcDir()) /
       "google3/runtime/testdata/"
       "test_lm_new_metadata.task")
          .string();
  auto model_assets = ModelAssets::Create(task_path);
  ASSERT_OK(model_assets);

  auto npu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  auto env_status = CreateEnvironment(*npu_settings, nullptr);
  ASSERT_OK(env_status);
  auto& env = env_status->env;

  auto options_status = env.GetOptions();
  ASSERT_TRUE(options_status.HasValue());
  const auto& options = *options_status;

  auto dispatch_lib_status =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
#if !defined(LITERT_DISABLE_NPU)
  ASSERT_TRUE(dispatch_lib_status.HasValue())
      << "kDispatchLibraryDir was not initialized when NPU was used first.";

  auto dispatch_lib_dir = std::get<const char*>(*dispatch_lib_status);
  std::string expected_path =
      std::filesystem::path(task_path).parent_path().string();
  EXPECT_EQ(std::string(dispatch_lib_dir), expected_path);

  auto compiler_plugin_lib_status =
      options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
  ASSERT_TRUE(compiler_plugin_lib_status.HasValue())
      << "kCompilerPluginLibraryDir was not initialized when NPU was used "
         "first.";

  auto compiler_plugin_lib_dir =
      std::get<const char*>(*compiler_plugin_lib_status);
  EXPECT_EQ(std::string(compiler_plugin_lib_dir), expected_path);
#else
  ASSERT_FALSE(dispatch_lib_status.HasValue());
#endif
}

TEST(LiteRtUtilTest,
     CreateEnvironment_NpuGenericCompilerPlugin_AppliesMagicNumbers) {
  // This model embeds a magic number for the context length.
  auto model = LoadModelFromFile("magic_test_context_length.tflite");
  ASSERT_OK(model);

  auto model_assets = ModelAssets::Create(TestTaskPath());
  ASSERT_OK(model_assets);
  auto npu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  // No aux model buffer => the NPU model takes the generic compiler-plugin
  // path, so magic-number configuration must be applied (as it is for CPU and
  // GPU).
  FakeNpuModelResources resources(*model, /*aux_buffer=*/std::nullopt);
  auto env_status = CreateEnvironment(*npu_settings, &resources);
  ASSERT_OK(env_status);

  auto options_status = env_status->env.GetOptions();
  ASSERT_TRUE(options_status.HasValue());
  auto magic_number_status =
      options_status->GetOption(EnvironmentOptions::Tag::kMagicNumberConfigs);
  EXPECT_TRUE(magic_number_status.HasValue())
      << "Magic-number configs should be applied for an NPU model that uses "
         "the generic LiteRT compiler-plugin path.";
}

TEST(LiteRtUtilTest, CreateEnvironment_NpuWithAuxExecutor_SkipsMagicNumbers) {
  auto model = LoadModelFromFile("magic_test_context_length.tflite");
  ASSERT_OK(model);

  auto model_assets = ModelAssets::Create(TestTaskPath());
  ASSERT_OK(model_assets);
  auto npu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  // A non-empty aux buffer => specialized TF_LITE_AUX NPU executor, which owns
  // magic-number handling itself, so CreateEnvironment must not add configs.
  FakeNpuModelResources resources(*model, /*aux_buffer=*/std::string("aux"));
  auto env_status = CreateEnvironment(*npu_settings, &resources);
  ASSERT_OK(env_status);

  auto options_status = env_status->env.GetOptions();
  ASSERT_TRUE(options_status.HasValue());
  auto magic_number_status =
      options_status->GetOption(EnvironmentOptions::Tag::kMagicNumberConfigs);
  EXPECT_FALSE(magic_number_status.HasValue())
      << "Magic-number configs must be skipped when the model ships a "
         "specialized TF_LITE_AUX NPU executor.";
}

TEST(
    LiteRtUtilTest,
    CreateEnvironment_NpuGenericCompilerPlugin_RespectsConfigureMagicNumbersFalse) {  // NOLINT(whitespace/line_length)
  auto model = LoadModelFromFile("magic_test_context_length.tflite");
  ASSERT_OK(model);

  auto model_assets = ModelAssets::Create(TestTaskPath());
  ASSERT_OK(model_assets);
  auto npu_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  AdvancedSettings advanced_settings;
  advanced_settings.configure_magic_numbers = false;
  npu_settings->GetMutableMainExecutorSettings().SetAdvancedSettings(
      advanced_settings);

  FakeNpuModelResources resources(*model, /*aux_buffer=*/std::nullopt);
  auto env_status = CreateEnvironment(*npu_settings, &resources);
  ASSERT_OK(env_status);

  auto options_status = env_status->env.GetOptions();
  ASSERT_TRUE(options_status.HasValue());
  auto magic_number_status =
      options_status->GetOption(EnvironmentOptions::Tag::kMagicNumberConfigs);
  EXPECT_FALSE(magic_number_status.HasValue())
      << "configure_magic_numbers=false must disable magic-number configs even "
         "on the generic compiler-plugin path.";
}

TEST(LiteRtUtilTest,
     CreateEnvironment_EmbeddingEngineSettings_Npu_IncludesNPUOptions) {
  std::string task_path = TestTaskPath();
  auto model_assets = ModelAssets::Create(task_path);
  ASSERT_OK(model_assets);

  auto npu_settings =
      EmbeddingEngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  auto owned_env = CreateEnvironment(*npu_settings, nullptr);
  ASSERT_OK(owned_env);
  auto& env = owned_env->env;

  auto expected_options = env.GetOptions();
  ASSERT_TRUE(expected_options.HasValue());
  const auto& options = *expected_options;

  auto dispatch_lib =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
#if !defined(LITERT_DISABLE_NPU)
  ASSERT_TRUE(dispatch_lib.HasValue());

  auto dispatch_lib_dir = std::get<const char*>(*dispatch_lib);
  std::string expected_path =
      std::filesystem::path(task_path).parent_path().string();
  EXPECT_EQ(std::string(dispatch_lib_dir), expected_path);

  auto compiler_plugin_lib =
      options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
  ASSERT_TRUE(compiler_plugin_lib.HasValue());

  auto compiler_plugin_lib_dir = std::get<const char*>(*compiler_plugin_lib);
  EXPECT_EQ(std::string(compiler_plugin_lib_dir), expected_path);
#else
  ASSERT_FALSE(dispatch_lib.HasValue());
#endif
}

TEST(LiteRtUtilTest,
     CreateEnvironment_EmbeddingEngineSettings_CPUGPUFirst_ExcludesNPUOptions) {
  std::string task_path = TestTaskPath();
  std::string expected_path =
      std::filesystem::path(task_path).parent_path().string();

  auto model_assets = ModelAssets::Create(task_path);
  ASSERT_OK(model_assets);

  auto cpu_settings =
      EmbeddingEngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(cpu_settings);
  cpu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  FakeModelResources fake_resources;
  auto owned_env = CreateEnvironment(*cpu_settings, &fake_resources);
  ASSERT_OK(owned_env);
  auto& env = owned_env->env;

  auto expected_options = env.GetOptions();
  ASSERT_TRUE(expected_options.HasValue());
  const auto& options = *expected_options;

  auto dispatch_lib =
      options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  ASSERT_FALSE(dispatch_lib.HasValue());

  auto compiler_plugin_lib =
      options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
  ASSERT_FALSE(compiler_plugin_lib.HasValue());

  // Initialize NPU second in the same process session.
  auto npu_settings =
      EmbeddingEngineSettings::CreateDefault(*model_assets, Backend::NPU);
  ASSERT_OK(npu_settings);
  npu_settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir("");

  auto npu_owned_env = CreateEnvironment(*npu_settings, nullptr);
  ASSERT_OK(npu_owned_env);
  auto& npu_env = npu_owned_env->env;

  auto npu_expected_options = npu_env.GetOptions();
  ASSERT_TRUE(npu_expected_options.HasValue());
  const auto& npu_options = *npu_expected_options;

  auto npu_dispatch_lib =
      npu_options.GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  auto npu_compiler_plugin_lib =
      npu_options.GetOption(EnvironmentOptions::Tag::kCompilerPluginLibraryDir);
#if !defined(LITERT_DISABLE_NPU)
  ASSERT_TRUE(npu_dispatch_lib.HasValue());
  ASSERT_TRUE(npu_compiler_plugin_lib.HasValue());

  auto npu_dispatch_lib_dir = std::get<const char*>(*npu_dispatch_lib);
  EXPECT_EQ(std::string(npu_dispatch_lib_dir), expected_path);
  auto npu_compiler_plugin_lib_dir =
      std::get<const char*>(*npu_compiler_plugin_lib);
  EXPECT_EQ(std::string(npu_compiler_plugin_lib_dir), expected_path);
#else
  ASSERT_FALSE(npu_dispatch_lib.HasValue());
  ASSERT_FALSE(npu_compiler_plugin_lib.HasValue());
#endif
}

}  // namespace
}  // namespace litert::lm
