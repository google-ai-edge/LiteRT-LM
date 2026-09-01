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

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl  // IWYU pragma: keep
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/util/logging.h"

namespace litert::lm {

namespace {

bool UsesNpu(Backend backend,
             const std::optional<VisionExecutorSettings>& vision_settings,
             const std::optional<AudioExecutorSettings>& audio_settings) {
  return (backend == Backend::NPU ||
          (vision_settings.has_value() &&
           vision_settings->GetBackend() == Backend::NPU) ||
          (audio_settings.has_value() &&
           audio_settings->GetBackend() == Backend::NPU));
}

struct LibDirResult {
  std::string library_dir;
  bool should_set_path = false;
  bool from_settings = false;
};

LibDirResult ResolveLibraryDir(
    const ExecutorSettingsBase& main_executor_settings) {
  std::string library_dir;
  bool from_settings = false;
  if (!main_executor_settings.GetLitertDispatchLibDir().empty()) {
    library_dir = main_executor_settings.GetLitertDispatchLibDir();
    from_settings = true;
  } else {
    std::string model_path(
        main_executor_settings.GetModelAssets().GetPath().value_or(""));
    std::filesystem::path path(model_path);
    library_dir = path.parent_path().string();
  }

  bool should_set_path = false;
  if (from_settings) {
    should_set_path = true;
  } else {
#ifdef __EMSCRIPTEN__
    should_set_path = !library_dir.empty() && library_dir != "/";
#else
    should_set_path = !library_dir.empty();
#endif
  }
  return {library_dir, should_set_path, from_settings};
}

void PopulateNpuOptions(
    Backend backend,
    const std::optional<VisionExecutorSettings>& vision_settings,
    const std::optional<AudioExecutorSettings>& audio_settings,
    const ExecutorSettingsBase& main_executor_settings,
    std::vector<EnvironmentOptions::Option>& env_options) {
#if !defined(LITERT_DISABLE_NPU)
  if (UsesNpu(backend, vision_settings, audio_settings)) {
    auto [library_dir, should_set_path, from_settings] =
        ResolveLibraryDir(main_executor_settings);

    if (should_set_path) {
      // TODO(b/540445921): move the 'library_dir' to its own space.
      env_options.push_back(::litert::EnvironmentOptions::Option{
          ::litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
          library_dir.c_str()});
      if (from_settings) {
        ABSL_VLOG(1) << "Setting dispatch library path from "
                        "main_executor_settings: "
                     << library_dir;
      } else {
        ABSL_VLOG(1) << "Setting dispatch library path: " << library_dir;
      }

      env_options.push_back(::litert::EnvironmentOptions::Option{
          ::litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          library_dir.c_str()});
      if (from_settings) {
        ABSL_VLOG(1) << "Setting compiler plugin library path from "
                        "main_executor_settings: "
                     << library_dir;
      } else {
        ABSL_VLOG(1) << "Setting compiler plugin library path: " << library_dir;
      }
    } else {
      ABSL_VLOG(1) << "No valid library path provided.";
    }
  }
#endif
}

}  // namespace

absl::StatusOr<OwnedEnvironment> CreateEnvironment(
    EngineSettings& engine_settings, ModelResources* model_resources) {
  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();
  Backend backend = main_executor_settings.GetBackend();

  std::vector<EnvironmentOptions::Option> env_options;
  auto helper = std::make_unique<MagicNumberConfigsHelper>();

  bool uses_generic_npu_compiler_plugin = false;
  if (model_resources != nullptr && backend == Backend::NPU) {
    auto aux_model_buffer =
        model_resources->GetTFLiteModelBuffer(ModelType::kTfLiteAux);
    uses_generic_npu_compiler_plugin =
        !aux_model_buffer.ok() || aux_model_buffer->empty();
  }

  if (model_resources != nullptr &&
      (backend == Backend::CPU || backend == Backend::GPU ||
       uses_generic_npu_compiler_plugin)) {
    if (!main_executor_settings.GetAdvancedSettings() ||
        main_executor_settings.GetAdvancedSettings()->configure_magic_numbers) {
      env_options =
          helper->GetLiteRtEnvOptions(*model_resources, main_executor_settings);
    }
  }

  PopulateNpuOptions(backend, engine_settings.GetVisionExecutorSettings(),
                     engine_settings.GetAudioExecutorSettings(),
                     main_executor_settings, env_options);

  if (auto severity = GetMinLogSeverity()) {
    env_options.push_back(::litert::EnvironmentOptions::Option{
        ::litert::EnvironmentOptions::Tag::kMinLoggerSeverity,
        static_cast<int64_t>(ToLiteRtLogSeverityInt8(*severity))});
  }

  LITERT_ASSIGN_OR_RETURN(auto env,
                          Environment::Create(EnvironmentOptions(env_options)));
  return OwnedEnvironment{std::move(helper), std::move(env)};
}

absl::StatusOr<OwnedEnvironment> CreateEnvironment(
    EmbeddingEngineSettings& engine_settings, ModelResources* model_resources) {
  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();
  Backend backend = main_executor_settings.GetBackend();

  std::vector<EnvironmentOptions::Option> env_options;
  auto helper = std::make_unique<MagicNumberConfigsHelper>();

  PopulateNpuOptions(backend, engine_settings.GetVisionExecutorSettings(),
                     engine_settings.GetAudioExecutorSettings(),
                     main_executor_settings, env_options);

  if (auto severity = GetMinLogSeverity()) {
    env_options.push_back(::litert::EnvironmentOptions::Option{
        ::litert::EnvironmentOptions::Tag::kMinLoggerSeverity,
        static_cast<int64_t>(ToLiteRtLogSeverityInt8(*severity))});
  }

  LITERT_ASSIGN_OR_RETURN(auto env,
                          Environment::Create(EnvironmentOptions(env_options)));
  return OwnedEnvironment{std::move(helper), std::move(env)};
}

}  // namespace litert::lm
