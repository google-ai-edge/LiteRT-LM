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

#include "runtime/framework/resource_management/resource_manager.h"

#include <cstdint>
#include <filesystem>  // NOLINT: Required for testdata path manipulation.
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/llm_processed_context.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr char kTestLoraModelPath[] =
    "litert_lm/runtime/testdata/test_lm_lora.litertlm";
constexpr char kTestLoraWeightsPath[] =
    "litert_lm/runtime/testdata/test_lora_rank32_f16_all_ones.tflite";
constexpr int kMaxNumTokens = 32;

std::string TestDataPath(absl::string_view relative_path) {
  return (std::filesystem::path(::testing::SrcDir()) / relative_path).string();
}

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateLitertLmModelResources(absl::string_view model_path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  ASSIGN_OR_RETURN(auto loader, LitertLmLoader::Create(std::move(scoped_file)));
  return ModelResourcesLitertLm::Create(std::move(loader));
}

class RecordingLlmExecutor : public LlmExecutor {
 public:
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<std::vector<int>>> Decode() override {
    return std::vector<std::vector<int>>();
  }

  absl::string_view ExecutorBackendName() const override {
    return "RecordingLlmExecutor";
  }

  absl::Status LoadLoRA(uint32_t lora_id,
                        const ModelAssets& model_assets) override {
    loaded_lora_ids_.push_back(lora_id);
    auto path = model_assets.GetPath();
    loaded_lora_paths_.push_back(path.ok() ? std::string(*path) : "");
    return absl::OkStatus();
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    used_lora_ids_.push_back(lora_id);
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override {
    created_lora_ids_.push_back(lora_id);
    auto processed_context = std::make_unique<LlmProcessedContext>(
        lora_id, absl::flat_hash_map<absl::string_view, TensorBuffer>());
    return std::make_unique<LlmContext>(
        std::move(processed_context),
        std::make_unique<RuntimeConfig>(std::move(runtime_config)),
        std::make_unique<RuntimeState>());
  }

  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> context_data) override {
    return UseLoRA(context_data->processed_context().lora_id());
  }

  const std::vector<uint32_t>& loaded_lora_ids() const {
    return loaded_lora_ids_;
  }

  const std::vector<std::string>& loaded_lora_paths() const {
    return loaded_lora_paths_;
  }

  const std::vector<std::optional<uint32_t>>& used_lora_ids() const {
    return used_lora_ids_;
  }

  const std::vector<std::optional<uint32_t>>& created_lora_ids() const {
    return created_lora_ids_;
  }

 private:
  std::vector<uint32_t> loaded_lora_ids_;
  std::vector<std::string> loaded_lora_paths_;
  std::vector<std::optional<uint32_t>> used_lora_ids_;
  mutable std::vector<std::optional<uint32_t>> created_lora_ids_;
};

std::string CreateTempLoraFile(absl::string_view filename) {
  const std::string lora_path =
      absl::StrCat(::testing::TempDir(), "/", filename);
  std::ofstream ofs(lora_path);
  ofs << "lora";
  return lora_path;
}

absl::Status SetScopedLoraFile(SessionConfig& session_config,
                               absl::string_view lora_path) {
  auto scoped_file = ScopedFile::Open(lora_path);
  if (!scoped_file.ok()) {
    return scoped_file.status();
  }
  session_config.SetScopedLoraFile(
      std::make_shared<ScopedFile>(std::move(*scoped_file)));
  return absl::OkStatus();
}

void ConfigureGreedySampler(SessionConfig& session_config) {
  proto::SamplerParameters& sampler_params =
      session_config.GetMutableSamplerParams();
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
}

TEST(ResourceManagerTest, CreateContextHandlerLoadsTextLora) {
  auto executor = std::make_unique<RecordingLlmExecutor>();
  RecordingLlmExecutor* executor_ptr = executor.get();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create("base_model"));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  ResourceManager resource_manager(
      /*model_resources=*/nullptr, std::move(executor),
      /*vision_executor_settings=*/nullptr,
      /*audio_executor_settings=*/nullptr, std::move(executor_settings),
      /*litert_env=*/nullptr);

  const std::string lora_path = CreateTempLoraFile("text_lora.tflite");
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetLoraPath(lora_path);
  ASSERT_OK(SetScopedLoraFile(session_config, lora_path));

  ASSERT_OK_AND_ASSIGN(auto context_handler,
                       resource_manager.CreateContextHandler(session_config));

  ASSERT_EQ(executor_ptr->loaded_lora_ids().size(), 1);
  EXPECT_EQ(executor_ptr->loaded_lora_ids()[0], 0);
  ASSERT_EQ(executor_ptr->loaded_lora_paths().size(), 1);
  EXPECT_EQ(executor_ptr->loaded_lora_paths()[0], lora_path);
  EXPECT_TRUE(executor_ptr->used_lora_ids().empty());
  ASSERT_EQ(executor_ptr->created_lora_ids().size(), 1);
  ASSERT_TRUE(executor_ptr->created_lora_ids()[0].has_value());
  EXPECT_EQ(*executor_ptr->created_lora_ids()[0], 0);

  auto shared_context_handler =
      std::shared_ptr<ContextHandler>(std::move(context_handler));
  ASSERT_OK_AND_ASSIGN(auto locked_executor,
                       resource_manager.AcquireExecutorWithContextHandler(
                           shared_context_handler));
  ASSERT_EQ(executor_ptr->used_lora_ids().size(), 1);
  ASSERT_TRUE(executor_ptr->used_lora_ids()[0].has_value());
  EXPECT_EQ(*executor_ptr->used_lora_ids()[0], 0);
}

TEST(ResourceManagerTest, CreateContextHandlerDoesNotReloadSameTextLoraPath) {
  auto executor = std::make_unique<RecordingLlmExecutor>();
  RecordingLlmExecutor* executor_ptr = executor.get();
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create("base_model"));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  ResourceManager resource_manager(
      /*model_resources=*/nullptr, std::move(executor),
      /*vision_executor_settings=*/nullptr,
      /*audio_executor_settings=*/nullptr, std::move(executor_settings),
      /*litert_env=*/nullptr);

  const std::string lora_path = CreateTempLoraFile("shared_text_lora.tflite");
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetLoraPath(lora_path);
  ASSERT_OK(SetScopedLoraFile(session_config, lora_path));

  ASSERT_OK_AND_ASSIGN(auto first_context_handler,
                       resource_manager.CreateContextHandler(session_config));
  ASSERT_OK_AND_ASSIGN(auto second_context_handler,
                       resource_manager.CreateContextHandler(session_config));

  ASSERT_EQ(executor_ptr->loaded_lora_ids().size(), 1);
  EXPECT_EQ(executor_ptr->loaded_lora_ids()[0], 0);
  ASSERT_EQ(executor_ptr->created_lora_ids().size(), 2);
  ASSERT_TRUE(executor_ptr->created_lora_ids()[0].has_value());
  ASSERT_TRUE(executor_ptr->created_lora_ids()[1].has_value());
  EXPECT_EQ(*executor_ptr->created_lora_ids()[0], 0);
  EXPECT_EQ(*executor_ptr->created_lora_ids()[1], 0);
}

TEST(ResourceManagerE2eTest, LoadsAndRunsTextLoraWithRealExecutor) {
  const std::string model_path = TestDataPath(kTestLoraModelPath);
  const std::string lora_path = TestDataPath(kTestLoraWeightsPath);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateLitertLmModelResources(model_path));
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       CreateLlmLiteRtCompiledModelExecutor(
                           executor_settings, env, *model_resources));
  ResourceManager resource_manager(
      model_resources.get(), std::move(executor),
      /*vision_executor_settings=*/nullptr,
      /*audio_executor_settings=*/nullptr, std::move(executor_settings), &env);

  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetLoraPath(lora_path);
  ConfigureGreedySampler(session_config);
  ASSERT_OK(SetScopedLoraFile(session_config, lora_path));
  ASSERT_OK_AND_ASSIGN(auto unique_context_handler,
                       resource_manager.CreateContextHandler(session_config));
  auto context_handler =
      std::shared_ptr<ContextHandler>(std::move(unique_context_handler));

  ASSERT_OK_AND_ASSIGN(auto locked_executor,
                       resource_manager.AcquireExecutorWithContextHandler(
                           context_handler));

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  ASSERT_OK(locked_executor->Prefill(inputs));
  ASSERT_OK_AND_ASSIGN(auto current_step, locked_executor->GetCurrentStep());
  EXPECT_EQ(current_step, input_tokens.size());

  ASSERT_OK_AND_ASSIGN(auto output_tokens, locked_executor->Decode());
  EXPECT_FALSE(output_tokens.empty());
  ASSERT_OK_AND_ASSIGN(current_step, locked_executor->GetCurrentStep());
  EXPECT_EQ(current_step, input_tokens.size() + 1);
}

}  // namespace
}  // namespace litert::lm
