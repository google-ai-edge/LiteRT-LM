// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// A fake executor that reports executor stats and counts stat resets.
class StatsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit StatsFakeLlmExecutor(int* num_resets)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_resets_(num_resets) {
    stats_.AccumulatePlainStep(absl::Milliseconds(10));
  }

  std::optional<ExecutorStats> GetExecutorStats() const override {
    return stats_;
  }
  void ResetExecutorStats() override { ++*num_resets_; }

 private:
  ExecutorStats stats_{.module_name = std::string(kLlmModuleName)};
  int* num_resets_;
};

// Tasks::Decode receives the `LockedLlmExecutor` wrapper rather than the
// underlying executor, so the stats APIs must be forwarded through it.
TEST(ResourceManagerTest, AcquiredExecutorForwardsExecutorStats) {
  int num_resets = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatsFakeLlmExecutor>(&num_resets),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());

  std::optional<ExecutorStats> stats = executor->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->module_name, kLlmModuleName);
  EXPECT_EQ(stats->plain_steps(), 1);

  executor->ResetExecutorStats();
  EXPECT_EQ(num_resets, 1);
}

// A fake executor that reports a given draft verification support and accepts
// every draft that it verifies.
class DraftVerificationFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit DraftVerificationFakeLlmExecutor(absl::Status support)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        support_(std::move(support)) {}

  absl::Status CheckDraftVerificationSupport() const override {
    return support_;
  }

  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft_token_ids,
      const ExecutorDecodeParams& decode_params) override {
    return DraftVerificationResult{
        .tokens =
            std::vector<int>(draft_token_ids.begin(), draft_token_ids.end()),
        .num_accepted = static_cast<int>(draft_token_ids.size())};
  }

 private:
  absl::Status support_;
};

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerificationSupport) {
  for (const absl::Status& support :
       {absl::OkStatus(), absl::FailedPreconditionError("No verify.")}) {
    SCOPED_TRACE(support.ToString());
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr,
            std::make_unique<DraftVerificationFakeLlmExecutor>(support),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                         resource_manager->AcquireExecutor());
    EXPECT_EQ(executor->CheckDraftVerificationSupport(), support);
  }
}

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerification) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<DraftVerificationFakeLlmExecutor>(absl::OkStatus()),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());
  EXPECT_THAT(executor->VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              IsOkAndHolds(FieldsAre(ElementsAre(3, 1, 4), 3)));
}

TEST(ResourceManagerTest, ExecutorsDontSupportDraftVerificationByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  // The defaults of `LlmExecutor`.
  EXPECT_THAT(executor.LlmExecutor::CheckDraftVerificationSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ResourceManagerTest, ExecutorsDontVerifyDraftsByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  executor.SetDraftVerificationSupport(absl::OkStatus());
  // The default of `LlmExecutor` for an executor that claims support.
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

// How the resource manager has installed contexts into an executor.
struct ContextSwitchCounts {
  int cloned_contexts = 0;
  int restored_contexts = 0;
  int updated_runtime_states = 0;
  std::vector<int> prefill_start_steps;
};

// A fake executor that counts context installs.
class ContextSwitchFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit ContextSwitchFakeLlmExecutor(
      ContextSwitchCounts* counts,
      std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}})
      : FakeLlmExecutor(/*vocab_size=*/16, prefill_tokens_set,
                        /*decode_tokens_set=*/{{4}}),
        counts_(counts) {}

  using FakeLlmExecutor::Prefill;
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    counts_->prefill_start_steps.push_back(GetCurrentStep().value_or(-1));
    return FakeLlmExecutor::Prefill(inputs, prefill_params);
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    ++counts_->cloned_contexts;
    return MakeContext();
  }

  using FakeLlmExecutor::RestoreContext;
  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    ++counts_->restored_contexts;
    return SetCurrentStep(llm_context->runtime_state().current_step);
  }
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return RuntimeConfig();
  }
  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    return absl::OkStatus();
  }
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return RuntimeState{.current_step = GetCurrentStep().value_or(0)};
  }
  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    ++counts_->updated_runtime_states;
    return SetCurrentStep(runtime_state.current_step);
  }

 private:
  std::unique_ptr<LlmContext> MakeContext() const {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(),
        std::make_unique<RuntimeState>(
            RuntimeState{.current_step = GetCurrentStep().value_or(0)}));
  }

  ContextSwitchCounts* counts_;
};

// Clones share the processed context of the session they were cloned from, so
// the resource manager switches between them with `UpdateRuntimeState()`
// rather than `RestoreContext()`. Executors that keep per-context state must
// treat both as a context switch (b/578989824).
TEST(ResourceManagerTest, SwitchesToClonesByUpdatingRuntimeState) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> root,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(root));
  EXPECT_EQ(counts.restored_contexts, 1);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone_1,
                       resource_manager->CloneContextHandler(root));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone_2,
                       resource_manager->CloneContextHandler(root));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(clone_1));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(clone_2));
  EXPECT_EQ(counts.restored_contexts, 1);
  EXPECT_EQ(counts.updated_runtime_states, 2);
}

absl::StatusOr<ExecutorInputs> MakeTextInputs(absl::Span<const int> tokens) {
  auto buffer = CopyToTensorBuffer<int>(
      tokens, {1, static_cast<int>(tokens.size())});
  if (!buffer.HasValue()) {
    return absl::InternalError(buffer.Error().Message());
  }
  return ExecutorInputs(ExecutorTextData(std::move(*buffer)), std::nullopt,
                        std::nullopt);
}

TEST(ResourceManagerTest,
     SyncsExecutorStepBeforePrefillingSuffixAfterFullPrefixMatch) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(
              &counts, std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}}),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> root,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(root));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({1, 2}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(2));
  }

  // Clone `root` at step 2, then advance `root` to step 4 so the shared
  // `ProcessedTokens` holds {1, 2, 3, 4} while `clone` remains at step 2.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone,
                       resource_manager->CloneContextHandler(root));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(root));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({3, 4}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(4));
  }

  // Acquire `clone` (restoring runtime current_step to 2) and prefill
  // {3, 4, 5, 6}: `RemoveMatchingTokens` strips {3, 4} up to step 4 ==
  // `processed_tokens->TokenCount()`, and the executor must be synced to step 4
  // before prefilling the remaining suffix {5, 6}.
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(clone));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(2));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({3, 4, 5, 6}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(6));
    ASSERT_OK_AND_ASSIGN(const ProcessedTokens* processed_tokens,
                         executor->GetProcessedTokens());
    EXPECT_THAT(processed_tokens->GetCopyOfTokens(),
                ElementsAre(ElementsAre(1, 2, 3, 4, 5, 6)));
  }

  EXPECT_THAT(counts.prefill_start_steps, ElementsAre(0, 2, 4));
  EXPECT_EQ(counts.cloned_contexts, 0);
}

constexpr absl::string_view kTestLmModelRelPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";
constexpr absl::string_view kTestVisionWithAdapterRelPath =
    "litert_lm/runtime/testdata/dummy_vision_with_adapter.litertlm";
constexpr absl::string_view kTestVisionWithoutAdapterRelPath =
    "litert_lm/runtime/testdata/dummy_vision_without_adapter.litertlm";
constexpr absl::string_view kTestStaticAudioRelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";
constexpr absl::string_view kTestStreamingAudioRelPath =
    "litert_lm/runtime/testdata/dummy_audio_streaming.litertlm";

std::string ResolveTestModelPath(absl::string_view rel_path) {
  return (std::filesystem::path(testing::SrcDir()) / std::string(rel_path))
      .string();
}

std::unique_ptr<FakeLlmExecutor> MakeMinimalFakeLlmExecutor() {
  return std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/16, /*prefill_tokens_set=*/std::vector<std::vector<int>>{},
      /*decode_tokens_set=*/std::vector<std::vector<int>>{});
}

absl::StatusOr<std::unique_ptr<ModelResources>> LoadLitertLmResources(
    const std::string& model_path) {
  ABSL_ASSIGN_OR_RETURN(ScopedFile scoped_file, ScopedFile::Open(model_path));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<LitertLmLoader> loader,
                        LitertLmLoader::Create(std::move(scoped_file)));
  return ModelResourcesLitertLm::Create(std::move(loader));
}

// Supplies persistent writable ScopedFiles per submodel and counts how many
// times `GetWeightCacheFile` requests a cache handle.
class CountingCacheProvider {
 public:
  explicit CountingCacheProvider(const std::filesystem::path& dir) : dir_(dir) {
    std::filesystem::create_directories(dir_);
  }

  ExecutorSettingsBase::CacheFileProvider AsProvider() {
    return [this](const ExecutorSettingsBase::CacheFileRequest& req)
               -> absl::StatusOr<std::shared_ptr<ScopedFile>> {
      ++calls_;
      auto& slot = files_[req.suffix];
      if (slot == nullptr) {
        const std::filesystem::path file_path =
            dir_ / absl::StrCat("cache", req.suffix);
        {
          std::ofstream create_empty_file(file_path, std::ios::binary);
        }
        ABSL_ASSIGN_OR_RETURN(ScopedFile opened,
                              ScopedFile::OpenWritable(file_path.string()));
        slot = std::make_shared<ScopedFile>(std::move(opened));
      }
      return slot;
    };
  }

  int calls() const { return calls_; }
  void ResetCalls() { calls_ = 0; }

 private:
  std::filesystem::path dir_;
  int calls_ = 0;
  absl::flat_hash_map<std::string, std::shared_ptr<ScopedFile>> files_;
};

class CountingFakeVisionExecutor : public VisionExecutor {
 public:
  explicit CountingFakeVisionExecutor(bool* destroyed)
      : destroyed_(destroyed) {}
  ~CountingFakeVisionExecutor() override {
    if (destroyed_ != nullptr) {
      *destroyed_ = true;
    }
  }

  absl::StatusOr<ExecutorVisionData> Encode(
      const TensorBuffer& input_image_tensor) override {
    return ExecutorVisionData();
  }

  absl::StatusOr<ExecutorVisionData> Encode(
      const absl::flat_hash_map<std::string, TensorBuffer>& input_tensors)
      override {
    return ExecutorVisionData();
  }

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override {
    return std::vector<int>{1, 10, 8};
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    VisionExecutorProperties props;
    props.num_tokens_per_image = 64;
    return props;
  }

 private:
  bool* destroyed_;
};

class CountingFakeAudioExecutor : public AudioExecutor {
 public:
  CountingFakeAudioExecutor(bool* destroyed, int* resets,
                            absl::StatusOr<ExecutorAudioData> flush_result)
      : destroyed_(destroyed),
        resets_(resets),
        flush_result_(std::move(flush_result)) {}
  ~CountingFakeAudioExecutor() override {
    if (destroyed_ != nullptr) {
      *destroyed_ = true;
    }
  }

  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    ExecutorAudioData data;
    data.SetValidTokens(2);
    return data;
  }

  absl::Status Reset() override {
    if (resets_ != nullptr) {
      ++*resets_;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorAudioData> Flush() override {
    if (!flush_result_.ok()) {
      return flush_result_.status();
    }
    ExecutorAudioData data;
    data.SetValidTokens(flush_result_->GetValidTokens());
    return data;
  }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    AudioExecutorProperties props;
    props.is_streaming_model = false;
    return props;
  }

 private:
  bool* destroyed_;
  int* resets_;
  absl::StatusOr<ExecutorAudioData> flush_result_;
};

TEST(ResourceManagerTest, InjectedExecutorsAreNotUnloadedAndAudioHelpersWork) {
  bool vision_destroyed = false;
  bool audio_destroyed = false;
  int audio_resets = 0;
  ExecutorAudioData flushed_data;
  flushed_data.SetValidTokens(5);

  {
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr, MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr,
            std::make_unique<CountingFakeAudioExecutor>(
                &audio_destroyed, &audio_resets, std::move(flushed_data)),
            std::make_unique<CountingFakeVisionExecutor>(&vision_destroyed)));

    resource_manager->MaybeUnloadVisionExecutor();
    resource_manager->MaybeUnloadAudioExecutor();
    EXPECT_FALSE(vision_destroyed);
    EXPECT_FALSE(audio_destroyed);
    EXPECT_OK(resource_manager->AcquireVisionExecutor());
    EXPECT_OK(resource_manager->AcquireAudioExecutor());
    ASSERT_OK_AND_ASSIGN(VisionExecutorProperties vision_props,
                         resource_manager->GetVisionExecutorProperties());
    EXPECT_EQ(vision_props.num_tokens_per_image, 64);
    ASSERT_OK_AND_ASSIGN(AudioExecutorProperties audio_props,
                         resource_manager->GetAudioExecutorProperties());
    EXPECT_FALSE(audio_props.is_streaming_model);

    ASSERT_OK_AND_ASSIGN(auto flushed, resource_manager->FlushAudioIfLoaded());
    ASSERT_TRUE(flushed.has_value());
    EXPECT_EQ(flushed->GetValidTokens(), 5);

    resource_manager->ResetAudioIfLoaded();
    EXPECT_EQ(audio_resets, 1);
  }
  EXPECT_TRUE(vision_destroyed);
  EXPECT_TRUE(audio_destroyed);
}

TEST(ResourceManagerTest, FlushAndResetAudioHandleUnloadedUnimplementedErrors) {
  // Unloaded executors with null settings: neither FlushAudioIfLoaded nor
  // ResetAudioIfLoaded fails, while property/load calls return InvalidArgument.
  {
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr, MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
    ASSERT_OK_AND_ASSIGN(auto flushed, resource_manager->FlushAudioIfLoaded());
    EXPECT_FALSE(flushed.has_value());
    resource_manager->ResetAudioIfLoaded();
    EXPECT_THAT(resource_manager->GetVisionExecutorProperties(),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(resource_manager->GetAudioExecutorProperties(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Unimplemented Flush() maps to std::nullopt.
  {
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr, MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr,
            std::make_unique<CountingFakeAudioExecutor>(
                /*destroyed=*/nullptr, /*resets=*/nullptr,
                absl::UnimplementedError("Flush not implemented"))));
    ASSERT_OK_AND_ASSIGN(auto flushed, resource_manager->FlushAudioIfLoaded());
    EXPECT_FALSE(flushed.has_value());
  }

  // Non-Unimplemented Flush() error is propagated.
  {
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr, MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr,
            std::make_unique<CountingFakeAudioExecutor>(
                /*destroyed=*/nullptr, /*resets=*/nullptr,
                absl::InternalError("Flush failed"))));
    EXPECT_THAT(resource_manager->FlushAudioIfLoaded(),
                StatusIs(absl::StatusCode::kInternal));
  }
}

TEST(ResourceManagerTest,
     ReloadableCpuVisionDefersSessionLoadLoadsOnDemandAndEvicts) {
  const std::string model_path =
      ResolveTestModelPath(kTestVisionWithAdapterRelPath);
  ASSERT_OK_AND_ASSIGN(auto model_resources, LoadLitertLmResources(model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(VisionExecutorSettings vision_settings,
                       VisionExecutorSettings::CreateDefault(
                           model_assets, /*encoder_backend=*/Backend::CPU,
                           /*adapter_backend=*/Backend::CPU));

  CountingCacheProvider cache_provider(
      std::filesystem::path(testing::TempDir()) / "vision_reload_scoped");
  vision_settings.SetCacheFileProvider(cache_provider.AsProvider());

  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          model_resources.get(), MakeMinimalFakeLlmExecutor(),
          std::make_unique<VisionExecutorSettings>(std::move(vision_settings)),
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  // Constructor probes encoder + adapter cache usability once.
  EXPECT_EQ(cache_provider.calls(), 2);
  cache_provider.ResetCalls();

  // Before any session enables vision, AcquireVisionExecutor must fail rather
  // than silently compiling an unprepared modality.
  EXPECT_THAT(resource_manager->AcquireVisionExecutor(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(cache_provider.calls(), 0);

  // GetVisionExecutorProperties inspects ModelResources without compiling.
  ASSERT_OK_AND_ASSIGN(VisionExecutorProperties props,
                       resource_manager->GetVisionExecutorProperties());
  EXPECT_GT(props.num_tokens_per_image, 0);
  EXPECT_EQ(cache_provider.calls(), 0);

  // Session creation with vision enabled validates presence via ModelResources
  // and defers compilation.
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetVisionModalityEnabled(true);
  ASSERT_OK(resource_manager->CreateContextHandler(session_config));
  EXPECT_EQ(cache_provider.calls(), 0);

  // First AcquireVisionExecutor compiles encoder + adapter on demand.
  {
    ASSERT_OK_AND_ASSIGN(auto vision_executor,
                         resource_manager->AcquireVisionExecutor());
    EXPECT_NE(vision_executor, nullptr);
  }
  EXPECT_EQ(cache_provider.calls(), 2);

  // Creating a second session while vision_executor_ is resident reuses it
  // without compiling again.
  ASSERT_OK(resource_manager->CreateContextHandler(session_config));
  EXPECT_EQ(cache_provider.calls(), 2);

  // Re-acquiring before eviction reuses the resident executor.
  {
    ASSERT_OK_AND_ASSIGN(auto vision_executor,
                         resource_manager->AcquireVisionExecutor());
    EXPECT_NE(vision_executor, nullptr);
  }
  EXPECT_EQ(cache_provider.calls(), 2);

  // Evicting drops the resident executor; subsequent acquire warm-reloads it.
  resource_manager->MaybeUnloadVisionExecutor();
  {
    ASSERT_OK_AND_ASSIGN(auto vision_executor,
                         resource_manager->AcquireVisionExecutor());
    EXPECT_NE(vision_executor, nullptr);
  }
  EXPECT_EQ(cache_provider.calls(), 4);
}

TEST(ResourceManagerTest,
     ReloadableCpuVisionWithoutAdapterWorksWithDirectoryAndBareRelativeCache) {
  const std::string model_path =
      ResolveTestModelPath(kTestVisionWithoutAdapterRelPath);
  ASSERT_OK_AND_ASSIGN(auto model_resources, LoadLitertLmResources(model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(VisionExecutorSettings vision_settings,
                       VisionExecutorSettings::CreateDefault(
                           model_assets, /*encoder_backend=*/Backend::CPU,
                           /*adapter_backend=*/Backend::CPU));

  const std::filesystem::path cache_dir =
      std::filesystem::path(testing::TempDir()) / "vision_dir_cache";
  std::filesystem::create_directories(cache_dir);
  vision_settings.SetCacheDir(cache_dir.string());
  const std::string encoder_suffix =
      absl::StrCat(VisionExecutorSettings::kEncoderName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix);
  ASSERT_OK_AND_ASSIGN(auto cache_file_var,
                       vision_settings.GetWeightCacheFile(
                           encoder_suffix, /*check_and_clean=*/false));
  const std::string expected_cache_path = std::get<std::string>(cache_file_var);

  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          model_resources.get(), MakeMinimalFakeLlmExecutor(),
          std::make_unique<VisionExecutorSettings>(std::move(vision_settings)),
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));

  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetVisionModalityEnabled(true);
  ASSERT_OK(resource_manager->CreateContextHandler(session_config));
  EXPECT_FALSE(FileExists(expected_cache_path));

  {
    ASSERT_OK_AND_ASSIGN(auto vision_executor,
                         resource_manager->AcquireVisionExecutor());
    EXPECT_NE(vision_executor, nullptr);
  }
  EXPECT_TRUE(FileExists(expected_cache_path));

  // Verify bare relative model_path in a writable current directory also
  // resolves parent_dir "." cleanly and defers compilation.
  const std::filesystem::path original_cwd = std::filesystem::current_path();
  struct CwdGuard {
    std::filesystem::path saved;
    ~CwdGuard() { std::filesystem::current_path(saved); }
  } cwd_guard{original_cwd};
  const std::filesystem::path rel_dir =
      std::filesystem::path(testing::TempDir()) / "bare_rel_cwd";
  std::filesystem::create_directories(rel_dir);
  std::filesystem::copy_file(model_path, rel_dir / "bare_vision.litertlm",
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::current_path(rel_dir);

  ASSERT_OK_AND_ASSIGN(auto rel_resources,
                       LoadLitertLmResources("bare_vision.litertlm"));
  ASSERT_OK_AND_ASSIGN(ModelAssets rel_assets,
                       ModelAssets::Create("bare_vision.litertlm"));
  ASSERT_OK_AND_ASSIGN(VisionExecutorSettings rel_settings,
                       VisionExecutorSettings::CreateDefault(
                           rel_assets, Backend::CPU, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      auto rel_manager,
      ResourceManager::Create(
          rel_resources.get(), MakeMinimalFakeLlmExecutor(),
          std::make_unique<VisionExecutorSettings>(std::move(rel_settings)),
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK(rel_manager->CreateContextHandler(session_config));
  EXPECT_FALSE(
      FileExists(absl::StrCat("bare_vision.litertlm", encoder_suffix)));
}

TEST(ResourceManagerTest,
     ReloadableCpuStaticAudioDefersSessionLoadLoadsOnDemandAndEvicts) {
  const std::string model_path = ResolveTestModelPath(kTestStaticAudioRelPath);
  ASSERT_OK_AND_ASSIGN(auto model_resources, LoadLitertLmResources(model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      AudioExecutorSettings audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, /*max_sequence_length=*/10,
          /*encoder_backend=*/Backend::CPU, /*adapter_backend=*/Backend::CPU));

  CountingCacheProvider cache_provider(
      std::filesystem::path(testing::TempDir()) / "audio_reload_scoped");
  audio_settings.SetCacheFileProvider(cache_provider.AsProvider());

  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          model_resources.get(), MakeMinimalFakeLlmExecutor(),
          /*vision_executor_settings=*/nullptr,
          std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
          /*litert_env=*/nullptr));
  const int init_probe_calls = cache_provider.calls();
  EXPECT_GT(init_probe_calls, 0);
  cache_provider.ResetCalls();

  // Unprepared acquire is rejected.
  EXPECT_THAT(resource_manager->AcquireAudioExecutor(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(cache_provider.calls(), 0);

  // Property lookup reads ModelResources without compiling.
  ASSERT_OK_AND_ASSIGN(AudioExecutorProperties props,
                       resource_manager->GetAudioExecutorProperties());
  EXPECT_FALSE(props.is_streaming_model);
  EXPECT_EQ(cache_provider.calls(), 0);

  // Session creation defers compilation for static CPU audio.
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(auto handler,
                       resource_manager->CreateContextHandler(session_config));
  EXPECT_FALSE(handler->HasAudioContext());
  EXPECT_EQ(cache_provider.calls(), 0);

  // FlushAudioIfLoaded and ResetAudioIfLoaded do not trigger compilation.
  ASSERT_OK_AND_ASSIGN(auto flushed, resource_manager->FlushAudioIfLoaded());
  EXPECT_FALSE(flushed.has_value());
  resource_manager->ResetAudioIfLoaded();
  EXPECT_EQ(cache_provider.calls(), 0);

  // First AcquireAudioExecutor compiles on demand; creating a second session
  // while resident defers without recompiling; MaybeUnloadAudioExecutor evicts
  // so the next AcquireAudioExecutor warm-reloads.
  {
    ASSERT_OK_AND_ASSIGN(auto audio_executor,
                         resource_manager->AcquireAudioExecutor());
    EXPECT_NE(audio_executor, nullptr);
  }
  const int first_load_calls = cache_provider.calls();
  EXPECT_GT(first_load_calls, 0);

  ASSERT_OK(resource_manager->CreateContextHandler(session_config));
  EXPECT_EQ(cache_provider.calls(), first_load_calls);

  resource_manager->MaybeUnloadAudioExecutor();
  {
    ASSERT_OK_AND_ASSIGN(auto audio_executor,
                         resource_manager->AcquireAudioExecutor());
    EXPECT_NE(audio_executor, nullptr);
  }
  EXPECT_EQ(cache_provider.calls(), 2 * first_load_calls);
}

TEST(ResourceManagerTest,
     StreamingAudioModelLoadsEagerlyCreatesContextAndNeverUnloads) {
  const std::string model_path =
      ResolveTestModelPath(kTestStreamingAudioRelPath);
  ASSERT_OK_AND_ASSIGN(auto model_resources, LoadLitertLmResources(model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path));
  ASSERT_OK_AND_ASSIGN(
      AudioExecutorSettings audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, /*max_sequence_length=*/10,
          /*encoder_backend=*/Backend::CPU, /*adapter_backend=*/Backend::CPU));

  CountingCacheProvider cache_provider(
      std::filesystem::path(testing::TempDir()) / "streaming_audio_scoped");
  audio_settings.SetCacheFileProvider(cache_provider.AsProvider());

  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          model_resources.get(), MakeMinimalFakeLlmExecutor(),
          /*vision_executor_settings=*/nullptr,
          std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
          /*litert_env=*/nullptr));
  cache_provider.ResetCalls();

  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(auto handler,
                       resource_manager->CreateContextHandler(session_config));
  EXPECT_TRUE(handler->HasAudioContext());
  const int eager_load_calls = cache_provider.calls();
  EXPECT_GT(eager_load_calls, 0);

  // Streaming audio is stateful and must remain resident across MaybeUnload.
  resource_manager->MaybeUnloadAudioExecutor();
  {
    ASSERT_OK_AND_ASSIGN(auto audio_executor,
                         resource_manager->AcquireAudioExecutor());
    EXPECT_NE(audio_executor, nullptr);
  }
  EXPECT_EQ(cache_provider.calls(), eager_load_calls);
}

TEST(ResourceManagerTest, NonReloadableConfigsLoadEagerlyAndFailFastOnMissing) {
  // 1. Missing vision/audio sections in a .litertlm container must fail fast
  // during CreateContextHandler for both reloadable and non-reloadable configs.
  const std::string text_only_model_path =
      ResolveTestModelPath(kTestLmModelRelPath);
  ASSERT_OK_AND_ASSIGN(auto text_model_resources,
                       LoadLitertLmResources(text_only_model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets text_assets,
                       ModelAssets::Create(text_only_model_path));
  const std::filesystem::path writable_cache_dir =
      std::filesystem::path(testing::TempDir()) / "fail_fast_cache";
  std::filesystem::create_directories(writable_cache_dir);
  const std::string writable_cache_dir_str = writable_cache_dir.string();

  for (absl::string_view cache_dir : {absl::string_view(writable_cache_dir_str),
                                      absl::string_view(":nocache")}) {
    ASSERT_OK_AND_ASSIGN(VisionExecutorSettings vision_settings,
                         VisionExecutorSettings::CreateDefault(
                             text_assets, Backend::CPU, Backend::CPU));
    vision_settings.SetCacheDir(std::string(cache_dir));
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(text_model_resources.get(),
                                MakeMinimalFakeLlmExecutor(),
                                std::make_unique<VisionExecutorSettings>(
                                    std::move(vision_settings)),
                                /*audio_executor_settings=*/nullptr,
                                /*litert_env=*/nullptr));
    SessionConfig session_config = SessionConfig::CreateDefault();
    session_config.SetVisionModalityEnabled(true);
    EXPECT_THAT(resource_manager->CreateContextHandler(session_config),
                StatusIs(absl::StatusCode::kNotFound));
  }

  for (absl::string_view cache_dir : {absl::string_view(writable_cache_dir_str),
                                      absl::string_view(":nocache")}) {
    ASSERT_OK_AND_ASSIGN(AudioExecutorSettings audio_settings,
                         AudioExecutorSettings::CreateDefault(
                             text_assets, /*max_sequence_length=*/10,
                             Backend::CPU, Backend::CPU));
    audio_settings.SetCacheDir(std::string(cache_dir));
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            text_model_resources.get(), MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
            /*litert_env=*/nullptr));
    SessionConfig session_config = SessionConfig::CreateDefault();
    session_config.SetAudioModalityEnabled(true);
    EXPECT_THAT(resource_manager->CreateContextHandler(session_config),
                StatusIs(absl::StatusCode::kNotFound));
  }

  // 2. `:nocache`, `:memory`, `SetDisableWeightCache(true)`, uncached
  // non-existent cache_dir, and `model_resources == nullptr` must compile
  // eagerly at session creation and never evict on MaybeUnloadVisionExecutor().
  const std::string vision_model_path =
      ResolveTestModelPath(kTestVisionWithoutAdapterRelPath);
  ASSERT_OK_AND_ASSIGN(auto vision_resources,
                       LoadLitertLmResources(vision_model_path));
  ASSERT_OK_AND_ASSIGN(ModelAssets vision_assets,
                       ModelAssets::Create(vision_model_path));

  for (absl::string_view mode :
       {"nocache", "memory", "disabled", "non_writable", "null_resources"}) {
    SCOPED_TRACE(mode);
    ASSERT_OK_AND_ASSIGN(VisionExecutorSettings vision_settings,
                         VisionExecutorSettings::CreateDefault(
                             vision_assets, Backend::CPU, Backend::CPU));
    ModelResources* resources_ptr = vision_resources.get();
    CountingCacheProvider provider(std::filesystem::path(testing::TempDir()) /
                                   absl::StrCat("non_reloadable_", mode));
    if (mode == "nocache") {
      vision_settings.SetCacheDir(":nocache");
    } else if (mode == "memory") {
      vision_settings.SetCacheDir(":memory");
    } else if (mode == "disabled") {
      vision_settings.SetCacheDir(writable_cache_dir.string());
      vision_settings.SetDisableWeightCache(true);
    } else if (mode == "non_writable") {
      vision_settings.SetCacheDir((std::filesystem::path(testing::TempDir()) /
                                   "missing_parent_dir_for_cache")
                                      .string());
    } else if (mode == "null_resources") {
      vision_settings.SetCacheFileProvider(provider.AsProvider());
      resources_ptr = nullptr;
    }

    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(resources_ptr, MakeMinimalFakeLlmExecutor(),
                                std::make_unique<VisionExecutorSettings>(
                                    std::move(vision_settings)),
                                /*audio_executor_settings=*/nullptr,
                                /*litert_env=*/nullptr));
    provider.ResetCalls();

    if (mode == "null_resources") {
      // Fallback property query with model_resources == nullptr loads eagerly.
      ASSERT_OK_AND_ASSIGN(VisionExecutorProperties props,
                           resource_manager->GetVisionExecutorProperties());
      EXPECT_GT(props.num_tokens_per_image, 0);
      EXPECT_GT(provider.calls(), 0);
      provider.ResetCalls();
    }

    SessionConfig session_config = SessionConfig::CreateDefault();
    session_config.SetVisionModalityEnabled(true);
    ASSERT_OK(resource_manager->CreateContextHandler(session_config));

    resource_manager->MaybeUnloadVisionExecutor();
    EXPECT_OK(resource_manager->AcquireVisionExecutor());
    EXPECT_EQ(provider.calls(), 0);
  }

  // 3. Static audio with model_resources == nullptr or non-zero LoRA rank is
  // non-reloadable: fallback GetAudioExecutorProperties() loads eagerly and
  // MaybeUnloadAudioExecutor() never evicts.
  const std::string audio_model_path =
      ResolveTestModelPath(kTestStaticAudioRelPath);
  ASSERT_OK_AND_ASSIGN(ModelAssets audio_assets,
                       ModelAssets::Create(audio_model_path));
  {
    ASSERT_OK_AND_ASSIGN(AudioExecutorSettings audio_settings,
                         AudioExecutorSettings::CreateDefault(
                             audio_assets, /*max_sequence_length=*/10,
                             Backend::CPU, Backend::CPU));
    CountingCacheProvider provider(std::filesystem::path(testing::TempDir()) /
                                   "audio_null_resources");
    audio_settings.SetCacheFileProvider(provider.AsProvider());
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr, MakeMinimalFakeLlmExecutor(),
            /*vision_executor_settings=*/nullptr,
            std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
            /*litert_env=*/nullptr));
    provider.ResetCalls();

    ASSERT_OK_AND_ASSIGN(AudioExecutorProperties props,
                         resource_manager->GetAudioExecutorProperties());
    EXPECT_FALSE(props.is_streaming_model);
    EXPECT_GT(provider.calls(), 0);
    provider.ResetCalls();

    resource_manager->MaybeUnloadAudioExecutor();
    EXPECT_OK(resource_manager->AcquireAudioExecutor());
    EXPECT_EQ(provider.calls(), 0);
  }
}

}  // namespace
}  // namespace litert::lm
