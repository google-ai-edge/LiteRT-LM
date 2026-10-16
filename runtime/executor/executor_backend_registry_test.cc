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

#include "runtime/executor/executor_backend_registry.h"

#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

class StubModelResources : public ModelResources {
 public:
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Stub tokenizer");
  }
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return &llm_metadata_;
  }

 private:
  proto::LlmMetadata llm_metadata_;
};

ExecutorBackendRegistry::CreatorFunc StubCreator() {
  return [](const LlmExecutorSettings&,
            ModelResources&) -> absl::StatusOr<std::unique_ptr<LlmExecutor>> {
    return absl::UnimplementedError("Test stub creator");
  };
}

TEST(ExecutorBackendRegistryTest, DynamicBackendNameRegistrationAndLookup) {
  BackendTraits custom_traits{
      .use_external_sampler = true,
      .default_sampler_backend = Backend::CPU,
  };
  const Backend assigned_backend = ExecutorBackendRegistry::Instance().Register(
      "custom_test_backend", StubCreator(), custom_traits);
  absl::Cleanup unregister = [assigned_backend] {
    ExecutorBackendRegistry::Instance()
        .Unregister(assigned_backend)
        .IgnoreError();
  };

  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered("custom_test_backend"));
  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered("CUSTOM_TEST_BACKEND"));
  EXPECT_TRUE(
      ExecutorBackendRegistry::Instance().IsRegistered(assigned_backend));

  ASSERT_OK_AND_ASSIGN(const Backend resolved_backend,
                       GetBackendFromString("custom_test_backend"));
  EXPECT_EQ(resolved_backend, assigned_backend);
  EXPECT_EQ(GetBackendString(assigned_backend), "CUSTOM_TEST_BACKEND");

  std::optional<BackendTraits> traits =
      ExecutorBackendRegistry::Instance().GetTraits(assigned_backend);
  ASSERT_TRUE(traits.has_value());
  EXPECT_TRUE(traits->use_external_sampler);
  EXPECT_EQ(traits->default_sampler_backend, Backend::CPU);

  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create("dummy.sbs"));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, assigned_backend));
  EXPECT_EQ(executor_settings.GetBackend(), assigned_backend);
}

TEST(ExecutorBackendRegistryTest, RuntimeTryRegisterAndUnregister) {
  ExecutorBackendRegistry registry;
  ASSERT_OK_AND_ASSIGN(const Backend backend,
                       registry.TryRegister("runtime_backend", StubCreator()));
  EXPECT_TRUE(registry.IsRegistered(backend));
  EXPECT_TRUE(registry.IsRegistered("runtime_backend"));

  EXPECT_OK(registry.Unregister("runtime_backend"));
  EXPECT_FALSE(registry.IsRegistered(backend));
  EXPECT_FALSE(registry.GetTraits(backend).has_value());
  ASSERT_OK_AND_ASSIGN(auto model_assets, ModelAssets::Create("dummy.sbs"));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  StubModelResources model_resources;
  EXPECT_THAT(registry.Create(backend, executor_settings, model_resources),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(registry.Unregister(backend),
              StatusIs(absl::StatusCode::kNotFound));

  // Re-registering the same name yields the same stable Backend ID.
  ASSERT_OK_AND_ASSIGN(const Backend re_registered,
                       registry.TryRegister("runtime_backend", StubCreator()));
  EXPECT_EQ(re_registered, backend);
}

TEST(ExecutorBackendRegistryTest, TryRegisterRejectsDuplicates) {
  ExecutorBackendRegistry registry;
  ASSERT_OK(registry.TryRegister("duplicate_backend", StubCreator()).status());
  EXPECT_THAT(registry.TryRegister("duplicate_backend", StubCreator()),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_OK(registry.TryRegister(Backend::CPU, StubCreator()));
  EXPECT_THAT(registry.TryRegister(Backend::CPU, StubCreator()),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(ExecutorBackendRegistryTest, TryRegisterRejectsInvalidArguments) {
  ExecutorBackendRegistry registry;
  EXPECT_THAT(registry.TryRegister("", StubCreator()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(registry.TryRegister("empty_creator_backend", nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelResourcesRegistryTest, DynamicFileFormatRegistrationAndLookup) {
  ASSERT_OK(ModelResourcesRegistry::Instance().TryRegister(
      ".custom_fmt",
      [](const ModelAssets&)
          -> absl::StatusOr<std::unique_ptr<ModelResources>> {
        return std::make_unique<StubModelResources>();
      },
      "CUSTMAGC"));
  absl::Cleanup unregister = [] {
    ModelResourcesRegistry::Instance().Unregister(".custom_fmt").IgnoreError();
  };

  EXPECT_TRUE(ModelResourcesRegistry::Instance().IsRegistered(".custom_fmt"));
  EXPECT_TRUE(ModelResourcesRegistry::Instance().IsRegistered("CUSTOM_FMT"));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("/tmp/model.custom_fmt"));
  EXPECT_TRUE(ModelResourcesRegistry::Instance().CanLoad(model_assets));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesRegistry::Instance().Create(model_assets));
  EXPECT_NE(resources, nullptr);
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       resources->GetLlmMetadata());
  EXPECT_NE(metadata, nullptr);

  const std::string tmp_path =
      absl::StrCat(::testing::TempDir(), "/magic_header_test.bin");
  {
    std::ofstream out(tmp_path, std::ios::binary);
    out << "CUSTMAGC1234";
  }
  ASSERT_OK_AND_ASSIGN(auto mmfile, MemoryMappedFile::Create(tmp_path));
  ASSERT_OK_AND_ASSIGN(auto mm_assets,
                       ModelAssets::Create(std::shared_ptr<MemoryMappedFile>(
                           std::move(mmfile))));
  EXPECT_TRUE(ModelResourcesRegistry::Instance().CanLoad(mm_assets));
  ASSERT_OK_AND_ASSIGN(auto mm_resources,
                       ModelResourcesRegistry::Instance().Create(mm_assets));
  EXPECT_NE(mm_resources, nullptr);
}

}  // namespace
}  // namespace litert::lm
