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

#include "omni/omni_registry.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/asr_session_config.pb.h"
#include "runtime/proto/tts_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::NotNull;

class FakeSessionFactory : public OmniSessionFactory {
 public:
  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull
      /*input_source*/) override {
    return absl::UnimplementedError("FakeSessionFactory::Create");
  }
};

TEST(OmniRegistryTest, ResolvesSessionConfigAndMetadataProtoTypeIds) {
  auto& registry = OmniRegistry::Get();
  EXPECT_THAT(registry.ResolveSessionConfigProtoTypeId("Asr"),
              IsOkAndHolds("litert.lm.proto.AsrSessionConfig"));
  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("Asr", ""),
              IsOkAndHolds("litert.lm.proto.AsrMetadata"));

  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("Asr", "Custom"),
              IsOkAndHolds("litert.lm.proto.CustomMetadata"));
  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("Asr", "a.b.c.MyMetadata"),
              IsOkAndHolds("a.b.c.MyMetadata"));

  EXPECT_THAT(registry.ResolveSessionConfigProtoTypeId("d.e.f.MySessionConfig"),
              IsOkAndHolds("d.e.f.MySessionConfig"));
  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("d.e.f.MySessionConfig",
                                                  /*metadata_type=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("d.e.f.MySessionConfig",
                                                  "a.b.c.MyMetadata"),
              IsOkAndHolds("a.b.c.MyMetadata"));
}

TEST(OmniRegistryTest,
     RegisterSessionFactoryForwardsSettingsAndValidatesTypes) {
  auto& registry = OmniRegistry::Get();
  OmniEngine::Settings captured_settings;
  float captured_overlap_ratio = 0.0f;
  int captured_input_milliseconds = 0;

  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"custom-registry-test-model",
                      "custom-registry-test-model-alias"},
      .get_session_factory =
          [&](absl::string_view /*model_name*/,
              const OmniEngine::Settings& settings,
              const MessageLite* absl_nullable session_cfg,
              const MessageLite* absl_nullable metadata,
              ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
              std::shared_ptr<lm::ModelResources> absl_nullable
              /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        captured_settings = settings;
        if (session_cfg != nullptr) {
          const auto* asr_cfg =
              static_cast<const lm::proto::AsrSessionConfig*>(session_cfg);
          captured_overlap_ratio = asr_cfg->overlap_ratio();
        }
        if (metadata != nullptr) {
          const auto* asr_meta =
              static_cast<const lm::proto::AsrMetadata*>(metadata);
          captured_input_milliseconds = asr_meta->input_milliseconds();
        }
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Asr",
      .model_remote_url = "https://example.com/custom_test_model.litertlm",
  }));

  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("custom-registry-test-model");
    registry.UnregisterModel("custom-registry-test-model-alias");
  };

  EXPECT_TRUE(registry.IsModelRegistered("custom-registry-test-model"));
  EXPECT_TRUE(registry.IsModelRegistered("custom-registry-test-model-alias"));
  ASSERT_OK_AND_ASSIGN(auto reg_model, registry.GetRegisteredModel(
                                           "custom-registry-test-model-alias"));
  EXPECT_EQ(reg_model.model_name, "custom-registry-test-model-alias");
  EXPECT_EQ(reg_model.canonical_model_name, "custom-registry-test-model");
  EXPECT_EQ(reg_model.model_remote_url,
            "https://example.com/custom_test_model.litertlm");
  EXPECT_EQ(reg_model.session_config_proto_type_id,
            "litert.lm.proto.AsrSessionConfig");
  EXPECT_EQ(reg_model.metadata_proto_type_id, "litert.lm.proto.AsrMetadata");

  OmniEngine::Settings caller_settings{
      .backend = OmniEngine::Settings::Backend::kGpu,
      .cache_dir = "/custom/cache",
      .num_threads = 8,
  };
  lm::proto::AsrSessionConfig caller_session_config;
  caller_session_config.set_overlap_ratio(0.5f);
  lm::proto::AsrMetadata caller_metadata;
  caller_metadata.set_input_milliseconds(3000);

  ASSERT_OK_AND_ASSIGN(auto factory,
                       registry.CreateSessionFactory(
                           "custom-registry-test-model", caller_settings,
                           &caller_session_config, &caller_metadata));
  EXPECT_THAT(factory, NotNull());
  EXPECT_EQ(captured_settings.backend, OmniEngine::Settings::Backend::kGpu);
  EXPECT_EQ(captured_settings.cache_dir, "/custom/cache");
  EXPECT_EQ(captured_settings.num_threads, 8);
  EXPECT_FLOAT_EQ(captured_overlap_ratio, 0.5f);
  EXPECT_EQ(captured_input_milliseconds, 3000);
}

TEST(OmniRegistryTest, ResolvesModelByPathWhenModelNameNotExact) {
  auto& registry = OmniRegistry::Get();
  std::string resolved_name;

  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"whisper-resolve-test"},
      .get_session_factory =
          [&](absl::string_view model_name,
              const OmniEngine::Settings& /*settings*/,
              const MessageLite* /*session_cfg*/,
              const MessageLite* /*metadata*/,
              ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
              std::shared_ptr<lm::ModelResources> /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        resolved_name = std::string(model_name);
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Asr",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("whisper-resolve-test");
  };

  OmniEngine::Settings path_settings{
      .model_path = "/tmp/my_whisper_resolve_test_v1.litertlm",
  };
  ASSERT_OK_AND_ASSIGN(
      auto factory_by_path,
      registry.CreateSessionFactory(/*model_name=*/"", path_settings));
  EXPECT_THAT(factory_by_path, NotNull());
  EXPECT_EQ(resolved_name, "whisper-resolve-test");
}

TEST(OmniRegistryTest, RejectsMismatchedProtoTypesAtFactoryCreation) {
  auto& registry = OmniRegistry::Get();
  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"proto-type-check-model"},
      .get_session_factory = [](absl::string_view, const OmniEngine::Settings&,
                                const MessageLite*,
                                const MessageLite*,
                                ::litert::Environment&, ::litert::Options&,
                                std::shared_ptr<lm::ModelResources>)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Asr",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("proto-type-check-model");
  };

  lm::proto::TtsSessionConfig wrong_session_config;
  EXPECT_THAT(registry.CreateSessionFactory("proto-type-check-model", {},
                                            &wrong_session_config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("litert.lm.proto.AsrSessionConfig")));
}

}  // namespace
}  // namespace litert::omni
