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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "runtime/util/file_util.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::StartsWith;

void WriteTestLitertLmWithSpeech2TextMetadata(
    const std::string& path,
    const lm::proto::Speech2TextMetadata& embedded_metadata) {
  const std::string serialized = embedded_metadata.SerializeAsString();
  constexpr uint64_t kDataOffset = 512;

  flatbuffers::FlatBufferBuilder builder(256);
  auto asr_meta_section = lm::schema::CreateSectionObject(
      builder, /*items=*/0, /*begin_offset=*/kDataOffset,
      /*end_offset=*/kDataOffset + serialized.size(),
      lm::schema::AnySectionDataType_Speech2TextMetadataProto);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> sections = {
      asr_meta_section};
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(sections));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  std::ofstream file(path, std::ios::binary);
  file.write("LITERTLM", 8);
  uint32_t major_version = 1;
  uint32_t minor_version = 0;
  uint32_t patch_version = 0;
  uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&minor_version), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&patch_version), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&padding), sizeof(uint32_t));
  uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset),
             sizeof(uint64_t));
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  std::string pad(kDataOffset - header_end_offset, '\0');
  file.write(pad.data(), pad.size());
  file.write(serialized.data(), serialized.size());
}

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
  EXPECT_THAT(registry.ResolveSessionConfigProtoTypeId("Speech2Text"),
              IsOkAndHolds("litert.lm.proto.Speech2TextSessionConfig"));
  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("Speech2Text", ""),
              IsOkAndHolds("litert.lm.proto.Speech2TextMetadata"));

  EXPECT_THAT(registry.ResolveMetadataProtoTypeId("Speech2Text", "Custom"),
              IsOkAndHolds("litert.lm.proto.CustomMetadata"));
  EXPECT_THAT(
      registry.ResolveMetadataProtoTypeId("Speech2Text", "a.b.c.MyMetadata"),
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
  LiteRtDelegatePrecision captured_gpu_precision =
      kLiteRtDelegatePrecisionDefault;
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
              ::litert::Environment& /*env*/, ::litert::Options& options,
              std::shared_ptr<lm::ModelResources> absl_nullable
              /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        captured_settings = settings;
        captured_gpu_precision = kLiteRtDelegatePrecisionDefault;
        if (auto gpu_options = options.GetOptions<::litert::GpuOptions>();
            gpu_options) {
          LrtGetGpuAcceleratorCompilationOptionsPrecision(
              &captured_gpu_precision, gpu_options->Get());
        }
        if (session_cfg != nullptr) {
          const auto* asr_cfg =
              static_cast<const lm::proto::Speech2TextSessionConfig*>(
                  session_cfg);
          captured_overlap_ratio = asr_cfg->overlap_ratio();
        }
        if (metadata != nullptr) {
          const auto* asr_meta =
              static_cast<const lm::proto::Speech2TextMetadata*>(metadata);
          captured_input_milliseconds = asr_meta->input_milliseconds();
        }
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
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
            "litert.lm.proto.Speech2TextSessionConfig");
  EXPECT_EQ(reg_model.metadata_proto_type_id,
            "litert.lm.proto.Speech2TextMetadata");

  OmniEngine::Settings caller_settings{
      .backend = OmniEngine::Settings::Backend::kGpu,
      .cache_dir = "/custom/cache",
      .num_threads = 8,
      .force_fp32_on_gpu = true,
  };
  lm::proto::Speech2TextSessionConfig caller_session_config;
  caller_session_config.set_overlap_ratio(0.5f);
  lm::proto::Speech2TextMetadata caller_metadata;
  caller_metadata.set_input_milliseconds(3000);

  ASSERT_OK_AND_ASSIGN(auto factory,
                       registry.CreateSessionFactory(
                           "custom-registry-test-model", caller_settings,
                           &caller_session_config, &caller_metadata));
  EXPECT_THAT(factory, NotNull());
  EXPECT_EQ(captured_settings.backend, OmniEngine::Settings::Backend::kGpu);
  EXPECT_EQ(captured_settings.cache_dir, "/custom/cache");
  EXPECT_EQ(captured_settings.num_threads, 8);
  EXPECT_TRUE(captured_settings.force_fp32_on_gpu);
  EXPECT_EQ(captured_gpu_precision, kLiteRtDelegatePrecisionFp32);
  EXPECT_FLOAT_EQ(captured_overlap_ratio, 0.5f);
  EXPECT_EQ(captured_input_milliseconds, 3000);

  caller_settings.force_fp32_on_gpu = false;
  ASSERT_OK_AND_ASSIGN(auto factory_fp16,
                       registry.CreateSessionFactory(
                           "custom-registry-test-model", caller_settings,
                           &caller_session_config, &caller_metadata));
  EXPECT_THAT(factory_fp16, NotNull());
  EXPECT_FALSE(captured_settings.force_fp32_on_gpu);
  EXPECT_EQ(captured_gpu_precision, kLiteRtDelegatePrecisionFp16);
}

TEST(OmniRegistryTest, ConfiguresWeightCachesOnlyForWritableCacheDir) {
  auto& registry = OmniRegistry::Get();
  std::string captured_cache_path;
  std::string captured_gpu_cache_key;
  std::string captured_gpu_serialization_dir;
  std::string captured_model_path;
  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"weight-cache-test-model"},
      .get_session_factory =
          [&](absl::string_view /*model_name*/,
              const OmniEngine::Settings& settings,
              const MessageLite* absl_nullable /*session_cfg*/,
              const MessageLite* absl_nullable /*metadata*/,
              ::litert::Environment& /*env*/, ::litert::Options& options,
              std::shared_ptr<lm::ModelResources> absl_nullable
              /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        captured_cache_path.clear();
        captured_gpu_cache_key.clear();
        captured_gpu_serialization_dir.clear();
        captured_model_path = settings.model_path;
        auto cpu_options = options.GetOptions<::litert::CpuOptions>();
        if (cpu_options) {
          auto path = cpu_options->GetXNNPackWeightCachePath();
          if (path) {
            captured_cache_path = std::string(*path);
          }
        }
        if (settings.backend == OmniEngine::Settings::Backend::kGpu) {
          auto gpu_options = options.GetOptions<::litert::GpuOptions>();
          if (gpu_options) {
            const char* key = nullptr;
            if (LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
                    &key, gpu_options->Get()) == kLiteRtStatusOk &&
                key != nullptr) {
              captured_gpu_cache_key = key;
            }
            const char* dir = nullptr;
            if (LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
                    &dir, gpu_options->Get()) == kLiteRtStatusOk &&
                dir != nullptr) {
              captured_gpu_serialization_dir = dir;
            }
          }
        }
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("weight-cache-test-model");
  };

  // Caches are only configured for an existing model file.
  const std::filesystem::path root =
      std::filesystem::path(::testing::TempDir()) / "omni_registry_cache";
  std::filesystem::remove_all(root);
  const std::filesystem::path model_dir = root / "models";
  std::filesystem::create_directories(model_dir);
  const std::string model_path = (model_dir / "model.tflite").string();
  std::ofstream(model_path) << "model";
  const std::string cache_dir = (root / "cache").string();

  // An explicit cache directory is created and holds a model-specific cache.
  ASSERT_OK(registry.CreateSessionFactory(
      "weight-cache-test-model",
      OmniEngine::Settings{.cache_dir = cache_dir, .model_path = model_path}));
  EXPECT_THAT(captured_cache_path,
              StartsWith(cache_dir + "/weight-cache-test-model_"));
  EXPECT_THAT(captured_cache_path, EndsWith(".xnnpack_cache"));

  // On GPU, programs and GPU-layout weights are serialized under the same
  // directory with the same model-specific key.
  ASSERT_OK(registry.CreateSessionFactory(
      "weight-cache-test-model",
      OmniEngine::Settings{.backend = OmniEngine::Settings::Backend::kGpu,
                           .cache_dir = cache_dir,
                           .model_path = model_path}));
  EXPECT_EQ(captured_gpu_serialization_dir, cache_dir);
  EXPECT_THAT(captured_gpu_cache_key, StartsWith("weight-cache-test-model_"));

  // An empty cache directory puts caches next to the model file.
  ASSERT_OK(registry.CreateSessionFactory(
      "weight-cache-test-model",
      OmniEngine::Settings{.model_path = model_path}));
  EXPECT_THAT(captured_cache_path,
              StartsWith(model_dir.string() + "/weight-cache-test-model_"));

  // An explicit cache directory that cannot be created is an error.
  EXPECT_THAT(registry.CreateSessionFactory(
                  "weight-cache-test-model",
                  OmniEngine::Settings{.cache_dir = "/non/existent/cache/dir",
                                       .model_path = model_path}),
              StatusIs(absl::StatusCode::kPermissionDenied));

  // A read-only cache directory that holds the model file (e.g. pre-installed
  // models) loads without caches; one without the model is still an error.
  const std::filesystem::path read_only_dir = root / "read_only";
  std::filesystem::create_directories(read_only_dir);
  const std::string read_only_model = (read_only_dir / "model.tflite").string();
  std::ofstream(read_only_model) << "model";
  std::filesystem::permissions(read_only_dir,
                               std::filesystem::perms::owner_write |
                                   std::filesystem::perms::group_write |
                                   std::filesystem::perms::others_write,
                               std::filesystem::perm_options::remove);
  if (!lm::IsDirectoryWritable(read_only_dir.string())) {  // Not as root.
    ASSERT_OK(registry.CreateSessionFactory(
        "weight-cache-test-model",
        OmniEngine::Settings{.cache_dir = read_only_dir.string(),
                             .model_path = read_only_model}));
    EXPECT_THAT(captured_cache_path, IsEmpty());
    EXPECT_THAT(registry.CreateSessionFactory(
                    "weight-cache-test-model",
                    OmniEngine::Settings{.cache_dir = read_only_dir.string(),
                                         .model_path = model_path}),
                StatusIs(absl::StatusCode::kPermissionDenied));
  }
  std::filesystem::permissions(read_only_dir,
                               std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);

  // ":nocache" disables the caches but still resolves the model file under the
  // default download directory.
  ASSERT_OK(registry.CreateSessionFactory(
      "weight-cache-test-model",
      OmniEngine::Settings{.cache_dir = ":nocache"}));
  EXPECT_THAT(captured_cache_path, IsEmpty());
  EXPECT_THAT(captured_model_path, StartsWith("/tmp/omni_models/"));
  ASSERT_OK(registry.CreateSessionFactory(
      "weight-cache-test-model",
      OmniEngine::Settings{.backend = OmniEngine::Settings::Backend::kGpu,
                           .cache_dir = ":nocache",
                           .model_path = model_path}));
  EXPECT_THAT(captured_gpu_cache_key, IsEmpty());
  EXPECT_THAT(captured_gpu_serialization_dir, IsEmpty());
}

TEST(OmniRegistryTest, DeletesOnlyStaleCachesOfTheSameModel) {
  auto& registry = OmniRegistry::Get();
  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"stale-cache-test-model"},
      .get_session_factory =
          [](absl::string_view, const OmniEngine::Settings&,
             const MessageLite* absl_nullable, const MessageLite* absl_nullable,
             ::litert::Environment&, ::litert::Options&,
             std::shared_ptr<lm::ModelResources> absl_nullable)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("stale-cache-test-model");
  };
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "omni_registry_stale";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string model_path = (dir / "model.tflite").string();
  std::ofstream(model_path) << "model";
  const auto touch = [&](absl::string_view name) {
    std::ofstream((dir / std::string(name)).string()) << "x";
  };
  // Built from an older version of the same model file: deleted.
  touch("stale-cache-test-model_1_2.xnnpack_cache");
  // Another model whose name starts with "stale-cache-test-model_": kept.
  touch("stale-cache-test-model_v2_1_2.xnnpack_cache");
  // Unrelated file: kept.
  touch("other-model_1_2.xnnpack_cache");

  ASSERT_OK(registry.CreateSessionFactory(
      "stale-cache-test-model",
      OmniEngine::Settings{.cache_dir = dir.string(),
                           .model_path = model_path}));

  EXPECT_FALSE(std::filesystem::exists(
      dir / "stale-cache-test-model_1_2.xnnpack_cache"));
  EXPECT_TRUE(std::filesystem::exists(
      dir / "stale-cache-test-model_v2_1_2.xnnpack_cache"));
  EXPECT_TRUE(std::filesystem::exists(dir / "other-model_1_2.xnnpack_cache"));
}

TEST(OmniRegistryTest, FindModelNameForPathMatchesLongestRegisteredName) {
  auto& registry = OmniRegistry::Get();
  const auto factory =
      [](absl::string_view, const OmniEngine::Settings&,
         const MessageLite* absl_nullable, const MessageLite* absl_nullable,
         ::litert::Environment&, ::litert::Options&,
         std::shared_ptr<lm::ModelResources> absl_nullable)
      -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
    return std::make_unique<FakeSessionFactory>();
  };
  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"find-test-model", "find-test-model-v2"},
      .get_session_factory = factory,
      .session_config_type_id = "Speech2Text",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("find-test-model");
    registry.UnregisterModel("find-test-model-v2");
  };

  EXPECT_THAT(registry.FindModelNameForPath("/x/Find_Test_Model_int8.litertlm"),
              IsOkAndHolds("find-test-model"));
  EXPECT_THAT(
      registry.FindModelNameForPath("/x/find_test_model_v2_int8_gpu.litertlm"),
      IsOkAndHolds("find-test-model-v2"));
  EXPECT_THAT(registry.FindModelNameForPath("/x/unrelated.litertlm"),
              StatusIs(absl::StatusCode::kNotFound));
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
              const MessageLite* /*metadata*/, ::litert::Environment& /*env*/,
              ::litert::Options& /*options*/,
              std::shared_ptr<lm::ModelResources> /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        resolved_name = std::string(model_name);
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
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
                                const MessageLite*, const MessageLite*,
                                ::litert::Environment&, ::litert::Options&,
                                std::shared_ptr<lm::ModelResources>)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("proto-type-check-model");
  };

  lm::proto::Text2SpeechSessionConfig wrong_session_config;
  EXPECT_THAT(registry.CreateSessionFactory("proto-type-check-model", {},
                                            &wrong_session_config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("litert.lm.proto.Speech2TextSessionConfig")));
}

TEST(OmniRegistryTest,
     MergesEmbeddedSpeech2TextMetadataAndReplacesDecodeStartTokens) {
  auto& registry = OmniRegistry::Get();
  lm::proto::Speech2TextMetadata captured_metadata;

  ASSERT_OK(registry.RegisterSessionFactory(RegistrySpec{
      .model_names = {"asr-metadata-merge-test-model"},
      .get_session_factory =
          [&](absl::string_view /*model_name*/,
              const OmniEngine::Settings& /*settings*/,
              const MessageLite* /*session_cfg*/,
              const MessageLite* absl_nullable metadata,
              ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
              std::shared_ptr<lm::ModelResources> /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        if (metadata != nullptr) {
          captured_metadata.CheckTypeAndMergeFrom(*metadata);
        }
        return std::make_unique<FakeSessionFactory>();
      },
      .session_config_type_id = "Speech2Text",
  }));
  absl::Cleanup cleanup = [&] {
    registry.UnregisterModel("asr-metadata-merge-test-model");
  };

  const std::string test_model_path =
      absl::StrCat(::testing::TempDir(), "/asr_metadata_merge_test.litertlm");
  lm::proto::Speech2TextMetadata embedded_metadata;
  embedded_metadata.set_input_milliseconds(30000);
  embedded_metadata.set_decode_stop_token_id(50257);
  embedded_metadata.add_decode_start_token_ids(50258);
  embedded_metadata.add_decode_start_token_ids(50259);
  WriteTestLitertLmWithSpeech2TextMetadata(test_model_path, embedded_metadata);

  OmniEngine::Settings settings{.model_path = test_model_path};

  // 1. Caller provides custom decode_start_token_ids: should replace embedded
  // decode_start_token_ids rather than appending, while preserving embedded
  // scalar fields not set by caller.
  lm::proto::Speech2TextMetadata caller_metadata;
  caller_metadata.add_decode_start_token_ids(999);
  ASSERT_OK_AND_ASSIGN(auto factory1,
                       registry.CreateSessionFactory(
                           "asr-metadata-merge-test-model", settings,
                           /*session_config=*/nullptr, &caller_metadata));
  EXPECT_THAT(factory1, NotNull());
  EXPECT_EQ(captured_metadata.input_milliseconds(), 30000);
  EXPECT_EQ(captured_metadata.decode_stop_token_id(), 50257);
  EXPECT_THAT(captured_metadata.decode_start_token_ids(), ElementsAre(999));

  // 2. Caller omits decode_start_token_ids: embedded decode_start_token_ids
  // should be preserved.
  captured_metadata.Clear();
  lm::proto::Speech2TextMetadata caller_metadata_no_tokens;
  caller_metadata_no_tokens.set_input_milliseconds(15000);
  ASSERT_OK_AND_ASSIGN(
      auto factory2,
      registry.CreateSessionFactory("asr-metadata-merge-test-model", settings,
                                    /*session_config=*/nullptr,
                                    &caller_metadata_no_tokens));
  EXPECT_THAT(factory2, NotNull());
  EXPECT_EQ(captured_metadata.input_milliseconds(), 15000);
  EXPECT_EQ(captured_metadata.decode_stop_token_id(), 50257);
  EXPECT_THAT(captured_metadata.decode_start_token_ids(),
              ElementsAre(50258, 50259));
}

}  // namespace
}  // namespace litert::omni
