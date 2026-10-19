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

#include "omni/text2speech/kokoro/kokoro_factory.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2speech/kokoro/espeak_assets.h"
#include "omni/text2speech/kokoro/kokoro_model_config.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

namespace schema = ::litert::lm::schema;

using ::testing::ElementsAre;

// ModelResources stub that only reports GenericBinaryData section names.
class FakeSectionNames : public lm::ModelResources {
 public:
  explicit FakeSectionNames(std::vector<std::string> names)
      : names_(std::move(names)) {}

  std::vector<std::string> GetGenericBinaryDataNames() const override {
    return names_;
  }

  // Unused parts of the interface.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  std::vector<std::string> names_;
};

TEST(KokoroFactoryTest, InitKokoroResourcesFailsWithNonexistentModels) {
  KokoroModelConfig config;
  config.acoustic_file = "nonexistent_acoustic.tflite";
  config.vocoder_file = "nonexistent_vocoder.tflite";

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ModelResources resources(shared_env);

  auto status =
      InitKokoroResources(config, "/tmp/invalid_path", "", lm::Backend::CPU, 1,
                          *shared_env, resources);
  EXPECT_FALSE(status.ok());
}

TEST(KokoroFactoryTest, CreateKokoroComponentsRejectsMissingModels) {
  KokoroModelConfig config;
  config.target_bucket = 128;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  TextChunkConfig chunk_config;
  chunk_config.max_buffer_size = 0;

  // Since ModelResources does not have compiled models loaded,
  // CreateKokoroComponents should propagate the error.
  std::vector<std::unique_ptr<internal::StageBase>> stages;
  Stage<Output>* output_stage = nullptr;
  auto status = CreateKokoroComponents(
      config, "/tmp", std::make_unique<StreamTextSource>(chunk_config),
      resources, stages, &output_stage);
  EXPECT_FALSE(status.ok());
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesSkipsNonVoiceSections) {
  FakeSectionNames lm_resources(
      {"af_heart", std::string(kokoro::kEspeakNgSectionName), "ja-lexicon",
       "zh-lexicon", "zh-textnorm", "ef_dora.bin"});

  EXPECT_THAT(GetAvailableKokoroVoices(/*model_folder=*/"", &lm_resources),
              ElementsAre("af_heart", "ef_dora"));
}

// Writes a minimal `.litertlm` file whose only payload is one
// GenericBinaryData section per name in `section_names`.
std::string WriteLitertLmWithSections(
    const std::string& path, const std::vector<std::string>& section_names) {
  flatbuffers::FlatBufferBuilder builder(1024);
  std::vector<flatbuffers::Offset<schema::SectionObject>> sections;
  uint64_t offset = 0;
  for (const std::string& name : section_names) {
    auto kv = schema::CreateKeyValuePair(builder, "name", name);
    auto items = builder.CreateVector(
        std::vector<flatbuffers::Offset<schema::KeyValuePair>>{kv});
    sections.push_back(schema::CreateSectionObject(
        builder, items, offset, offset + 16,
        schema::AnySectionDataType_GenericBinaryData));
    offset += 16;
  }
  auto metadata = schema::CreateLiteRTLMMetaData(
      builder, 0,
      schema::CreateSectionMetadata(builder, builder.CreateVector(sections)));
  builder.Finish(metadata);

  std::ofstream file(path, std::ios::binary);
  file.write("LITERTLM", 8);
  const uint32_t versions[4] = {1, 0, 0, 0};  // major, minor, patch, padding
  file.write(reinterpret_cast<const char*>(versions), sizeof(versions));
  const uint64_t header_end = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end), sizeof(header_end));
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  const std::string payload(offset + 16, 'a');
  file.write(payload.data(), payload.size());
  return path;
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesReadsLitertLmBundle) {
  const std::string bundle = WriteLitertLmWithSections(
      (std::filesystem::path(::testing::TempDir()) / "kokoro_voices.litertlm")
          .string(),
      {"zf_xiaobei", std::string(kokoro::kEspeakNgSectionName), "af_heart",
       "zh-lexicon"});

  // Only the bundle path is given, as `TextToSpeech::ListVoices()` does.
  EXPECT_THAT(GetAvailableKokoroVoices(bundle),
              ElementsAre("af_heart", "zf_xiaobei"));
  // A directory containing the bundle works too.
  EXPECT_THAT(GetAvailableKokoroVoices(
                  std::filesystem::path(bundle).parent_path().string()),
              ElementsAre("af_heart", "zf_xiaobei"));
}

}  // namespace
}  // namespace litert::omni::tts
