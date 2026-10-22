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
#include "omni/text2speech/kokoro/common.h"
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

// Byte size of one Kokoro voice pack section.
constexpr size_t kVoicePackBytes = kokoro::kVoicePackSize * sizeof(float);

// A GenericBinaryData section name and its payload size in bytes.
struct SectionSpec {
  std::string name;
  size_t size = kVoicePackBytes;
};

// ModelResources stub that only reports GenericBinaryData sections.
class FakeSections : public lm::ModelResources {
 public:
  explicit FakeSections(const std::vector<SectionSpec>& sections) {
    for (const SectionSpec& section : sections) {
      sections_.emplace_back(section.name, std::string(section.size, '\0'));
    }
  }

  std::vector<std::string> GetGenericBinaryDataNames() const override {
    std::vector<std::string> names;
    names.reserve(sections_.size());
    for (const auto& [name, data] : sections_) {
      names.push_back(name);
    }
    return names;
  }

  absl::StatusOr<absl::string_view> GetGenericBinaryDataBuffer(
      absl::string_view name) override {
    for (const auto& [section_name, data] : sections_) {
      if (section_name == name) return absl::string_view(data);
    }
    return absl::NotFoundError(name);
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
  std::vector<std::pair<std::string, std::string>> sections_;
};

TEST(KokoroFactoryTest, InitKokoroResourcesFailsWithNonexistentModels) {
  KokoroModelConfig config;
  EXPECT_FALSE(config.force_fp32_on_gpu);
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

  for (bool force_fp32_on_gpu : {false, true}) {
    config.force_fp32_on_gpu = force_fp32_on_gpu;
    auto gpu_status =
        InitKokoroResources(config, "/tmp/invalid_path", "", lm::Backend::GPU,
                            1, *shared_env, resources);
    EXPECT_FALSE(gpu_status.ok());
  }
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
  // Every section has the size of a voice pack, so the non-voice sections are
  // rejected by their names alone.
  FakeSections lm_resources({{"af_heart"},
                             {"ja-lexicon"},
                             {"zh-lexicon"},
                             {"zh-textnorm"},
                             {std::string(kTraditionalChineseSectionName)},
                             {"ef_dora.bin"}});

  EXPECT_THAT(GetAvailableKokoroVoices(/*model_folder=*/"", &lm_resources),
              ElementsAre("af_heart", "ef_dora"));
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesRequiresVoicePackSize) {
  FakeSections lm_resources({{"af_heart", /*size=*/16},
                             {"am_adam", kVoicePackBytes + sizeof(float)},
                             {"am_echo", kVoicePackBytes - sizeof(float)},
                             {"bf_emma", kVoicePackBytes}});

  EXPECT_THAT(GetAvailableKokoroVoices(/*model_folder=*/"", &lm_resources),
              ElementsAre("bf_emma"));
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesRequiresVoiceNamePattern) {
  FakeSections lm_resources({{"xf_unknown"},  // Unknown language letter.
                             {"ax_heart"},    // Unknown gender letter.
                             {"af_"},         // Missing name.
                             {"afheart"},     // Missing separator.
                             {"af_he-art"},   // Unexpected character.
                             {"zf_xiaobei"}});

  EXPECT_THAT(GetAvailableKokoroVoices(/*model_folder=*/"", &lm_resources),
              ElementsAre("zf_xiaobei"));
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesFiltersVoiceFiles) {
  const std::filesystem::path model_dir =
      std::filesystem::path(::testing::TempDir()) / "kokoro_voice_files";
  const std::filesystem::path voices_dir = model_dir / "voices";
  std::filesystem::create_directories(voices_dir);
  const auto write_file = [&voices_dir](const std::string& name, size_t size) {
    std::ofstream file(voices_dir / name, std::ios::binary);
    const std::string payload(size, '\0');
    file.write(payload.data(), payload.size());
  };
  write_file("af_heart.bin", kVoicePackBytes);
  write_file("af_short.bin", 16);
  write_file("zh-t2s.bin", kVoicePackBytes);
  write_file("notes.bin", kVoicePackBytes);
  write_file("bf_emma.txt", kVoicePackBytes);

  EXPECT_THAT(GetAvailableKokoroVoices(model_dir.string()),
              ElementsAre("af_heart"));
}

// Writes a minimal `.litertlm` file whose only payload is one
// GenericBinaryData section per entry in `sections`.
std::string WriteLitertLmWithSections(
    const std::string& path, const std::vector<SectionSpec>& sections) {
  flatbuffers::FlatBufferBuilder builder(1024);
  std::vector<flatbuffers::Offset<schema::SectionObject>> section_objects;
  uint64_t offset = 0;
  for (const SectionSpec& section : sections) {
    auto kv = schema::CreateKeyValuePair(builder, "name", section.name);
    auto items = builder.CreateVector(
        std::vector<flatbuffers::Offset<schema::KeyValuePair>>{kv});
    section_objects.push_back(schema::CreateSectionObject(
        builder, items, offset, offset + section.size,
        schema::AnySectionDataType_GenericBinaryData));
    offset += section.size;
  }
  auto metadata = schema::CreateLiteRTLMMetaData(
      builder, 0,
      schema::CreateSectionMetadata(builder,
                                    builder.CreateVector(section_objects)));
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
      {{"zf_xiaobei"},
       {"af_heart"},
       {"zh-lexicon", /*size=*/16},
       {std::string(kTraditionalChineseSectionName), /*size=*/16}});

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
