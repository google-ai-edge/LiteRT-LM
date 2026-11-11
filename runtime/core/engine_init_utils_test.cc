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

#include "runtime/core/engine_init_utils.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_streaming.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/file_data_stream.h"
#include "runtime/util/litert_lm_streaming_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

std::string TestdataPath(absl::string_view filename) {
  return (std::filesystem::path(::testing::SrcDir()) /
          "litert_lm/runtime/testdata" /
          std::string(filename))
      .string();
}

class InMemoryDataStream : public DataStream {
 public:
  explicit InMemoryDataStream(absl::string_view payload) : payload_(payload) {}

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override {
    return ReadAndPreserve(buffer, offset, size);
  }

  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override {
    if (offset + size > payload_.size()) {
      return absl::OutOfRangeError("Read out of bounds");
    }
    std::memcpy(buffer, payload_.data() + offset, size);
    return absl::OkStatus();
  }

  absl::Status Discard(uint64_t offset, uint64_t size) override {
    return absl::OkStatus();
  }

 private:
  std::string payload_;
};

TEST(EngineInitUtilsTest, CopyStreamingModelHintsCopiesConstraintsAndTypes) {
  ASSERT_OK_AND_ASSIGN(
      auto stream, FileDataStream::Create(
                       TestdataPath("test_lm_with_section_hints.litertlm")));
  LitertLmStreamingLoader loader(std::move(stream));
  ASSERT_OK(loader.LoadHeader());

  ModelResourcesStreaming resources;
  CopyStreamingModelHints(loader,
                          {proto::LlmMetadata::TF_LITE_PREFILL_DECODE,
                           proto::LlmMetadata::TF_LITE_VISION_ENCODER},
                          resources);

  EXPECT_EQ(resources.GetTFLiteModelBackendConstraint(
                proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
            "cpu");
  EXPECT_EQ(resources.GetTFLiteModelPreferActivationType(
                proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
            "fp16");
  EXPECT_EQ(resources.GetTFLiteModelBackendConstraint(
                proto::LlmMetadata::TF_LITE_VISION_ENCODER),
            std::nullopt);
}

TEST(EngineInitUtilsTest, CacheStreamedModelSectionStoresBufferAndHints) {
  const std::string fake_model_bytes = "fake_tflite_flatbuffer_bytes";
  flatbuffers::FlatBufferBuilder builder;
  auto offset =
      schema::CreateSectionObject(builder, /*items=*/0, /*begin_offset=*/0,
                                  /*end_offset=*/fake_model_bytes.size(),
                                  schema::AnySectionDataType_TFLiteModel);
  builder.Finish(offset);
  const schema::SectionObject* section_obj =
      flatbuffers::GetRoot<schema::SectionObject>(builder.GetBufferPointer());

  SectionInfo section{
      .section = section_obj,
      .backend_constraint = "gpu",
      .prefer_activation_type = "fp32",
      .data_stream = std::make_shared<InMemoryDataStream>(fake_model_bytes),
  };

  ModelResourcesStreaming resources;
  ASSERT_OK(CacheStreamedModelSection(
      section, proto::LlmMetadata::TF_LITE_VISION_ENCODER, resources));

  EXPECT_EQ(resources.GetTFLiteModelBackendConstraint(
                proto::LlmMetadata::TF_LITE_VISION_ENCODER),
            "gpu");
  EXPECT_EQ(resources.GetTFLiteModelPreferActivationType(
                proto::LlmMetadata::TF_LITE_VISION_ENCODER),
            "fp32");
  ASSERT_OK_AND_ASSIGN(absl::string_view cached_buffer,
                       resources.GetTFLiteModelBuffer(
                           proto::LlmMetadata::TF_LITE_VISION_ENCODER));
  EXPECT_EQ(cached_buffer, fake_model_bytes);
}

#if defined(ENABLE_SENTENCEPIECE_TOKENIZER) && \
    !defined(LITERT_LM_SENTENCEPIECE_DISABLED)
TEST(EngineInitUtilsTest, LoadStreamedSentencePieceTokenizerRecordsBenchmark) {
  ASSERT_OK_AND_ASSIGN(
      auto stream, FileDataStream::Create(TestdataPath("test_lm.litertlm")));
  LitertLmStreamingLoader loader(std::move(stream));

  std::optional<SectionInfo> sp_section;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto section, loader.GetNextSection());
    if (!section.has_value()) {
      break;
    }
    if (section->section->data_type() ==
        schema::AnySectionDataType_SP_Tokenizer) {
      sp_section = std::move(section);
      break;
    }
  }
  ASSERT_TRUE(sp_section.has_value());

  std::optional<BenchmarkInfo> benchmark_info(proto::BenchmarkParams{});
  ASSERT_OK_AND_ASSIGN(auto tokenizer, LoadStreamedSentencePieceTokenizer(
                                           *sp_section, benchmark_info));
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_GT(tokenizer->GetVocabSize(), 0);
  EXPECT_TRUE(benchmark_info->GetInitPhases().contains(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTokenizer))));
}

TEST(EngineInitUtilsTest,
     LoadStreamedSentencePieceTokenizerRejectsInvalidBuffer) {
  const std::string invalid_sp = "not_a_valid_sentencepiece_model";
  flatbuffers::FlatBufferBuilder builder;
  auto offset =
      schema::CreateSectionObject(builder, /*items=*/0, /*begin_offset=*/0,
                                  /*end_offset=*/invalid_sp.size(),
                                  schema::AnySectionDataType_SP_Tokenizer);
  builder.Finish(offset);
  const schema::SectionObject* section_obj =
      flatbuffers::GetRoot<schema::SectionObject>(builder.GetBufferPointer());

  SectionInfo section{
      .section = section_obj,
      .data_stream = std::make_shared<InMemoryDataStream>(invalid_sp),
  };
  std::optional<BenchmarkInfo> benchmark_info;
  EXPECT_FALSE(
      LoadStreamedSentencePieceTokenizer(section, benchmark_info).ok());
}
#else
TEST(EngineInitUtilsTest,
     LoadStreamedSentencePieceTokenizerReturnsUnimplementedWhenDisabled) {
  const std::string dummy_sp = "dummy_sp_bytes";
  flatbuffers::FlatBufferBuilder builder;
  auto offset = schema::CreateSectionObject(
      builder, /*items=*/0, /*begin_offset=*/0,
      /*end_offset=*/dummy_sp.size(), schema::AnySectionDataType_SP_Tokenizer);
  builder.Finish(offset);
  const schema::SectionObject* section_obj =
      flatbuffers::GetRoot<schema::SectionObject>(builder.GetBufferPointer());

  SectionInfo section{
      .section = section_obj,
      .data_stream = std::make_shared<InMemoryDataStream>(dummy_sp),
  };
  std::optional<BenchmarkInfo> benchmark_info;
  EXPECT_THAT(LoadStreamedSentencePieceTokenizer(section, benchmark_info),
              StatusIs(absl::StatusCode::kUnimplemented));
}
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
TEST(EngineInitUtilsTest, LoadStreamedHuggingFaceTokenizerRecordsBenchmark) {
  ASSERT_OK_AND_ASSIGN(
      auto stream,
      FileDataStream::Create(TestdataPath("test_hf_tokenizer.litertlm")));
  LitertLmStreamingLoader loader(std::move(stream));
  ASSERT_OK_AND_ASSIGN(auto section, loader.GetNextSection());
  ASSERT_TRUE(section.has_value());

  std::optional<BenchmarkInfo> benchmark_info(proto::BenchmarkParams{});
  ASSERT_OK_AND_ASSIGN(auto tokenizer, LoadStreamedHuggingFaceTokenizer(
                                           *section, benchmark_info));
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_TRUE(benchmark_info->GetInitPhases().contains(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTokenizer))));
}
#else
TEST(EngineInitUtilsTest,
     LoadStreamedHuggingFaceTokenizerReturnsUnimplementedWhenDisabled) {
  ASSERT_OK_AND_ASSIGN(
      auto stream,
      FileDataStream::Create(TestdataPath("test_hf_tokenizer.litertlm")));
  LitertLmStreamingLoader loader(std::move(stream));
  ASSERT_OK_AND_ASSIGN(auto section, loader.GetNextSection());
  ASSERT_TRUE(section.has_value());

  std::optional<BenchmarkInfo> benchmark_info;
  EXPECT_THAT(LoadStreamedHuggingFaceTokenizer(*section, benchmark_info),
              StatusIs(absl::StatusCode::kUnimplemented));
}
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

TEST(EngineInitUtilsTest, LoadStreamedHuggingFaceTokenizerRejectsInvalidZlib) {
  const std::string invalid_zlib = "not_valid_zlib_stream";
  flatbuffers::FlatBufferBuilder builder;
  auto offset =
      schema::CreateSectionObject(builder, /*items=*/0, /*begin_offset=*/0,
                                  /*end_offset=*/invalid_zlib.size(),
                                  schema::AnySectionDataType_HF_Tokenizer_Zlib);
  builder.Finish(offset);
  const schema::SectionObject* section_obj =
      flatbuffers::GetRoot<schema::SectionObject>(builder.GetBufferPointer());

  SectionInfo section{
      .section = section_obj,
      .data_stream = std::make_shared<InMemoryDataStream>(invalid_zlib),
  };
  std::optional<BenchmarkInfo> benchmark_info;
  EXPECT_THAT(LoadStreamedHuggingFaceTokenizer(section, benchmark_info),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineInitUtilsTest,
     ExtractModalityExecutorSettingsCopiesConfiguredModalities) {
  ASSERT_OK_AND_ASSIGN(ModelAssets assets, ModelAssets::Create("dummy_path"));
  ASSERT_OK_AND_ASSIGN(EngineSettings text_only_settings,
                       EngineSettings::CreateDefault(assets, Backend::CPU));
  ModalityExecutorSettings empty_modalities =
      ExtractModalityExecutorSettings(text_only_settings);
  EXPECT_EQ(empty_modalities.vision, nullptr);
  EXPECT_EQ(empty_modalities.audio, nullptr);

  ASSERT_OK_AND_ASSIGN(
      EngineSettings multimodal_settings,
      EngineSettings::CreateDefault(assets, Backend::CPU,
                                    /*vision_backend=*/Backend::GPU,
                                    /*audio_backend=*/Backend::CPU));
  ASSERT_OK(
      multimodal_settings.GetMutableVisionExecutorSettings()->SetAdapterBackend(
          Backend::GPU));
  ModalityExecutorSettings configured_modalities =
      ExtractModalityExecutorSettings(multimodal_settings);
  ASSERT_NE(configured_modalities.vision, nullptr);
  EXPECT_EQ(configured_modalities.vision->GetBackend(), Backend::GPU);
  EXPECT_EQ(configured_modalities.vision->GetAdapterBackend(), Backend::GPU);
  ASSERT_NE(configured_modalities.audio, nullptr);
  EXPECT_EQ(configured_modalities.audio->GetBackend(), Backend::CPU);
  // Original settings remain populated so SessionConfig validation knows which
  // modalities are enabled on the engine.
  EXPECT_TRUE(multimodal_settings.GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(multimodal_settings.GetAudioExecutorSettings().has_value());
}

TEST(EngineInitUtilsTest,
     InitializeStreamedModalitiesResetsSettingsWhenModelsMissing) {
  ASSERT_OK_AND_ASSIGN(
      ModelAssets assets,
      ModelAssets::Create(TestdataPath("test_lm_new_metadata.task")));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(assets, Backend::CPU,
                                    /*vision_backend=*/Backend::CPU,
                                    /*audio_backend=*/Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OwnedEnvironment> owned_env,
      CreateOwnedEnvironment(settings, /*model_resources=*/nullptr));

  ModelResourcesStreaming empty_resources;
  ASSERT_OK_AND_ASSIGN(StreamedVisionExecutorInit vision_init,
                       InitializeStreamedVisionExecutor(
                           settings, empty_resources,
                           /*llm_metadata=*/nullptr, owned_env->env));
  EXPECT_EQ(vision_init.settings, nullptr);
  EXPECT_EQ(vision_init.executor, nullptr);
  EXPECT_FALSE(settings.GetVisionExecutorSettings().has_value());

  ASSERT_OK_AND_ASSIGN(StreamedAudioExecutorInit audio_init,
                       InitializeStreamedAudioExecutor(
                           settings, empty_resources, owned_env->env));
  EXPECT_EQ(audio_init.settings, nullptr);
  EXPECT_EQ(audio_init.executor, nullptr);
  EXPECT_FALSE(settings.GetAudioExecutorSettings().has_value());
  // Re-running after settings were cleared exercises the disabled-modality
  // early-return path.
  ASSERT_OK_AND_ASSIGN(StreamedVisionExecutorInit disabled_vision_init,
                       InitializeStreamedVisionExecutor(
                           settings, empty_resources,
                           /*llm_metadata=*/nullptr, owned_env->env));
  EXPECT_EQ(disabled_vision_init.settings, nullptr);
  EXPECT_EQ(disabled_vision_init.executor, nullptr);

  ASSERT_OK_AND_ASSIGN(StreamedAudioExecutorInit disabled_audio_init,
                       InitializeStreamedAudioExecutor(
                           settings, empty_resources, owned_env->env));
  EXPECT_EQ(disabled_audio_init.settings, nullptr);
  EXPECT_EQ(disabled_audio_init.executor, nullptr);
}

absl::Status LoadStreamedBundleIntoResources(
    absl::string_view filename, ModelResourcesStreaming& resources) {
  ABSL_ASSIGN_OR_RETURN(auto stream,
                        FileDataStream::Create(TestdataPath(filename)));
  LitertLmStreamingLoader loader(std::move(stream));
  while (true) {
    ABSL_ASSIGN_OR_RETURN(auto section, loader.GetNextSection());
    if (!section.has_value()) {
      break;
    }
    if (section->section->data_type() ==
            schema::AnySectionDataType_TFLiteModel &&
        section->buffer_key.model_type.has_value()) {
      ABSL_ASSIGN_OR_RETURN(
          const proto::LlmMetadata::TfLiteModelType model_type,
          ParseLlmTfLiteModelType(*section->buffer_key.model_type));
      ABSL_RETURN_IF_ERROR(
          CacheStreamedModelSection(*section, model_type, resources));
    } else if (section->section->data_type() ==
                   schema::AnySectionDataType_TFLiteWeights &&
               section->buffer_key.model_type.has_value()) {
      ABSL_ASSIGN_OR_RETURN(
          const proto::LlmMetadata::TfLiteModelType model_type,
          ParseLlmTfLiteModelType(*section->buffer_key.model_type));
      ABSL_RETURN_IF_ERROR(resources.SetWeightsFromStream(
          model_type, *section->data_stream, section->Size()));
    } else {
      ABSL_RETURN_IF_ERROR(section->Discard());
    }
  }
  return absl::OkStatus();
}

TEST(EngineInitUtilsTest,
     InitializeStreamedVisionExecutorCreatesExecutorFromStreamedBundle) {
  ASSERT_OK_AND_ASSIGN(
      ModelAssets assets,
      ModelAssets::Create(TestdataPath("dummy_vision_with_adapter.litertlm")));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(assets, Backend::CPU,
                                    /*vision_backend=*/Backend::CPU,
                                    /*audio_backend=*/std::nullopt));
  settings.GetMutableVisionExecutorSettings()->SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OwnedEnvironment> owned_env,
      CreateOwnedEnvironment(settings, /*model_resources=*/nullptr));

  ModelResourcesStreaming resources;
  ASSERT_OK(LoadStreamedBundleIntoResources(
      "dummy_vision_with_adapter.litertlm", resources));

  ASSERT_OK_AND_ASSIGN(StreamedVisionExecutorInit vision_init,
                       InitializeStreamedVisionExecutor(
                           settings, resources,
                           /*llm_metadata=*/nullptr, owned_env->env));
  EXPECT_NE(vision_init.settings, nullptr);
  EXPECT_NE(vision_init.executor, nullptr);
  EXPECT_TRUE(settings.GetVisionExecutorSettings().has_value());

  // Also exercise the non-CPU adapter backend warning path.
  ASSERT_OK(settings.GetMutableVisionExecutorSettings()->SetAdapterBackend(
      Backend::GPU));
  (void)InitializeStreamedVisionExecutor(settings, resources,
                                         /*llm_metadata=*/nullptr,
                                         owned_env->env);
}

TEST(EngineInitUtilsTest,
     InitializeStreamedAudioExecutorCreatesExecutorFromStreamedBundle) {
  ASSERT_OK_AND_ASSIGN(
      ModelAssets assets,
      ModelAssets::Create(TestdataPath("dummy_audio_only.litertlm")));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(assets, Backend::CPU,
                                    /*vision_backend=*/std::nullopt,
                                    /*audio_backend=*/Backend::CPU));
  settings.GetMutableAudioExecutorSettings()->SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OwnedEnvironment> owned_env,
      CreateOwnedEnvironment(settings, /*model_resources=*/nullptr));

  ModelResourcesStreaming resources;
  ASSERT_OK(
      LoadStreamedBundleIntoResources("dummy_audio_only.litertlm", resources));

  ASSERT_OK_AND_ASSIGN(
      StreamedAudioExecutorInit audio_init,
      InitializeStreamedAudioExecutor(settings, resources, owned_env->env));
  EXPECT_NE(audio_init.settings, nullptr);
  EXPECT_NE(audio_init.executor, nullptr);
  EXPECT_TRUE(settings.GetAudioExecutorSettings().has_value());
}

}  // namespace
}  // namespace litert::lm
