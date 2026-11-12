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

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/file_data_stream.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::status::StatusIs;

constexpr char kTestLitertLmPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr char kTestTaskPath[] =
    "litert_lm/runtime/testdata/test_lm_new_metadata.task";

std::string BuildMinimalArtisanTfliteBytes() {
  odml::infra::proto::LlmParameters params;
  params.set_vocab_size(32000);
  params.set_start_token_id(2);
  std::string params_bytes = params.SerializeAsString();

  constexpr uint64_t kHeaderPaddedSize = 1024;
  flatbuffers::FlatBufferBuilder b;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(b),
      tflite::CreateBuffer(b, /*data=*/0, kHeaderPaddedSize,
                           params_bytes.size()),
  };
  std::vector<flatbuffers::Offset<tflite::Metadata>> metadata = {
      tflite::CreateMetadataDirect(b, "odml.infra.proto.LlmParameters", 1),
  };
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
      tflite::CreateSubGraph(b),
  };
  auto model = tflite::CreateModel(
      b, /*version=*/3, /*operator_codes=*/0, b.CreateVector(subgraphs),
      b.CreateString("artisan_test"), b.CreateVector(buffers),
      /*metadata_buffer=*/0, b.CreateVector(metadata));
  tflite::FinishModelBuffer(b, model);
  std::string out(reinterpret_cast<const char*>(b.GetBufferPointer()),
                  b.GetSize());
  out.resize(kHeaderPaddedSize, '\0');
  out.append(params_bytes);
  // Streaming `ModelData::Create(ReadDataFn)` reads a 1 MiB header chunk.
  constexpr size_t kStreamingHeaderChunkBytes = 1024 * 1024;
  if (out.size() < kStreamingHeaderChunkBytes) {
    out.resize(kStreamingHeaderChunkBytes, '\0');
  }
  return out;
}

absl::StatusOr<std::filesystem::path> WriteSyntheticArtisanLitertLm(
    absl::string_view filename) {
  const auto source_path =
      std::filesystem::path(::testing::SrcDir()) / kTestLitertLmPath;
  ABSL_ASSIGN_OR_RETURN(auto mapped,
                        MemoryMappedFile::Create(source_path.string()));
  ABSL_ASSIGN_OR_RETURN(auto loader, LitertLmLoader::Create(std::move(mapped)));
  auto sp_buf = loader->GetSentencePieceTokenizer();
  if (!sp_buf.has_value() || sp_buf->Size() == 0) {
    return absl::InternalError(
        "Failed to read SentencePiece tokenizer from test_lm.litertlm.");
  }
  const std::string sp_bytes(sp_buf->StrView());

  proto::LlmMetadata llm_metadata;
  llm_metadata.mutable_start_token()->mutable_token_ids()->add_ids(2);
  llm_metadata.set_max_num_tokens(128);
  const std::string metadata_bytes = llm_metadata.SerializeAsString();

  const std::string artisan_tflite_bytes = BuildMinimalArtisanTfliteBytes();
  const size_t alignment = MemoryMappedFile::GetOffsetAlignment();
  auto align_up = [alignment](uint64_t offset) -> uint64_t {
    const uint64_t rem = offset % alignment;
    return rem == 0 ? offset : offset + (alignment - rem);
  };

  const uint64_t metadata_start = align_up(4096);
  const uint64_t metadata_end = metadata_start + metadata_bytes.size();
  const uint64_t sp_start = align_up(metadata_end);
  const uint64_t sp_end = sp_start + sp_bytes.size();
  const uint64_t artisan_start = align_up(sp_end);
  const uint64_t artisan_end = artisan_start + artisan_tflite_bytes.size();
  const uint64_t file_end = align_up(artisan_end);

  flatbuffers::FlatBufferBuilder builder(1024);
  auto str_val =
      schema::CreateStringValueDirect(builder, "tf_lite_artisan_text_decoder");
  std::vector<flatbuffers::Offset<schema::KeyValuePair>> items = {
      schema::CreateKeyValuePairDirect(
          builder, "model_type", schema::VData_StringValue, str_val.Union())};
  std::vector<flatbuffers::Offset<schema::SectionObject>> sections = {
      schema::CreateSectionObjectDirect(
          builder, /*items=*/nullptr, metadata_start, metadata_end,
          schema::AnySectionDataType_LlmMetadataProto),
      schema::CreateSectionObjectDirect(
          builder, /*items=*/nullptr, sp_start, sp_end,
          schema::AnySectionDataType_SP_Tokenizer),
      schema::CreateSectionObjectDirect(builder, &items, artisan_start,
                                        artisan_end,
                                        schema::AnySectionDataType_TFLiteModel),
  };
  builder.Finish(schema::CreateLiteRTLMMetaData(
      builder, 0, schema::CreateSectionMetadataDirect(builder, &sections)));

  auto out_path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream file(out_path, std::ios::binary);
  file.write("LITERTLM", 8);
  const uint32_t header_words[4] = {1, 0, 0, 0};
  file.write(reinterpret_cast<const char*>(header_words), sizeof(header_words));
  const uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());

  auto write_section_at = [&file](uint64_t current_pos, uint64_t target_pos,
                                  const std::string& payload) -> uint64_t {
    if (target_pos > current_pos) {
      std::string pad(target_pos - current_pos, '\0');
      file.write(pad.data(), pad.size());
    }
    file.write(payload.data(), payload.size());
    return target_pos + payload.size();
  };

  uint64_t pos = header_end_offset;
  pos = write_section_at(pos, metadata_start, metadata_bytes);
  pos = write_section_at(pos, sp_start, sp_bytes);
  pos = write_section_at(pos, artisan_start, artisan_tflite_bytes);
  if (file_end > pos) {
    std::string tail_pad(file_end - pos, '\0');
    file.write(tail_pad.data(), tail_pad.size());
  }
  file.close();
  return out_path;
}

TEST(EngineGpuArtisanTest, RegistersAdvancedLegacyEngineInDefaultSelection) {
  ASSERT_OK_AND_ASSIGN(auto engine_types,
                       EngineFactory::Instance().ListEngineTypes());
  EXPECT_THAT(engine_types,
              Contains(EngineFactory::EngineType::kAdvancedLegacyTfLite));
  EXPECT_THAT(
      engine_types,
      Contains(EngineFactory::EngineType::kAdvancedLiteRTCompiledModel));
}

TEST(EngineGpuArtisanTest,
     RoutesArtisanLitertLmToGpuArtisanExecutorInMmapMode) {
  ASSERT_OK_AND_ASSIGN(
      const auto model_path,
      WriteSyntheticArtisanLitertLm("synthetic_artisan_mmap.litertlm"));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  // `EngineFactory::CreateDefault` routes `Backend::GPU_ARTISAN` to
  // `EngineAdvancedLegacyImpl::Create`, which loads `LlmMetadata`,
  // `SentencePieceTokenizer`, and `ModelResourcesLitertLmGoogle`, and invokes
  // `LlmGpuArtisanExecutor::Create`, reaching
  // `CreateGpuConfigFromLlmParametersProto` without requiring GPU hardware.
  auto engine_or = EngineFactory::CreateDefault(std::move(settings));
  EXPECT_THAT(engine_or, StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("transformer_parameters")));
}

TEST(EngineGpuArtisanTest,
     RoutesArtisanLitertLmToGpuArtisanExecutorInStreamingMode) {
  ASSERT_OK_AND_ASSIGN(
      const auto model_path,
      WriteSyntheticArtisanLitertLm("synthetic_artisan_stream.litertlm"));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<FileDataStream> stream,
                       FileDataStream::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(stream));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  auto engine_or = EngineFactory::CreateDefault(std::move(settings));
  EXPECT_THAT(engine_or, StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("transformer_parameters")));
}

TEST(EngineGpuArtisanTest, RejectsNonArtisanLitertLmWithoutCrashing) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestLitertLmPath;
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  auto engine_or = EngineFactory::CreateDefault(std::move(settings));
  ASSERT_FALSE(engine_or.ok());
  EXPECT_THAT(std::string(engine_or.status().message()),
              Not(HasSubstr("No available engine for backend: GPU_ARTISAN")));
}

TEST(EngineGpuArtisanTest, RejectsTaskModelForGpuArtisanBackend) {
  const auto task_path =
      std::filesystem::path(::testing::SrcDir()) / kTestTaskPath;
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(task_path.string()));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  auto engine_or = EngineFactory::Create(
      EngineFactory::EngineType::kAdvancedLegacyTfLite, std::move(settings));
#if LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
  EXPECT_THAT(engine_or, StatusIs(absl::StatusCode::kInvalidArgument));
#else
  EXPECT_FALSE(engine_or.ok());
#endif  // LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
}

#if LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
TEST(EngineGpuArtisanTest, RejectsLegacyCpuBackendWhenExcluded) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestLitertLmPath;
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings settings,
      EngineSettings::CreateDefault(model_assets, Backend::CPU));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  auto engine_or = EngineFactory::Create(
      EngineFactory::EngineType::kAdvancedLegacyTfLite, std::move(settings));
  EXPECT_THAT(engine_or, StatusIs(absl::StatusCode::kInvalidArgument));
}
#endif  // LITERT_LM_EXCLUDE_LEGACY_EXECUTORS

}  // namespace
}  // namespace litert::lm
