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

#include "runtime/util/litert_lm_streaming_loader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/components/model_resources.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/file_data_stream.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

std::string GetTestModelPath() {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  return model_path.string();
}

TEST(LitertLmStreamingLoaderTest, LoadTestModel) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));

  LitertLmStreamingLoader loader(std::move(stream));

  // The test model should have multiple sections (e.g., Tokenizer, Metadata,
  // TFLiteModel)
  int num_sections = 0;
  while (true) {
    ASSERT_OK_AND_ASSIGN(auto section_opt, loader.GetNextSection());
    if (!section_opt.has_value()) {
      break;
    }
    ++num_sections;

    // The section stream should be accessible and readable
    EXPECT_NE(section_opt->data_stream, nullptr);
    size_t size = section_opt->section->end_offset() -
                  section_opt->section->begin_offset();
    EXPECT_GT(size, 0);
  }

  // Expect at least tokenizer, metadata, and tflite model.
  EXPECT_GE(num_sections, 3);
}

TEST(LitertLmStreamingLoaderTest, GetTfLiteModelSectionHints) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_with_section_hints.litertlm";
  ASSERT_OK_AND_ASSIGN(auto stream,
                       FileDataStream::Create(model_path.string()));

  LitertLmStreamingLoader loader(std::move(stream));
  ASSERT_OK(loader.LoadHeader());

  EXPECT_EQ(loader.GetTFLiteModelPreferActivationType(
                proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
            "fp16");
  EXPECT_EQ(loader.GetTFLiteModelBackendConstraint(
                proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
            "cpu");
}

TEST(LitertLmStreamingLoaderTest, LoadHfTokenizerModel) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  ASSERT_OK_AND_ASSIGN(auto stream,
                       FileDataStream::Create(model_path.string()));

  LitertLmStreamingLoader loader(std::move(stream));
  ASSERT_OK_AND_ASSIGN(auto section_opt, loader.GetNextSection());
  ASSERT_TRUE(section_opt.has_value());
  EXPECT_EQ(section_opt->section->data_type(),
            schema::AnySectionDataType_HF_Tokenizer_Zlib);
}

class MockDataStream : public DataStream {
 public:
  MOCK_METHOD(absl::Status, ReadAndDiscard,
              (void* buffer, uint64_t offset, uint64_t size), (override));
  MOCK_METHOD(absl::Status, ReadAndPreserve,
              (void* buffer, uint64_t offset, uint64_t size), (override));
  MOCK_METHOD(absl::Status, Discard, (uint64_t offset, uint64_t size),
              (override));
};

TEST(LitertLmStreamingLoaderTest, InvalidMagicNumber) {
  auto mock_stream = std::make_shared<MockDataStream>();
  EXPECT_CALL(*mock_stream,
              ReadAndDiscard(testing::_, 0, sizeof(HeaderPreamble)))
      .WillOnce([](void* buffer, uint64_t offset, uint64_t size) {
        HeaderPreamble preamble;
        memcpy(preamble.magic, "INVALID!", 8);
        preamble.header_end_offset = 1024;
        memcpy(buffer, &preamble, sizeof(HeaderPreamble));
        return absl::OkStatus();
      });

  LitertLmStreamingLoader loader(std::move(mock_stream));
  EXPECT_THAT(
      loader.GetNextSection().status(),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Invalid magic number")));
}

TEST(LitertLmStreamingLoaderTest, HeaderTooLarge) {
  auto mock_stream = std::make_shared<MockDataStream>();
  EXPECT_CALL(*mock_stream,
              ReadAndDiscard(testing::_, 0, sizeof(HeaderPreamble)))
      .WillOnce([](void* buffer, uint64_t offset, uint64_t size) {
        HeaderPreamble preamble;
        memcpy(preamble.magic, "LITERTLM", 8);
        preamble.header_end_offset =
            20 * 1024 *
            1024;  // 20MB is larger than kLitertLmHeaderMaxSize (16KB)
        memcpy(buffer, &preamble, sizeof(HeaderPreamble));
        return absl::OkStatus();
      });

  LitertLmStreamingLoader loader(std::move(mock_stream));
  EXPECT_THAT(loader.GetNextSection().status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Header end offset is too large")));
}

TEST(LitertLmStreamingLoaderTest, HeaderTooSmall) {
  auto mock_stream = std::make_shared<MockDataStream>();
  EXPECT_CALL(*mock_stream,
              ReadAndDiscard(testing::_, 0, sizeof(HeaderPreamble)))
      .WillOnce([](void* buffer, uint64_t offset, uint64_t size) {
        HeaderPreamble preamble;
        memcpy(preamble.magic, "LITERTLM", 8);
        preamble.header_end_offset =
            8;  // Smaller than sizeof(HeaderPreamble) which is 32
        memcpy(buffer, &preamble, sizeof(HeaderPreamble));
        return absl::OkStatus();
      });

  LitertLmStreamingLoader loader(std::move(mock_stream));
  EXPECT_THAT(loader.GetNextSection().status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Header end offset is too small")));
}

const schema::SectionObject* BuildSectionObject(
    flatbuffers::FlatBufferBuilder& builder, uint64_t begin_offset,
    uint64_t end_offset,
    schema::AnySectionDataType data_type =
        schema::AnySectionDataType_GenericBinaryData) {
  auto offset = schema::CreateSectionObject(builder, /*items=*/0, begin_offset,
                                            end_offset, data_type);
  builder.Finish(offset);
  return flatbuffers::GetRoot<schema::SectionObject>(
      builder.GetBufferPointer());
}

TEST(SectionInfoTest, ReadBufferAndParseProtoSucceed) {
  proto::LlmMetadata expected;
  expected.set_max_num_tokens(2048);
  const std::string serialized = expected.SerializeAsString();

  flatbuffers::FlatBufferBuilder builder;
  const schema::SectionObject* section_obj = BuildSectionObject(
      builder, /*begin_offset=*/100, /*end_offset=*/100 + serialized.size(),
      schema::AnySectionDataType_LlmMetadataProto);

  auto mock_stream = std::make_shared<NiceMock<MockDataStream>>();
  EXPECT_CALL(*mock_stream, ReadAndDiscard(_, 0, serialized.size()))
      .WillOnce([&serialized](void* buffer, uint64_t, uint64_t size) {
        std::memcpy(buffer, serialized.data(), size);
        return absl::OkStatus();
      });

  SectionInfo section_info{.section = section_obj, .data_stream = mock_stream};
  EXPECT_EQ(section_info.Size(), serialized.size());

  ASSERT_OK_AND_ASSIGN(
      proto::LlmMetadata parsed,
      section_info.ParseProto<proto::LlmMetadata>("LlmMetadata"));
  EXPECT_EQ(parsed.max_num_tokens(), 2048);
}

TEST(SectionInfoTest, EmptySectionSkipsStreamRead) {
  flatbuffers::FlatBufferBuilder builder;
  const schema::SectionObject* section_obj =
      BuildSectionObject(builder, /*begin_offset=*/64, /*end_offset=*/64);

  auto mock_stream = std::make_shared<StrictMock<MockDataStream>>();
  SectionInfo section_info{.section = section_obj, .data_stream = mock_stream};
  EXPECT_EQ(section_info.Size(), 0);

  ASSERT_OK_AND_ASSIGN(auto buffer, section_info.ReadBuffer());
  EXPECT_TRUE(buffer.empty());
}

TEST(SectionInfoTest, DiscardForwardsSectionSize) {
  flatbuffers::FlatBufferBuilder builder;
  const schema::SectionObject* section_obj =
      BuildSectionObject(builder, /*begin_offset=*/32, /*end_offset=*/96);

  auto mock_stream = std::make_shared<NiceMock<MockDataStream>>();
  EXPECT_CALL(*mock_stream, Discard(0, 64)).WillOnce(Return(absl::OkStatus()));

  SectionInfo section_info{.section = section_obj, .data_stream = mock_stream};
  EXPECT_OK(section_info.Discard());
}

TEST(SectionInfoTest, ParseProtoReturnsInternalErrorOnCorruptData) {
  const std::string corrupt_bytes =
      "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01";
  flatbuffers::FlatBufferBuilder builder;
  const schema::SectionObject* section_obj = BuildSectionObject(
      builder, /*begin_offset=*/0, /*end_offset=*/corrupt_bytes.size(),
      schema::AnySectionDataType_LlmMetadataProto);

  auto mock_stream = std::make_shared<NiceMock<MockDataStream>>();
  EXPECT_CALL(*mock_stream, ReadAndDiscard(_, 0, corrupt_bytes.size()))
      .WillOnce([&corrupt_bytes](void* buffer, uint64_t, uint64_t size) {
        std::memcpy(buffer, corrupt_bytes.data(), size);
        return absl::OkStatus();
      });

  SectionInfo section_info{.section = section_obj, .data_stream = mock_stream};
  EXPECT_THAT(section_info.ParseProto<proto::LlmMetadata>("LlmMetadata"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Failed to parse LlmMetadata")));
}

TEST(SectionInfoTest, RejectsNullOrInvertedSectionBounds) {
  SectionInfo null_section;
  EXPECT_EQ(null_section.Size(), 0);
  EXPECT_THAT(null_section.ReadBuffer(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(null_section.Discard(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  flatbuffers::FlatBufferBuilder valid_builder;
  const schema::SectionObject* valid_obj =
      BuildSectionObject(valid_builder, /*begin_offset=*/0, /*end_offset=*/16);
  SectionInfo null_stream_section{.section = valid_obj, .data_stream = nullptr};
  EXPECT_THAT(null_stream_section.ReadBuffer(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("data stream is null")));
  EXPECT_THAT(null_stream_section.Discard(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("data stream is null")));

  flatbuffers::FlatBufferBuilder builder;
  const schema::SectionObject* inverted_obj =
      BuildSectionObject(builder, /*begin_offset=*/200, /*end_offset=*/100);
  auto mock_stream = std::make_shared<NiceMock<MockDataStream>>();
  SectionInfo inverted_section{.section = inverted_obj,
                               .data_stream = mock_stream};
  EXPECT_EQ(inverted_section.Size(), 0);
  EXPECT_THAT(inverted_section.ReadBuffer(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid section range")));
  EXPECT_THAT(inverted_section.Discard(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid section range")));
}

}  // namespace
}  // namespace litert::lm
