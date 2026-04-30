// Copyright 2025 The ODML Authors.
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

#include "schema/core/litertlm_read.h"

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tflite/model_builder.h"  // from @litert

namespace litert {
namespace lm {
namespace schema {
namespace {

absl::StatusOr<std::string> ReadFileToString(const std::string& filename) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata" /
      filename;

  std::ifstream input_stream(input_filename);
  if (!input_stream.is_open()) {
    return absl::InternalError(
        absl::StrCat("Could not open file: ", input_filename.string()));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return content;
}

TEST(LiteRTLMReadTest, IsLiteRTLMFileValidFile) {
  ASSERT_OK_AND_ASSIGN(std::string content,
                       ReadFileToString("test_tok_tfl_llm.litertlm"));
  EXPECT_TRUE(IsLiteRTLMFile(content));

  std::istringstream stream(content);
  EXPECT_TRUE(IsLiteRTLMFile(stream));
}

TEST(LiteRTLMReadTest, IsLiteRTLMFileInvalidFile) {
  ASSERT_OK_AND_ASSIGN(std::string content,
                       ReadFileToString("attention.tflite"));
  EXPECT_FALSE(IsLiteRTLMFile(content));

  std::istringstream stream(content);
  EXPECT_FALSE(IsLiteRTLMFile(stream));
}

TEST(LiteRTLMReadTest, HeaderReadFile) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LitertlmHeader header;

  absl::Status status =
      ReadHeaderFromLiteRTLM(input_filename.string(), &header);

  ASSERT_OK(status);
  const LiteRTLMMetaData* metadata = header.metadata;
  auto system_metadata = metadata->system_metadata();
  ASSERT_TRUE(!!system_metadata);
  auto entries = system_metadata->entries();
  ASSERT_TRUE(!!entries);         // Ensure entries is not null
  ASSERT_EQ(entries->size(), 2);  // Check the number of key-value pairs.
}

TEST(LiteRTLMReadTest, HeaderReadIstream) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LitertlmHeader header;

  std::ifstream input_file_stream(input_filename, std::ios::binary);
  ASSERT_TRUE(input_file_stream.is_open());
  absl::Status status = ReadHeaderFromLiteRTLM(input_file_stream, &header);
  ASSERT_TRUE(status.ok());
  const LiteRTLMMetaData* metadata = header.metadata;
  auto system_metadata = metadata->system_metadata();
  ASSERT_TRUE(!!system_metadata);
  auto entries = system_metadata->entries();
  ASSERT_TRUE(!!entries);         // Ensure entries is not null
  ASSERT_EQ(entries->size(), 2);  // Check the number of key-value pairs.
}

TEST(LiteRTLMReadTest, TokenizerRead) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  sentencepiece::SentencePieceProcessor sp_proc;
  absl::Status result =
      ReadSPTokenizerFromSection(input_filename.string(), 0, &sp_proc);
  ASSERT_OK(result);
}

TEST(LiteRTLMReadTest, LlmMetadataRead) {
  using litert::lm::proto::LlmMetadata;
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LlmMetadata params;
  absl::Status result =
      ReadLlmMetadataFromSection(input_filename.string(), 2, &params);
  ASSERT_OK(result);
}

TEST(LiteRTLMReadTest, TFLiteRead) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result = ReadTFLiteFileFromSection(input_filename.string(), 1,
                                                  &model, &mapped_file);
  ASSERT_OK(result);
  // Verify that buffer backing TFLite is still valid and reading data works.
  ASSERT_EQ(model->GetModel()->subgraphs()->size(), 1);
}

// NB: This API is only available on non-Windows platforms.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
TEST(LiteRTLMReadTest, TFLiteReadOwnedAllocation) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> model;
  absl::Status result =
      ReadTFLiteFileFromSection(input_filename.string(), 1, &model);
  ASSERT_TRUE(result.ok());
  // Verify that buffer backing TFLite is still valid and reading data works.
  ASSERT_EQ(model->GetModel()->subgraphs()->size(), 1);
}
#endif

TEST(LiteRTLMReadTest, TFLiteReadBinaryData) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::vector<uint8_t> data;
  absl::Status result =
      ReadBinaryDataFromSection(input_filename.string(), 3, &data);
  ASSERT_OK(result);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(data.data()), data.size()),
            "Dummy Binary Data Content");
}

TEST(LiteRTLMReadTest, TFLiteReadAny) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result =
      ReadAnyTFLiteFile(input_filename.string(), &tflite_model, &mapped_file);
  ASSERT_OK(result);
}

TEST(LiteRTLMReadTest, TFLiteRead_InvalidSection) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result = ReadTFLiteFileFromSection(input_filename.string(), 0,
                                                  &tflite_model, &mapped_file);
  ASSERT_FALSE(result.ok());
  ASSERT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiteRTLMReadTest, TFLiteRead_HFTokenizer) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_hf_tokenizer.litertlm";

  ASSERT_OK_AND_ASSIGN(std::string expected_tokenizer_json,
                       ReadFileToString("tokenizer.json"));

  std::string actual_tokenizer_json;
  absl::Status result =
      ReadAnyHfTokenizerJson(input_filename.string(), &actual_tokenizer_json);
  ASSERT_OK(result);
  EXPECT_EQ(actual_tokenizer_json, expected_tokenizer_json);
}

TEST(LiteRTLMReadTest, DecompressData_InvalidSize) {
  uint64_t huge_size = 2ULL << 30;  // 2 GB
  std::vector<uint8_t> output;
  absl::Status status = DecompressData(
      reinterpret_cast<const uint8_t*>(&huge_size), sizeof(uint64_t), &output);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("exceeds maximum allowed size"));
}

}  // namespace
}  // namespace schema
}  // namespace lm
}  // namespace litert
