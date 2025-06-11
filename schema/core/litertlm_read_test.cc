#include "schema/core/litertlm_read.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tflite/model_builder.h"  // from @litert

namespace litert {
namespace lm {
namespace schema {
namespace {

TEST(LiteRTLMReadTest, HeaderReadFile) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LitertlmHeader header;

  absl::Status status =
      ReadHeaderFromLiteRTLM(input_filename.string(), &header);

  ASSERT_TRUE(status.ok());
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
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, LlmMetadataRead) {
  using litert::lm::proto::LlmMetadata;
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  LlmMetadata params;
  absl::Status result =
      ReadLlmMetadataFromSection(input_filename.string(), 2, &params);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, TFLiteRead) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result = ReadTFLiteFileFromSection(
      input_filename.string(), 1, &model, &mapped_file);
  ASSERT_TRUE(result.ok());
  // Verify that buffer backing TFLite is still valid and reading data works.
  ASSERT_EQ(model->GetModel()->subgraphs()->size(), 1);
}

TEST(LiteRTLMReadTest, TFLiteReadOwnedAllocation) {
  const std::string input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> model;
  absl::Status result = ReadTFLiteFileFromSection(input_filename, 1, &model);
  ASSERT_TRUE(result.ok());
  // Verify that buffer backing TFLite is still valid and reading data works.
  ASSERT_EQ(model->GetModel()->subgraphs()->size(), 1);
}

TEST(LiteRTLMReadTest, TFLiteReadBinaryData) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::string data;
  absl::Status result =
      ReadBinaryDataFromSection(input_filename.string(), 3, &data);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(data, "Dummy Binary Data Content");
}

TEST(LiteRTLMReadTest, TFLiteReadAny) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result =
      ReadAnyTFLiteFile(input_filename.string(), &tflite_model, &mapped_file);
  ASSERT_TRUE(result.ok());
}

TEST(LiteRTLMReadTest, TFLiteRead_InvalidSection) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::unique_ptr<tflite::FlatBufferModel> tflite_model;
  std::unique_ptr<MemoryMappedFile> mapped_file;
  absl::Status result = ReadTFLiteFileFromSection(
      input_filename.string(), 0, &tflite_model, &mapped_file);
  ASSERT_FALSE(result.ok());
  ASSERT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace schema
}  // namespace lm
}  // namespace litert
