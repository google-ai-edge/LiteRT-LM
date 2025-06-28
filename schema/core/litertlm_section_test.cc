#include "schema/core/litertlm_section.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm::schema {
namespace {

TEST(LiteRTLMSectionTest, TestFileBackedSectionStream) {
  // Get the file path
  const auto file_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/attention.tflite";
  const auto output_file_path =
      std::filesystem::path(::testing::TempDir()) / "attention_copy.tflite";

  // Define an output file path
  std::ofstream output_file(output_file_path, std::ios::binary);
  EXPECT_TRUE(output_file.is_open());

  // Create the file-backed Section stream object
  FileBackedSectionStream fbss(file_path.string());
  absl::Status result = fbss.Prepare();
  ASSERT_TRUE(result.ok());

  size_t fbss_size = fbss.BufferSize();
  auto& fbss_stream = fbss.GetStream();
  output_file << fbss_stream.rdbuf();
  EXPECT_EQ(output_file.tellp(), fbss_size);

  output_file.close();

  // Read the file back and check contents.
  std::ifstream input_file(output_file_path, std::ios::binary);
  ASSERT_TRUE(input_file.is_open());

  // Read the original file into a buffer
  std::ifstream original_file(file_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(original_file.is_open());
  size_t original_size = original_file.tellg();
  original_file.seekg(0, std::ios::beg);
  std::vector<char> original_buffer(original_size);
  original_file.read(original_buffer.data(), original_size);
  original_file.close();

  // Read the copied file into a buffer
  std::vector<char> copied_buffer(fbss_size);
  input_file.read(copied_buffer.data(), fbss_size);
  input_file.close();

  // Compare the buffers
  ASSERT_EQ(original_size, fbss_size);
  EXPECT_TRUE(std::equal(original_buffer.begin(), original_buffer.end(),
                         copied_buffer.begin(), copied_buffer.end()));
}

TEST(LiteRTLMSectionTest, TestProtoSectionStream) {
  using litert::lm::proto::LlmMetadata;

  // Constants for the Token Generation Data
  const std::string start_token = "<start>";
  const std::vector<std::string> stop_tokens = {"<stop>", "<eos>"};
  const auto output_file_path =
      std::filesystem::path(::testing::TempDir()) / "llm_metadata.pb";

  // Create an LlmMetadata protocol buffer
  LlmMetadata metadata;

  // Set the start_token
  metadata.mutable_start_token()->set_token_str(start_token);

  // Set the stop_tokens
  for (const std::string& stop_token : stop_tokens) {
    metadata.add_stop_tokens()->set_token_str(stop_token);
  }

  // ** Write the file using typical protobuf serialization **
  std::string serialized_params = metadata.SerializeAsString();
  // Convert the serialized string to a vector of unsigned chars
  std::vector<unsigned char> buffer(serialized_params.begin(),
                                    serialized_params.end());
  // Write the buffer to a file
  std::ofstream output_file(output_file_path, std::ios::binary);
  ASSERT_TRUE(output_file.is_open());
  output_file.write(reinterpret_cast<const char*>(buffer.data()),
                    buffer.size());
  std::ofstream::pos_type bytes_written = output_file.tellp();
  ASSERT_GT(bytes_written, 0);
  output_file.close();

  // ** Write the file using SectionStream interface **
  ProtoBufSectionStream<LlmMetadata> pbss(metadata);
  absl::Status result = pbss.Prepare();
  ASSERT_TRUE(result.ok());

  size_t pbss_size = pbss.BufferSize();
  auto& pbss_stream = pbss.GetStream();
  const auto output_file_streamed_path =
      std::filesystem::path(::testing::TempDir()) / "llm_metadata_streamed.pb";
  std::ofstream output_streamed(output_file_streamed_path, std::ios::binary);
  ASSERT_TRUE(output_streamed.is_open());

  output_streamed << pbss_stream.rdbuf();
  EXPECT_EQ(output_streamed.tellp(), pbss_size);

  output_streamed.close();

  // ** Read the file back in and check the contents **
  std::ifstream input_streamed(output_file_streamed_path, std::ios::binary);
  ASSERT_TRUE(input_streamed.is_open());
  std::stringstream ss;
  ss << input_streamed.rdbuf();  // Read the entire file into a stringstream
  std::string serialized_read_back = ss.str();
  input_streamed.close();

  LlmMetadata params_read_back;
  ASSERT_TRUE(params_read_back.ParseFromString(serialized_read_back));

  // Compare the fields.
  EXPECT_EQ(metadata.start_token().token_str(),
            params_read_back.start_token().token_str());
  EXPECT_EQ(metadata.stop_tokens().size(),
            params_read_back.stop_tokens().size());
  for (size_t i = 0; i < metadata.stop_tokens().size(); ++i) {
    EXPECT_EQ(metadata.stop_tokens(i).token_str(),
              params_read_back.stop_tokens(i).token_str());
  }
}

}  // namespace
}  // namespace litert::lm::schema
