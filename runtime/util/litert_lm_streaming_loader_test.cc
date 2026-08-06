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
#include "runtime/util/data_stream.h"
#include "runtime/util/file_data_stream.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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

}  // namespace
}  // namespace litert::lm
