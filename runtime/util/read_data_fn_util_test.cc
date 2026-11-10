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

#include "runtime/util/read_data_fn_util.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

using ::testing::_;
using ::testing::Return;

class MockDataStream : public DataStream {
 public:
  MOCK_METHOD(absl::Status, ReadAndPreserve,
              (void* buffer, uint64_t offset, uint64_t size), (override));
  MOCK_METHOD(absl::Status, ReadAndDiscard,
              (void* buffer, uint64_t offset, uint64_t size), (override));
  MOCK_METHOD(absl::Status, Discard, (uint64_t offset, uint64_t size),
              (override));
};

TEST(ReadDataFnUtilTest, MakeReadDataFnKeep) {
  auto mock_data_stream = std::make_shared<MockDataStream>();
  const std::string data = "hello world";
  EXPECT_CALL(*mock_data_stream, ReadAndPreserve(_, 0, 5))
      .WillOnce([&data](void* buffer, uint64_t offset, uint64_t size) {
        memcpy(buffer, data.data() + offset, size);
        return absl::OkStatus();
      });

  auto read_data_fn = MakeReadDataFn(mock_data_stream, data.size());
  void* buffer =
      read_data_fn(0, 5, odml::infra::llm_utils::ModelData::ReadMode::KEEP);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(strncmp(static_cast<char*>(buffer), "hello", 5), 0);
  free(buffer);
}

TEST(ReadDataFnUtilTest, MakeReadDataFnDiscard) {
  auto mock_data_stream = std::make_shared<MockDataStream>();
  const std::string data = "hello world";
  EXPECT_CALL(*mock_data_stream, ReadAndDiscard(_, 6, 5))
      .WillOnce([&data](void* buffer, uint64_t offset, uint64_t size) {
        memcpy(buffer, data.data() + offset, size);
        return absl::OkStatus();
      });

  auto read_data_fn = MakeReadDataFn(mock_data_stream, data.size());
  void* buffer =
      read_data_fn(6, 5, odml::infra::llm_utils::ModelData::ReadMode::DISCARD);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(strncmp(static_cast<char*>(buffer), "world", 5), 0);
  free(buffer);
}

TEST(ReadDataFnUtilTest, MakeReadDataFnDiscardAll) {
  auto mock_data_stream = std::make_shared<MockDataStream>();
  const std::string data = "hello world";
  EXPECT_CALL(*mock_data_stream, Discard(0, data.size()))
      .WillOnce(Return(absl::OkStatus()));

  auto read_data_fn = MakeReadDataFn(mock_data_stream, data.size());
  void* buffer = read_data_fn(
      0, 1, odml::infra::llm_utils::ModelData::ReadMode::DISCARD_ALL);
  ASSERT_NE(buffer, nullptr);
  free(buffer);
}

TEST(ReadDataFnUtilTest, MakeReadDataFnReadError) {
  auto mock_data_stream = std::make_shared<MockDataStream>();
  EXPECT_CALL(*mock_data_stream, ReadAndPreserve(_, 0, 5))
      .WillOnce(Return(absl::InternalError("Read failed")));

  auto read_data_fn = MakeReadDataFn(mock_data_stream, 10);
  void* buffer =
      read_data_fn(0, 5, odml::infra::llm_utils::ModelData::ReadMode::KEEP);
  EXPECT_EQ(buffer, nullptr);
}

}  // namespace
}  // namespace litert::lm
