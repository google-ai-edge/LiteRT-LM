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

#include "runtime/util/streamed_weights_manager.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

class MemoryDataStream : public DataStream {
 public:
  explicit MemoryDataStream(std::string data) : data_(std::move(data)) {}

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override {
    return ReadAndPreserve(buffer, offset, size);
  }

  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override {
    if (offset + size > data_.size()) {
      return absl::OutOfRangeError("Read beyond end of stream");
    }
    std::memcpy(buffer, data_.data() + offset, size);
    return absl::OkStatus();
  }

  absl::Status Discard(uint64_t offset, uint64_t size) override {
    discarded_ = true;
    return absl::OkStatus();
  }

  bool discarded() const { return discarded_; }

 private:
  std::string data_;
  bool discarded_ = false;
};

TEST(StreamedWeightsManagerTest, SetAndGetCurrentlyCompilingModel) {
  SetCurrentlyCompilingModel(ModelType::kTfLiteVisionEncoder);
  EXPECT_EQ(GetCurrentlyCompilingModel(), ModelType::kTfLiteVisionEncoder);

  SetCurrentlyCompilingModel(ModelType::kUnknown);
  EXPECT_EQ(GetCurrentlyCompilingModel(), ModelType::kUnknown);
}

TEST(StreamedWeightsManagerTest, StoreReadAndClearWeightsStream) {
  auto stream = std::make_shared<MemoryDataStream>("hello_weights_data");
  StoreWeightsStream(ModelType::kTfLiteTextEncoder, stream);

  char buffer[6] = {0};
  EXPECT_TRUE(ReadStoredWeights(static_cast<int>(ModelType::kTfLiteTextEncoder),
                                /*offset=*/6, /*size=*/5, buffer)
                  .ok());
  EXPECT_EQ(std::string(buffer, 5), "weigh");

  EXPECT_TRUE(ClearStoredWeightsStream(ModelType::kTfLiteTextEncoder).ok());
  EXPECT_TRUE(stream->discarded());

  EXPECT_EQ(ReadStoredWeights(static_cast<int>(ModelType::kTfLiteTextEncoder),
                              /*offset=*/0, /*size=*/5, buffer)
                .code(),
            absl::StatusCode::kNotFound);
}

TEST(StreamedWeightsManagerTest, ClearAllStoredWeightsStreams) {
  auto stream1 = std::make_shared<MemoryDataStream>("weights_1");
  auto stream2 = std::make_shared<MemoryDataStream>("weights_2");
  StoreWeightsStream(ModelType::kTfLiteVisionEncoder, stream1);
  StoreWeightsStream(ModelType::kTfLiteAudioEncoderHw, stream2);

  EXPECT_TRUE(ClearStoredWeightsStreams().ok());
  EXPECT_TRUE(stream1->discarded());
  EXPECT_TRUE(stream2->discarded());

  char buffer[4] = {0};
  EXPECT_EQ(ReadStoredWeights(static_cast<int>(ModelType::kTfLiteVisionEncoder),
                              /*offset=*/0, /*size=*/4, buffer)
                .code(),
            absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace litert::lm
