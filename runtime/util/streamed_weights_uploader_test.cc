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

#include "runtime/util/streamed_weights_uploader.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

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
    discarded_.emplace_back(offset, size);
    return absl::OkStatus();
  }

  const std::vector<std::pair<uint64_t, uint64_t>>& discarded() const {
    return discarded_;
  }

 private:
  std::string data_;
  std::vector<std::pair<uint64_t, uint64_t>> discarded_;
};

TEST(StreamedWeightsUploaderTest, DiscardsSectionWhenDestroyed) {
  auto file_stream = std::make_shared<MemoryDataStream>("header_weights");
  auto weights_stream = file_stream->OpenSubStream(/*offset=*/7, /*size=*/7);
  ASSERT_TRUE(weights_stream.ok());

  {
    StreamedWeightsUploader uploader(*std::move(weights_stream));
    litert::Options options;
    uploader.SetOnOptions(options);
    EXPECT_TRUE(file_stream->discarded().empty());
  }

  // Dropping the last reference to the section's stream discards the section,
  // whether or not compilation read all of it.
  EXPECT_THAT(file_stream->discarded(), ElementsAre(Pair(7, 7)));
}

}  // namespace
}  // namespace litert::lm
