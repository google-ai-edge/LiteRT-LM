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

#include "runtime/conversation/channel_util.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

TEST(ExtractChannelTextTest, EmptyResponses) {
  Responses responses(TaskState::kDone);
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content, IsEmpty());
  EXPECT_THAT(responses.GetTexts(), IsEmpty());
}

TEST(ExtractChannelTextTest, MultipleTextsError) {
  Responses responses(TaskState::kProcessing, {"Text 1", "Text 2"});
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  EXPECT_FALSE(channel_content.ok());
  EXPECT_THAT(channel_content.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
}

TEST(ExtractChannelTextTest, SingleChannelOccurrence) {
  Responses responses(TaskState::kProcessing,
                      {"Hello <think>hmm</think> World!"});
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content,
              UnorderedElementsAre(
                  std::pair<const std::string, std::string>("thought", "hmm")));
  EXPECT_THAT(responses.GetTexts()[0], Eq("Hello  World!"));
}

TEST(ExtractChannelTextTest, MultipleOccurrencesOfSameChannel) {
  Responses responses(TaskState::kProcessing,
                      {"Hello <think>hmm</think> World <think>yeah</think>!"});
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content,
              UnorderedElementsAre(std::pair<const std::string, std::string>(
                  "thought", "hmmyeah")));
  EXPECT_THAT(responses.GetTexts()[0], Eq("Hello  World !"));
}

TEST(ExtractChannelTextTest, MultipleDifferentChannels) {
  Responses responses(TaskState::kProcessing,
                      {"Hello <think>hmm</think> World <joke>lol</joke>!"});
  std::vector<Channel> channels = {
      {"thought", "<think>", "</think>"},
      {"joke", "<joke>", "</joke>"},
  };

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content,
              UnorderedElementsAre(
                  std::pair<const std::string, std::string>("thought", "hmm"),
                  std::pair<const std::string, std::string>("joke", "lol")));
  EXPECT_THAT(responses.GetTexts()[0], Eq("Hello  World !"));
}

TEST(ExtractChannelTextTest, NoChannelFound) {
  Responses responses(TaskState::kProcessing, {"Hello World!"});
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content, IsEmpty());
  EXPECT_THAT(responses.GetTexts()[0], Eq("Hello World!"));
}

TEST(ExtractChannelTextTest, MissingEndDelimiter) {
  Responses responses(TaskState::kProcessing, {"Hello <think>hmm"});
  std::vector<Channel> channels = {{"thought", "<think>", "</think>"}};

  auto channel_content = ExtractChannelContent(channels, responses);
  ASSERT_OK(channel_content);
  EXPECT_THAT(*channel_content,
              UnorderedElementsAre(
                  std::pair<const std::string, std::string>("thought", "hmm")));
  EXPECT_THAT(responses.GetTexts()[0], Eq("Hello "));
}

TEST(InsertChannelContentIntoMessageTest, MessageInsertion) {
  Message message = {{"role", "assistant"}, {"content", "Hello!"}};
  absl::flat_hash_map<std::string, std::string> channel_content = {
      {"thought", "hmm"}};

  InsertChannelContentIntoMessage(channel_content, message);

  EXPECT_THAT(message["channels"]["thought"], Eq("hmm"));
}

}  // namespace
}  // namespace litert::lm
