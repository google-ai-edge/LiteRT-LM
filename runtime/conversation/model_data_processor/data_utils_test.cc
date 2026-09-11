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

#include "runtime/conversation/model_data_processor/data_utils.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

void WriteFile(absl::string_view path, absl::string_view contents) {
  std::ofstream ofstr(std::string(path), std::ios::out);
  ofstr << contents;
}

TEST(DataUtilsTest, LoadItemData_TextItem) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "text"},
                           {"text", "some text"},
                       }));
  EXPECT_EQ(std::string(static_cast<const char*>(memory_mapped_file->data()),
                        memory_mapped_file->length()),
            "some text");
}

TEST(DataUtilsTest, LoadItemData_ImageItemWithPath) {
  auto path = std::filesystem::path(::testing::TempDir()) / "test_image.jpg";
  WriteFile(path.string(), "image_contents");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "image"},
                           {"path", path.string()},
                       }));
  EXPECT_EQ(std::string(static_cast<const char*>(memory_mapped_file->data()),
                        memory_mapped_file->length()),
            "image_contents");
}

TEST(DataUtilsTest, LoadItemData_ImageItemWithBlob) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "image"},
                           {"blob", "aW1hZ2VfY29udGVudHM="},
                       }));
  EXPECT_EQ(std::string(static_cast<const char*>(memory_mapped_file->data()),
                        memory_mapped_file->length()),
               "image_contents");
}

TEST(DataUtilsTest, LoadItemData_AudioItemWithPath) {
  auto path = std::filesystem::path(::testing::TempDir()) / "test_audio.wav";
  WriteFile(path.string(), "audio_contents");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "audio"},
                           {"path", path.string()},
                       }));
  EXPECT_EQ(std::string(static_cast<const char*>(memory_mapped_file->data()),
                        memory_mapped_file->length()),
            "audio_contents");
}

TEST(DataUtilsTest, LoadItemData_AudioItemWithBlob) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "audio"},
                           {"blob", "YXVkaW9fY29udGVudHM="},
                       }));
  EXPECT_EQ(std::string(static_cast<const char*>(memory_mapped_file->data()),
                        memory_mapped_file->length()),
            "audio_contents");
}

TEST(DataUtilsTest, LoadItemData_UnsupportedItemType) {
  auto status = LoadItemData({
                                 {"type", "unsupported_type"},
                             })
                    .status();
  EXPECT_THAT(status,
              testing::status::StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Unsupported item type: unsupported_type"));
}

TEST(DataUtilsTest, LoadItemData_InvalidItem) {
  auto status = LoadItemData({
                                 {"type", "image"},
                             })
                    .status();
  EXPECT_THAT(status,
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Audio or image item must contain a path or blob."));
}

TEST(DataUtilsTest, LoadItemData_ImageItemWithInvalidBase64Blob) {
  auto result = LoadItemData({
      {"type", "image"},
      {"blob", "invalid base64"},
  });
  EXPECT_THAT(result.status(),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("Failed to decode base64 blob."));
}

TEST(DataUtilsTest, LoadItemData_ToolResponseItem) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({{"type", "tool_response"},
                                     {"tool_response", "some response"}}));
  EXPECT_EQ(memory_mapped_file, nullptr);
}

TEST(DataUtilsTest, LoadItemData_MissingType) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({{"not_type", "some value"}}));
  EXPECT_EQ(memory_mapped_file, nullptr);
}

TEST(DataUtilsTest, NormalizeContent_String) {
  const ordered_json content = "Hello";
  const ordered_json expected = {{{"type", "text"}, {"text", "Hello"}}};
  EXPECT_EQ(NormalizeContent(content), expected);
}

TEST(DataUtilsTest, NormalizeContent_Object) {
  const ordered_json content = {{"type", "text"}, {"text", "Hello"}};
  const ordered_json expected = {{{"type", "text"}, {"text", "Hello"}}};
  EXPECT_EQ(NormalizeContent(content), expected);
}

TEST(DataUtilsTest, NormalizeContent_Array) {
  const ordered_json content = {{{"type", "text"}, {"text", "Hello"}}};
  EXPECT_EQ(NormalizeContent(content), content);
}

TEST(DataUtilsTest, NormalizeMessageContent_StringContent) {
  const ordered_json msg = {{"role", "user"}, {"content", "Hello"}};
  const ordered_json expected = {
      {"role", "user"}, {"content", {{{"type", "text"}, {"text", "Hello"}}}}};
  EXPECT_EQ(NormalizeMessageContent(msg), expected);
}

TEST(DataUtilsTest, NormalizeMessageContent_ArrayContent) {
  const ordered_json msg = {
      {"role", "user"}, {"content", {{{"type", "text"}, {"text", "Hello"}}}}};
  EXPECT_EQ(NormalizeMessageContent(msg), msg);
}

TEST(DataUtilsTest, NormalizeMessageContent_NoContent) {
  const ordered_json msg = {{"role", "assistant"},
                            {"tool_calls", ordered_json::array()}};
  EXPECT_EQ(NormalizeMessageContent(msg), msg);
}

TEST(DataUtilsTest, NormalizeMessageContent_ObjectContent) {
  const ordered_json msg = {{"role", "user"},
                            {"content", {{"type", "text"}, {"text", "Hello"}}}};
  const ordered_json expected = {
      {"role", "user"}, {"content", {{{"type", "text"}, {"text", "Hello"}}}}};
  EXPECT_EQ(NormalizeMessageContent(msg), expected);
}

}  // namespace
}  // namespace litert::lm
