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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
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

TEST(DataUtilsTest, LoadItemData_ImageItemWithBinaryBlob) {
  std::vector<uint8_t> raw_bytes = {'i', 'm', 'a', 'g', 'e', '_', 'c',
                                    'o', 'n', 't', 'e', 'n', 't', 's'};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "image"},
                           {"blob", ordered_json::binary(std::move(raw_bytes))},
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

TEST(DataUtilsTest, LoadItemData_AudioItemWithBinaryBlob) {
  std::vector<uint8_t> raw_bytes = {'a', 'u', 'd', 'i', 'o', '_', 'c',
                                    'o', 'n', 't', 'e', 'n', 't', 's'};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> memory_mapped_file,
                       LoadItemData({
                           {"type", "audio"},
                           {"blob", ordered_json::binary(std::move(raw_bytes))},
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

TEST(DataUtilsTest, ResponseTextToMessage_WithoutTools) {
  ASSERT_OK_AND_ASSIGN(
      const ordered_json message,
      ResponseTextToMessage("Hello world", /*preface=*/std::nullopt,
                            "```tool_code", "```", SyntaxType::kJson,
                            /*options=*/{}));
  const ordered_json expected = {
      {"role", "assistant"},
      {"content", {{{"type", "text"}, {"text", "Hello world"}}}}};
  EXPECT_EQ(message, expected);
}

TEST(DataUtilsTest, ResponseTextToMessage_WithTools) {
  Preface preface = JsonPreface{
      .tools = ordered_json::array(
          {{{"type", "function"}, {"function", {{"name", "get_weather"}}}}}),
  };
  ASSERT_OK_AND_ASSIGN(
      const ordered_json message,
      ResponseTextToMessage(
          R"(Let me check that.```tool_code
{"name": "get_weather", "arguments": {"city": "Mountain View"}}
```)",
          preface, "```tool_code", "```", SyntaxType::kJson, /*options=*/{}));
  EXPECT_EQ(message["role"], "assistant");
  ASSERT_TRUE(message.contains("content"));
  EXPECT_EQ(message["content"],
            ordered_json::array(
                {{{"type", "text"}, {"text", "Let me check that."}}}));
  ASSERT_TRUE(message.contains("tool_calls"));
  ASSERT_EQ(message["tool_calls"].size(), 1);
  EXPECT_EQ(message["tool_calls"][0]["function"]["name"], "get_weather");
}

TEST(DataUtilsTest, FormatToolCalls_NonArrayReturnsError) {
  auto status = FormatToolCalls(ordered_json::object({{"not", "an array"}}),
                                [](const ordered_json& v) { return v.dump(); })
                    .status();
  EXPECT_THAT(status,
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Tool calls must be an array."));
}

TEST(DataUtilsTest, FormatToolCalls_EmptyArray) {
  ASSERT_OK_AND_ASSIGN(
      const ordered_json formatted,
      FormatToolCalls(ordered_json::array(),
                      [](const ordered_json& v) { return v.dump(); }));
  EXPECT_EQ(formatted, ordered_json::array());
}

TEST(DataUtilsTest, FormatToolCalls_ValidWithObjectArguments) {
  const ordered_json tool_calls = ordered_json::array({
      {
          {"type", "function"},
          {"function",
           {{"name", "get_weather"},
            {"arguments", {{"city", "MTV"}, {"units", "celsius"}}}}},
      },
  });
  ASSERT_OK_AND_ASSIGN(const ordered_json formatted,
                       FormatToolCalls(tool_calls, [](const ordered_json& v) {
                         return absl::StrCat("formatted_",
                                             v.get<std::string>());
                       }));
  const ordered_json expected = ordered_json::array({
      {
          {"type", "function"},
          {"function",
           {{"name", "get_weather"},
            {"arguments",
             {{"city", "formatted_MTV"}, {"units", "formatted_celsius"}}}}},
      },
  });
  EXPECT_EQ(formatted, expected);
}

TEST(DataUtilsTest, FormatToolCalls_NonObjectArgumentsPassedThrough) {
  const ordered_json tool_calls = ordered_json::array({
      {
          {"function",
           {{"name", "custom_fn"}, {"arguments", "raw_arg_string"}}},
      },
  });
  ASSERT_OK_AND_ASSIGN(const ordered_json formatted,
                       FormatToolCalls(tool_calls, [](const ordered_json& v) {
                         return absl::StrCat("formatted_",
                                             v.get<std::string>());
                       }));
  const ordered_json expected = ordered_json::array({
      {
          {"type", "function"},
          {"function",
           {{"name", "custom_fn"}, {"arguments", "raw_arg_string"}}},
      },
  });
  EXPECT_EQ(formatted, expected);
}

TEST(DataUtilsTest, FormatToolCalls_SkipsInvalidEntries) {
  const ordered_json tool_calls = ordered_json::array({
      "not an object",
      ordered_json::object(),
      {{"no_function", true}},
      {{"function", "not an object"}},
      {{"function", ordered_json::object()}},  // missing name
      {{"function", {{"name", "valid_fn"}}}},  // valid, no arguments
  });
  ASSERT_OK_AND_ASSIGN(const ordered_json formatted,
                       FormatToolCalls(tool_calls, [](const ordered_json& v) {
                         return v.dump();
                       }));
  const ordered_json expected = ordered_json::array({
      {
          {"type", "function"},
          {"function", {{"name", "valid_fn"}}},
      },
  });
  EXPECT_EQ(formatted, expected);
}

TEST(DataUtilsTest, FormatToolCalls_CallbackErrorPropagated) {
  const ordered_json tool_calls = ordered_json::array({
      {
          {"function",
           {{"name", "fail_fn"}, {"arguments", {{"arg1", "val1"}}}}},
      },
  });
  auto status = FormatToolCalls(tool_calls, [](const ordered_json& v) {
                  return absl::InternalError("Formatting failed");
                }).status();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(status.message(), testing::HasSubstr("Formatting failed"));
}

TEST(DataUtilsTest, FormatToolsArray_NonArrayReturnsError) {
  auto status = FormatToolsArray("not an array", [](const ordered_json& t) {
                  return t.dump();
                }).status();
  EXPECT_THAT(status,
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), testing::HasSubstr("Tools must be an array."));
}

TEST(DataUtilsTest, FormatToolsArray_EmptyArray) {
  ASSERT_OK_AND_ASSIGN(
      const ordered_json formatted,
      FormatToolsArray(ordered_json::array(),
                       [](const ordered_json& t) { return t.dump(); }));
  EXPECT_EQ(formatted, ordered_json::array());
}

TEST(DataUtilsTest, FormatToolsArray_ValidArray) {
  const ordered_json tools = ordered_json::array({
      {{"name", "tool_a"}},
      {{"name", "tool_b"}},
  });
  ASSERT_OK_AND_ASSIGN(const ordered_json formatted,
                       FormatToolsArray(tools, [](const ordered_json& t) {
                         return absl::StrCat("decl:",
                                             t["name"].get<std::string>());
                       }));
  const ordered_json expected = ordered_json::array({
      "decl:tool_a",
      "decl:tool_b",
  });
  EXPECT_EQ(formatted, expected);
}

TEST(DataUtilsTest, FormatToolsArray_CallbackErrorPropagated) {
  const ordered_json tools = ordered_json::array({
      {{"name", "tool_a"}},
  });
  auto status = FormatToolsArray(tools, [](const ordered_json& t) {
                  return absl::InternalError("Tool format error");
                }).status();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(status.message(), testing::HasSubstr("Tool format error"));
}

}  // namespace
}  // namespace litert::lm
