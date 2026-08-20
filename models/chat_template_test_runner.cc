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

#include <algorithm>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "google/protobuf/text_format.h"  // from @com_google_protobuf

ABSL_FLAG(std::string, chat_template, "", "Path to Jinja chat template file.");
ABSL_FLAG(std::string, input_dir, "",
          "Path to directory containing input JSON files.");
ABSL_FLAG(std::string, golden_dir, "",
          "Path to directory containing golden TXT files.");
ABSL_FLAG(bool, update_golden, false,
          "Update golden files with rendered output instead of comparing.");
ABSL_FLAG(std::vector<std::string>, pbtext_files, {},
          "Comma-separated list of LlmMetadataProto.pbtext files to verify.");
ABSL_FLAG(std::vector<std::string>, mirror_chat_templates, {},
          "Comma-separated list of mirror chat template files to verify.");

namespace litert::lm {
namespace {

absl::StatusOr<std::string> ExtractJinjaPromptTemplateFromPbtext(
    absl::string_view pbtext_content) {
  ::litert::lm::proto::LlmMetadata metadata;
  if (!google::protobuf::TextFormat::ParseFromString(pbtext_content, &metadata)) {
    return absl::InvalidArgumentError(
        "Failed to parse LlmMetadata pbtext content");
  }
  if (!metadata.has_jinja_prompt_template()) {
    return absl::NotFoundError(
        "jinja_prompt_template field not found in pbtext");
  }
  return metadata.jinja_prompt_template();
}

using json = nlohmann::ordered_json;

absl::StatusOr<std::string> ReadFileContents(
    const std::filesystem::path& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(
        absl::StrCat("Could not open file: ", path.string()));
  }
  return std::string((std::istreambuf_iterator<char>(input_stream)),
                     std::istreambuf_iterator<char>());
}

absl::Status WriteFileContents(const std::filesystem::path& path,
                               absl::string_view content) {
  std::ofstream output_stream(path);
  if (!output_stream.is_open()) {
    return absl::InternalError(
        absl::StrCat("Could not open file for writing: ", path.string()));
  }
  output_stream << content;
  if (!output_stream.good()) {
    return absl::InternalError(
        absl::StrCat("Failed writing content to file: ", path.string()));
  }
  return absl::OkStatus();
}

std::filesystem::path ResolvePath(absl::string_view path_str) {
  std::filesystem::path p{std::string(path_str)};
  if (p.is_absolute()) {
    return p;
  }
  const auto src_dir = std::filesystem::path(::testing::SrcDir());
  if (std::filesystem::exists(src_dir / path_str)) {
    return src_dir / path_str;
  }
  if (std::filesystem::exists(src_dir / "litert_lm" / path_str)) {
    return src_dir / "litert_lm" / path_str;
  }
  if (std::filesystem::exists(src_dir / "_main" / path_str)) {
    return src_dir / "_main" / path_str;
  }
  return src_dir / p;
}

std::filesystem::path ResolveSourcePath(absl::string_view path_str) {
  std::filesystem::path p{std::string(path_str)};
  if (p.is_absolute()) {
    return p;
  }
  const char* workspace_dir = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (workspace_dir != nullptr && workspace_dir[0] != '\0') {
    return std::filesystem::path(workspace_dir) / path_str;
  }
  return ResolvePath(path_str);
}

PromptTemplateInput ParseInputJson(const json& j) {
  PromptTemplateInput input;
  if (j.is_array()) {
    input.messages = j;
  } else if (j.is_object()) {
    if (j.contains("messages")) {
      input.messages = j["messages"];
    }
    if (j.contains("tools")) {
      input.tools = j["tools"];
    }
    if (j.contains("add_generation_prompt")) {
      input.add_generation_prompt = j["add_generation_prompt"].get<bool>();
    }
    if (j.contains("extra_context")) {
      input.extra_context = j["extra_context"];
    }
    if (j.contains("bos_token")) {
      input.bos_token = j["bos_token"];
    }
    if (j.contains("eos_token")) {
      input.eos_token = j["eos_token"];
    }
  }
  return input;
}

TEST(ChatTemplateTestRunner, RenderAndCompareGoldens) {
  const std::string chat_template_flag = absl::GetFlag(FLAGS_chat_template);
  const std::string input_dir_flag = absl::GetFlag(FLAGS_input_dir);
  const std::string golden_dir_flag = absl::GetFlag(FLAGS_golden_dir);
  const bool update_golden = absl::GetFlag(FLAGS_update_golden);

  ASSERT_FALSE(chat_template_flag.empty())
      << "--chat_template must be specified.";
  ASSERT_FALSE(input_dir_flag.empty()) << "--input_dir must be specified.";
  ASSERT_FALSE(golden_dir_flag.empty()) << "--golden_dir must be specified.";

  const auto template_path = ResolvePath(chat_template_flag);
  const auto input_dir = ResolvePath(input_dir_flag);
  const auto golden_dir_read = ResolvePath(golden_dir_flag);
  const auto golden_dir_write = ResolveSourcePath(golden_dir_flag);

  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       ReadFileContents(template_path));
  PromptTemplate prompt_template(template_content);

  ASSERT_TRUE(std::filesystem::exists(input_dir))
      << "Input directory does not exist: " << input_dir;

  std::vector<std::filesystem::path> json_files;
  for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      json_files.push_back(entry.path());
    }
  }
  std::sort(json_files.begin(), json_files.end());

  ASSERT_GT(json_files.size(), 0)
      << "No .json test input files found in input_dir: " << input_dir;

  int test_count = 0;
  for (const auto& json_path : json_files) {
    const std::string stem = json_path.stem().string();

    ASSERT_OK_AND_ASSIGN(const std::string json_str,
                         ReadFileContents(json_path));
    json j = json::parse(json_str, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) {
      ADD_FAILURE() << "Failed to parse JSON file: " << json_path;
      continue;
    }

    // We test both non-thinking and thinking modes if the golden files exist.
    struct TestVariant {
      std::string suffix;
      bool enable_thinking;
    };
    std::vector<TestVariant> variants = {
        {".txt", false},
        {"-thinking.txt", true},
    };

    for (const auto& variant : variants) {
      const auto golden_path = golden_dir_read / (stem + variant.suffix);
      // Skip the thinking variant if the golden file doesn't exist (unless
      // updating)
      if (!update_golden && !std::filesystem::exists(golden_path) &&
          variant.enable_thinking) {
        continue;
      }

      PromptTemplateInput input = ParseInputJson(j);
      input.extra_context["enable_thinking"] = variant.enable_thinking;

      ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                           prompt_template.Apply(input));

      if (update_golden) {
        const auto write_golden_path =
            golden_dir_write / (stem + variant.suffix);
        std::filesystem::create_directories(golden_dir_write);
        ASSERT_OK(WriteFileContents(write_golden_path, rendered_prompt))
            << "Failed to write golden file: " << write_golden_path;
        ABSL_LOG(INFO) << "Updated golden file: " << write_golden_path;
      } else {
        ASSERT_TRUE(std::filesystem::exists(golden_path))
            << "Matching golden file not found: " << golden_path
            << " (for input: " << json_path << ")";

        ASSERT_OK_AND_ASSIGN(const std::string golden_content,
                             ReadFileContents(golden_path));

        EXPECT_EQ(rendered_prompt, golden_content)
            << "Mismatch for test case: " << stem << variant.suffix << "\n"
            << "Input file: " << json_path << "\n"
            << "Golden file: " << golden_path;
      }
    }

    test_count++;
  }

  ABSL_LOG(INFO) << (update_golden ? "Updated " : "Successfully verified ")
                 << test_count << " chat template test case(s).";

  const std::vector<std::string> pbtext_files_flag =
      absl::GetFlag(FLAGS_pbtext_files);
  for (const std::string& pbtext_flag : pbtext_files_flag) {
    if (pbtext_flag.empty()) continue;
    const auto pbtext_path = ResolvePath(pbtext_flag);
    ASSERT_TRUE(std::filesystem::exists(pbtext_path))
        << "pbtext file does not exist: " << pbtext_path;
    ASSERT_OK_AND_ASSIGN(const std::string pbtext_content,
                         ReadFileContents(pbtext_path));
    ASSERT_OK_AND_ASSIGN(const std::string extracted_template,
                         ExtractJinjaPromptTemplateFromPbtext(pbtext_content));
    EXPECT_EQ(extracted_template, template_content)
        << "jinja_prompt_template in " << pbtext_flag
        << " does not match chat_template (" << chat_template_flag << ")";
    ABSL_LOG(INFO) << "Successfully verified pbtext jinja_prompt_template in: "
                   << pbtext_flag;
  }

  const std::vector<std::string> mirror_chat_templates_flag =
      absl::GetFlag(FLAGS_mirror_chat_templates);
  for (const std::string& mirror_flag : mirror_chat_templates_flag) {
    if (mirror_flag.empty()) continue;
    const auto mirror_path = ResolvePath(mirror_flag);
    ASSERT_TRUE(std::filesystem::exists(mirror_path))
        << "Mirror chat template file does not exist: " << mirror_path;
    ASSERT_OK_AND_ASSIGN(const std::string mirror_content,
                         ReadFileContents(mirror_path));
    EXPECT_EQ(mirror_content, template_content)
        << "Mirror chat template " << mirror_flag
        << " does not match chat_template (" << chat_template_flag << ")";
    ABSL_LOG(INFO) << "Successfully verified mirror chat template in: "
                   << mirror_flag;
  }
}

}  // namespace
}  // namespace litert::lm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
