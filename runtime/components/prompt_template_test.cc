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

#include "runtime/components/prompt_template.h"

#include <algorithm>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/civil_time.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

constexpr absl::string_view kTestModelTemplates[] = {
    "google-gemma-3n-e2b-it", "Qwen-Qwen3-0.6B", "HuggingFaceTB-SmolLM3-3B",
    "microsoft-Phi-4-mini-instruct", "bos-token-test"};

json GetMessageSystem() {
  return {
      {"role", "system"},
      {"content", "I am The System!"},
  };
}

json GetMessageUserTextTurn1() {
  return {
      {"role", "user"},
      {"content", "I need help"},
  };
}

json GetMessageAssistantText() {
  return {
      {"role", "assistant"},
      {"content", "Hi, what can I do for you?"},
  };
}

json GetMessageUserTextTurn2() {
  return {
      {"role", "user"},
      {"content", "Write a poem about a cat"},
  };
}

json GetTools() {
  return json::parse(R"({
      "type": "function",
      "function": {
        "name": "GetWeather",
        "description": "Get the weather of the location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The location to get the weather for."
            }
          },
          "required": ["location"]
        }
      }
    })");
}

std::string GetTestdataPath(const std::string& file_name) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / file_name)
      .string();
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

class PromptTemplateTest : public ::testing::TestWithParam<absl::string_view> {
};

TEST_P(PromptTemplateTest, CreateTest) {
  const std::string test_model_template = std::string(GetParam());
  const std::string test_file_name = test_model_template + ".jinja";
  const std::string golden_file_name =
      test_model_template + "-jinja-golden.txt";

  const std::string test_file_path = GetTestdataPath(test_file_name);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  PromptTemplateInput input{
      .messages =
          json::array({GetMessageSystem(), GetMessageUserTextTurn1(),
                       GetMessageAssistantText(), GetMessageUserTextTurn2()}),
      .tools = json::array({GetTools()}),
      .add_generation_prompt = true,
      .extra_context = json::object({{"enable_thinking", false}}),
      .now = absl::FromCivil(absl::CivilHour(2025, 7, 29, 12),
                             absl::UTCTimeZone()),
  };
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(input));

  const std::string golden_file_path = GetTestdataPath(golden_file_name);
  ASSERT_OK_AND_ASSIGN(const std::string golden_content,
                       GetContents(golden_file_path));
  EXPECT_EQ(rendered_prompt, golden_content);
}

INSTANTIATE_TEST_SUITE_P(
    PromptTemplateTest, PromptTemplateTest,
    testing::ValuesIn(kTestModelTemplates),
    [](const testing::TestParamInfo<PromptTemplateTest::ParamType>& info) {
      std::string name = std::string(info.param);
      std::replace(name.begin(), name.end(), '.', '_');
      std::replace(name.begin(), name.end(), '-', '_');
      return name;
    });

}  // namespace
}  // namespace litert::lm
