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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
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
    "google-gemma-3n-e2b-it", "HuggingFaceTB-SmolLM3-3B",
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

TEST(PromptTemplateCustomTest, StripReplacementTest) {
  {
    PromptTemplate prompt_template("{{ '  hello  '.strip() }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "hello");
  }
  {
    PromptTemplate prompt_template("{{ '  hello  '.lstrip() }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "hello  ");
  }
  {
    PromptTemplate prompt_template("{{ '  hello  '.rstrip() }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "  hello");
  }
  {
    PromptTemplate prompt_template("{{ 'ohelloo'.strip('o') }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "hell");
  }
  {
    PromptTemplate prompt_template("{{ 'ohelloo'.lstrip('o') }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "helloo");
  }
  {
    PromptTemplate prompt_template("{{ 'ohelloo'.rstrip('o') }}");
    PromptTemplateInput input;
    auto res = prompt_template.Apply(input);
    ASSERT_OK(res);
    EXPECT_EQ(*res, "ohell");
  }
}

TEST(PromptTemplateCustomTest, Qwen3MultiTurnTemplateRenderingTest) {
  const std::string test_file_path =
      (std::filesystem::path(::testing::SrcDir()) /
       "litert_lm/models/qwen3/chat_template.jinja")
          .string();
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  const json msg_system = {
      {"role", "system"},
      {"content", {{{"type", "text"}, {"text", "I am The System!"}}}},
  };
  const json msg_user_1 = {
      {"role", "user"},
      {"content", {{{"type", "text"}, {"text", "I need help"}}}},
  };
  const json msg_assistant = {
      {"role", "assistant"},
      {"content", {{{"type", "text"}, {"text", "Hi, what can I do for you?"}}}},
  };
  const json msg_user_2 = {
      {"role", "user"},
      {"content", {{{"type", "text"}, {"text", "Write a poem about a cat"}}}},
  };

  for (bool enable_thinking : {true, false}) {
    json extra_context = json::object({{"enable_thinking", enable_thinking}});

    PromptTemplateInput old_input{
        .messages = json::array({msg_system, msg_user_1, msg_assistant}),
        .tools = json::array({GetTools()}),
        .add_generation_prompt = false,
        .extra_context = extra_context,
    };
    ASSERT_OK_AND_ASSIGN(const std::string old_string,
                         prompt_template.Apply(old_input));

    PromptTemplateInput new_input{
        .messages =
            json::array({msg_system, msg_user_1, msg_assistant, msg_user_2}),
        .tools = json::array({GetTools()}),
        .add_generation_prompt = true,
        .extra_context = extra_context,
    };
    ASSERT_OK_AND_ASSIGN(const std::string new_string,
                         prompt_template.Apply(new_input));

    EXPECT_TRUE(absl::StartsWith(new_string, old_string))
        << "enable_thinking: " << enable_thinking
        << "\nold_string: " << old_string << "\nnew_string: " << new_string;
  }
}

TEST(PromptTemplateCustomTest, CapabilitiesTest) {
  PromptTemplate standard_template("{{ messages[0].content }}");
  EXPECT_FALSE(standard_template.GetCapabilities().supports_single_turn);

  PromptTemplate single_turn_template(
      "{% if is_appending_to_prefill %}{{ message.content }}{% endif %}");
  EXPECT_TRUE(single_turn_template.GetCapabilities().supports_single_turn);
}

TEST(PromptTemplateCustomTest, GenerationTagReplacementTest) {
  PromptTemplate prompt_template(
      "{% generation %}{{ 'hello' }}{% endgeneration %}"
      "{%- generation -%}{{ ' world' }}{%- endgeneration -%}");
  PromptTemplateInput input;
  ASSERT_OK_AND_ASSIGN(const std::string rendered,
                       prompt_template.Apply(input));
  EXPECT_EQ(rendered, "hello world");
}

TEST(PromptTemplateCustomTest, EmptyMessagesAllTestdataTemplates) {
  const std::filesystem::path testdata_dir = GetTestdataPath("");
  ASSERT_TRUE(std::filesystem::exists(testdata_dir))
      << "Testdata directory does not exist: " << testdata_dir;

  std::vector<std::filesystem::path> jinja_files;
  for (const auto& entry : std::filesystem::directory_iterator(testdata_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".jinja") {
      jinja_files.push_back(entry.path());
    }
  }
  std::sort(jinja_files.begin(), jinja_files.end());
  ASSERT_FALSE(jinja_files.empty())
      << "No .jinja files found in: " << testdata_dir;

  for (const auto& jinja_path : jinja_files) {
    ASSERT_OK_AND_ASSIGN(const std::string template_content,
                         GetContents(jinja_path.string()));
    PromptTemplate prompt_template(template_content);

    json synthetic_tools;
    if (absl::StrContains(template_content, "tool | trim")) {
      synthetic_tools = json::array({"declaration of dummy_tool"});
    } else {
      synthetic_tools = json::array({
          {
              {"type", "function"},
              {"function",
               {
                   {"name", "dummy_function"},
                   {"description", "A dummy function description"},
                   {"parameters",
                    {
                        {"type", "object"},
                        {"properties",
                         {
                             {"param1",
                              {
                                  {"type", "string"},
                                  {"description", "dummy string parameter"},
                              }},
                         }},
                        {"required", json::array({"param1"})},
                    }},
               }},
          },
      });
    }

    const std::vector<std::optional<json>> tool_variants = {
        std::nullopt,
        json::array({}),
        synthetic_tools,
    };

    for (const auto& tools : tool_variants) {
      for (bool enable_thinking : {false, true}) {
        for (bool add_generation_prompt : {false, true}) {
          PromptTemplateInput input;
          input.messages = json::array({});
          if (tools.has_value()) {
            input.tools = *tools;
          }
          input.add_generation_prompt = add_generation_prompt;
          input.extra_context["enable_thinking"] = enable_thinking;

          EXPECT_OK(prompt_template.Apply(input))
              << "Failed rendering empty messages for template: "
              << jinja_path.filename().string() << " with tools="
              << (tools.has_value() ? tools->dump() : "nullopt")
              << ", enable_thinking=" << enable_thinking
              << ", add_generation_prompt=" << add_generation_prompt;
        }
      }
    }
  }
}

}  // namespace
}  // namespace litert::lm
