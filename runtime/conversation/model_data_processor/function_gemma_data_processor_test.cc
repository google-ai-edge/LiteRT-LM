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

#include "runtime/conversation/model_data_processor/function_gemma_data_processor.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/function_gemma_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::status::IsOkAndHolds;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

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

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  if (!text_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == text_input->GetRawTextString().value();
}

class FunctionGemmaDataProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        GetTestdataPath("function_gemma_sentencepiece.model"));
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(FunctionGemmaDataProcessorTest, ToInputDataVectorTextOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\ntest prompt\n<end_of_turn>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text("<start_of_turn>user\ntest prompt\n<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST_F(FunctionGemmaDataProcessorTest, ToMessage) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST_F(FunctionGemmaDataProcessorTest, ToMessageWithToolCalls) {
  FunctionGemmaDataProcessorConfig config;
  JsonPreface preface{.tools = nlohmann::ordered_json::parse(
                          R"json([{
                            "name": "tool_name",
                            "parameters": {
                              "type": "object",
                              "properties": {
                                "x": {
                                  "type": "integer"
                                }
                              }
                            }
                          }])json")};

  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(
              TaskState::kProcessing,
              {"This is some text.\n"
               "<start_function_call>call:tool_name{x:1}<end_function_call>"
               "<start_function_call>call:tool_name{x:2}<end_function_call>"}),
          std::monostate{}));

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "This is some text.\n"
      }
    ],
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "tool_name",
          "arguments": {
            "x": 1
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "tool_name",
          "arguments": {
            "x": 2
          }
        }
      }
    ]
  })json"));
}

TEST_F(FunctionGemmaDataProcessorTest,
       PromptTemplateToInputDataVectorTextOnly) {
  const std::string test_file_path =
      GetTestdataPath("google-function-gemma.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"}, {"content", "How are you?"}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"}, {"content", "What is the capital of France?"}},
  };
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputText expected_text(R"(<start_of_turn>developer
Hello world!<end_of_turn>
<start_of_turn>user
How are you?<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
)");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST_F(FunctionGemmaDataProcessorTest, FormatTools) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json([
    {
      "name": "get_weather",
      "description": "Gets weather information.",
      "parameters": {
        "properties": {
          "location": {
            "type": "string",
            "description": "Weather location."
          }
        },
        "required": ["location"]
      }
    },
    {
      "name": "get_stock_price",
      "description": "Gets stock price.",
      "parameters": {
        "properties": {
          "symbol": {
            "type": "string",
            "description": "Stock symbol."
          }
        },
        "required": ["symbol"]
      }
    }
  ])json");

  ASSERT_OK_AND_ASSIGN(const nlohmann::ordered_json formatted_tools,
                       processor->FormatTools(tools));

  nlohmann::ordered_json expected = {
      ("declaration:get_weather{"
       "description:<escape>Gets weather information.<escape>,"
       "parameters:{"
       "properties:{"
       "location:{"
       "type:<escape>STRING<escape>,"
       "description:<escape>Weather location.<escape>"
       "}"   // location
       "},"  // properties
       "required:[<escape>location<escape>]"
       "}"  // parameters
       "}"  // declaration
       ),
      ("declaration:get_stock_price{"
       "description:<escape>Gets stock price.<escape>,"
       "parameters:{"
       "properties:{"
       "symbol:{"
       "type:<escape>STRING<escape>,"
       "description:<escape>Stock symbol.<escape>"
       "}"   // symbol
       "},"  // properties
       "required:[<escape>symbol<escape>]"
       "}"  // parameters
       "}"  // declaration
       )};
  EXPECT_EQ(formatted_tools, expected);
}

TEST_F(FunctionGemmaDataProcessorTest, FormatToolsWithInvalidInput) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  // `tools` is not an array.
  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "description": "Gets weather information.",
    "parameters": {
      "properties": {
        "location": {
          "type": "string",
          "description": "Weather location."
        }
      },
      "required": ["location"]
    }
  })json");

  EXPECT_THAT(processor->FormatTools(tools),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithStringContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", "test prompt"},
  };

  // The template input is identical to the original message if the content is a
  // string.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(FunctionGemmaDataProcessorTest, MessageToTemplateInputWithTextContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", {{{"type", "text"}, {"text", "test prompt"}}}},
  };

  // Text content items should be unchanged.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(FunctionGemmaDataProcessorTest, MessageToTemplateInputNoContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
  };

  // The template input should be the same as the original message if there is
  // no content or tool calls.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(FunctionGemmaDataProcessorTest, MessageToTemplateInputWithToolCalls) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "This is some text."
      }
    ],
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "tool1",
          "arguments": {
            "x": 1
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "tool2",
          "arguments": {
            "y": "foo"
          }
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "This is some text."
      }
    ],
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "tool1",
          "arguments": {
            "x": "1"
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "tool2",
          "arguments": {
            "y": "<escape>foo<escape>"
          }
        }
      }
    ]
  })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponsesNameAndValue) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "name": "tool_1",
          "value": {
            "key1": "value1",
            "key2": "value2"
          }
        }
      },
      {
        "type": "tool_response",
        "tool_response": {
          "name": "tool_2",
          "value": {
            "key3": "value3",
            "key4": "value4"
          }
        }
      }
    ]
  })json");

  // The tool responses should be formatted as text items with the tool name
  // and value converted to FC format.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{key1:<escape>value1<escape>,key2:<escape>value2<escape>}"
                  },
                  {
                    "type": "text",
                    "text": "tool_2{key3:<escape>value3<escape>,key4:<escape>value4<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponseToolNameAndValue) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "tool_name": "tool_1",
          "value": {
            "key1": "value1"
          }
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{key1:<escape>value1<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponseNameAndArgs) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "name": "tool_1",
          "key1": "value1"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{key1:<escape>value1<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponsesToolNameAndArgs) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "tool_name": "tool_1",
          "key1": "value1"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{key1:<escape>value1<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponseWithNonObjectValue) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "name": "tool_1",
          "value": "foo"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{value:<escape>foo<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponseWithNonObjectResponse) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "name": "tool_1",
          "response": "foo"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{response:<escape>foo<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolResponsesNoName) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "key1": "value1"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "{key1:<escape>value1<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolContentAsObject) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": {
      "name": "get_weather",
      "temperature": 72,
      "units": "Fahrenheit"
    }
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": "get_weather{temperature:72,units:<escape>Fahrenheit<escape>}"
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolContentAsObjectWithNameAndResponse) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": {
      "name": "tool_1",
      "response": {
        "key1": "value1"
      }
    }
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": "tool_1{key1:<escape>value1<escape>}"
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolContentAsArrayWithNameAndResponse) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "name": "tool_1",
        "response": {
          "key1": "value1"
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "tool_1{key1:<escape>value1<escape>}"
                  }
                ]
              })json")));
}

TEST_F(FunctionGemmaDataProcessorTest,
       MessageToTemplateInputWithToolContentAsString) {
  ASSERT_OK_AND_ASSIGN(auto processor, FunctionGemmaDataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": "get_weather{temperature:72,units:<escape>Fahrenheit<escape>}"
  })json");

  // String content should be kept as is.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": "get_weather{temperature:72,units:<escape>Fahrenheit<escape>}"
              })json")));
}

struct RenderTemplateTestCase {
  std::string jinja_template_file;
  bool use_template_for_fc_format;
};

class FunctionGemmaRenderTemplateTest
    : public FunctionGemmaDataProcessorTest,
      public ::testing::WithParamInterface<RenderTemplateTestCase> {};

TEST_P(FunctionGemmaRenderTemplateTest, RenderTemplateUserTurn) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n");
}

TEST_P(FunctionGemmaRenderTemplateTest, RenderTemplateAssistantTurnTextOnly) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Sorry, I can't help with that."
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = false};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Sorry, I can't help with that.<end_of_turn>\n");
}

TEST_P(FunctionGemmaRenderTemplateTest, RenderTemplateWithToolDeclarations) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets weather information.",
        "parameters": {
          "properties": {
            "location": {
              "description": "Weather location.",
              "nullable": false,
              "type": "string"
            }
          },
          "required": ["location"],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_stock_price",
        "description": "Gets stock price.",
        "parameters": {
          "properties": {
            "symbol": {
              "description": "Stock symbol.",
              "nullable": false,
              "type": "string"
            }
          },
          "required": ["symbol"],
          "type": "object"
        }
      }
    }
  ])json");

  nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content": "How is the weather in Paris and London?"
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Format the tools.
  ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json formatted_tools,
                       processor->FormatTools(tools));

  // Render the template.
  PromptTemplateInput template_input = {.messages = messages,
                                        .tools = formatted_tools,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_THAT(rendered_prompt,
              Eq("<start_of_turn>developer\n"
                 "<start_function_declaration>"
                 "declaration:get_weather{"
                 "description:<escape>Gets weather information.<escape>,"
                 "parameters:{"
                 "properties:{"
                 "location:{"
                 "description:<escape>Weather location.<escape>,"
                 "nullable:false,"
                 "type:<escape>STRING<escape>"
                 "}"   // location
                 "},"  // properties
                 "required:[<escape>location<escape>],"
                 "type:<escape>OBJECT<escape>"
                 "}"  // parameters
                 "}"  // declaration
                 "<end_function_declaration>"
                 "<start_function_declaration>"
                 "declaration:get_stock_price{"
                 "description:<escape>Gets stock price.<escape>,"
                 "parameters:{"
                 "properties:{"
                 "symbol:{"
                 "description:<escape>Stock symbol.<escape>,"
                 "nullable:false,"
                 "type:<escape>STRING<escape>"
                 "}"   // symbol
                 "},"  // properties
                 "required:[<escape>symbol<escape>],"
                 "type:<escape>OBJECT<escape>"
                 "}"  // parameters
                 "}"  // declaration
                 "<end_function_declaration>"
                 "<end_of_turn>\n"
                 "<start_of_turn>user\n"
                 "How is the weather in Paris and London?<end_of_turn>\n"
                 "<start_of_turn>model\n"));
}

TEST_P(FunctionGemmaRenderTemplateTest, RenderTemplateWithToolCalls) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "London"
            }
          }
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = false};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  //
  // Note that a model turn containing tool calls is terminated by
  // "<start_function_response>" instead of "<end_of_turn>".
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<start_function_call>"
            "call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
            "<start_function_call>"
            "call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_response>");
}

TEST_P(FunctionGemmaRenderTemplateTest, RenderTemplateWithToolResponses) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "London"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": [
        {
          "name": "get_weather",
          "response": {
            "location": "Paris",
            "temperature": 20,
            "unit": "C",
            "weather": "Sunny"
          }
        },
        {
          "name": "get_weather",
          "response": {
            "location": "London",
            "temperature": 15,
            "unit": "C",
            "weather": "Cloudy"
          }
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  //
  // Note that the generation prompt is suppressed after the tool response,
  // despite add_generation_prompt = true.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<start_function_call>"
            "call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
            "<start_function_call>"
            "call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>Paris<escape>,"
            "temperature:20,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Sunny<escape>"
            "}"  // response:get_weather
            "<end_function_response>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>London<escape>,"
            "temperature:15,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Cloudy<escape>"
            "}"  // response:get_weather
            "<end_function_response>");
}

TEST_P(FunctionGemmaRenderTemplateTest,
       RenderTemplateWithMultipleToolMessages) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "London"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": {
        "name": "get_weather",
        "response": {
          "location": "Paris",
          "temperature": 20,
          "unit": "C",
          "weather": "Sunny"
        }
      }
    },
    {
      "role": "tool",
      "content": {
        "name": "get_weather",
        "response": {
          "location": "London",
          "temperature": 15,
          "unit": "C",
          "weather": "Cloudy"
        }
      }
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  //
  // Note that the generation prompt is suppressed after the tool response,
  // despite add_generation_prompt = true.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<start_function_call>"
            "call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
            "<start_function_call>"
            "call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>Paris<escape>,"
            "temperature:20,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Sunny<escape>"
            "}"  // response:get_weather
            "<end_function_response>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>London<escape>,"
            "temperature:15,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Cloudy<escape>"
            "}"  // response:get_weather
            "<end_function_response>");
}

TEST_P(FunctionGemmaRenderTemplateTest,
       RenderTemplateWithModelResponseAfterToolResponse) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "London"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": [
        {
          "name": "get_weather",
          "response": {
            "location": "Paris",
            "temperature": 20,
            "unit": "C",
            "weather": "Sunny"
          }
        },
        {
          "name": "get_weather",
          "response": {
            "location": "London",
            "temperature": 15,
            "unit": "C",
            "weather": "Cloudy"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "The weather in Paris is sunny and the weather in London is cloudy."
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = false};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris and London?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<start_function_call>"
            "call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
            "<start_function_call>"
            "call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>Paris<escape>,"
            "temperature:20,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Sunny<escape>"
            "}"  // response:get_weather
            "<end_function_response>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>London<escape>,"
            "temperature:15,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Cloudy<escape>"
            "}"  // response:get_weather
            "<end_function_response>"
            "The weather in Paris is sunny and the weather in London is cloudy."
            "<end_of_turn>\n");
}

TEST_P(FunctionGemmaRenderTemplateTest,
       RenderTemplateWithEmptyAssistantMessage) {
  const RenderTemplateTestCase& test_case = GetParam();

  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath(test_case.jinja_template_file);
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": [
        {
          "name": "get_weather",
          "response": {
            "location": "Paris",
            "temperature": 20,
            "unit": "C",
            "weather": "Sunny"
          }
        }
      ]
    },
    {
      "role": "assistant"
    },
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in New York?"
        }
      ]
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>user\n"
            "How is the weather in Paris?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<start_function_call>"
            "call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
            "<start_function_response>"
            "response:get_weather{"
            "location:<escape>Paris<escape>,"
            "temperature:20,"
            "unit:<escape>C<escape>,"
            "weather:<escape>Sunny<escape>"
            "}"  // response:get_weather
            "<end_function_response>"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            "How is the weather in New York?<end_of_turn>\n"
            "<start_of_turn>model\n");
}

INSTANTIATE_TEST_SUITE_P(
    FcFormatCodeOrTemplate, FunctionGemmaRenderTemplateTest,
    testing::ValuesIn<RenderTemplateTestCase>({
        {.jinja_template_file = "google-function-gemma.jinja",
         .use_template_for_fc_format = false},
        {.jinja_template_file = "google-function-gemma-hf.jinja",
         .use_template_for_fc_format = true},
    }));

TEST_F(FunctionGemmaDataProcessorTest,
       MobileActionsTemplateSystemInstructionsSplitting) {
  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath("google-function-gemma-mobile-actions.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets weather information.",
        "parameters": {
          "properties": {
            "location": {
              "description": "Weather location.",
              "nullable": false,
              "type": "string"
            }
          },
          "required": ["location"],
          "type": "object"
        }
      }
    }
  ])json");

  nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "System message part 1. This will appear BEFORE tool declarations."
        },
        {
          "type": "text",
          "text": "System message part 2. This will appear AFTER tool declarations."
        }
      ]
    },
    {
      "role": "user",
      "content": "How is the weather in Paris?"
    }
  ])json");

  // Create the model data processor.
  FunctionGemmaDataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       FunctionGemmaDataProcessor::Create(config));

  // Format the tools.
  ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json formatted_tools,
                       processor->FormatTools(tools));

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .tools = formatted_tools,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  // Compare to the expected prompt.
  EXPECT_THAT(
      rendered_prompt,
      Eq("<start_of_turn>developer\n"
         "System message part 1. This will appear BEFORE tool declarations."
         "\n\n"
         "<start_function_declaration>"
         "declaration:get_weather{"
         "description:<escape>Gets weather information.<escape>,"
         "parameters:{"
         "properties:{"
         "location:{"
         "description:<escape>Weather location.<escape>,"
         "nullable:false,"
         "type:<escape>STRING<escape>"
         "}"
         "},"
         "required:[<escape>location<escape>],"
         "type:<escape>OBJECT<escape>"
         "}"
         "}"
         "<end_function_declaration>"
         "System message part 2. This will appear AFTER tool declarations."
         "<end_of_turn>\n"
         "<start_of_turn>user\n"
         "How is the weather in Paris?<end_of_turn>\n"
         "<start_of_turn>model\n"));
}

}  // namespace
}  // namespace litert::lm
