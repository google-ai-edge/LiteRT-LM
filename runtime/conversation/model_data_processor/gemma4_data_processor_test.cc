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

#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"

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
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
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

constexpr char kImageTestdataDir[] =
    "litert_lm/runtime/components/preprocessor/testdata/";

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

// Checks that the image tensor map has the expected keys and that the image
// tensor has at most `max_num_patches` patches.
MATCHER_P(HasInputImage, max_num_patches, "") {
  if (!std::holds_alternative<InputImage>(arg)) {
    return false;
  }
  if (!std::get<InputImage>(arg).IsTensorBufferMap()) {
    return false;
  }
  auto tensor_map = std::get<InputImage>(arg).GetPreprocessedImageTensorMap();
  if (!tensor_map.ok()) {
    return false;
  }
  if (!(*tensor_map)->contains("images")) {
    return false;
  }
  if (!(*tensor_map)->contains("positions_xy")) {
    return false;
  }
  auto image_tensor = (*tensor_map)->at("images").Duplicate();
  auto image_tensor_type = (*image_tensor).TensorType();
  if ((*image_tensor_type).Layout().Dimensions()[1] > max_num_patches) {
    return false;
  }
  return true;
}

MATCHER(HasInputImageEnd, "") {
  if (!std::holds_alternative<InputImageEnd>(arg)) {
    return false;
  }
  return true;
}

class Gemma4DataProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        GetTestdataPath("gemma4_sentencepiece.model"));
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(Gemma4DataProcessorTest, ToInputDataVectorTextOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  const std::string rendered_template_prompt =
      "<|turn>user\ntest prompt\n<turn|>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text("<|turn>user\ntest prompt\n<turn|>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST_F(Gemma4DataProcessorTest, ToInputDataVectorTextAndImage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(
                                           /*Gemma4DataProcessorConfig=*/
                                           {.max_num_patches = 2520}));
  const std::string rendered_template_prompt =
      "<|turn>user\nHere is an image of apples <|image|><turn|>";

  std::string image_path = (std::filesystem::path(::testing::SrcDir()) /
                            kImageTestdataDir / "apple.png")
                               .string();
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content",
       {{{"type", "text"}, {"text", "Here is an image of apples "}},
        {{"type", "image"}, {"path", image_path}}}}};

  {
    ASSERT_OK_AND_ASSIGN(
        const std::vector<InputData> input_data,
        processor->ToInputDataVector(rendered_template_prompt,
                                     json::array({message}), {}));

    InputText expected_text1(
        "<|turn>user\nHere is an image of apples <|image>");
    InputText expected_text2("\n\n");
    InputText expected_text3("<turn|>");
    EXPECT_THAT(input_data,
                ElementsAre(HasInputText(&expected_text1), HasInputImage(2520),
                            HasInputImageEnd(), HasInputText(&expected_text2),
                            HasInputText(&expected_text3)));
  }

  {
    // Override the visual token budget to 100 at runtime.
    ASSERT_OK_AND_ASSIGN(
        const std::vector<InputData> input_data,
        processor->ToInputDataVector(
            rendered_template_prompt, json::array({message}),
            Gemma4DataProcessorArguments{.visual_token_budget = 100}));

    InputText expected_text1(
        "<|turn>user\nHere is an image of apples <|image>");
    InputText expected_text2("\n\n");
    InputText expected_text3("<turn|>");
    EXPECT_THAT(input_data,
                ElementsAre(HasInputText(&expected_text1), HasInputImage(900),
                            HasInputImageEnd(), HasInputText(&expected_text2),
                            HasInputText(&expected_text3)));
  }

  {
    // Override the visual token budget to 300 at runtime, which is larger than
    // the max_num_patches / 9 in the config. The visual token budget should be
    // capped to max_num_patches / 9.
    ASSERT_OK_AND_ASSIGN(
        const std::vector<InputData> input_data,
        processor->ToInputDataVector(
            rendered_template_prompt, json::array({message}),
            Gemma4DataProcessorArguments{.visual_token_budget = 300}));

    InputText expected_text1(
        "<|turn>user\nHere is an image of apples <|image>");
    InputText expected_text2("\n\n");
    InputText expected_text3("<turn|>");
    EXPECT_THAT(input_data,
                ElementsAre(HasInputText(&expected_text1), HasInputImage(2520),
                            HasInputImageEnd(), HasInputText(&expected_text2),
                            HasInputText(&expected_text3)));
  }
}

TEST_F(Gemma4DataProcessorTest, ToMessage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST_F(Gemma4DataProcessorTest, ToMessageWithToolCalls) {
  Gemma4DataProcessorConfig config;
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
                       Gemma4DataProcessor::Create(config, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"This is some text.\n"
                     "<|tool_call>call:tool_name{x:1}<tool_call|>"
                     "<|tool_call>call:tool_name{x:2}<tool_call|>"}),
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

TEST_F(Gemma4DataProcessorTest, PromptTemplateToInputDataVectorTextOnly) {
  const std::string test_file_path = GetTestdataPath("google-gemma-4.jinja");
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

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputText expected_text(R"(<|turn>system
Hello world!<turn|>
<|turn>user
How are you?<turn|>
<|turn>model
I am doing well, thanks for asking.<turn|>
<|turn>user
What is the capital of France?<turn|>
<|turn>model
)");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST_F(Gemma4DataProcessorTest, FormatTools) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
       "description:<|\"|>Gets weather information.<|\"|>,"
       "parameters:{"
       "properties:{"
       "location:{"
       "type:<|\"|>STRING<|\"|>,"
       "description:<|\"|>Weather location.<|\"|>"
       "}"   // location
       "},"  // properties
       "required:[<|\"|>location<|\"|>]"
       "}"  // parameters
       "}"  // declaration
       ),
      ("declaration:get_stock_price{"
       "description:<|\"|>Gets stock price.<|\"|>,"
       "parameters:{"
       "properties:{"
       "symbol:{"
       "type:<|\"|>STRING<|\"|>,"
       "description:<|\"|>Stock symbol.<|\"|>"
       "}"   // symbol
       "},"  // properties
       "required:[<|\"|>symbol<|\"|>]"
       "}"  // parameters
       "}"  // declaration
       )};
  EXPECT_EQ(formatted_tools, expected);
}

TEST_F(Gemma4DataProcessorTest, FormatToolsWithInvalidInput) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithStringContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", "test prompt"},
  };

  // The template input is identical to the original message if the content is a
  // string.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithTextContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", {{{"type", "text"}, {"text", "test prompt"}}}},
  };

  // Text content items should be unchanged.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputNoContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
  };

  // The template input should be the same as the original message if there is
  // no content or tool calls.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithToolCalls) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
            "y": "<|\"|>foo<|\"|>"
          }
        }
      }
    ]
  })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponsesNameAndValue) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{key1:<|\"|>value1<|\"|>,key2:<|\"|>value2<|\"|>}"
                  },
                  {
                    "type": "text",
                    "text": "tool_2{key3:<|\"|>value3<|\"|>,key4:<|\"|>value4<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponseToolNameAndValue) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{key1:<|\"|>value1<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponseNameAndArgs) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{key1:<|\"|>value1<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponsesToolNameAndArgs) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{key1:<|\"|>value1<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponseWithNonObjectValue) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{value:<|\"|>foo<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolResponseWithNonObjectResponse) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{response:<|\"|>foo<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithToolResponsesNoName) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "{key1:<|\"|>value1<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithToolContentAsObject) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                "content": "get_weather{temperature:72,units:<|\"|>Fahrenheit<|\"|>}"
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolContentAsObjectWithNameAndResponse) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                "content": "tool_1{key1:<|\"|>value1<|\"|>}"
              })json")));
}

TEST_F(Gemma4DataProcessorTest,
       MessageToTemplateInputWithToolContentAsArrayWithNameAndResponse) {
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = false;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));
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
                    "text": "tool_1{key1:<|\"|>value1<|\"|>}"
                  }
                ]
              })json")));
}

TEST_F(Gemma4DataProcessorTest, MessageToTemplateInputWithToolContentAsString) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": "get_weather{temperature:72,units:<|\"|>Fahrenheit<|\"|>}"
  })json");

  // String content should be kept as is.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": "get_weather{temperature:72,units:<|\"|>Fahrenheit<|\"|>}"
              })json")));
}

struct RenderTemplateTestCase {
  std::string jinja_template_file;
  bool use_template_for_fc_format;
};

class Gemma4RenderTemplateTest
    : public Gemma4DataProcessorTest,
      public ::testing::WithParamInterface<RenderTemplateTestCase> {};

TEST_P(Gemma4RenderTemplateTest, RenderTemplateUserTurn) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
            "<|turn>user\n"
            "How is the weather in Paris and London?<turn|>\n"
            "<|turn>model\n");
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateAssistantTurnTextOnly) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
            "<|turn>user\n"
            "How is the weather in Paris and London?<turn|>\n"
            "<|turn>model\n"
            "Sorry, I can't help with that.<turn|>\n");
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateWithToolDeclarations) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
  EXPECT_THAT(
      rendered_prompt,
      Eq("<|turn>system\n\n\n"
         "<|tool>declaration:get_weather{description:<|\"|>Gets weather "
         "information.<|\"|>,parameters:{properties:{location:{description:<|"
         "\"|>Weather "
         "location.<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>location<|"
         "\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
         "<|tool>declaration:get_stock_price{description:<|\"|>Gets stock "
         "price.<|\"|>,parameters:{properties:{symbol:{description:<|\"|>Stock "
         "symbol.<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>symbol<|\"|>]"
         ",type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
         "<|turn>user\nHow is the weather in Paris and London?<turn|>\n"
         "<|turn>model\n"));
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateWithToolCalls) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
  // Note: The template currently terminates this turn with "<turn|>" instead of
  // "<|tool_response>".
  EXPECT_EQ(
      rendered_prompt,
      "<|turn>user\n"
      "How is the weather in Paris and London?<turn|>\n"
      "<|turn>model\n"
      "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|><"
      "turn|>\n");
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateWithToolResponses) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
  EXPECT_EQ(
      rendered_prompt,
      "<|turn>user\n"
      "How is the weather in Paris and London?<turn|>\n"
      "<|turn>model\n"
      "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|><"
      "turn|>\n"
      "<|tool_response>response:get_weather{location:<|\"|>Paris<|\"|>,"
      "temperature:20,unit:<|\"|>C<|\"|>,weather:<|\"|>Sunny<|\"|>}<tool_"
      "response|>"
      "<|tool_response>response:get_weather{location:<|\"|>London<|\"|>,"
      "temperature:15,unit:<|\"|>C<|\"|>,weather:<|\"|>Cloudy<|\"|>}<tool_"
      "response|>");
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateWithMultipleToolMessages) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
  EXPECT_EQ(
      rendered_prompt,
      "<|turn>user\n"
      "How is the weather in Paris and London?<turn|>\n"
      "<|turn>model\n"
      "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|><"
      "turn|>\n"
      "<turn|>\n"
      "<turn|>\n"
      "<|turn>model\n");
}

TEST_P(Gemma4RenderTemplateTest,
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
  EXPECT_EQ(
      rendered_prompt,
      "<|turn>user\n"
      "How is the weather in Paris and London?<turn|>\n"
      "<|turn>model\n"
      "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"
      "<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|><"
      "turn|>\n"
      "<|tool_response>response:get_weather{location:<|\"|>Paris<|\"|>,"
      "temperature:20,unit:<|\"|>C<|\"|>,weather:<|\"|>Sunny<|\"|>}<tool_"
      "response|>"
      "<|tool_response>response:get_weather{location:<|\"|>London<|\"|>,"
      "temperature:15,unit:<|\"|>C<|\"|>,weather:<|\"|>Cloudy<|\"|>}<tool_"
      "response|>"
      "<|turn>model\n"
      "The weather in Paris is sunny and the weather in London is "
      "cloudy.<turn|>\n");
}

TEST_P(Gemma4RenderTemplateTest, RenderTemplateWithEmptyAssistantMessage) {
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
  Gemma4DataProcessorConfig config;
  config.use_template_for_fc_format = test_case.use_template_for_fc_format;
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma4DataProcessor::Create(config));

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
            "<|turn>user\n"
            "How is the weather in Paris?<turn|>\n"
            "<|turn>model\n"
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_"
            "call|><turn|>\n"
            "<|tool_response>response:get_weather{location:<|\"|>Paris<|\"|>,"
            "temperature:20,unit:<|\"|>C<|\"|>,weather:<|\"|>Sunny<|\"|>}<tool_"
            "response|><|turn>model\n"
            "<turn|>\n"
            "<|turn>user\n"
            "How is the weather in New York?<turn|>\n"
            "<|turn>model\n");
}

INSTANTIATE_TEST_SUITE_P(FcFormatCodeOrTemplate, Gemma4RenderTemplateTest,
                         testing::ValuesIn<RenderTemplateTestCase>({
                             {.jinja_template_file = "google-gemma-4.jinja",
                              .use_template_for_fc_format = true},
                         }));

}  // namespace
}  // namespace litert::lm
