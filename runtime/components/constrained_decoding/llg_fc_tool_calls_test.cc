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

#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/tokenizer/sentencepiece_tokenizer.h"

namespace litert::lm {
namespace {
using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

using ::testing::status::StatusIs;

class TestSentencePieceTokenizer : public Tokenizer {
 public:
  explicit TestSentencePieceTokenizer(
      std::unique_ptr<::litert::support::SentencePieceTokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {}

  TokenizerType GetTokenizerType() const override {
    return tokenizer_->GetTokenizerType();
  }

  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    return tokenizer_->TextToTokenIds(text);
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return tokenizer_->TokenToId(token);
  }

  absl::StatusOr<std::string> TokenIdsToText(absl::Span<const int> ids,
                                             bool skip_special_tokens) override {
    return tokenizer_->TokenIdsToText(ids);
  }

  std::vector<std::string> GetTokens() const override {
    return tokenizer_->GetTokens();
  }

  int GetVocabSize() const override { return tokenizer_->GetVocabSize(); }

  const ::litert::support::SentencePieceTokenizer& GetInternalTokenizer()
      const {
    return *tokenizer_;
  }

 private:
  std::unique_ptr<::litert::support::SentencePieceTokenizer> tokenizer_;
};

class LlgFcToolCallsTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer_or =
        ::litert::support::SentencePieceTokenizer::CreateFromFile(
            "runtime/components/testdata/"
            "gemma4_sentencepiece.model");
    ASSERT_OK(tokenizer_or);
    tokenizer_ =
        std::make_unique<TestSentencePieceTokenizer>(std::move(*tokenizer_or));
  }

  std::unique_ptr<TestSentencePieceTokenizer> tokenizer_;
  LlGuidanceConfig config_{
      .eos_id = 1,
      .special_tokens = {"<|tool_call>", "<tool_call|>", "<|tool_response>",
                         "<tool_response|>", "<|\"|>"}
  };

  LlgConstraintsOptions GetDefaultFcOptions(LlgConstraintMode mode) {
    LlgConstraintsOptions options;
    options.funcall_format = FuncallFormat::kFc;
    options.constraint_mode = mode;

    // Gemma 4 token expectations:
    options.code_fence_start = "<|tool_call>";
    options.code_fence_end = "<tool_call|>";
    options.function_response_start = "<|tool_response>";
    options.open_quote = "<|\"|>";
    options.close_quote = "<|\"|>";

    return options;
  }

  absl::StatusOr<bool> AcceptsInternal(Constraint& constraint,
                                       absl::string_view text) {
    ABSL_ASSIGN_OR_RETURN(TokenIds ids, tokenizer_->TextToTokenIds(text));
    auto state = constraint.Start();
    for (int i = 0; i < ids.size(); ++i) {
      int id = ids[i];
      ABSL_ASSIGN_OR_RETURN(auto bitmap, constraint.ComputeBitmap(*state));

      if (!bitmap->Get(id)) {
        return false;
      }
      ABSL_ASSIGN_OR_RETURN(state, constraint.ComputeNext(*state, id));
    }
    ABSL_ASSIGN_OR_RETURN(auto final_bitmap, constraint.ComputeBitmap(*state));
    return final_bitmap->Get(*config_.eos_id);
  }

  void AssertAccepts(Constraint& constraint, absl::string_view text) {
    auto accepts_or = AcceptsInternal(constraint, text);
    if (!accepts_or.ok()) {
      ADD_FAILURE() << "AcceptsInternal failed for text: \"" << text
                    << "\"\nStatus: " << accepts_or.status();
      return;
    }
    if (!*accepts_or) {
      ADD_FAILURE() << "Constraint failed to ACCEPT text: \""
                    << absl::Utf8SafeCEscape(text) << "\"";
    }
  }

  void AssertRejects(Constraint& constraint, absl::string_view text) {
    auto accepts_or = AcceptsInternal(constraint, text);
    if (!accepts_or.ok() || !*accepts_or) return;
    if (*accepts_or) {
      ADD_FAILURE() << "Constraint failed to REJECT text: \""
                    << absl::Utf8SafeCEscape(text) << "\"";
    }
  }

  std::unique_ptr<Constraint> CreateConstraint(
      const nlohmann::ordered_json& tools,
      const LlgConstraintsOptions& options) {
    auto provider_status_or =
        LlgConstraintProvider::Create(*tokenizer_, config_);
    if (!provider_status_or.ok()) {
      ADD_FAILURE() << "Failed to create provider: "
                    << provider_status_or.status();
      return nullptr;
    }
    auto provider = std::move(*provider_status_or);

    auto res = CreateLarkGrammarForFcToolCalls(tools, options);
    EXPECT_OK(res);
    if (!res.ok()) return nullptr;

    auto constraint_status_or = provider->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kLark,
                                .constraint_string = *res});
    if (!constraint_status_or.ok()) {
      ADD_FAILURE() << "Failed to create constraint: "
                    << constraint_status_or.status();
      return nullptr;
    }
    return std::move(*constraint_status_or);
  }
};

TEST_F(LlgFcToolCallsTest, TextOnly) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextOnly));

  AssertAccepts(*constraint, "This is just plain text.");
  AssertAccepts(*constraint, "Some html tags <div>some text</div>");
  AssertRejects(*constraint,
                "Something <|tool_call>call:get_weather{}<tool_call|>");
}

TEST_F(LlgFcToolCallsTest, TextAndOrFunctionCalls) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tool2 = nlohmann::ordered_json::parse(R"json({
    "name": "find_movies",
    "parameters": {
      "type": "object",
      "properties": {
        "genres": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1, tool2});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextAndOrFunctionCalls));

  // Text only
  AssertAccepts(*constraint, "A normal text");
  // Single function call.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)");
  // Single function call with text before.
  AssertAccepts(
      *constraint,
      R"(Some normal text<|tool_call>call:find_movies{genres:[<|"|>Action<|"|>]}<tool_call|><|tool_response>)");
  // Multiple function calls.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>}<tool_call|><|tool_call>call:find_movies{genres:[<|"|>Action<|"|>]}<tool_call|><|tool_response>)");
  // Multiple function calls with text before.
  AssertAccepts(
      *constraint,
      R"(Some normal text ... <|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_call>call:find_movies{genres:[<|"|>Action<|"|>,<|"|>Comedy<|"|>]}<tool_call|><|tool_response>)");

  // Rejects function call without <|tool_response> suffix.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>)");
  // Rejects function call with wrong function name.
  AssertRejects(*constraint,
                R"(<|tool_call>call:get_weath{}<tool_call|><|tool_response>)");
  // Rejects function call with extra text after it.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{}<tool_call|><|tool_response>extra text)");
}

TEST_F(LlgFcToolCallsTest, FunctionCallsOnly) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tool2 = nlohmann::ordered_json::parse(R"json({
    "name": "find_movies",
    "parameters": {
      "type": "object",
      "properties": {
        "genres": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    }
  })json");
  nlohmann::ordered_json tool3 = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");
  nlohmann::ordered_json tool4 = nlohmann::ordered_json::parse(R"json({
    "name": "set_timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration": {
          "type": "integer"
        },
        "sound": {
          "type": "boolean"
        }
      },
      "required": ["duration"]
    }
  })json");
  nlohmann::ordered_json tools =
      nlohmann::ordered_json::array({tool1, tool2, tool3, tool4});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Single function call.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)");
  // Single function call without params.
  AssertAccepts(*constraint,
                R"(<|tool_call>call:get_time{}<tool_call|><|tool_response>)");
  // Multiple function calls with different primitive parameters.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:find_movies{genres:[<|"|>Action<|"|>]}<tool_call|><|tool_call>call:set_timer{duration:10,sound:true}<tool_call|><|tool_call>call:set_timer{duration:5,sound:false}<tool_call|><|tool_response>)");

  // Rejects Text only
  AssertRejects(*constraint, "A normal text");
  // Rejects single function call with text before
  AssertRejects(
      *constraint,
      R"(Some normal text<|tool_call>call:find_movies{genres:[<|"|>Action<|"|>,<|"|>Comedy<|"|>]}<tool_call|><|tool_response>)");
  // Rejects multiple function calls with text before.
  AssertRejects(
      *constraint,
      R"(Some normal text <|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_call>call:find_movies{genres:[<|"|>Action<|"|>,<|"|>Comedy<|"|>]}<tool_call|><|tool_response>)");
  // Rejects function call without <|tool_response> suffix.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>)");
  // Rejects function call with wrong function name.
  AssertRejects(*constraint,
                R"(<|tool_call>call:get_weath{}<tool_call|><|tool_response>)");
  // Rejects function call with extra text after it.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{}<tool_call|><|tool_response>extra text)");
}

TEST_F(LlgFcToolCallsTest, EmptyTools_TextOnly_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextOnly));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(*constraint,
                "Text with <|tool_call>call:some_tool{}<tool_call|>");
}

TEST_F(LlgFcToolCallsTest, EmptyTools_TextAndOrFunctionCalls_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextAndOrFunctionCalls));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(*constraint,
                "Text with <|tool_call>call:some_tool{}<tool_call|>");
}

TEST_F(LlgFcToolCallsTest, EmptyTools_FunctionCallsOnly_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto res = CreateLarkGrammarForFcToolCalls(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));
  EXPECT_THAT(res, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LlgFcToolCallsTest, ParameterNameConstraint) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept valid parameters.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>}<tool_call|><|tool_response>)");
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>)");

  // Reject unexpected parameter name.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>,extra:<|"|>data<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, NoParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept valid call.
  AssertAccepts(*constraint,
                R"(<|tool_call>call:get_time{}<tool_call|><|tool_response>)");

  // Reject call with unexpected parameters.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_time{timezone:<|"|>PST<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, RequiredParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept with required parameter.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>}<tool_call|><|tool_response>)");

  // Reject missing required parameter.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, OptionalParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "ping",
    "parameters": {
      "type": "object",
      "properties": {
        "timeout": { "type": "integer" }
      }
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Valid without optional parameter.
  AssertAccepts(*constraint,
                R"(<|tool_call>call:ping{}<tool_call|><|tool_response>)");

  // Valid with optional parameter.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:ping{timeout:5}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, RequiredAndOptionalParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "set_timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration": {
          "type": "integer"
        },
        "sound": {
          "type": "boolean"
        }
      },
      "required": ["duration"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept with required parameter and optional parameter.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10,sound:true}<tool_call|><|tool_response>)");

  // Accept with required parameter only.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10}<tool_call|><|tool_response>)");

  // Reject with optional parameter only.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_timer{sound:true}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, PrimitiveTypes) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "set_timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration": {
          "type": "integer"
        },
        "sound": {
          "type": "boolean"
        }
      },
      "required": ["duration"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept valid types.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10}<tool_call|><|tool_response>)");

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10,sound:true}<tool_call|><|tool_response>)");

  // Reject invalid type (string instead of integer).
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:<|"|>10<|"|>,sound:true}<tool_call|><|tool_response>)");

  // Reject invalid type (string instead of boolean).
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10,sound:<|"|>true<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, EnumParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "set_device_state",
    "parameters": {
      "type": "object",
      "properties": {
        "device": {
          "type": "string"
        },
        "state": {
          "type": "string",
          "enum": ["on", "off"]
        }
      },
      "required": ["device", "state"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept valid enum value.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_device_state{device:<|"|>light<|"|>,state:<|"|>on<|"|>}<tool_call|><|tool_response>)");

  // Reject invalid enum value.
  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_device_state{device:<|"|>light<|"|>,state:<|"|>dimmed<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, FcArgumentTypes) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "complex_tool",
    "parameters": {
      "type": "object",
      "properties": {
        "str": { "type": "string" },
        "num": { "type": "number" },
        "int": { "type": "integer" },
        "bool": { "type": "boolean" },
        "list": { "type": "array", "items": { "type": "string" } },
        "dict": { "type": "object", "additionalProperties": { "type": "string" } },
        "null_val": { "type": "null" }
      },
      "required": ["str", "num", "int", "bool", "list", "dict", "null_val"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accepts all supported types.
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:complex_tool{str:<|"|>abc<|"|>,num:1.2,int:3,bool:true,list:[<|"|>a<|"|>,<|"|>b<|"|>],dict:{k:<|"|>v<|"|>},null_val:null}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, FcNullValue) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string" }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertRejects(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>MV<|"|>,unit:null}<tool_call|><|tool_response>)");

  nlohmann::ordered_json tool2 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": ["string", "null"] }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools2 = nlohmann::ordered_json::array({tool2});
  auto constraint2 = CreateConstraint(
      tools2, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint2,
      R"(<|tool_call>call:get_weather{location:<|"|>MV<|"|>,unit:null}<tool_call|><|tool_response>)");

  nlohmann::ordered_json tool3 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": ["string", "null"] }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools3 = nlohmann::ordered_json::array({tool3});
  auto constraint3 = CreateConstraint(
      tools3, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint3,
      R"(<|tool_call>call:get_weather{location:null}<tool_call|><|tool_response>)");
  AssertRejects(
      *constraint3,
      R"(<|tool_call>call:get_weather{}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, FcNestedStructures) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "nested_tool",
    "parameters": {
      "type": "object",
      "properties": {
        "nested_arr": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "integer" } }
        },
        "nested_obj": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "additionalProperties": { "type": "string" }
          }
        }
      }
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:nested_tool{nested_arr:[[1,2],[3]],nested_obj:{a:{b:<|"|>c<|"|>}}}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, RequiredParametersStrictOrder) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "set_timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration": {
          "type": "integer"
        },
        "sound": {
          "type": "boolean"
        }
      },
      "required": ["sound", "duration"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{sound:true,duration:10}<tool_call|><|tool_response>)");

  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10,sound:true}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, RequiredParametersBeforeOptional) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "set_timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration": {
          "type": "integer"
        },
        "sound": {
          "type": "boolean"
        }
      },
      "required": ["duration"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:set_timer{duration:10,sound:true}<tool_call|><|tool_response>)");

  AssertRejects(
      *constraint,
      R"(<|tool_call>call:set_timer{sound:true,duration:10}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, OptionalParametersFlexibleOrder) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "search",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "filter": { "type": "string" }
      }
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:search{query:<|"|>cat<|"|>,filter:<|"|>images<|"|>}<tool_call|><|tool_response>)");
  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:search{filter:<|"|>images<|"|>,query:<|"|>cat<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, DuplicateOptionalParametersAllowed) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "search",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "filter": { "type": "string" }
      },
      "required": ["query"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:search{query:<|"|>cat<|"|>,filter:<|"|>images<|"|>,filter:<|"|>videos<|"|>}<tool_call|><|tool_response>)");

  AssertRejects(
      *constraint,
      R"(<|tool_call>call:search{query:<|"|>cat<|"|>,query:<|"|>dog<|"|>}<tool_call|><|tool_response>)");

  AssertRejects(
      *constraint,
      R"(<|tool_call>call:search{filter:<|"|>images<|"|>,query:<|"|>cat<|"|>}<tool_call|><|tool_response>)");
}

TEST_F(LlgFcToolCallsTest, MultipleFunctionCalls) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tool2 = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1, tool2});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_weather{location:<|"|>Mountain View<|"|>}<tool_call|><|tool_call>call:get_time{}<tool_call|><|tool_response>)");

  AssertAccepts(
      *constraint,
      R"(<|tool_call>call:get_time{}<tool_call|><|tool_call>call:get_time{}<tool_call|><|tool_response>)");
}

}  // namespace
}  // namespace litert::lm