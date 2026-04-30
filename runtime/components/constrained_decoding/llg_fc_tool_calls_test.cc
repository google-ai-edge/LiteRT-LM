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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

struct TokenDef {
  std::string text;
  int id;
  bool is_control;
};

class SimpleTokenizer : public Tokenizer {
 public:
  static constexpr int kPadId = 0;
  static constexpr int kEosId = 1;
  static constexpr int kVocabSize = 600;

  SimpleTokenizer() {
    for (int i = 0; i < 256; ++i) {
      char c = static_cast<char>(i);
      if (c != '<' && c != '>') {
        vocab_[std::string(1, c)] = i;
        id_to_piece_[i] = std::string(1, c);
      }
    }

    const std::vector<TokenDef> token_defs = {
        {"<pad>", kPadId, true},
        {"<eos>", kEosId, true},
        {"<start_function_call>", 501, true},
        {"<end_function_call>", 502, true},
        {"<escape>", 503, true},
        {"<start_function_response>", 504, true},
        {"<div>", 505, false},
        {"</div>", 506, false},
    };

    for (const auto& def : token_defs) {
      vocab_[def.text] = def.id;
      id_to_piece_[def.id] = def.text;
      if (def.is_control) {
        control_token_texts_.insert(def.text);
      }
    }
  }

  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kSentencePiece;
  }

  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    TokenIds ids;
    absl::string_view remaining_text = text;

    while (!remaining_text.empty()) {
      int best_match_len = 0;
      int token_id = -1;

      for (int len = remaining_text.length(); len >= 1; --len) {
        absl::string_view prefix = remaining_text.substr(0, len);
        auto it = vocab_.find(prefix);
        if (it != vocab_.end()) {
          best_match_len = len;
          token_id = it->second;
          break;
        }
      }

      if (token_id != -1) {
        ids.push_back(token_id);
        remaining_text.remove_prefix(best_match_len);
      } else {
        return absl::InternalError(
            absl::StrCat("Failed to tokenize at: ", remaining_text));
      }
    }
    return ids;
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
      return it->second;
    }
    return absl::NotFoundError(absl::StrCat("Token not found: ", token));
  }

  absl::StatusOr<std::string> TokenIdsToText(const TokenIds& ids) override {
    std::string text;
    for (int id : ids) {
      auto it = id_to_piece_.find(id);
      if (it != id_to_piece_.end()) {
        text += it->second;
      } else {
        return absl::InternalError(absl::StrCat("Unknown token ID: ", id));
      }
    }
    return text;
  }

  std::vector<std::string> GetTokens() const override {
    std::vector<std::string> tokens(kVocabSize);
    for (int i = 0; i < tokens.size(); ++i) {
      tokens[i] = "[UNUSED_" + std::to_string(i) + "]";
    }

    for (const auto& pair : id_to_piece_) {
      int id = pair.first;
      if (id >= 0 && id < kVocabSize) {
        const std::string& token_str = pair.second;
        if (control_token_texts_.count(token_str)) {
          tokens[id] = "\xff" + token_str;
        } else {
          tokens[id] = token_str;
        }
      }
    }
    return tokens;
  }

 private:
  absl::flat_hash_map<std::string, int> vocab_;
  absl::flat_hash_map<int, std::string> id_to_piece_;
  std::set<std::string> control_token_texts_;
};

class LlgFcToolCallsTest : public testing::Test {
 protected:
  SimpleTokenizer tokenizer_;
  LlGuidanceConfig config_{.eos_id = 1};

  LlgConstraintsOptions GetDefaultFcOptions(LlgConstraintMode mode) {
    LlgConstraintsOptions options;
    options.funcall_format = FuncallFormat::kFc;
    options.constraint_mode = mode;
    options.code_fence_start = "<start_function_call>";
    options.code_fence_end = "<end_function_call>";
    options.function_response_start = "<start_function_response>";
    options.open_quote = "<escape>";
    options.close_quote = "<escape>";
    return options;
  }

  absl::StatusOr<bool> AcceptsInternal(Constraint& constraint,
                                       absl::string_view text) {
    ASSIGN_OR_RETURN(TokenIds ids, tokenizer_.TextToTokenIds(text));
    auto state = constraint.Start();
    for (int i = 0; i < ids.size(); ++i) {
      int id = ids[i];
      ASSIGN_OR_RETURN(auto bitmap, constraint.ComputeBitmap(*state));

      if (!bitmap->Get(id)) {
        return false;
      }
      ASSIGN_OR_RETURN(state, constraint.ComputeNext(*state, id));
    }
    ASSIGN_OR_RETURN(auto final_bitmap, constraint.ComputeBitmap(*state));
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
        LlgConstraintProvider::Create(tokenizer_, config_);
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
  AssertRejects(
      *constraint,
      "Something <start_function_call>call:get_weather{}<end_function_call>");
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
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_response>)");
  // Single function call with text before.
  AssertAccepts(
      *constraint,
      R"(Some normal text<start_function_call>call:find_movies{genres:[<escape>Action<escape>]}<end_function_call><start_function_response>)");
  // Multiple function calls.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_call>call:find_movies{genres:[<escape>Action<escape>]}<end_function_call><start_function_response>)");
  // Multiple function calls with text before.
  AssertAccepts(
      *constraint,
      R"(Some normal text ... <start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_call>call:find_movies{genres:[<escape>Action<escape>,<escape>Comedy<escape>]}<end_function_call><start_function_response>)");

  // Rejects function call without <start_function_response> suffix.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call>)");
  // Rejects function call with wrong function name.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weath{}<end_function_call><start_function_response>)");
  // Rejects function call with extra text after it.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{}<end_function_call><start_function_response>extra text)");
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
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_response>)");
  // Single function call without params.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_time{}<end_function_call><start_function_response>)");
  // Multiple function calls with different primitive parameters.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:find_movies{genres:[<escape>Action<escape>]}<end_function_call><start_function_call>call:set_timer{duration:10,sound:true}<end_function_call><start_function_call>call:set_timer{duration:5,sound:false}<end_function_call><start_function_response>)");

  // Rejects Text only
  AssertRejects(*constraint, "A normal text");
  // Rejects single function call with text before
  AssertRejects(
      *constraint,
      R"(Some normal text<start_function_call>call:find_movies{genres:[<escape>Action<escape>,<escape>Comedy<escape>]}<end_function_call><start_function_response>)");
  // Rejects multiple function calls with text before.
  AssertRejects(
      *constraint,
      R"(Some normal text <start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_call>call:find_movies{genres:[<escape>Action<escape>,<escape>Comedy<escape>]}<end_function_call><start_function_response>)");
  // Rejects function call without <start_function_response> suffix.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call>)");
  // Rejects function call with wrong function name.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weath{}<end_function_call><start_function_response>)");
  // Rejects function call with extra text after it.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{}<end_function_call><start_function_response>extra text)");
}

TEST_F(LlgFcToolCallsTest, EmptyTools_TextOnly_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextOnly));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(
      *constraint,
      "Text with <start_function_call>call:some_tool{}<end_function_call>");
}

TEST_F(LlgFcToolCallsTest, EmptyTools_TextAndOrFunctionCalls_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextAndOrFunctionCalls));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(
      *constraint,
      "Text with <start_function_call>call:some_tool{}<end_function_call>");
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
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_response>)");
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_response>)");

  // Reject unexpected parameter name.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,extra:<escape>data<escape>}<end_function_call><start_function_response>)");
}

TEST_F(LlgFcToolCallsTest, NoParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint = CreateConstraint(
      tools, GetDefaultFcOptions(LlgConstraintMode::kFunctionCallsOnly));

  // Accept valid call.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_time{}<end_function_call><start_function_response>)");

  // Reject call with unexpected parameters.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_time{timezone:<escape>PST<escape>}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_response>)");

  // Reject missing required parameter.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{}<end_function_call><start_function_response>)");
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
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:ping{}<end_function_call><start_function_response>)");

  // Valid with optional parameter.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:ping{timeout:5}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:set_timer{duration:10,sound:true}<end_function_call><start_function_response>)");

  // Accept with required parameter only.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:set_timer{duration:10}<end_function_call><start_function_response>)");

  // Reject with optional parameter only.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_timer{sound:true}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:set_timer{duration:10}<end_function_call><start_function_response>)");

  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:set_timer{duration:10,sound:true}<end_function_call><start_function_response>)");

  // Reject invalid type (string instead of integer).
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_timer{duration:<escape>10<escape>,sound:true}<end_function_call><start_function_response>)");

  // Reject invalid type (string instead of boolean).
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_timer{duration:10,sound:<escape>true<escape>}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:set_device_state{device:<escape>light<escape>,state:<escape>on<escape>}<end_function_call><start_function_response>)");

  // Reject invalid enum value.
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_device_state{device:<escape>light<escape>,state:<escape>dimmed<escape>}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:complex_tool{str:<escape>abc<escape>,num:1.2,int:3,bool:true,list:[<escape>a<escape>,<escape>b<escape>],dict:{k:<escape>v<escape>},null_val:null}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:get_weather{location:<escape>MV<escape>,unit:null}<end_function_call><start_function_response>)");

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
      R"(<start_function_call>call:get_weather{location:<escape>MV<escape>,unit:null}<end_function_call><start_function_response>)");

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
      R"(<start_function_call>call:get_weather{location:null}<end_function_call><start_function_response>)");
  AssertRejects(
      *constraint3,
      R"(<start_function_call>call:get_weather{}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:nested_tool{nested_arr:[[1,2],[3]],nested_obj:{a:{b:<escape>c<escape>}}}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:set_timer{sound:true,duration:10}<end_function_call><start_function_response>)");

  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_timer{duration:10,sound:true}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:set_timer{duration:10,sound:true}<end_function_call><start_function_response>)");

  AssertRejects(
      *constraint,
      R"(<start_function_call>call:set_timer{sound:true,duration:10}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:search{query:<escape>cat<escape>,filter:<escape>images<escape>}<end_function_call><start_function_response>)");
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:search{filter:<escape>images<escape>,query:<escape>cat<escape>}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:search{query:<escape>cat<escape>,filter:<escape>images<escape>,filter:<escape>videos<escape>}<end_function_call><start_function_response>)");

  AssertRejects(
      *constraint,
      R"(<start_function_call>call:search{query:<escape>cat<escape>,query:<escape>dog<escape>}<end_function_call><start_function_response>)");

  AssertRejects(
      *constraint,
      R"(<start_function_call>call:search{filter:<escape>images<escape>,query:<escape>cat<escape>}<end_function_call><start_function_response>)");
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
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_call>call:get_time{}<end_function_call><start_function_response>)");

  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_time{}<end_function_call><start_function_call>call:get_time{}<end_function_call><start_function_response>)");
}

}  // namespace
}  // namespace litert::lm
