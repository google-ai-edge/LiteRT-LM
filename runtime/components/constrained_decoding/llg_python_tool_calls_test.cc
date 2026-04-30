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

#include "runtime/components/constrained_decoding/llg_python_tool_calls.h"

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

namespace litert::lm {
namespace {

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

class LlgPythonToolCallsTest : public testing::Test {
 protected:
  SimpleTokenizer tokenizer_;
  LlGuidanceConfig config_{.eos_id = 1};

  LlgConstraintsOptions GetDefaultPythonOptions(LlgConstraintMode mode) {
    LlgConstraintsOptions options;
    options.funcall_format = FuncallFormat::kPython;
    options.code_fence_start = "```tool_code\n";
    options.code_fence_end = "\n```";
    options.open_quote = "\"";
    options.close_quote = "\"";
    options.constraint_mode = mode;
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
    auto provider_or = LlgConstraintProvider::Create(tokenizer_, config_);
    if (!provider_or.ok()) {
      ADD_FAILURE() << "Failed to create provider: " << provider_or.status();
      return nullptr;
    }
    auto provider = std::move(*provider_or);

    auto res = CreateLarkGrammarForPythonToolCalls(tools, options);
    if (!res.ok()) {
      ADD_FAILURE() << "Failed to create grammar: " << res.status();
      return nullptr;
    }
    auto constraint_or = provider->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kLark,
                                .constraint_string = *res});
    if (!constraint_or.ok()) {
      ADD_FAILURE() << "Failed to create constraint: "
                    << constraint_or.status();
      return nullptr;
    }
    return std::move(*constraint_or);
  }
};

TEST_F(LlgPythonToolCallsTest, PythonFunctionCalls) {
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
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View",unit="celsius")
```)");

  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="SF")
get_weather(location="LA", unit="fahrenheit")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonTextAndFunctionCalls) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kTextAndOrFunctionCalls);

  auto constraint = CreateConstraint(tools, options);

  // Accepts text then function call.
  AssertAccepts(*constraint,
                R"(I will get the weather for Mountain View.
```tool_code
get_weather(location="Mountain View")
```)");

  // Accepts only text.
  AssertAccepts(*constraint, "Just some text.");

  // Accepts only function call.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonQuotes) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accepts arguments with single quotes.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location='Mountain View')
```)");

  // Accepts arguments with double quotes.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonFlexibleWhitespace) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accepts whitespace between symbols.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather( location = "MV" , unit = "C" )
```)");

  // Accepts minimal whitespace.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="MV",unit="C" )
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonParameterNameConstraint) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept valid parameters.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View")
```)");
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View", unit="celsius")
```)");

  // Reject unexpected parameter name.
  AssertRejects(*constraint,
                R"(```tool_code
get_weather(location="Mountain View", extra="data")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonNoParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept valid call.
  AssertAccepts(*constraint,
                R"(```tool_code
get_time()
```)");

  // Reject call with unexpected parameters.
  AssertRejects(*constraint,
                R"(```tool_code
get_time(timezone="PST")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonRequiredParameter) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept with required parameter.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View")
```)");

  // Reject missing required parameter.
  AssertRejects(*constraint,
                R"(```tool_code
get_weather()
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonOptionalParameter) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Valid without optional parameter.
  AssertAccepts(*constraint,
                R"(```tool_code
ping()
```)");

  // Valid with optional parameter.
  AssertAccepts(*constraint,
                R"(```tool_code
ping(timeout=5)
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonEnumParameters) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept valid enum value.
  AssertAccepts(*constraint,
                R"(```tool_code
set_device_state(device="light", state="on")
```)");

  // Reject invalid enum value.
  AssertRejects(*constraint,
                R"(```tool_code
set_device_state(device="light", state="dimmed")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonRequiredParametersStrictOrder) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept valid order: sound then duration
  AssertAccepts(*constraint,
                R"(```tool_code
set_timer(sound=True, duration=10)
```)");

  // Reject invalid order: duration then sound
  AssertRejects(*constraint,
                R"(```tool_code
set_timer(duration=10, sound=True)
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonRequiredParametersBeforeOptional) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept valid order (required parameter first).
  AssertAccepts(*constraint,
                R"(```tool_code
set_timer(duration=10, sound=True)
```)");

  // Reject optional parameter before required parameter.
  AssertRejects(*constraint,
                R"(```tool_code
set_timer(sound=True, duration=10)
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonOptionalParametersFlexibleOrder) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Valid calls with optional parameters in different order.
  AssertAccepts(*constraint,
                R"(```tool_code
search(query="cat", filter="images")
```)");
  AssertAccepts(*constraint,
                R"(```tool_code
search(filter="images", query="cat")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonDuplicateOptionalParametersAllowed) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Valid calls with optional duplicate
  AssertAccepts(*constraint,
                R"(```tool_code
search(query="cat", filter="images", filter="videos")
```)");

  // Invalid (duplicate required parameter)
  AssertRejects(*constraint,
                R"(```tool_code
search(query="cat", query="dog")
```)");

  // Invalid (optional before required)
  AssertRejects(*constraint,
                R"(```tool_code
search(filter="images", query="cat")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonMultipleFunctionCallsDifferentTools) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accept multiple different function calls
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="Mountain View")
get_time()
```)");

  // Accept multiple same function calls
  AssertAccepts(*constraint,
                R"(```tool_code
get_time()
get_time()
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonRequiredAndOptionalArguments) {
  nlohmann::ordered_json tool1 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string" },
        "fake": { "type": "string" }
      },
      "required": ["location", "unit"]
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool1});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accepts tool calls with only required arguments.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="MV",unit="C")
```)");

  // Accepts tool calls with required and optional arguments.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="MV",unit="C",fake="yes")
```)");

  // Rejects tool calls with missing required arguments.
  AssertRejects(*constraint,
                R"(```tool_code
get_weather(location="MV")
```)");

  // Rejects wrong order for required arguments.
  AssertRejects(*constraint,
                R"(```tool_code
get_weather(unit="C",location="MV")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonArgumentTypes) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Accepts all supported types.
  AssertAccepts(*constraint,
                R"(```tool_code
complex_tool(str="abc",num=1.2,int=3,bool=True,list=["a","b"],dict={"k":"v"},null_val=None)
```)");

  // Accepts types with alternative syntax.
  AssertAccepts(*constraint,
                R"(```tool_code
complex_tool(str='abc',num=-1,int=0,bool=False,list=[],dict={},null_val=None)
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonOptionalNone) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  // Optional parameter can be None.
  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="MV",unit=None)
```)");

  // Required parameter cannot be None.
  AssertRejects(*constraint,
                R"(```tool_code
get_weather(location=None)
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonNestedStructures) {
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

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  AssertAccepts(*constraint,
                R"(```tool_code
nested_tool(nested_arr=[[1,2],[3]], nested_obj={"a":{"b":"c"}})
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonEscapedStrings) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "say_it",
    "parameters": {
      "type": "object",
      "properties": { "text": { "type": "string" } }
    }
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  AssertAccepts(*constraint,
                R"(```tool_code
say_it(text="He said \"Hello\"")
```)");
}

TEST_F(LlgPythonToolCallsTest, PythonNullValue) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
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
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  LlgConstraintsOptions options =
      GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly);

  auto constraint = CreateConstraint(tools, options);

  AssertAccepts(*constraint,
                R"(```tool_code
get_weather(location="MV", unit=None)
```)");

  nlohmann::ordered_json tool2 = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": ["string", "null"] }
      },
      "required": ["location"]
    }
  })json");
  nlohmann::ordered_json tools2 = nlohmann::ordered_json::array({tool2});
  auto constraint2 = CreateConstraint(
      tools2, GetDefaultPythonOptions(LlgConstraintMode::kFunctionCallsOnly));

  AssertAccepts(*constraint2,
                R"(```tool_code
get_weather(location=None)
```)");
  AssertRejects(*constraint2,
                R"(```tool_code
get_weather()
```)");
}

}  // namespace
}  // namespace litert::lm
