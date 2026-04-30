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

#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

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
#include "runtime/components/tokenizer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

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

class LlguidanceSchemaUtilsTest : public testing::Test {
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

    auto res = CreateLarkGrammarForTools(tools, options);
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

TEST_F(LlguidanceSchemaUtilsTest, TextOnly) {
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
      tools, GetDefaultFcOptions(LlgConstraintMode::kTextOnly));

  AssertAccepts(*constraint, "A normal text");
  AssertRejects(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_response>)");
}

TEST_F(LlguidanceSchemaUtilsTest, PythonTextAndFunctionCalls) {
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

}  // namespace
}  // namespace litert::lm
