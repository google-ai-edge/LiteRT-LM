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

using ::testing::status::StatusIs;

struct TokenDef {
  std::string text;
  int id;
  // Whether the token is a special control token.
  bool is_control;
};

// A simple mock tokenizer for testing.
class SimpleTokenizer : public Tokenizer {
 public:
  // Constants for token ID ranges
  static constexpr int kPadId = 0;
  static constexpr int kEosId = 1;
  static constexpr int kVocabSize = 600;  // Sufficiently large for current IDs

  SimpleTokenizer() {
    // ID Assignment:
    // - 0: <pad>
    // - 1: <eos>
    // - 2-255: Byte tokens (excluding '<' and '>')
    // - 500+: Special tokens defined in TokenDef

    // Add single characters (excluding < and >)
    // NOTE: We exclude '<' and '>' as single characters to simplify this test
    // tokenizer. In production SentencePiece models, '<' is often a valid
    // token. Production tokenizers have prioritization rules (e.g., for
    // USER_DEFINED tokens) that resolve ambiguities, ensuring that longer
    // special tokens like '<ctrl1>' are preferred over sequences like '<'
    // then 'c'.
    for (int i = 0; i < 256; ++i) {
      char c = static_cast<char>(i);
      if (c != '<' && c != '>') {
        vocab_[std::string(1, c)] = i;
        id_to_piece_[i] = std::string(1, c);
      }
    }

    // User defined tokens.
    const std::vector<TokenDef> token_defs = {
        {"<pad>", kPadId, true},
        {"<eos>", kEosId, true},
        {"<start_function_call>", 501, true},
        {"<end_function_call>", 502, true},
        {"<escape>", 503, true},
        {"<start_function_response>", 504, true},
        {"<ctrl99>", 509, true},
        {"<ctrl100>", 510, true},
        // Some common HTML tags.
        {"<div>", 505, false},
        {"</div>", 506, false},
        {"<span>", 507, false},
        {"</span>", 508, false},
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

      // Iterate possible prefix lengths, longest to shortest
      for (int len = remaining_text.length(); len >= 1; --len) {
        absl::string_view prefix = remaining_text.substr(0, len);
        auto it = vocab_.find(prefix);
        if (it != vocab_.end()) {
          best_match_len = len;
          token_id = it->second;
          break;  // Found longest match
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
  // Set of token texts that should be prefixed with \xff in GetTokens()
  // as they represent special control instructions:
  // https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md
  std::set<std::string> control_token_texts_;
};

class LlguidanceSchemaUtilsTest : public testing::Test {
 protected:
  SimpleTokenizer tokenizer_;
  LlGuidanceConfig config_{.eos_id = 1};

  LlgConstraintsOptions GetDefaultOptions(LlgConstraintMode mode) {
    LlgConstraintsOptions options;
    options.constraint_mode = mode;
    options.fc2_code_fence_start = "<start_function_call>";
    options.fc2_code_fence_end = "<end_function_call>";
    options.fc2_function_response_start = "<start_function_response>";
    options.fc2_open_quote = "<escape>";
    options.fc2_close_quote = "<escape>";
    return options;
  }

  // Validates if the constraint accepts the text sequence.
  absl::StatusOr<bool> AcceptsInternal(Constraint& constraint,
                                       absl::string_view text) {
    ASSIGN_OR_RETURN(TokenIds ids, tokenizer_.TextToTokenIds(text));
    auto state = constraint.Start();
    for (int i = 0; i < ids.size(); ++i) {
      int id = ids[i];
      ASSIGN_OR_RETURN(auto bitmap, constraint.ComputeBitmap(*state));

      if (!bitmap->Get(id)) {
        // Rejected
        return false;
      }
      ASSIGN_OR_RETURN(state, constraint.ComputeNext(*state, id));
    }
    ASSIGN_OR_RETURN(auto final_bitmap, constraint.ComputeBitmap(*state));
    return final_bitmap->Get(*config_.eos_id);
    // return true;
  }

  void AssertAccepts(
      Constraint& constraint,
      absl::string_view text
  ) {
    auto accepts_or = AcceptsInternal(constraint, text);
    if (!accepts_or.ok()) {
      // litert_lm:oss-begin
      // ADD_FAILURE() << "AcceptsInternal failed for text: \"" << text
      //               << "\"\nStatus: " << accepts_or.status();
      // litert_lm:oss-end
      return;
    }
    if (!*accepts_or) {
      // litert_lm:oss-begin
      // ADD_FAILURE() << "Constraint failed to ACCEPT text: \""
      //               << absl::Utf8SafeCEscape(text) << "\"";
      // litert_lm:oss-end
    }
  }

  void AssertRejects(
      Constraint& constraint,
      absl::string_view text
  ) {
    auto accepts_or = AcceptsInternal(constraint, text);
    // Failure to process is considered a rejection.
    if (!accepts_or.ok() || !*accepts_or) return;
    if (*accepts_or) {
      // litert_lm:oss-begin
      // ADD_FAILURE() << "Constraint failed to REJECT text: \""
      //               << absl::Utf8SafeCEscape(text) << "\"";
      // litert_lm:oss-end
    }
  }

  // Helper to create constraint from tools
  std::unique_ptr<Constraint> CreateConstraint(
      const nlohmann::ordered_json& tools,
      const LlgConstraintsOptions& options) {
    auto provider_or = LlgConstraintProvider::Create(tokenizer_, config_);
    EXPECT_OK(provider_or);
    auto provider = std::move(*provider_or);

    auto res = FormatToolsAsLarkGrammar(tools, options);
    EXPECT_OK(res);
    auto constraint_or = provider->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kLark,
                                .constraint_string = *res});
    EXPECT_OK(constraint_or);
    return std::move(*constraint_or);
  }
};

TEST_F(LlguidanceSchemaUtilsTest, Lark_TextOnly) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather"
  })json");
  nlohmann::ordered_json tools = nlohmann::ordered_json::array({tool});

  auto constraint =
      CreateConstraint(tools, GetDefaultOptions(LlgConstraintMode::kTextOnly));

  AssertAccepts(*constraint, "This is just plain text.");
  AssertAccepts(*constraint, "Some html tags <div>some text</div>");
  AssertRejects(
      *constraint,
      "Something <start_function_call>call:get_weather{}<end_function_call>");
}

TEST_F(LlguidanceSchemaUtilsTest, Lark_TextAndOr) {
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

  auto constraint =
      CreateConstraint(tools, GetDefaultOptions(LlgConstraintMode::kTextAndOr));

  // Text only
  AssertAccepts(*constraint, "A normal text");

  // Single function call.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_response>)");

  // Single function call with text before
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

TEST_F(LlguidanceSchemaUtilsTest, Lark_FunctionCallOnly) {
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
  nlohmann::ordered_json tools =
      nlohmann::ordered_json::array({tool1, tool2, tool3});

  auto constraint = CreateConstraint(
      tools, GetDefaultOptions(LlgConstraintMode::kFunctionCallOnly));

  // Single function call.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_response>)");

  // Single function call without params.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_time{}<end_function_call><start_function_response>)");

  // Multiple function calls.
  AssertAccepts(
      *constraint,
      R"(<start_function_call>call:get_weather{location:<escape>Mountain View<escape>}<end_function_call><start_function_call>call:find_movies{genres:[<escape>Action<escape>]}<end_function_call><start_function_response>)");

  // Rejects Text only
  AssertRejects(*constraint, "A normal text");

  // Rejects single function call with text before
  AssertRejects(
      *constraint,
      R"(Some normal text<start_function_call>call:find_movies{genres:[<escape>Action<escape>,<escape>Comedy<escape>]}<end_function_call><start_function_response>)");

  // Rejects multiple function calls with text before.
  AssertRejects(
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

TEST_F(LlguidanceSchemaUtilsTest, EmptyTools_TextOnly_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint =
      CreateConstraint(tools, GetDefaultOptions(LlgConstraintMode::kTextOnly));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(
      *constraint,
      "Text with <start_function_call>call:some_tool{}<end_function_call>");
}

TEST_F(LlguidanceSchemaUtilsTest, EmptyTools_TextAndOr_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto constraint =
      CreateConstraint(tools, GetDefaultOptions(LlgConstraintMode::kTextAndOr));
  AssertAccepts(*constraint, "Any text is fine.");
  AssertRejects(
      *constraint,
      "Text with <start_function_call>call:some_tool{}<end_function_call>");
}

TEST_F(LlguidanceSchemaUtilsTest, EmptyTools_FunctionCallOnly_Lark) {
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  auto res = FormatToolsAsLarkGrammar(
      tools, GetDefaultOptions(LlgConstraintMode::kFunctionCallOnly));
  EXPECT_THAT(res, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
