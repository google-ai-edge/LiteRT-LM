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

#include "runtime/conversation/model_data_processor/minicpm5_data_processor.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpm5_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

constexpr absl::string_view kCdataPrefix = "<![CDATA[";
constexpr absl::string_view kCdataSuffix = "]]>";
constexpr size_t kMinCdataLength = kCdataPrefix.size() + kCdataSuffix.size();

}  // namespace

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>

MiniCpm5DataProcessor::Create(MiniCpm5DataProcessorConfig config,
                              std::optional<Preface> preface) {
  return absl::WrapUnique(
      new MiniCpm5DataProcessor(std::move(config), std::move(preface)));
}

absl::StatusOr<nlohmann::ordered_json>
MiniCpm5DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_array()) {
    const auto& content = message["content"];
    if (content.size() == 1 && content[0].contains("text")) {
      auto result = nlohmann::ordered_json::object(
          {{"role", message["role"]}, {"content", content[0]["text"]}});
      return result;
    }
  }
  return message;
}

absl::StatusOr<std::vector<InputData>>
MiniCpm5DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const MiniCpm5DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  input_data.emplace_back(InputText(rendered_template_prompt));
  return input_data;
}

absl::StatusOr<Message> MiniCpm5DataProcessor::ToMessageImpl(
    const Responses& responses,
    const MiniCpm5DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json message = {{"role", "assistant"}};

  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    std::string text_str(response_text);
    size_t start_pos = text_str.find("<function");
    if (start_pos != std::string::npos) {
      static const LazyRE2 kFuncRegex = {
          R"regex(<function\s+name\s*=\s*["']([^"']+)["']\s*>)regex"};
      std::string func_name;
      absl::string_view search_view(text_str);
      search_view.remove_prefix(start_pos);

      if (RE2::Consume(&search_view, *kFuncRegex, &func_name)) {
        nlohmann::ordered_json args_obj = nlohmann::ordered_json::object();

        static const LazyRE2 kParamRegex = {
            R"regex((?s)<param\s+name\s*=\s*["']([^"']+)["']\s*>(.*?)</param>)regex"};
        std::string pname, pval;
        while (RE2::FindAndConsume(&search_view, *kParamRegex, &pname, &pval)) {
          // Handle CDATA if present
          if (pval.size() >= kMinCdataLength &&
              pval.starts_with(kCdataPrefix) && pval.ends_with(kCdataSuffix)) {
            pval = pval.substr(kCdataPrefix.size(),
                               pval.size() - kMinCdataLength);
          }
          args_obj[pname] = pval;
        }

        std::string prefix_content = text_str.substr(0, start_pos);
        if (!prefix_content.empty()) {
          message["content"] = nlohmann::ordered_json::array(
              {{{"type", "text"}, {"text", prefix_content}}});
        }
        message["tool_calls"] = nlohmann::ordered_json::array(
            {{{"type", "function"},
              {"function", {{"name", func_name}, {"arguments", args_obj}}}}});
        return message;
      }
    }
  }

  message["content"] = nlohmann::ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return message;
}

absl::string_view MiniCpm5DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view MiniCpm5DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
