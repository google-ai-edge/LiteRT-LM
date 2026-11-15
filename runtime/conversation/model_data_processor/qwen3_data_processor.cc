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

#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Qwen3DataProcessor::Create(
    Qwen3DataProcessorConfig config, std::optional<Preface> preface) {
  return absl::WrapUnique(
      new Qwen3DataProcessor(std::move(config), std::move(preface)));
}

absl::StatusOr<Message> Qwen3DataProcessor::ToMessageImpl(
    const Responses& responses, const Qwen3DataProcessorArguments& args) const {
  return ResponseTextToMessage(
      responses.GetTexts()[0], preface_, config_.code_fence_start,
      config_.code_fence_end, SyntaxType::kJson,
      {.escape_fence_strings = config_.escape_fence_strings,
       .tool_code_regex = config_.tool_code_regex,
       .return_error_on_parse_failure = ReturnErrorOnParseFailure()});
}

absl::string_view Qwen3DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Qwen3DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
