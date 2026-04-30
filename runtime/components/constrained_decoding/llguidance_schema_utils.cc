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

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"
#include "runtime/components/constrained_decoding/llg_python_tool_calls.h"

namespace litert::lm {

absl::StatusOr<std::string> CreateLarkGrammarForTools(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  switch (options.funcall_format) {
    case FuncallFormat::kFc:
      return CreateLarkGrammarForFcToolCalls(tools, options);
    case FuncallFormat::kPython:
      return CreateLarkGrammarForPythonToolCalls(tools, options);
  }
  return absl::InvalidArgumentError("Unknown function call format.");
}

}  // namespace litert::lm
