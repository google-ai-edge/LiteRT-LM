// Copyright 2026 The Google AI Edge Authors.
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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/fc_parser_utils.h"
#include "runtime/components/tool_use/json_parser_utils.h"
#include "runtime/components/tool_use/python_parser_utils.h"

namespace litert::lm {

absl::StatusOr<nlohmann::ordered_json> ParsePythonExpression(
    absl::string_view text) {
  return absl::UnimplementedError(
      "Tool use parsing is disabled in this build.");
}

absl::StatusOr<nlohmann::ordered_json> ParseJsonExpression(
    absl::string_view text) {
  return absl::UnimplementedError(
      "Tool use parsing is disabled in this build.");
}

absl::StatusOr<nlohmann::ordered_json> ParseFcExpression(
    absl::string_view text) {
  return absl::UnimplementedError(
      "Tool use parsing is disabled in this build.");
}

}  // namespace litert::lm
