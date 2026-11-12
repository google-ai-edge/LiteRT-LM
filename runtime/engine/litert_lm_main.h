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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_MAIN_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_MAIN_H_

#include <iostream>
#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl

ABSL_DECLARE_FLAG(std::string, backend);
ABSL_DECLARE_FLAG(std::string, model_path);
ABSL_DECLARE_FLAG(std::string, input_prompt);
ABSL_DECLARE_FLAG(std::string, input_prompt_file);

namespace litert::lm {

// Runs the LiteRT-LM CLI flow using currently parsed Abseil flags, writing
// model output to `out` and user-facing diagnostics to `err`. Returns
// `EXIT_SUCCESS` on success and `EXIT_FAILURE` on error.
int RunLiteRtLmFromFlags(std::ostream& out = std::cout,
                         std::ostream& err = std::cerr);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_MAIN_H_
